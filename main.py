# If we are on colab: this clones the repo and installs the dependencies
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, read_image
import torch
import argparse
import os
import numpy as np
from osgeo import gdal, osr, ogr
import timeit
from PIL import Image
from utils.pair_util import list_files, inference, get_center_aim, pixel_to_geolocation, visualize_and_save_matches, \
    get_m_nums
from utils.logger import Logger
import csv


def geo2pixel(geotransform, lon, lat):
    lon = float(lon)
    lat = float(lat)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)

    # coord_transform = osr.CoordinateTransformation(src_srs, osr.SpatialReference(target_srs))
    # point.Transform(coord_transform)

    x = int((point.GetX() - geotransform[0]) / geotransform[1])
    y = abs(int((geotransform[3] - point.GetY()) / geotransform[5])
)
    return x, y

def save_coordinates_to_csv(csv_file, image_name, coord):
    """将图像文件名和对应的地理坐标保存到 CSV 文件"""
    # 如果文件不存在，则创建文件并写入表头
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Image Name", "Longitude", "Latitude"])  # 表头
        writer.writerow([image_name, coord[0], coord[1]])  # 写入图像名称和对应的坐标

def crop_geotiff_by_center_point(longitude, latitude, input_tif_path, crop_size_px, crop_size_py):

    # 打开原始图像数据集
    in_ds = gdal.Open(input_tif_path)
    if in_ds is None:
        raise ValueError("无法打开输入的GeoTIFF文件")

    # 获取原数据集的地理参考信息
    geotransform = in_ds.GetGeoTransform()

    # 将经纬度坐标转换为图像坐标
    x, y = geo2pixel(geotransform, longitude, latitude)

    # 根据裁剪半径计算实际裁剪矩形框大小（这里简化为正方形裁剪）
    block_xsize = int(min(crop_size_px, in_ds.RasterXSize - x))
    block_ysize = int(min(crop_size_py, in_ds.RasterYSize - y))

    # 调整裁剪区域以确保裁剪圆心位于裁剪矩形中心
    offset_x = int(max(x - block_xsize // 2, 0))
    offset_y = int(max(y - block_ysize // 2, 0))

    # 从每个波段中读取裁剪区域的数据
    in_band1 = in_ds.GetRasterBand(1)
    in_band2 = in_ds.GetRasterBand(2)
    in_band3 = in_ds.GetRasterBand(3)
    out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band2 = in_band2.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band3 = in_band3.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)

    # 设置裁剪后图像的仿射变换参数
    top_left_x = geotransform[0] + offset_x * geotransform[1]
    top_left_y = geotransform[3] + offset_y * geotransform[5]
    dst_transform = (top_left_x, geotransform[1], geotransform[2], top_left_y, geotransform[4], geotransform[5])

    rgb_crop = np.dstack((out_band1, out_band2, out_band3))
    return rgb_crop, dst_transform, offset_x, offset_y

def parse_opt():
    parser = argparse.ArgumentParser(description="Benchmark script for LightGlue")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="device to benchmark on",
    )
    parser.add_argument(
        "--num_keypoints",
        nargs="+",
        type=int,
        default=1024,
        help="number of keypoints (list separated by spaces)",
    )
    parser.add_argument(
        "--image_ste_path", default="/home/c301/Yinpengyu/卫片/太康县/test/clipped.tif", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--image_uav_path", default="/home/c301/Yinpengyu/卫片/太康县/test/patch/", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--save_path", default="/home/c301/Yinpengyu/卫片/太康县/test/output/res_img", type=str,
        help="path where figure should be saved"
    )
    parser.add_argument(
        "--fault_path", default="/home/c301/Yinpengyu/卫片/太康县/test/output/fault_res_img", type=str,
        help="path where fault figure should be saved"
    )
    parser.add_argument(
        "--start_lon", default=114.631412, type=float, help="path where fault figure should be saved"
    )
    parser.add_argument(
        "--start_lat", default=34.195191, type=float, help="path where fault figure should be saved"
    )
    parser.add_argument(
        "--test_num", default=10, type=int, help="the maximum number of test images to be processed"
    )
    args = parser.parse_args()
    return args


def main():
    logger = Logger(log_file=os.path.join(args.save_path, 'log.txt'))  # 创建日志工具，指定输出文件
    logger.log(f"Running inference on device:f{device}")
    torch.set_grad_enabled(False)
    # 读取UAV目录
    images_uav, images_uav_format = list_files(args.image_uav_path)
    # 模型初始化
    max_num_keypoints = args.num_keypoints
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # 初始化输出文件夹
    if os.path.exists(args.save_path):
        pass
    else:
        os.makedirs(args.save_path)

    if os.path.exists(args.fault_path):
        pass
    else:
        os.makedirs(args.fault_path)

    # 初始化 CSV 文件路径
    csv_file = os.path.join(args.save_path, 'image_coordinates.csv')
    # 初始化坐标，窗宽, 起点
    coord = (args.start_lon, args.start_lat)
    # win_size = 5000
    inx = 0
    image_uav = images_uav[inx]
    th = 0
    while inx < args.test_num:
        # 获取当前时间戳（毫秒级别）
        start_time = timeit.default_timer()
        image_uav = images_uav[inx]
        image_uav_1 = load_image(image_uav)
        winy = image_uav_1.shape[1]
        winx = image_uav_1.shape[2]
        image_ste, img_ste_geo, ox, oy = crop_geotiff_by_center_point(longitude=coord[0], latitude=coord[1],
                                                                      input_tif_path=args.image_ste_path,
                                                                      crop_size_px=winx,
                                                                      crop_size_py=winy)
        # 获取裁图时间戳
        end_time_1 = timeit.default_timer()  # 计算执行时间（毫秒）
        execution_time_ms = (end_time_1 - start_time) * 1000
        fps = 1000 / execution_time_ms
        logger.log(f"裁图时间: {execution_time_ms} 毫秒, FPS={fps}")
        img_ste_pos = [img_ste_geo[0], img_ste_geo[3]]
        img_uav_id = os.path.splitext(os.path.split(image_uav)[1])[0]  # 无人机图像的ID  目标图像的ID
        image_uav = Image.open(image_uav).convert('RGB')
        output_path = os.path.join(args.save_path, f"{img_uav_id}_{img_ste_pos}.{images_uav_format}")
        fault_path = os.path.join(args.fault_path, f"{img_uav_id}_{img_ste_pos}.{images_uav_format}")
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = inference(image_ste, image_uav, extractor, matcher, device)
        # 获取推理时间戳
        end_time_2 = timeit.default_timer()  # 计算执行时间（毫秒）
        execution_time_ms = (end_time_2 - end_time_1) * 1000
        fps = 1000 / execution_time_ms
        logger.log(f"推理时间: {execution_time_ms} 毫秒, FPS={fps}")
        if matches_num > max_num_keypoints / 15:
            aim = get_center_aim(winy, winx, m_kpts_ste, m_kpts_uav)
            # aimx, aimy = aim[0] + ox, aim[1] + oy
            # draw_points_on_image('/media/orin_agx/512G/dataset/2024-03-26_西江数据/XiJiang_DOM.tif',
            #          (aimx, aimy),
            #         '/media/orin_agx/512G/LightGlue/output/match/match.jpg' )
            aim_geo = pixel_to_geolocation(aim[0], aim[1], img_ste_geo)
            coord = aim_geo
            # with open('log_0327.txt', 'a') as log_file:
            #     log_message = ", ".join(str(item) for item in [uav_mapping[img_uav_id + '.JPG'], round(aim_geo[0], 6), round(aim_geo[1], 6), matches_num])
            #     log_file.write(log_message + '\n')
            # 将图像名称和对应的地理坐标保存到 CSV 文件
            save_coordinates_to_csv(csv_file, img_uav_id, coord)

            inx += 1
            th = 0
            # 获取结束时间戳
            end_time = timeit.default_timer()  # 计算执行时间（毫秒）
            execution_time_ms = (end_time - end_time_2) * 1000
            fps = 1000 / execution_time_ms
            logger.log(f"匹配成功：{img_uav_id}.jpg")
            visualize_and_save_matches(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, output_path)
        elif th > 3:
            inx += 1
            th = 0
        else:
            visualize_and_save_matches(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, fault_path)
            directions = [(0, 1000), (0, -1000), (-1000, 0), (1000, 0)]
            aims = []
            for dx, dy in directions:
                n_coord = (coord[0] + dx * img_ste_geo[1], coord[1] + dy * img_ste_geo[5])
                image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(longitude=n_coord[0], latitude=n_coord[1],
                                                                            input_tif_path=args.image_ste_path,
                                                                            crop_size_px=winx,
                                                                            crop_size_py=winy)
                matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = inference(image_ste, image_uav, extractor, matcher,
                                                                             device)
                aims.append((n_coord, matches_num))
            # 选取匹配数量最高的结果
            max_aim = max(aims, key=get_m_nums)
            coord, _ = max_aim
            logger.log(f"边界拓展：{img_uav_id}.jpg")
            # 获取结束时间戳
            end_time = timeit.default_timer()  # 计算执行时间（毫秒）
            execution_time_ms = (end_time - end_time_2) * 1000
            fps = 1000 / execution_time_ms
            th += 1


if __name__ == "__main__":
    args = parse_opt()
    if args.device != "auto":
        device = torch.device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
