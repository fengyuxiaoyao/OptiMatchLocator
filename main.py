# If we are on colab: this clones the repo and installs the dependencies
# from lightglue import LightGlue, SuperPoint
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from model.utils import load_image, read_image
from model.mavlink import CustomSITL
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
import time
import keyboard


def geo2pixel(geotransform, lon, lat):
    lon = float(lon)
    lat = float(lat)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)

    x = int((point.GetX() - geotransform[0]) / geotransform[1])
    y = abs(int((geotransform[3] - point.GetY()) / geotransform[5]))
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
        type=int,
        default=1024,
        help="number of keypoints",
    )
    parser.add_argument(
        "--image_ste_path", default="D:/TestData/clipped.tif", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--image_uav_path", default="/mnt/d/TestData/patch/", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--save_path", default="D:/TestData/output/res_img", type=str,
        help="path where figure should be saved"
    )
    parser.add_argument(
        "--fault_path", default="D:/TestData/output/fault_res_img", type=str,
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
    parser.add_argument(
        "--weights_path",
        default="D:/TestData/weights/superpoint_lightglue_v0-1_arxiv.pth",
        type=str, help="path to weights file"
    )
    args = parser.parse_args()
    return args


def main():
    # 初始化输出文件夹
    try:
        os.makedirs(args.save_path, exist_ok=True)
    except OSError as e:
        print(f"创建保存路径时出错: {e}")

    logger = Logger(log_file=os.path.join(args.save_path, 'log.txt'))  # 创建日志工具，指定输出文件
    logger.log(f"Running inference on device:f{device}")
    torch.set_grad_enabled(False)

    # 模型初始化
    max_num_keypoints = args.num_keypoints
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints,
                           weight_path=os.path.join(os.path.dirname(__file__), "weights", "superpoint_v1.pth")).eval().to(device)
    matcher = LightGlue(features=None, weights=args.weights_path).eval().to(device)

    if os.path.exists(args.fault_path):
        pass
    else:
        os.makedirs(args.fault_path)

    # 初始化 CSV 文件路径
    csv_file = os.path.join(args.save_path, 'image_coordinates.csv')
    # 初始化坐标，窗宽, 起点
    coord = (args.start_lon, args.start_lat)
    # win_size = 5000

    th = 0

    logger.log("Initializing SITL...")
    sitl = CustomSITL()

    sitl.connect_to_serial()
    if sitl.connect_to_sitl(0, 0, 1):
        logger.log("SITL 环境准备就绪")
        # 等待系统稳定
    # GPS_TYPE无法实现空中切换，只能在地面切换
    time.sleep(5)
    # sitl.wait_for_ekf_ready()
    sitl.arm_vehicle()
    # sitl.takeoff(60)
    sitl.send_guided_change_heading(1, 90, sitl.turn_rate)
    sitl.takeoff_without_gps()

    # 注册按键事件处理器
    keyboard.on_press(sitl.key_press_handler)
    keyboard.on_release(sitl.key_release_handler)

    # 启动控制循环
    print("开始监听按键控制...")
    print("使用 WSAD 控制姿态，方向键控制油门和偏航")
    print("空格键：除油门外所有通道回中")
    print("按 ESC 退出")
    print("开始控制循环...")
    sitl.rc_values[2] = 1700
    sitl.connection.mav.rc_channels_override_send(
        sitl.connection.target_system,
        sitl.connection.target_component,
        *sitl.rc_values
    )
    winy = 1024
    winx = 1024
    while sitl.is_running:
        sitl.connection.mav.rc_channels_override_send(
            sitl.connection.target_system,
            sitl.connection.target_component,
            *sitl.rc_values
        )
        REAL_lat, REAL_lon, COMPUTED_alt, SIM_lat, SIM_lon = sitl.get_global_position()
        coord = (SIM_lon, SIM_lat)
        logger.log(f"真实坐标: {REAL_lat}, {REAL_lon}, 计算高度: {COMPUTED_alt}, 飞控仿真坐标: {SIM_lat}, {SIM_lon}")
        image_ste, img_ste_geo, ox, oy = crop_geotiff_by_center_point(longitude=SIM_lon, latitude=SIM_lat,
                                                                      input_tif_path=args.image_ste_path,
                                                                      crop_size_px=winx*2,
                                                                      crop_size_py=winy*2)
        real_img, real_geo, ox, oy = crop_geotiff_by_center_point(longitude=REAL_lon, latitude=REAL_lat,
                                                                  input_tif_path=args.image_ste_path,
                                                                  crop_size_px=winy,
                                                                  crop_size_py=winx)

        end_time_1 = timeit.default_timer()  # 计算执行时间（毫秒）
        # 计算真实坐标
        img_ste_pos = [img_ste_geo[0], img_ste_geo[3]]
        output_path = os.path.join(args.save_path, f"{SIM_lat}, {SIM_lon}.jpg")
        fault_path = os.path.join(args.fault_path, f"{SIM_lat}, {SIM_lon}.jpg")
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = inference(image_ste, real_img, extractor, matcher, device)
        # 获取推理时间戳
        end_time_2 = timeit.default_timer()  # 计算执行时间（毫秒）
        execution_time_ms = (end_time_2 - end_time_1) * 1000
        fps = 1000 / execution_time_ms
        logger.log(f"推理时间: {execution_time_ms} 毫秒, FPS={fps}")
        if matches_num > max_num_keypoints / 15:
            aim = get_center_aim(winy, winx, m_kpts_ste, m_kpts_uav)
            aim_geo = pixel_to_geolocation(aim[0], aim[1], img_ste_geo)
            sitl.update_global_position(int(aim_geo[1]*1e7), int(aim_geo[0]*1e7), COMPUTED_alt)
            logger.log(f"真实坐标: {REAL_lat}, {REAL_lon}, 计算坐标: {aim_geo[1]}, {aim_geo[0]}, 飞控仿真坐标: {SIM_lat}, {SIM_lon}")
            coord = aim_geo
            # 将图像名称和对应的地理坐标保存到 CSV 文件
            save_coordinates_to_csv(csv_file, end_time_1, coord)

            # 获取结束时间戳
            end_time = timeit.default_timer()  # 计算执行时间（毫秒）
            execution_time_ms = (end_time - end_time_2) * 1000
            fps = 1000 / execution_time_ms
            logger.log(f"匹配成功：{REAL_lat}, {REAL_lon}")
            visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, output_path)
        else:
            visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, fault_path)
            directions = [(0, 1000), (0, -1000), (-1000, 0), (1000, 0)]
            aims = []
            for dx, dy in directions:
                n_coord = (coord[0] + dx * img_ste_geo[1], coord[1] + dy * img_ste_geo[5])
                image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(longitude=n_coord[0], latitude=n_coord[1],
                                                                            input_tif_path=args.image_ste_path,
                                                                            crop_size_px=winx,
                                                                            crop_size_py=winy)
                matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = inference(image_ste, real_img, extractor, matcher,
                                                                             device)
                aims.append((n_coord, matches_num))
            # 选取匹配数量最高的结果
            max_aim = max(aims, key=get_m_nums)
            coord, _ = max_aim
            logger.log(f"搜寻失败，尝试边界拓展：{SIM_lat}, {SIM_lon}.jpg")
            # 获取结束时间戳
            end_time = timeit.default_timer()  # 计算执行时间（毫秒）
            execution_time_ms = (end_time - end_time_2) * 1000
            fps = 1000 / execution_time_ms


if __name__ == "__main__":
    args = parse_opt()
    if args.device != "auto":
        device = torch.device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
