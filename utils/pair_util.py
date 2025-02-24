import cv2, torch
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
import csv
import numpy as np
import glob
import os
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, Grayscale
import torchvision.transforms.functional as TF

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

import random


def get_variable(coord, p):
    probability = random.random()  # 生成0到1之间的随机数

    # 假设a为1的概率是p，那么a为2的概率就是1-p
    # 这里我们假设a为1的概率是0.5，a为2的概率也是0.5，你可以根据实际情况调整这些概率
    if probability < p:
        return coord
    else:
        return (-100000000, 100000000)


def draw_points_on_image(image_path, points, output_path):
    # 读取输入图像
    image = cv2.imread(image_path)

    # 将图像转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 在图像上标点

    x, y = int(points[0]), int(points[1])
    cv2.circle(image_rgb, (x, y), 25, (255, 0, 0), -1)  # 在图像上标点
    image_rgb = cv2.resize(image_rgb, (5000, 5000))

    # 显示带有标点的图像
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    # 保存带有标点的图像
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def pad_to_match_dimension(tensor_to_pad, reference_tensor):
    """
    将 tensor_to_pad 填充至与 reference_tensor 相同的维度。

    参数:
        tensor_to_pad (torch.Tensor): 需要填充的张量。
        reference_tensor (torch.Tensor): 作为参考的张量，被用于确定填充后的维度。

    返回:
        torch.Tensor: 填充后的张量，其维度与 reference_tensor 相同。
    """
    in_shape = len(tensor_to_pad.shape)
    if in_shape > 2:
        tensor_to_pad = tensor_to_pad.squeeze(0)
        reference_tensor = reference_tensor.squeeze(0)
    # 计算填充的行数和列数
    padding_rows = reference_tensor.shape[0] - tensor_to_pad.shape[0]
    padding_cols = reference_tensor.shape[1] - tensor_to_pad.shape[1]

    # 使用 torch.nn.functional.pad() 函数填充
    # 该函数以（左填充，右填充，上填充，下填充）的方式进行填充
    # 这里我们将右填充和下填充设置为相应的行数和列数，其余填充设置为0
    padded_tensor = torch.nn.functional.pad(tensor_to_pad, (0, padding_cols, 0, padding_rows))
    if in_shape > 2:
        padded_tensor = padded_tensor.unsqueeze(0)
    return padded_tensor


def get_m_nums(aim):
    return aim[1]


def process_matches(matches_num, image_uav, m_kpts_ste, m_kpts_uav,
                    uav_mapping, img_uav_id, image_ste, matches_S_U, output_path):
    aim = get_center_aim(image_uav, m_kpts_ste, m_kpts_uav)
    visualize_and_save_matches(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, output_path)
    return aim


def visualize_and_save_matches(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, output_path):
    """
    Visualize images and their matches and save the result to the output_path.

    Parameters:
        image_ste (numpy.ndarray): Image from source.
        image_uav (numpy.ndarray): Image from target.
        m_kpts_ste (list): Keypoints from source.
        m_kpts_uav (list): Keypoints from target.
        matches_S_U (dict): Matches between source and target.
        output_path (str): Path to save the visualization.
    """
    image_ste = np.array(image_ste)
    # image_ste = image_ste.cpu().permute(1, 2, 0).numpy()
    image_uav = np.array(image_uav)
    axes = viz2d.plot_images([image_ste, image_uav])
    viz2d.plot_matches(m_kpts_ste, m_kpts_uav, color="lime", lw=0.1)
    viz2d.add_text(0, f'Stop after {matches_S_U["stop"]} layers', fs=20)
    plt.savefig(output_path)
    plt.close()


def extract_number(filename):
    return int(filename.split('/')[-1].split('.')[0])


def read_coordinates(file_path):
    """
    从给定路径的txt文件中读取坐标信息，并以字典列表的形式返回，其中每个字典包含id、经度和纬度。
    :param file_path: 包含坐标的txt文件路径
    :return: 字典列表，格式为 [{'id': id, 'longitude': longitude, 'latitude': latitude}, ...]
    """
    # 初始化结果列表
    coordinates_by_id = []
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        # # 跳过表头（如果有的话）
        # next(reader)  # 如果第一行是标题行，则取消注释此行
        for row in reader:
            try:
                # 假设第一列是id，后面两列分别是经度和纬度
                id_ = row[0]
                longitude = float(row[2])
                latitude = float(row[1])
                # 将每行的数据打包成一个字典，并添加到列表中
                coordinates_by_id.append({
                    'id': id_,
                    'longitude': longitude,
                    'latitude': latitude
                })
            except (IndexError, ValueError):
                # 如果某一行数据格式不正确，则忽略该行
                pass

    return coordinates_by_id


def inference(image_ste, image_uav, extractor, matcher, device):
    # transform = Compose([
    #     ToTensor(),                          # 将PIL图像转为Tensor
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    # ])
    transform = ToTensor()
    image_ste = transform(image_ste).to(device)
    image_uav = transform(image_uav).to(device)
    # image_ste = torch.from_numpy(np.array(image_ste)).to(device).permute(2, 0, 1).float()
    # image_uav = torch.from_numpy(np.array(image_uav)).to(device).permute(2, 0, 1).float()

    feats_ste = extractor.extract(image_ste)
    feats_uav = extractor.extract(image_uav)
    if feats_ste['keypoints'].shape != feats_uav['keypoints'].shape:
        feats_ste['keypoints'] = pad_to_match_dimension(feats_ste['keypoints'], feats_uav['keypoints'])
        feats_ste['keypoint_scores'] = pad_to_match_dimension(feats_ste['keypoint_scores'],
                                                              feats_uav['keypoint_scores'])
        feats_ste['descriptors'] = pad_to_match_dimension(feats_ste['descriptors'], feats_uav['descriptors'])
    matches_S_U = matcher({"image0": feats_ste, "image1": feats_uav})
    feats_ste, feats_uav, matches_S_U = [rbd(x) for x in [feats_ste, feats_uav, matches_S_U]]
    kpts_ste, kpts_uav, matches = feats_ste["keypoints"], feats_uav["keypoints"], matches_S_U["matches"]
    m_kpts_ste, m_kpts_uav = kpts_ste[matches[..., 0]], kpts_uav[matches[..., 1]]
    matches_num = matches_S_U["matches"].shape[0]

    return matches_S_U, matches_num, m_kpts_ste, m_kpts_uav


def get_center_aim(h, w, m_kpts_ste, m_kpts_uav):
    m_kpts_ste, m_kpts_uav = m_kpts_ste.cpu().numpy(), m_kpts_uav.cpu().numpy()
    Ma, _ = cv2.findHomography(m_kpts_uav, m_kpts_ste, cv2.RANSAC, 5.0)
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0], [int(w / 2), int(h / 2)]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, Ma)
    moments = cv2.moments(dst)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    center = (cX, cY)
    return center


def list_files(directory):
    p = str(Path(directory).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    image_format = files[0].split('.')[-1].lower()
    file_list = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    file_list = sorted(file_list, key=extract_number)
    return file_list, image_format


def pixel_to_geolocation(x_pixel, y_pixel, geotransform):
    """
    将图像坐标系中的像素坐标转换为实际地理坐标（经纬度）

    参数：
    x_pixel (float): 图像x轴上的像素位置
    y_pixel (float): 图像y轴上的像素位置
    geotransform (tuple of 6 floats): 地理变换元组，形式如：(top_left_x, pixel_width, rotation0, top_left_y, rotation1, pixel_height)

    返回：
    (lon, lat): 经纬度坐标对
    """

    # 地理变换参数解释
    # top_left_x, top_left_y 是图像左上角在地理坐标系中的坐标
    # pixel_width 和 pixel_height 分别是每个像素对应的地理单位长度（通常是米）
    # rotation0 和 rotation1 是旋转参数，在大多数情况下它们是0

    origin_x = geotransform[0]
    pixel_width = geotransform[1]
    origin_y = geotransform[3]
    pixel_height = geotransform[5]

    # 考虑到GDAL中图像的原点位于左上角且y轴方向向下增长，需要做调整
    lon = origin_x + x_pixel * pixel_width
    lat = origin_y + y_pixel * pixel_height  # 注意这里是减法

    return lon, lat


def crop_image_by_center_point(x, y, img, crop_size_px, crop_size_py):
    # 计算裁剪区域的左上角坐标
    left = max(x - crop_size_px // 2, 0)
    top = max(y - crop_size_py // 2, 0)

    # 计算裁剪区域的右下角坐标
    right = min(x + crop_size_px // 2, img.width)
    bottom = min(y + crop_size_py // 2, img.height)

    # 裁剪图像
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img, left, top


def center_crop_with_coords(image_tensor, center, crop_size_x, crop_size_y):
    """
    根据中心坐标裁剪图像，并返回裁剪后的图像以及裁剪图像的左上角坐标在原图中的位置

    :param image_tensor: 原始图像（tensor形式）
    :param center: 中心坐标 (x, y)
    :param crop_size: 要裁剪的大小 (width, height)
    :return: 裁剪后的图像（tensor形式），裁剪图像的左上角坐标在原图中的位置 (x, y)
    """
    # 获取图像的尺寸
    image_height, image_width = image_tensor.shape[-2:]

    # 计算裁剪区域的左上角和右下角坐标
    crop_width, crop_height = crop_size_x, crop_size_y
    x_center, y_center = center
    x1 = max(0, int(x_center - crop_width / 2))
    y1 = max(0, int(y_center - crop_height / 2))
    x2 = min(image_width, int(x_center + crop_width / 2))
    y2 = min(image_height, int(y_center + crop_height / 2))

    # 裁剪图像
    cropped_image = TF.crop(image_tensor, y1, x1, y2 - y1, x2 - x1)

    return cropped_image, x1, y1


def geo_to_pixel(geo_x, geo_y, tfw_path):
    with open(tfw_path, 'r') as tfw_file:
        lines = tfw_file.readlines()
        pixel_size_x = float(lines[0])
        pixel_size_y = float(lines[3])
        origin_x = float(lines[4])
        origin_y = float(lines[5])

    pixel_x = int((geo_x - origin_x) / pixel_size_x)
    pixel_y = int((origin_y - geo_y) / pixel_size_y * (-1))

    return pixel_x, pixel_y


def pixel_to_geo(pixel_x, pixel_y, tfw_path):
    with open(tfw_path, 'r') as tfw_file:
        lines = tfw_file.readlines()
        pixel_size_x = float(lines[0])
        pixel_size_y = float(lines[3])
        origin_x = float(lines[4])
        origin_y = float(lines[5])

    geo_x = origin_x + pixel_x * pixel_size_x
    geo_y = origin_y + pixel_y * pixel_size_y

    return geo_x, geo_y




