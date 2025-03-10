from typing import Tuple
from dataclasses import dataclass
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from server.mavlink import CustomSITL
from utils.logger import Logger
from utils.pair import inference, get_center_aim, pixel_to_geolocation, visualize_and_save_matches
from utils.pair import crop_geotiff_by_center_point, save_coordinates_to_csv, get_m_nums
import argparse
import torch
import timeit
import keyboard
import os

@dataclass
class AppConfig:
    # 修复类型注解
    device: torch.device
    args: argparse.Namespace
    extractor: SuperPoint
    matcher: LightGlue
    logger: Logger
    sitl: CustomSITL

class OptMatch:

    def __init__(self, app_config: AppConfig):
        self.config = app_config  # 添加配置存储

    def process_image_matching(self, image_ste, real_img):
        """核心图像匹配处理流程
        Args:
            image_ste: 卫星基准图像
            real_img: 无人机实时图像
        Returns:
            tuple: 匹配结果 (matches_S_U, 匹配数量, 关键点集合)
        """
        start_time = timeit.default_timer()

        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = inference(
            image_ste, real_img,
            self.config.extractor,  # 使用实例存储的配置
            self.config.matcher,
            self.config.device
        )

        elapsed_time = (timeit.default_timer() - start_time) * 1000
        self.config.logger.log(f"推理时间: {elapsed_time:.2f} 毫秒, FPS={1000 / elapsed_time:.1f}")
        return matches_S_U, matches_num, m_kpts_ste, m_kpts_uav


    def process_image_data(self, config: AppConfig, position_data, win_size: Tuple[int, int], csv_file: str):  # 添加self参数
        """处理图像数据"""
        REAL_lat, REAL_lon, COMPUTED_alt, SIM_lat, SIM_lon = position_data
        config.logger.log(f"定位数据: 真实({REAL_lat}, {REAL_lon}), 仿真({SIM_lat}, {SIM_lon}), 高度:{COMPUTED_alt}")
        winx, winy = win_size
        output_path = config.args.save_path
        fault_path = config.args.fault_path
        # 图像裁剪处理
        image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
            longitude=SIM_lon, latitude=SIM_lat,
            input_tif_path=config.args.image_ste_path,
            crop_size_px=winx * 2,
            crop_size_py=winy * 2
        )

        real_img, real_geo, _, _ = crop_geotiff_by_center_point(
            longitude=REAL_lon, latitude=REAL_lat,
            input_tif_path=config.args.image_ste_path,
            crop_size_px=winx,
            crop_size_py=winy
        )

        # 核心匹配处理（修复方法调用）
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = self.process_image_matching(
            image_ste, real_img
        )

        if matches_num > config.args.num_keypoints / 15:
            aim = get_center_aim(winy, winx, m_kpts_ste, m_kpts_uav)
            aim_geo = pixel_to_geolocation(aim[0], aim[1], img_ste_geo)
            config.sitl.update_global_position(int(aim_geo[1] * 1e7), int(aim_geo[0] * 1e7), COMPUTED_alt)
            config.logger.log(
                f"真实坐标: {REAL_lat}, {REAL_lon}, 计算坐标: {aim_geo[1]}, {aim_geo[0]}, 飞控仿真坐标: {SIM_lat}, {SIM_lon}")
            coord = aim_geo
            # 将图像名称和对应的地理坐标保存到 CSV 文件
            save_coordinates_to_csv(csv_file, timeit.default_timer(), coord, (REAL_lon, REAL_lat), (SIM_lon, SIM_lat))

            config.logger.log(f"匹配成功：{REAL_lat}, {REAL_lon}")
            visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, output_path)
        else:
            visualize_and_save_matches(image_ste, real_img, m_kpts_ste, m_kpts_uav, matches_S_U, fault_path)
            directions = [(0, 1000), (0, -1000), (-1000, 0), (1000, 0)]
            aims = []
            for dx, dy in directions:
                n_coord = (SIM_lon + dx * img_ste_geo[1], SIM_lat + dy * img_ste_geo[5])
                # 修复参数路径引用
                image_ste, img_ste_geo, _, _ = crop_geotiff_by_center_point(
                    longitude=n_coord[0], latitude=n_coord[1],
                    input_tif_path=config.args.image_ste_path,  # 修复args->config.args
                    crop_size_px=winx,
                    crop_size_py=winy)
                matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = self.process_image_matching(
                    image_ste, real_img
                )
                aims.append((n_coord, matches_num))
            # 选取匹配数量最高的结果
            max_aim = max(aims, key=get_m_nums)
            coord, _ = max_aim
            config.logger.log(f"搜寻失败，尝试边界拓展：{SIM_lat}, {SIM_lon}.jpg")

    def run(self):
        """主控制循环"""
        win_size = (1024, 1024)
        csv_file = os.path.join(self.config.args.save_path, 'image_coordinates.csv')  # 使用实例配置

        # 注册输入监听
        keyboard.on_press(self.config.sitl.key_press_handler)
        keyboard.on_release(self.config.sitl.key_release_handler)

        # 控制循环
        while self.config.sitl.is_running:
            # 更新无人机状态
            self.config.sitl.connection.mav.rc_channels_override_send(
                self.config.sitl.connection.target_system,
                self.config.sitl.connection.target_component,
                *self.config.sitl.rc_values
            )

            # 获取定位数据
            position_data = self.config.sitl.get_global_position()

            # 图像处理流程
            self.process_image_data(self.config, position_data, win_size, csv_file)