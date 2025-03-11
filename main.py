# If we are on colab: this clones the repo and installs the dependencies
# from lightglue import LightGlue, SuperPoint
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from server.mavlink import CustomSITL
import torch
import argparse
import os
import timeit
from utils.pair import inference, get_center_aim, pixel_to_geolocation, visualize_and_save_matches, \
    get_m_nums, save_coordinates_to_csv, crop_geotiff_by_center_point
from utils.logger import Logger
import time
import keyboard
from typing import Tuple
from dataclasses import dataclass
from server.optmatch import OptMatch


@dataclass
class AppConfig:
    device: torch.device
    args: argparse.Namespace
    extractor: SuperPoint
    matcher: LightGlue
    logger: Logger
    sitl: CustomSITL


def setup_environment(args) -> Tuple[Logger, torch.device]:
    """初始化环境和硬件连接"""
    try:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.fault_path, exist_ok=True)
    except OSError as e:
        print(f"创建保存路径时出错: {e}")

    logger = Logger(log_file=os.path.join(args.save_path, 'log.txt'))
    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log(f"Running inference on device: {device}")
    torch.set_grad_enabled(False)
    return logger, device

def initialize_models(args, device) -> Tuple[SuperPoint, LightGlue]:
    """初始化AI模型"""
    extractor = SuperPoint(
        max_num_keypoints=args.num_keypoints,
        weight_path=os.path.join(os.path.dirname(__file__), "weights", "superpoint_v1.pth")
    ).eval().to(device)

    matcher = LightGlue(features=None, weights=args.weights_path).eval().to(device)
    return extractor, matcher

def setup_sitl_connection(logger: Logger) -> CustomSITL:
    """初始化无人机连接"""
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
    sitl.start_thread()
    sitl.takeoff_without_gps()

    # 由于不使用舵机控制飞行，可暂时不使用键盘按键功能，避免操作误触
    # keyboard.on_press(sitl.key_press_handler)
    # keyboard.on_release(sitl.key_release_handler)
    return sitl

def main():
    args = parse_opt()
    logger, device = setup_environment(args)
    extractor, matcher = initialize_models(args, device)
    sitl = setup_sitl_connection(logger)

    config = AppConfig(
        device=device,
        args=args,
        extractor=extractor,
        matcher=matcher,
        logger=logger,
        sitl=sitl
    )

    opt_matcher = OptMatch(config)
    try:
        opt_matcher.run()
    except KeyboardInterrupt:
        logger.log("程序被用户中断")
    finally:
        # 添加资源释放逻辑
        keyboard.unhook_all()
        logger.close()
        if sitl.connection:
            sitl.connection.close()



def parse_opt():
    parser = argparse.ArgumentParser(description="Benchmark script for LightGlue")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="cuda",
        help="device to benchmark on",
    )
    parser.add_argument(
        "--num_keypoints",

        type=int,
        default=1024,
        help="number of keypoints",
    )
    parser.add_argument(
        "--image_ste_path", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--image_uav_path", default="/mnt/d/TestData/patch/", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--save_path", type=str,
        help="path where figure should be saved"
    )
    parser.add_argument(
        "--fault_path", type=str,
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
        type=str, help="path to weights file"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

