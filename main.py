# If we are on colab: this clones the repo and installs the dependencies
# from lightglue import LightGlue, SuperPoint
from model.superpoint import SuperPoint
from model.lightglue import LightGlue
from model.utils import load_image, read_image
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
from pymavlink import mavutil
import time
import subprocess
import threading
import socket
import os
import queue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CustomSITL:
    def __init__(self):
        # Mission Planner 安装路径(需要根据实际情况修改)
       
        self.mp_path = os.path.join(os.path.expanduser("~"), "Documents", "Mission Planner")
        self.sitl_path = os.path.join(self.mp_path, "sitl")
        self.ardupilot_path = os.path.join(self.sitl_path, "ArduPlane.exe")
        self.sitl_process = None
        self.connection = None
        self.connect_serial=None
        
        #模拟遥控器信号
        self.rc_values = [65535] * 18
        self.turn_rate=20
        self.turn_angle=60
        self.current_yaw = 90  # 存储当前yaw值
        self.last_attitude_msg = None  # 存储最新的姿态消息
         # 控制标志
        self.is_running = True
        self.active_controls = set()  # 当前按下的按键集合

        self.coord_lon = 0  
        self.coord_lat = 0  
        self.real_coord_lon = 0  
        self.real_coord_lat=0
        # 用last与current计算航向与速度
        self.current_alt = 0  # 存储当前高度
        self.current_lat = 0  # 存储当前纬度
        self.current_lon = 0  # 存储当前经度
        #GPS环境开关
        self.gps=0
        self.gps_time_usec=0

        self.EARTH_RADIUS=6378137    # 地球半径(米)
        self.last_time=0
        self.vn = 0
        self.ve = 0
        self.vd = 0
        self.interval = 0.1  # 发送消息的间隔时间(秒)

        self.queue_csv = queue.Queue()

    def connect_to_sitl(self,ARMING_CHECK):
        """连接到 SITL"""
        try:
            self.connection = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
            # self.connection = mavutil.mavlink_connection('udpin:127.0.0.1:14551')
            self.connection.target_system = 1
            self.connection.target_component = 1
            
            # 设置超时
            self.connection.source_system = 255
            self.connection.source_component = 0

            """禁用起飞前检查"""
            self.connection.mav.param_set_send(
                self.connection.target_system,
                self.connection.target_component,
                b'ARMING_CHECK',  # 参数名
                ARMING_CHECK,               # 设置为0表示禁用所有检查
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
        # 等待参数设置确认
            time.sleep(1)
            print("已禁用起飞前检查")
            print("成功连接到 SITL")
            # 确保系统ID和组件ID已正确设置
            print(f"System ID: {self.connection.target_system}")
            print(f"Component ID: {self.connection.target_component}")               

            # 设置消息发送间隔，此处设置的为attitude和global_position_int消息的发送间隔为5Hz
            self.connection.mav.command_long_send(
                self.connection.target_system, self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,  # confirmation
                mavutil.mavlink.MAVLINK_MSG_ID_SIMSTATE,  # param1: message_id
                1e6 / 10,  # param2: interval in microseconds
                0,  # param3
                0,  # param4
                0,  # param5
                0,  # param6
                0  # param7
            )
            self.connection.mav.command_long_send(
                self.connection.target_system, self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,  # confirmation
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,  # param1: message_id
                1e6 / 10,  # param2: interval in microseconds
                0,  # param3
                0,  # param4
                0,  # param5
                0,  # param6
                0  # param7
            )
            self.connection.mav.command_long_send(
                self.connection.target_system, self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,  # confirmation
                mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,  # param1: message_id
                1e6 / 10,  # param2: interval in microseconds
                0,  # param3
                0,  # param4
                0,  # param5
                0,  # param6
                0  # param7
            )
            self.connection.mav.command_long_send(
                self.connection.target_system, self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,  # confirmation
                mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,  # param1: message_id
                1e6 / 10,  # param2: interval in microseconds
                0,  # param3
                0,  # param4
                0,  # param5
                0,  # param6
                0  # param7
            )
            return True
        except Exception as e:
            print(f"连接 SITL 失败: {str(e)}")
            return False

    def connect_to_serial(self):
        """直接通过tcp连接，模拟串口连接过程"""
        self.connection_serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection_serial.connect(('127.0.0.1', 5763))
    def stop_sitl(self):
        """停止 SITL"""
        if self.sitl_process:
            self.sitl_process.terminate()
            print("SITL 已停止")
        if self.connection:
            self.connection.close()
    def set_mode(self, mode,timeout = 20):
        """设置飞行模式"""
        mode_map = {
            'STABILIZE': 2,
            'FBWA': 5,
            'AUTO': 10,
            'GUIDED': 15,
            'LOITER': 12,
            'RTL': 11,
            'FBWB': 6,
        }
        
        if mode not in mode_map:
            print(f"不支持的模式: {mode}")
            return False
    # 使用 command_long_send 来设置模式
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,  # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_map[mode],
            0, 0, 0, 0, 0
        )
        
        # 等待模式切换确认，添加超时机制
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                if msg:
                    current_mode = msg.custom_mode
                    print(f"当前模式: {current_mode}, 目标模式: {mode_map[mode]}")
                    if current_mode == mode_map[mode]:
                        print(f"成功切换到 {mode} 模式")
                        return True
                    else:
                        self.connection.mav.command_long_send(
                            self.connection.target_system,
                            self.connection.target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                            0,  # confirmation
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            mode_map[mode],
                            0, 0, 0, 0, 0
                        )
            except Exception as e:
                print(f"接收消息时出错: {str(e)}")
        
        print(f"切换到 {mode} 模式超时")
        return False
    def arm_vehicle(self):
        """解锁并起飞到指定高度"""
        print("开始解锁和起飞程序...")        
        
        # 解锁
        print("发送解锁命令")
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        
        # 等待解锁确认
        start_time = time.time()
        armed = False
        while time.time() - start_time < 30:  # 10秒超时
            msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                armed = True
                print("飞机已解锁")
                break
            else:
                self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        if not armed:
            print("解锁失败")
            return False         
        msg = self.connection.recv_match(
        type=['ATTITUDE', 'GLOBAL_POSITION_INT'],
        blocking=True
            )
        if msg.get_type() == 'ATTITUDE':
            self.current_yaw = msg.yaw*180/3.1415926
            if self.current_yaw < 0:
                self.current_yaw += 360
            else :
                self.current_yaw = self.current_yaw % 360
        return True

    def wait_for_ekf_ready(self):
        """等待 EKF 完全收敛"""
        print("等待 EKF 收敛...")
        while True:
            msg = self.connection.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=1)
            if msg:
                flags = msg.flags
                # 检查所有必要的 EKF 标志
                if (flags & mavutil.mavlink.EKF_ATTITUDE and
                    flags & mavutil.mavlink.EKF_VELOCITY_HORIZ and
                    flags & mavutil.mavlink.EKF_VELOCITY_VERT and
                    flags & mavutil.mavlink.EKF_POS_HORIZ_REL and
                    flags & mavutil.mavlink.EKF_POS_HORIZ_ABS and
                    flags & mavutil.mavlink.EKF_POS_VERT_ABS):
                    print("EKF 已收敛")
                    return True
            time.sleep(0.1)
    def send_gps_input(self, lat, lon, alt):
        """
        发送GPS位置信息
        lat: 当前纬度
        lon: 当前经度
        alt: 当前高度 (米)
        暂时不使用和计算速度信息，位置信息的误差容许值调的较宽泛
        """
        # 记录时间

        # 计算速度 (如果有上一点数据)
        # if self.last_lat is not None and self.last_lon is not None and self.last_alt is not None and self.last_time is not None:
        #     time_diff = current_time - self.last_time
        #     vn, ve, vd = self.calculate_velocity(
        #         self.last_lat/ 1e7, self.last_lon/ 1e7, self.last_alt,
        #         lat/ 1e7, lon/ 1e7, alt,
        #         time_diff
        #     )
        # else:
        #     vn, ve, vd = 0, 0, 0  # 无上一点数据，速度设为 0
        #     self.last_time = current_time

        # 发送 MAVLink GPS_INPUT 消息
        
        self.connection.mav.gps_input_send(
            0,  # 时间戳（微秒）
            1,                        # gps_id
            0b00000000,               # ignore_flags (仅忽略 VD 速度)
            0,                        # time_week_ms
            0,                        # time_week
            3,                        # fix_type (3D fix)
            (int)(lat*1e7),           # lat - 纬度(度 * 1e7)
            (int)(lon*1e7),           # lon - 经度(度 * 1e7)
            alt,                      # alt - 高度(米)
            1.2,                      # hdop - 水平精度因子
            1.2,                      # vdop - 垂直精度因子
            self.vn,                       # vn - 北向速度
            self.ve,                       # ve - 东向速度
            self.vd,                     # vd - 垂直速度
            3,                        # speed_accuracy
            3,                      # horiz_accuracy
            3,                      # vert_accuracy
            8,                        # satellites_visible
            0                         # yaw
        )

    def takeoff_without_gps(self):
        """无GPS条件下的固定翼起飞"""
        # 确保在 STABILIZE 模式
        self.set_mode("FBWA")
        time.sleep(1)
        try:
            # 初始化所有通道为中位值
            # 1. 逐步增加油门
            self.rc_values[4]=0
            print("正在加速...")
            for throttle in range(1200, 1800, 50):  # 从中位值逐步增加到80%油门
                self.rc_values[2] = throttle  # 通道3是油门
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)
                
            # 2. 等待速度建立
            print("保持速度")
            for _ in range(60):  # 保持3秒
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                    
                )             
                time.sleep(0.1)

            # 3. 抬升机头
            print("抬升机头...")
            self.rc_values[1] = 1700  # 通道2上拉（俯仰）
            for _ in range(300):  # 保持3秒               
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                )           
                time.sleep(0.1)
            # 4. 保持一段时间让飞机爬升
            # 5. 恢复平飞姿态
            print("调整为平飞...")
            for _ in range(30):  # 保持3秒
                self.rc_values[1] = 1500  # 通道2上拉（俯仰）
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                )           
                time.sleep(0.1)
            # 确保模式稳定
            print("起飞结束")
            
            self.connection.mav.param_set_send(
                self.connection.target_system,
                self.connection.target_component,
                b'SIM_RC_FAIL',
                1,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
            return True

        except Exception as e:
            print(f"起飞过程出错: {str(e)}")
            # 恢复所有通道到中位
            self.rc_values = [65535] * 18
            self.connection.mav.rc_channels_override_send(
                self.connection.target_system,
                self.connection.target_component,
                *self.rc_values
            )
            return False

    def send_mav(self):
        while True:
            self.connection.mav.rc_channels_override_send(
                self.connection.target_system,
                self.connection.target_component,
                *self.rc_values
            )
            time.sleep(0.1)

    def receive_msg(self, msg_type):
        while True:
            while True:
        # 尝试非阻塞方式获取任意消息
                old_msg = self.connection.recv_msg()
                if old_msg is None:  # 如果没有更多消息可接收
                    break
            msg = self.connection.recv_match(type=msg_type, blocking=True)
            # 根据消息类型分别处理
            if msg_type == 'SIMSTATE':
                self.real_coord_lat = msg.lat / 1e7
                self.real_coord_lon = msg.lng / 1e7

            elif msg_type == 'VFR_HUD':
                self.current_alt = msg.alt

            elif msg_type == 'LOCAL_POSITION_NED':
                self.vn, self.ve, self.vd = msg.vx, msg.vy, msg.vz

            elif msg_type == 'GLOBAL_POSITION_INT':
                self.coord_lon = msg.lon / 1e7
                self.coord_lat = msg.lat / 1e7

# 在主程序中启动这些线程



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

def csv_saver_thread(csv_file,q_csv):
    while True:
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode = 'a', newline='') as file:
            writer = csv.writer(file)
            image_name, aim, global_pos, simstate = q_csv.get()
            print(f"收到坐标信息：{image_name}, {aim}, {global_pos}, {simstate}")
            if not file_exists:
                writer.writerow(["Image Name", "Aim_Longitude", "Aim_Latitude", "Global Longitude", "Global Latitude", "Sim_Longitude", "Sim_Latitude"])
            writer.writerow([image_name, aim[0], aim[1],global_pos[0],global_pos[1],simstate[0],simstate[1]])
        # file.flush()  # 立即刷新到磁盘（可选，推荐）

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
        default="cuda",
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
        "--image_ste_path", default="C:/Users/fumq7/Desktop/taikang/clipped.tif", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--image_uav_path", default="C:/Users/fumq7/Desktop/taikang/patch/", type=str,
        help="path where figure to be paired"
    )
    parser.add_argument(
        "--save_path", default="C:/Users/fumq7/Desktop/taikang/res_img", type=str,
        help="path where figure should be saved"
    )
    parser.add_argument(
        "--fault_path", default="C:/Users/fumq7/Desktop/taikang/fault_res_img", type=str,
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
        default="C:/Users/fumq7/Desktop/OptiMatchLocator/weights/superpoint_lightglue_v0-1_arxiv.pth",
        type=str, help="path to weights file"
    )
    args = parser.parse_args()
    return args


def main():
    logger = Logger(log_file=os.path.join(args.save_path, 'log.txt'))  # 创建日志工具，指定输出文件
    print(torch.__version__)
    logger.log(torch.cuda.is_available())
    logger.log(f"Running inference on device:f{device}")

    torch.set_grad_enabled(False)
    # 读取UAV目录-uav通过裁剪获取，不再通过文件夹读取
    # images_uav, images_uav_format = list_files(args.image_uav_path)
    # 模型初始化
    max_num_keypoints = args.num_keypoints
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints,
                           weight_path=os.path.join(os.path.dirname(__file__), "weights", "superpoint_v1.pth")).eval().to(device)
    matcher = LightGlue(features=None, weights=args.weights_path).eval().to(device)

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


    # win_size = 5000
    inx = 0
    # image_uav = images_uav[inx]
    th = 0
    #创建sitl连接
    sitl = CustomSITL()
    sitl.connect_to_serial()
    if sitl.connect_to_sitl(0):
        print("SITL 环境准备就绪")
    time.sleep(5)
    sitl.wait_for_ekf_ready()
    sitl.arm_vehicle()
    sender_thread = threading.Thread(
        target=sitl.send_mav,
        daemon=True  # daemon=True 使主程序退出时自动关闭子线程
    )
    for msg_type in ['SIMSTATE', 'VFR_HUD', 'LOCAL_POSITION_NED', 'GLOBAL_POSITION_INT']:
        threading.Thread(target=sitl.receive_msg, args=(msg_type,), daemon=True).start()
    threading.Thread(target=csv_saver_thread, args=(csv_file, sitl.queue_csv), daemon=True).start()
    sitl.takeoff_without_gps()
    sender_thread.start()
    while True:
        logger.log(f"start processing image {inx}")

        #将接收与更新消息直接写入子线程循环中，快速更新消息，主程序只从class字段中获取信息

        #下面是获取匹配信息
        start_time = timeit.default_timer()
        # image_uav = images_uav[inx]
        # image_uav_1 = load_image(image_uav)
        # winy = image_uav_1.shape[1]
        # winx = image_uav_1.shape[2]
        coord = (sitl.coord_lon, sitl.coord_lat)
        real_coord = (sitl.real_coord_lon, sitl.real_coord_lat)
        logger.log(f"当前坐标：{coord}")
        logger.log(f"当前真实坐标：{real_coord}")
        winy = 1000
        winx = 1000
        image_ste, img_ste_geo, ox, oy = crop_geotiff_by_center_point(longitude=coord[0], latitude=coord[1],
                                                                      input_tif_path=args.image_ste_path,
                                                                      crop_size_px=winx,
                                                                      crop_size_py=winy)
        image_uav, img_uav_geo, _, _ = crop_geotiff_by_center_point(longitude=real_coord[0], latitude=real_coord[1],
                                                                    input_tif_path=args.image_ste_path,
                                                                    crop_size_px=winx,
                                                                    crop_size_py=winy)
        # 获取裁图时间戳
        end_time_1 = timeit.default_timer()  # 计算执行时间（毫秒）
        execution_time_ms = (end_time_1 - start_time) * 1000
        fps = 1000 / execution_time_ms
        logger.log(f"裁图时间: {execution_time_ms} 毫秒, FPS={fps}")
        img_ste_pos = [img_ste_geo[0], img_ste_geo[3]]
        img_uav_id = inx #不从文件夹读取图像命名uav_id，使用迭代次数命名
        output_path = os.path.join(args.save_path, f"{img_uav_id}_{img_ste_pos}.tif")#暂时强制改为tif
        fault_path = os.path.join(args.fault_path, f"{img_uav_id}_{img_ste_pos}.tif")
        matches_S_U, matches_num, m_kpts_ste, m_kpts_uav = inference(image_ste, image_uav, extractor, matcher, device)
        # 获取推理时间戳
        end_time_2 = timeit.default_timer()  # 计算执行时间（毫秒）
        execution_time_ms = (end_time_2 - end_time_1) * 1000
        fps = 1000 / execution_time_ms
        logger.log(f"推理时间: {execution_time_ms} 毫秒, FPS={fps}")
        if inx == 10:
            sitl.connection.mav.param_set_send(
                sitl.connection.target_system,
                sitl.connection.target_component,
                b'SIM_GPS_DISABLE',
                1,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
        while th<3:
            if matches_num > max_num_keypoints / 15:
                aim = get_center_aim(winy, winx, m_kpts_ste, m_kpts_uav)
                aim_geo = pixel_to_geolocation(aim[0], aim[1], img_ste_geo)
                sitl.current_lat = aim_geo[1]
                sitl.current_lon = aim_geo[0]
                # sitl.current_lat = real_coord[1]
                # sitl.current_lon = real_coord[0]
                # 将图像名称和对应的地理坐标保存到 CSV 文件
                sitl.queue_csv.put((img_uav_id, aim_geo,coord,real_coord))
                # save_coordinates_to_csv(csv_file, img_uav_id, coord)
                th = 0
                # 获取结束时间戳
                end_time = timeit.default_timer()  # 计算执行时间（毫秒）
                execution_time_ms = (end_time - end_time_2) * 1000
                fps = 1000 / execution_time_ms
                logger.log(f"匹配成功：{img_uav_id}.jpg")
                sitl.send_gps_input(sitl.current_lat, sitl.current_lon, sitl.current_alt)
                logger.log(f"发送GPS位置信息: lat={sitl.current_lat}, lon={sitl.current_lon}, alt={sitl.current_alt}")
                # visualize_and_save_matches(image_ste, image_uav, m_kpts_ste, m_kpts_uav, matches_S_U, output_path)
                inx += 1
                break
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
