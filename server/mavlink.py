import subprocess
import socket
import time
import os
from pymavlink import mavutil
import math
import inspect
import keyboard
import matplotlib.pyplot as plt
import threading
from queue import Queue


class CustomSITL:
    def __init__(self,logger):
        self.connection = None
        self.connect_serial = None
        self.logger = logger
        self.waypoints1 = [
            {'lat': 39.8900811, 'lon': 114.4266629, 'alt': 50},
            {'lat': 39.8907067, 'lon': 114.4397736, 'alt': 50},
            {'lat': 39.8787859, 'lon': 114.4448376, 'alt': 100},

            {'lat': 39.8746361, 'lon': 114.4308043, 'alt': 150},
            {'lat': 39.8824414, 'lon': 114.4164276, 'alt': 200},
        ]


        # 模拟遥控器信号
        self.rc_values = [65535] * 18
        self.turn_rate = 20
        self.turn_angle = 60
        self.current_yaw = 90  # 存储当前yaw值
        self.last_attitude_msg = None  # 存储最新的姿态消息
        # 控制标志

        self.last_alt = 0  # 存储当前高度
        self.last_lat = 0  # 存储当前纬度
        self.last_lon = 0  # 存储当前经度```````
        # 用last与current计算航向与速度
        self.current_alt = 0  # 存储当前高度
        self.real_lat = 0  # 存储simstate真实纬度
        self.real_lon = 0  # 存储simstate真实经度
        self.global_position_lat=0 #存储global_position_int中飞控计算纬度
        self.global_position_lon=0 #存储global_position_int中飞控计算经度

        self.vn, self.ve, self.vd = 0, 0, 0  # 速度信息
        self.message_queue = Queue()
        # GPS环境开关
        self.gps = 0

        self.EARTH_RADIUS = 6378137  # 地球半径(米)
        self.last_time = 0

    def connect_to_sitl(self, ARMING_CHECK, SIM_GPS_DISABLE, GPS_TYPE):
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
                ARMING_CHECK,  # 设置为0表示禁用所有检查
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
            # 等待参数设置确认
            time.sleep(1)
            print("已禁用起飞前检查")
            # self.connection.mav.param_set_send(
            #     self.connection.target_system,
            #     self.connection.target_component,
            #     b'SIM_GPS_DISABLE',
            #     SIM_GPS_DISABLE,
            #     mavutil.mavlink.MAV_PARAM_TYPE_INT32
            # )
            # 0是开启环境内GPS信号（默认值），1关闭内部GPS模拟信号
            print("成功连接到 SITL")

            # 确保系统ID和组件ID已正确设置
            print(f"System ID: {self.connection.target_system}")
            print(f"Component ID: {self.connection.target_component}")

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

    def SIM_GPS_DISABLE(self,value):
        self.connection.mav.param_set_send(
            self.connection.target_system,
            self.connection.target_component,
            b'SIM_GPS_DISABLE',
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )

    def wait_for_message(self, message_type, timeout, seq=None):
        """等待特定类型的消息，带序号检查"""
        # present heading: VFR_HUD {airspeed : 25.004688262939453, groundspeed : 0.0, heading : 87, throttle : 38, alt : 128.88999938964844, climb : 0.7966185212135315}
        start = time.time()
        while time.time() - start < timeout:
            msg = self.connection.recv_match(type=message_type, blocking=True, timeout=1)
            if msg:
                if seq is None or (hasattr(msg, 'seq') and msg.seq == seq):
                    return msg
                    # 清理其他消息
            while self.connection.recv_match(blocking=False):
                pass
        return None

    def upload_mission(self, waypoints):
        try:
            print("开始上传任务...")

            # 清除现有任务并确认
            print("清除现有任务...")
            for _ in range(3):  # 尝试3次
                self.connection.waypoint_clear_all_send()
                msg = self.connection.recv_match(type=['MISSION_ACK'], blocking=True, timeout=2)
                if msg:
                    print("清除成功")
                    break
            print(f"发送航点数量: {len(waypoints)}")
            self.connection.mav.mission_count_send(
                self.connection.target_system,
                self.connection.target_component,
                len(waypoints)
            )
            # 上传航点
            for i in range(len(waypoints)):
                success = False
                for retry in range(3):  # 每个航点尝试3次
                    # 等待请求，过滤掉其他消息
                    start_time = time.time()
                    while time.time() - start_time < 5:  # 5秒超时
                        msg = self.wait_for_message(['MISSION_REQUEST_INT', 'MISSION_REQUEST'], 1)
                        if not msg:
                            print(f"未收到航点 {i} 的请求，重试...")
                        if msg and (
                                msg.get_type() == 'MISSION_REQUEST_INT' or msg.get_type() == 'MISSION_REQUEST') and msg.seq == i:
                            # 收到正确的请求，发送航点
                            self.connection.mav.mission_item_int_send(
                                self.connection.target_system,
                                self.connection.target_component,
                                i,
                                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                0, 1,
                                0, 0, 0, 0,
                                int(waypoints[i]['lat'] * 1e7),  # 转换为整数坐标
                                int(waypoints[i]['lon'] * 1e7),
                                waypoints[i]['alt']
                            )
                            print(f"发送航点 {i}")
                            success = True
                            break

                    if success:
                        break
                    else:
                        print(f"重试发送航点 {i}")
                        # 重新发送航点数量
                        self.connection.mav.mission_count_send(
                            self.connection.target_system,
                            self.connection.target_component,
                            len(waypoints)
                        )
                        time.sleep(1)

                if not success:
                    print(f"航点 {i} 上传失败")
                    return False

                time.sleep(0.1)  # 短暂延时

            # 等待最终确认
            print("等待最终确认...")
            for _ in range(3):
                msg = self.connection.recv_match(type=['MISSION_ACK'], blocking=True, timeout=2)
                if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    print("任务上传完成并确认")
                    break
                time.sleep(1)

            # 验证任务
            print("开始验证任务...")
            self.connection.waypoint_request_list_send()
            msg = self.connection.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=2)
            if not msg or msg.count != len(waypoints):
                print(f"验证失败: 预期 {len(waypoints)} 个航点，实际 {msg.count if msg else 0} 个")
                return False

            print(f"验证成功: 共 {msg.count} 个航点")

            self.set_mode("AUTO")
            return True

        except Exception as e:
            print(f"错误: {str(e)}")
            return False

    def set_mode(self, mode, timeout=20):
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
            while True:
                m=self.connection.recv_msg()
                if m is None:
                    break
            try:
                # msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                msg=self.connection.messages['HEARTBEAT']
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
                        time.sleep(0.5)
                else:
                    print("未收到 HEARTBEAT 消息")
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
            self.current_yaw = msg.yaw * 180 / 3.1415926
            if self.current_yaw < 0:
                self.current_yaw += 360
            else:
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

    def takeoff(self, alt):
        """
        takeoff命令只有在多旋翼无人机上起作用，固定翼无人机需手动使用油门起飞
        固定翼飞机使用takeoffwithoutGPS起飞
        """
        self.set_mode("GUIDED")
        self.set_mode("AUTO")

        # 2. 添加起飞任务
        self.connection.mav.mission_item_send(
            self.connection.target_system,
            self.connection.target_component,
            0,  # 序号
            0,  # 当前航点帧
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # 当前航点
            1,  # 自动继续
            15,  # 起飞角度（通常10-15度）
            0, 0, 0,  # 未使用
            0, 0, alt  # 目标高度
        )

        print(f"开始起飞到 {alt} 米")

        start_time = time.time()
        reached_alt = False
        print("等待达到目标高度...")

        while time.time() - start_time < 100:  # 60秒超时
            msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                current_alt = msg.relative_alt / 1000.0  # 转换为米
                if abs(current_alt - alt) <= 1.0:  # 在1米误差范围内
                    reached_alt = True
                    print(f"已达到目标高度: {alt}米")
                    break

        if not reached_alt:
            print("未能在规定时间内达到目标高度")
            return False

    def set_local_position(self, x, y, z):
        """
        x: 北向位移(米)
        y: 东向位移(米)
        z: 向下位移(米，通常为负值)
        """
        self.set_mode("GUIDED")
        self.connection.mav.set_position_target_local_ned_send(
            0,  # 时间戳
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b110111111000,  # type_mask
            x, y, z,  # 位置
            0, 0, 0,  # 速度
            0, 0, 0,  # 加速度
            0, 0  # yaw, yaw_rate
        )

    def calculate_velocity(self, lat1, lon1, alt1, lat2, lon2, alt2, time_diff):
        """
        计算速度信息 (vn, ve, vd)
        lat1, lon1, alt1: 上一点坐标
        lat2, lon2, alt2: 当前点坐标
        time_diff: 时间间隔 (秒)
        """
        if time_diff == 0:
            return 0, 0, 0  # 避免除零错误

        # 转换经纬度到弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # 计算北向速度 vn (纬度方向)
        vn = (lat2 - lat1) * self.EARTH_RADIUS / time_diff

        # 计算东向速度 ve (经度方向, 需要乘以 cos(纬度))
        ve = (lon2 - lon1) * self.EARTH_RADIUS * math.cos((lat1 + lat2) / 2) / time_diff

        # 计算垂直速度 vd (高度方向)
        vd = (alt2 - alt1) / time_diff

        return vn, ve, vd

    def send_gps_input(self, lat, lon, alt):
        """
        发送GPS位置信息
        lat: 当前纬度
        lon: 当前经度
        alt: 当前高度 (米)
        暂时不使用和计算速度信息，位置信息的误差容许值调的较宽泛
        """
        # 记录时间
        current_time = time.time()

        # 计算速度 (如果有上一点数据)。暂时不计算速度信息
        # if self.last_lat is not None and self.last_lon is not None and self.last_alt is not None and self.last_time is not None:
        #     time_diff = current_time - self.last_time
        #     vn, ve, vd = self.calculate_velocity(
        #         self.last_lat / 1e7, self.last_lon / 1e7, self.last_alt,
        #         lat / 1e7, lon / 1e7, alt,
        #         time_diff
        #     )
        # else:
        #     vn, ve, vd = 0, 0, 0  # 无上一点数据，速度设为 0
        #     self.last_time = current_time

        # 发送 MAVLink GPS_INPUT 消息
        self.connection.mav.gps_input_send(
            int(current_time * 1e6),  # 时间戳（微秒）
            1,  # gps_id
            0b00000000,  # ignore_flags (仅忽略 VD 速度)
            0,  # time_week_ms
            0,  # time_week
            3,  # fix_type (3D fix)
            lat,  # lat - 纬度(度 * 1e7)
            lon,  # lon - 经度(度 * 1e7)
            alt,  # alt - 高度(米)
            2.0,  # hdop - 水平精度因子
            2.0,  # vdop - 垂直精度因子
            self.vn,  # vn - 北向速度
            self.ve,  # ve - 东向速度
            self.vd,  # vd - 垂直速度
            3.0,  # speed_accuracy
            3.0,  # horiz_accuracy
            3.0,  # vert_accuracy
            8,  # satellites_visible
            0  # yaw
        )

    def send_gps_serial(self, lat, lon, alt):
        """
        模拟nmea串口字符串发送GPS位置信息
        lat: 纬度(度)
        lon: 经度(度)
        alt: 高度(米,相对地面)
        """

        nmea_gga = f"$GPGGA,{time.strftime('%H%M%S')},{lat:.4f},N,{lon:.4f},E,1,08,0.9,{alt:.1f},M,,M,,*"

        def calculate_checksum(nmea_sentence):
            # 计算 NMEA 语句的校验和
            checksum = 0
            for char in nmea_sentence:
                if char == '$':
                    continue
                if char == '*':
                    break
                checksum ^= ord(char)
            return f"{checksum:02X}"  # 返回两位十六进制数

        checksum = calculate_checksum(nmea_gga)
        nmea_gga = f"{nmea_gga}*{checksum}"
        # 发送 NMEA 数据
        if self.connection:
            self.connection_serial.sendall((nmea_gga + "\r\n").encode())
            print(f"Sent: {nmea_gga.strip()}")
        else:
            print("Connection is not established.")
        self.connection_serial.send(f"GPS: {lat}, {lon}, {alt}\n".encode())

    def set_attitude_target(self, roll, pitch, yaw, thrust):
        """
        控制飞机姿态
        参数:
        roll: 横滚角(弧度)
        pitch: 俯仰角(弧度)
        yaw: 偏航角(弧度)
        thrust: 油门(0-1范围)
        """

        def euler_to_quaternion(roll, pitch, yaw):
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)

            q = [0] * 4
            q[0] = cr * cp * cy + sr * sp * sy  # w
            q[1] = sr * cp * cy - cr * sp * sy  # x
            q[2] = cr * sp * cy + sr * cp * sy  # y
            q[3] = cr * cp * sy - sr * sp * cy  # z
            return q

        # 计算四元数
        q = euler_to_quaternion(roll, pitch, yaw)

        # type_mask:
        # bit 1: body roll rate
        # bit 2: body pitch rate
        # bit 3: body yaw rate
        # bit 7: throttle
        # bit 8: attitude
        type_mask = 0b00000111  # 忽略角速率，使用姿态四元数和油门

        self.connection.mav.set_attitude_target_send(
            0,  # 时间戳 (ms)
            self.connection.target_system,  # 目标系统
            self.connection.target_component,  # 目标组件
            type_mask,  # type_mask
            q,  # 四元数 (w, x, y, z)
            0, 0, 0,  # 角速度 (roll, pitch, yaw 弧度/秒)
            thrust  # 油门 (0-1)
        )
        start_time = time.time()
        reached_target = False

        while time.time() - start_time < 5 and not reached_target:
            # 获取当前姿态
            msg = self.connection.recv_match(type='ATTITUDE', blocking=True, timeout=0.1)
            if msg is not None:
                # 计算误差
                roll_error = abs(msg.roll - roll)
                pitch_error = abs(msg.pitch - pitch)
                yaw_error = abs(msg.yaw - yaw)

                # 打印当前状态
                print(f"当前姿态 - Roll: {math.degrees(msg.roll):.1f}°, "
                      f"Pitch: {math.degrees(msg.pitch):.1f}°, "
                      f"Yaw: {math.degrees(msg.yaw):.1f}°")
                print(f"姿态误差 - Roll: {math.degrees(roll_error):.1f}°, "
                      f"Pitch: {math.degrees(pitch_error):.1f}°, "
                      f"Yaw: {math.degrees(yaw_error):.1f}°")

                # 检查是否达到目标
                if (roll_error < 0.1 and
                        pitch_error < 0.1 and
                        yaw_error < 0.1):
                    reached_target = True
                    print("已达到目标姿态!")
                    break

            time.sleep(0.1)  # 短暂延时避免过度占用CPU

        if not reached_target:
            print("超时：未能达到目标姿态")

        return reached_target

    def send_guided_change_heading(self, heading_type, target_heading, heading_rate):
        """
        可以实现在全程无GPS情况下，发送改变航向的控制命令
        heading_type: 航向类型 (0: course-over-ground, 1: raw vehicle heading)
        target_heading: 目标航向(度, 0-359.99)
        heading_rate: 改变航向的速率(米/秒/秒)
        """
        self.set_mode("GUIDED")
        self.connection.mav.command_long_send(
            self.connection.target_system,  # target system
            self.connection.target_component,  # target component
            mavutil.mavlink.MAV_CMD_GUIDED_CHANGE_HEADING,  # command (MAV_CMD_GUIDED_CHANGE_HEADING)
            0,  # confirmation
            heading_type,  # param1: 航向类型
            target_heading,  # param2: 目标航向(度)
            heading_rate,  # param3: 航向改变速率
            0,  # param4: (空)
            0,  # param5: (空)
            0,  # param6: (空)
            0  # param7: (空)
        )

    def send_guided_waypoint(self, lat, lon, alt_relative):
        """
        指定一个全球坐标点，让固定翼飞机在GUIDED模式下飞向该点
        单点目标飞行，与mission航线任务飞行对应

        参数:
        lat: 纬度
        lon: 经度
        alt_relative: 相对起飞点的高度(米)
        """
        # 首先切换到GUIDED模式
        self.set_mode("GUIDED")

        # 发送MISSION_ITEM消息
        self.connection.mav.mission_item_send(
            self.connection.target_system,  # target system
            self.connection.target_component,  # target component
            0,  # sequence number (0)
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # frame
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,  # command (WAYPOINT)
            2,  # current (2 = guided mode waypoint)
            1,  # autocontinue
            0, 0, 0, 0,  # param1-4 (未使用)
            lat, lon, alt_relative  # x(lat), y(lon), z(alt)
        )

        # 等待命令被接受
        msg = self.connection.recv_match(type=['MISSION_ACK'], timeout=3)
        if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
            print("Waypoint accepted")
            return True
        else:
            print("Waypoint not accepted")
            return False

    def update_gps_position(self, current_lat, current_lon, alt, velocity_n, velocity_u, time_step):
        """
        计算GPS位置更新
        velocity_n: 北向速度 (m/s)
        velocity_u: 垂直速度 (m/s)
        time_step: 时间步长 (s)
        """
        # 计算这个时间步长内移动的距离
        distance = velocity_n * time_step  # 移动距离(米)

        # 计算纬度变化（地球周长约40075017米）
        delta_lat = (distance * 360) / 40075017  # 将距离转换为纬度变化
        new_lat = current_lat + delta_lat

        # 计算经度变化
        earth_circumference = 40075017  # 地球周长(米)
        # 在当前纬度下的地球周长
        radius_at_latitude = earth_circumference * math.cos(math.radians(current_lat))
        # 将距离转换为经度变化
        delta_lon = (distance * 360) / radius_at_latitude
        new_lon = current_lon + delta_lon

        # 计算高度变化
        new_alt = alt + (velocity_u * time_step)

        return new_lat, current_lon, new_alt

    def takeoff_without_gps(self):
        """无GPS条件下的固定翼起飞"""
        # 确保在 STABILIZE 模式
        self.set_mode("FBWA")
        time.sleep(1)
        try:
            # 初始化所有通道为中位值
            # 1. 逐步增加油门
            self.rc_values[4] = 0
            self.logger.log(f"开始加速")
            for throttle in range(1200, 1800, 50):  # 从中位值逐步增加到80%油门
                self.rc_values[2] = throttle  # 通道3是油门
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)

            # 2. 等待速度建立
            self.logger.log(f"保持速度")
            for _ in range(30):  # 保持3秒
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values

                )
                time.sleep(0.1)

            # 3. 抬升机头
            self.logger.log(f"抬升机头")
            self.rc_values[1] = 1700  # 通道2上拉（俯仰）
            for _ in range(150):  # 保持3秒
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)
            # 4. 保持一段时间让飞机爬升
            # 5. 恢复平飞姿态
            self.logger.log(f"调整为平飞")
            for _ in range(30):  # 保持3秒
                self.rc_values[1] = 1500  # 通道2上拉（俯仰）
                self.connection.mav.rc_channels_override_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    *self.rc_values
                )
                time.sleep(0.1)
            # 确保模式稳定
            self.logger.log(f"起飞结束")
            #关闭遥控器模拟信号，必要时会显示飞机失控，模拟飞行过远遥控器信号失联情况
            self.connection.mav.param_set_send(
                self.connection.target_system,
                self.connection.target_component,
                b'SIM_RC_FAIL',
                1,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
            #启动thread维持飞机油门
            thread = threading.Thread(target=self.thorottle_thread)
            thread.daemon = True  # 设为守护线程，主线程结束时子线程也结束
            thread.start()
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

    # 模拟摇杆控制的相关函数
    def reset_channels(self):
        """除油门外所有通道回中"""
        # 保存当前油门值
        current_throttle = self.rc_values[2]

        # 重置所有通道到中位
        self.rc_values[:4] = [1500] * 4

        # 恢复油门值
        self.rc_values[2] = current_throttle

        # 清除当前活动的控制（除了油门相关的按键）

    def key_press_handler(self, event):
        """按键按下处理"""
        key = event.name
        print("press: ", key)
        if key == 'W':  # 抬头
            self.rc_values[1] += 10  # 通道2抬升
            self.active_controls.add(key)
        elif key == 'S':  # 低头
            self.rc_values[1] -= 10  # 通道2下压
            self.active_controls.add(key)
        elif key == 'A':  # 左转
            self.rc_values[0] -= 10  # 副翼向左
            self.active_controls.add(key)
        elif key == 'D':  # 右转
            self.rc_values[0] += 10  # 副翼向右
            self.active_controls.add(key)
        elif key == 'up':  # 油门加
            self.rc_values[2] += 10  # 通道3 油门增加
        elif key == 'down':  # 油门减
            self.rc_values[2] -= 10  # 通道3 油门减小
        elif key == 'left':  # 左偏航
            self.rc_values[3] -= 10  # 通道4 方向舵左
        elif key == 'right':  # 右偏航
            self.rc_values[3] += 10  # 通道4 方向舵右
        elif key == '+':  # 加号增大转弯角度
            self.turn_angle += 10
            print("present turn_angle:", self.turn_angle)
        elif key == '-':  # 减号减小转弯角度
            self.turn_angle -= 10
            print("present turn_angle:", self.turn_angle)
        elif key == '4':  # 小键盘 4在当前基础上向左转
            # msg = self.wait_for_message("ATTITUDE",2)
            # print("present yaw:", msg.yaw*180/3.1415926)
            # target_yaw = msg.yaw*180/3.1415926 - self.turn_angle
            # if target_yaw < 0:
            #     target_yaw += 360
            self.current_yaw -= self.turn_angle
            if self.current_yaw < 0:
                self.current_yaw += 360
            print("target yaw:", self.current_yaw)
            self.send_guided_change_heading(1, self.current_yaw, self.turn_rate)
        elif key == '6':  # 小键盘 6 在当前基础上向右转
            # msg = self.wait_for_message("ATTITUDE",2)
            # print("present yaw:", msg.yaw*180/3.1415926)
            # target_yaw = msg.yaw*180/3.1415926 + self.turn_angle
            # if target_yaw > 360:
            #     target_yaw -= 360
            self.current_yaw += self.turn_angle
            if self.current_yaw > 360:
                self.current_yaw -= 360
            print("target yaw:", self.current_yaw)
            self.send_guided_change_heading(1, self.current_yaw, self.turn_rate)
        elif key == '8':  # 数字键8加大转弯速率
            self.turn_rate += 5  # 通道1调整
            print("present turn_rate:", self.turn_rate)
        elif key == '2':  # 数字键2 减小转弯速率
            self.turn_rate -= 5  #
            if self.turn_rate < 0:
                self.turn_rate = 5
            print("present turn_rate:", self.turn_rate)
        elif key == 'space':  # 除油门外所有通道回中
            self.reset_channels()
        elif key == "0":
            self.gps = 1 - self.gps
            self.connection.mav.param_set_send(
                self.connection.target_system,
                self.connection.target_component,
                b'SIM_GPS_DISABLE',
                self.gps,
                mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )
            print("GPS_DISABLE: ", self.gps)

        # print("keyname:",key)
        # if key in self.key_mapping:
        #     if key == 'space':
        #         self.reset_channels()
        #     else:
        #         self.active_controls.add(key)

    def key_release_handler(self, event):
        """按键释放处理"""
        print("触发release")
        key = event.name

    def get_global_position(self):
        return self.real_lat, self.real_lon, self.current_alt, self.global_position_lat, self.global_position_lon
    
    def refresh_msg(self):
        while True:
            while True:
                msg=self.connection.recv_msg()
                if msg is None:
                    break
            # if msg_type == 'SIMSTATE':
            self.real_lat=self.connection.messages['SIMSTATE'].lat / 1e7
            self.real_lon=self.connection.messages['SIMSTATE'].lng / 1e7
            # elif msg_type == 'GLOBAL_POSITION_INT':
            self.global_position_lat=self.connection.messages['GLOBAL_POSITION_INT'].lat / 1e7
            self.global_position_lon=self.connection.messages['GLOBAL_POSITION_INT'].lon / 1e7
            # elif msg_type == 'VFR_HUD':
            self.current_alt=self.connection.messages['VFR_HUD'].alt
            # elif msg_type == 'LOCAL_POSITION_NED':
            self.vn,self.ve,self.vd=self.connection.messages['LOCAL_POSITION_NED'].vx,self.connection.messages['LOCAL_POSITION_NED'].vy,self.connection.messages['LOCAL_POSITION_NED'].vz
    
    def start_thread(self):
        # msg_types = ['SIMSTATE', 'GLOBAL_POSITION_INT', 'VFR_HUD', 'LOCAL_POSITION_NED']

        # for msg_type in msg_types:
        thread = threading.Thread(target=self.refresh_msg)
        thread.daemon = True  # 设为守护线程，主线程结束时子线程也结束
            
        thread.start()
    
    def thorottle_thread(self):
        while True:          
            self.connection.mav.rc_channels_override_send(
                self.connection.target_system,
                self.connection.target_component,
                *self.rc_values
            )
            time.sleep(0.1)

    def update_global_position(self, current_lat, current_lon, current_alt):
        """更新当前位置"""
        self.send_gps_input(current_lat, current_lon, current_alt)
        self.last_alt = current_alt
        self.last_lat = current_lat
        self.last_lon = current_lon

    # def control_loop(self):
    #     """主控制循环"""

    #     self.rc_values[2] = 1700

    #     print("开始控制循环...")
    #     # self.set_local_position(1000,1000,-200)
    #     # self.send_guided_change_heading(1,150,20)
    #     # self.send_guided_change_heading(1, 90, self.turn_rate)
    #     self.connection.mav.rc_channels_override_send(
    #         self.connection.target_system,
    #         self.connection.target_component,
    #         *self.rc_values
    #     )
    #     while self.is_running:

    #         self.connection.mav.rc_channels_override_send(
    #             self.connection.target_system,
    #             self.connection.target_component,
    #             *self.rc_values
    #         )
    #         msg = self.connection.recv_match(
    #             type=['SIMSTATE', 'VFR_HUD', 'GPS_RAW_INT'],
    #             blocking=True
    #         )
    #         # 角度信息暂时不用
    #         # if msg.get_type() == 'ATTITUDE':
    #         #     self.current_yaw = msg.yaw*180/3.1415926
    #         #     if self.current_yaw < 0:
    #         #         self.current_yaw += 360
    #         #     else :
    #         #         self.current_yaw = self.current_yaw % 360
    #         # 从SIMSTATE获取位置信息
    #         if msg.get_type() == 'SIMSTATE':
    #             self.current_lat = msg.lat
    #             self.current_lon = msg.lng
    #         # 从VFR_HUD获取高度信息，获取的是海拔高度，非相对地面高度，应该是根据气压高程计获取
    #         elif msg.get_type() == 'VFR_HUD':
    #             self.current_alt = msg.alt  # 高度

    #             # self.send_gps_serial(lat, lon, alt)
    #         # 用mavlink消息发送经纬度与高程信息给飞控，暂时未解算并发送速度信息
    #         self.send_gps_input(self.current_lat, self.current_lon, self.current_alt)
    #         self.last_alt = self.current_alt
    #         self.last_lat = self.current_lat
    #         self.last_lon = self.current_lon
    #         time.sleep(0.05)  # 20Hz 控制频率

    # def run(self):
    #     """启动控制器"""
    #     try:
    #         # 注册按键事件处理器
    #         keyboard.on_press(self.key_press_handler)
    #         keyboard.on_release(self.key_release_handler)

    #         # 启动控制循环
    #         print("开始监听按键控制...")
    #         print("使用 WSAD 控制姿态，方向键控制油门和偏航")
    #         print("空格键：除油门外所有通道回中")
    #         print("按 ESC 退出")
    #         self.control_loop()

    #     except KeyboardInterrupt:
    #         print("\n程序已终止")
    #     finally:
    #         self.cleanup()

    # def cleanup(self):
    #     """清理资源"""
    #     self.is_running = False
    #     # 恢复所有通道到中位
    #     self.rc_values[:4] = [1500] * 4
    #     self.connection.mav.rc_channels_override_send(
    #         self.connection.target_system,
    #         self.connection.target_component,
    #         *self.rc_values
    #     )
    #     # 取消按键监听
    #     keyboard.unhook_all()
