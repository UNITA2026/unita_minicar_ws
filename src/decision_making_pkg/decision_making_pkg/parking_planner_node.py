#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import LaserScan

try:
    from interfaces_pkg.msg import MotionCommand
except ImportError:
    class MotionCommand:
        steering = 0
        left_speed = 0
        right_speed = 0

# ==========================================
# [설정 파라미터]
# ==========================================
PUB_CTRL_TOPIC = "topic_control_signal"
PUB_STATE_TOPIC = "/parking/planner_state"

SUB_LIDAR_TOPIC = "/scan"
SUB_CONE_DET_TOPIC = "/parking2/cones_detected"
SUB_CONE_MID_TOPIC = "/parking2/cone_mid_x_norm" 
SUB_V_PAIR_TOPIC = "/parking/vertical_pair_detected"
SUB_LANE_ERR_TOPIC = "/parking/lane_center_error_px"
SUB_STOP_LINE_TOPIC = "/parking/stop_line_detected"

# 속도 설정
FORWARD_PWM = 60         
REVERSE_PWM = 50         
STOP_PWM = 0           

STEER_MAX_LEFT = -7      
STEER_MAX_RIGHT = 7      
STEER_CENTER = 0

LIDAR_CHECK_MIN_DEG = 103.0
LIDAR_CHECK_MAX_DEG = 108.0
OBSTACLE_DIST_THR = 0.8 
GAP_DIST_THR = 1.5       

# 오른쪽 충돌 방지 오프셋
TARGET_OFFSET = -0.1   

# ✅ [수정 1] 스윙 정지 조건 (중앙 정렬 오차 허용 범위)
# 0.5(중앙) +- 0.05 (0.45 ~ 0.55 사이면 정지)
CENTER_TOLERANCE = 0.1 

CONE_P_GAIN = 3.0

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

class ParkingPlannerNode(Node):
    def __init__(self):
        super().__init__("parking_planner_node")
        self.qos_reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # ---------------- 상태 정의 ----------------
        self.S_SEARCH_OBSTACLE = 0 
        self.S_SEARCH_GAP = 1       
        self.S_SWING_LEFT = 2       
        self.S_WAIT_GEAR = 3        
        self.S_REVERSE_ENTRY = 4    
        self.S_PARALLEL_STOP = 5    
        self.S_REVERSE_STRAIGHT = 6 
        self.S_FINISH = 99          

        self.state = self.S_SEARCH_OBSTACLE
        self.state_start_time = 0.0      
        
        self.min_lidar_dist = 99.0
        self.is_cone_detected = False      
        self.cone_mid_x = -1.0             # 초기값 -1 (감지 안됨)
        self.is_v_pair_detected = False    
        self.lane_error_px = 0.0           
        self.is_stop_line_detected = False 
        
        self.create_subscription(LaserScan, SUB_LIDAR_TOPIC, self.cb_lidar, self.qos_sensor)
        self.create_subscription(Bool, SUB_CONE_DET_TOPIC, self.cb_cone, 10)
        self.create_subscription(Float32, SUB_CONE_MID_TOPIC, self.cb_cone_mid, 10)
        self.create_subscription(Bool, SUB_V_PAIR_TOPIC, self.cb_v_pair, 10)
        self.create_subscription(Float32, SUB_LANE_ERR_TOPIC, self.cb_lane_err, 10)
        self.create_subscription(Bool, SUB_STOP_LINE_TOPIC, self.cb_stop_line, 10)

        self.pub_ctrl = self.create_publisher(MotionCommand, PUB_CTRL_TOPIC, self.qos_reliable)
        self.pub_state = self.create_publisher(String, PUB_STATE_TOPIC, 10)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Parking Planner: Center Align Stop & Blind Reverse Mode.")

    def get_time_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def cb_cone(self, msg: Bool): self.is_cone_detected = msg.data
    def cb_cone_mid(self, msg: Float32): 
        # -1.0이면 감지 안된 것
        self.cone_mid_x = msg.data
    def cb_v_pair(self, msg: Bool): self.is_v_pair_detected = msg.data
    def cb_lane_err(self, msg: Float32): self.lane_error_px = msg.data
    def cb_stop_line(self, msg: Bool): self.is_stop_line_detected = msg.data
    def cb_lidar(self, msg: LaserScan):
        if not msg.ranges: return
        target_min_rad = math.radians(LIDAR_CHECK_MIN_DEG)
        target_max_rad = math.radians(LIDAR_CHECK_MAX_DEG)
        idx_min = int((target_min_rad - msg.angle_min) / msg.angle_increment)
        idx_max = int((target_max_rad - msg.angle_min) / msg.angle_increment)
        arr_len = len(msg.ranges)
        idx_min = clamp(idx_min, 0, arr_len - 1)
        idx_max = clamp(idx_max, 0, arr_len - 1)
        if idx_min > idx_max: idx_min, idx_max = idx_max, idx_min
        roi = np.array(msg.ranges[idx_min:idx_max+1])
        valid = roi[(roi > 0.05) & (roi < 10.0)]
        self.min_lidar_dist = np.min(valid) if len(valid) > 0 else 99.0

    def control_loop(self):
        current_time = self.get_time_sec()
        
        steer_cmd = STEER_CENTER
        speed_cmd = 0
        state_str = "UNKNOWN"

        # [1] 장애물 탐색
        if self.state == self.S_SEARCH_OBSTACLE:
            state_str = "SEARCH_OBSTACLE"
            speed_cmd = FORWARD_PWM
            if self.min_lidar_dist < OBSTACLE_DIST_THR: self.state = self.S_SEARCH_GAP

        # [2] 빈 공간 탐색
        elif self.state == self.S_SEARCH_GAP:
            state_str = "SEARCH_GAP"
            speed_cmd = FORWARD_PWM
            if self.min_lidar_dist > GAP_DIST_THR: self.state = self.S_SWING_LEFT

        # [3] 스윙 (왼쪽 전진) -> 꼬깔이 화면 중앙에 올 때까지
        elif self.state == self.S_SWING_LEFT:
            state_str = "SWING_LEFT"
            steer_cmd = STEER_MAX_LEFT
            speed_cmd = FORWARD_PWM
            
            # ✅ [수정 1] 꼬깔이 화면 중앙(0.5) 부근에 오면 정지
            # cone_mid_x가 유효하고(0.0 이상), 0.5 근처인지 확인
            if self.cone_mid_x >= 0.0:
                diff = abs(self.cone_mid_x - 0.5)
                if diff < CENTER_TOLERANCE:
                    self.get_logger().info(f"Cone Centered ({self.cone_mid_x:.2f})! Stopping.")
                    self.state_start_time = current_time
                    self.state = self.S_WAIT_GEAR
            
            # (안전장치) 너무 오래 돌면 강제 정지 (예: 10초)
            # if current_time - self.state_start_time > 10.0: ...

        # [4] 정지 및 핸들 우측 최대 꺾기 (대기)
        elif self.state == self.S_WAIT_GEAR:
            speed_cmd = STOP_PWM
            
            # ✅ [수정 2] 무조건 오른쪽 최대로 꺾고 대기 (진입 준비)
            steer_cmd = STEER_MAX_RIGHT 

            elapsed = current_time - self.state_start_time
            state_str = f"WAIT_RIGHT ({elapsed:.1f}s)"
            
            if elapsed > 2.0:
                self.get_logger().info("Reverse Entry Start!")
                self.state = self.S_REVERSE_ENTRY

        # [5] 후진 진입 (우측 최대 고정)
        elif self.state == self.S_REVERSE_ENTRY:
            speed_cmd = -REVERSE_PWM 
            
            # ✅ [수정 3] 핸들은 무조건 우측 최대 고정
            steer_cmd = 0
            state_str = "ENTRY_RIGHT_MAX"
            
            # 평행(Vertical Pair) 감지되면 -> 정지 모드 (Step 5.5)
            if self.is_v_pair_detected:
                self.get_logger().warn(">>> Parallel Detected! STOP & LOCK <<<")
                self.state_start_time = current_time
                self.state = self.S_PARALLEL_STOP

        # [5.5] 평행 정지 (1.5초간 자세 잡기 & 핸들 0)
        elif self.state == self.S_PARALLEL_STOP:
            speed_cmd = STOP_PWM       
            steer_cmd = STEER_CENTER   # 핸들 0으로 정렬
            
            elapsed = current_time - self.state_start_time
            state_str = f"ALIGN_STOP ({elapsed:.1f}s)"
            
            if elapsed > 1.5:
                self.get_logger().info("Locked! Going Blind Straight.")
                self.state = self.S_REVERSE_STRAIGHT

        # [6] 직진 후진 (Absolute Lock)
        elif self.state == self.S_REVERSE_STRAIGHT:
            state_str = "REVERSE_STRAIGHT"
            speed_cmd = -REVERSE_PWM
            
            # ✅ [수정 4] 무조건 핸들 0 고정 (추종 X)
            steer_cmd = STEER_CENTER  
            
            if self.is_stop_line_detected: 
                self.get_logger().warn(">>> Stop Line Detected! FINISH <<<")
                self.state = self.S_FINISH

        # [7] 완료
        elif self.state == self.S_FINISH:
            state_str = "FINISH"
            steer_cmd = STEER_CENTER
            speed_cmd = STOP_PWM

        self.publish_command(steer_cmd, speed_cmd)
        self.pub_state.publish(String(data=f"[{state_str}]"))

    def publish_command(self, steer, speed):
        msg = MotionCommand()
        msg.steering = int(steer)
        msg.left_speed = int(speed)
        msg.right_speed = int(speed)
        self.pub_ctrl.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParkingPlannerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.publish_command(0, 0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()