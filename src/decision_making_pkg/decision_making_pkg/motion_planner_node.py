#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
)

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand


# --------------- Tunable Params (Top) ---------------
# Topics
SUB_PATH_TOPIC_NAME = "path_planning_result"          # 수신할 경로 토픽
PUB_TOPIC_NAME = "topic_control_signal"               # 발행할 모션 명령 토픽
SUB_DETECTION_TOPIC_NAME = "detections"               # 객체 검출 결과 토픽
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"  # 신호등 인식 토픽

# Feature toggles
USE_TRAFFIC_LIDAR_STOP = True                         # 신호등/라이다 정지 로직 사용 여부
USE_PD = True                                         # PD 보정 조향 사용 여부

# Vehicle/Image
CAR_CENTER_POINT = [320, 179]                         # 차량 기준 픽셀 좌표 (x, y)
CAR_CENTER_X = 320                                    # PD 횡오차 기준 중심 x
VEHICLE_HEADING_RAD = -1.57079632679                  # 차량 진행방향 라디안(기본 위쪽)
TIMER = 0.1                                           # 제어 주기(초), 0.1=10Hz

# Pure Pursuit
LOOKAHEAD_DISTANCE = 170.0                            # 목표점 거리(작을수록 민감, 클수록 완만)
WHEELBASE = 50.0                                      # 가상 휠베이스(클수록 조향 계산 완만)
MAX_STEER_ANGLE_RAD = 0.55                            # 조향각 정규화 기준(작을수록 출력 커짐)
MAX_STEER_CMD = 9.0                                   # 최종 조향 명령 최대 절대값

# PD
KP = 0.01                                             # 횡오차 비례 이득(즉각 조향 강도) 초기값 0.05
KD = 0.045                                             # 변화량 이득(진동 억제/선행 보정)
MAX_PD_STEER = 4.0                                    # PD 보정 최대 절대값
LOOKAHEAD_Y = 155                                     # PD가 참조할 y 라인(화면 아래쪽일수록 가까움)

# Speed
BASE_SPEED = 120                                      # 기본 주행 속도
MIN_SPEED = 80                                       # 최소 속도 하한
MAX_SPEED = 150                                # 최대 속도 상한
STEER_SPEED_GAIN = 12.0                               # 조향 클수록 감속시키는 계수
# ----------------------------------------------------


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class UnitaPurePursuitNode(Node):
    def __init__(self):
        super().__init__("motion_planner_node")

        # -------------------------
        # Topic parameters
        # -------------------------
        self.sub_path_topic = self.declare_parameter("sub_path_topic", SUB_PATH_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value

        self.use_traffic_lidar_stop = bool(
            self.declare_parameter("use_traffic_lidar_stop", USE_TRAFFIC_LIDAR_STOP).value
        )
        self.sub_detection_topic = self.declare_parameter("sub_detection_topic", SUB_DETECTION_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter("sub_traffic_light_topic", SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.timer_period = float(self.declare_parameter("timer", TIMER).value)

        # -------------------------
        # Pure Pursuit parameters
        # -------------------------
        self.lookahead_distance = float(self.declare_parameter("lookahead_distance", LOOKAHEAD_DISTANCE).value)
        self.wheelbase = float(self.declare_parameter("wheelbase", WHEELBASE).value)
        self.max_steer_angle = float(self.declare_parameter("max_steer_angle_rad", MAX_STEER_ANGLE_RAD).value)
        self.max_steer_cmd = float(self.declare_parameter("max_steer_cmd", MAX_STEER_CMD).value)

        cp = self.declare_parameter("car_center_point", CAR_CENTER_POINT).value
        self.car_center_point = (int(cp[0]), int(cp[1]))
        self.vehicle_heading = float(self.declare_parameter("vehicle_heading_rad", VEHICLE_HEADING_RAD).value)

        # -------------------------
        # PD parameters
        # -------------------------
        self.use_pd = bool(self.declare_parameter("use_pd", USE_PD).value)
        self.prev_dx = 0.0
        self.car_center_x = int(self.declare_parameter("car_center_x", CAR_CENTER_X).value)

        self.Kp = float(self.declare_parameter("Kp", KP).value)
        self.Kd = float(self.declare_parameter("Kd", KD).value)
        self.lookahead_y = int(self.declare_parameter("lookahead_y", LOOKAHEAD_Y).value)

        self.max_steer = float(self.declare_parameter("max_steer", MAX_PD_STEER).value)
        self.base_speed = int(self.declare_parameter("base_speed", BASE_SPEED).value)
        self.min_speed = int(self.declare_parameter("min_speed", MIN_SPEED).value)
        self.max_speed = int(self.declare_parameter("max_speed", MAX_SPEED).value)
        self.steer_speed_gain = float(self.declare_parameter("steer_speed_gain", STEER_SPEED_GAIN).value)

        # QoS
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Data holders
        self.path_data = None
        self.detection_data = None
        self.traffic_light_data = None
        self.lidar_data = None

        # Subscribers
        self.path_sub = self.create_subscription(
            PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile
        )

        if self.use_traffic_lidar_stop:
            self.detection_sub = self.create_subscription(
                DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile
            )
            self.traffic_light_sub = self.create_subscription(
                String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile
            )

        # Publisher
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    # -------------------------
    # Callbacks
    # -------------------------
    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    # -------------------------
    # Helpers
    # -------------------------
    def find_lookahead_point(self, path):
        car_x, car_y = self.car_center_point

        forward_points = [p for p in path if p[1] <= car_y]
        if not forward_points:
            forward_points = path

        for p in forward_points:
            dx = p[0] - car_x
            dy = p[1] - car_y
            if math.hypot(dx, dy) >= self.lookahead_distance:
                return p

        return forward_points[0] if forward_points else None

    def compute_pp_steer_cmd(self, path):
        if not path:
            return 0.0
        if self.lookahead_distance <= 0.0 or self.max_steer_angle <= 0.0:
            return 0.0

        lookahead = self.find_lookahead_point(path)
        if lookahead is None:
            return 0.0

        car_x, car_y = self.car_center_point
        lx, ly = lookahead

        target_angle = math.atan2(ly - car_y, lx - car_x)
        alpha = normalize_angle(target_angle - self.vehicle_heading)

        steer_angle = math.atan2(
            2.0 * self.wheelbase * math.sin(alpha),
            self.lookahead_distance,
        )

        steer_cmd = (steer_angle / self.max_steer_angle) * self.max_steer_cmd
        steer_cmd = clamp(steer_cmd, -self.max_steer_cmd, self.max_steer_cmd)
        return float(steer_cmd)

    def compute_pd_steer_cmd(self, path):
        if not path or len(path) < 5:
            self.prev_dx = 0.0
            return 0.0

        x_target, _y_target = min(path, key=lambda p: abs(p[1] - self.lookahead_y))
        dx = float(x_target - self.car_center_x)

        steer = self.Kp * dx + self.Kd * (dx - self.prev_dx)
        self.prev_dx = dx

        steer = clamp(steer, -self.max_steer, self.max_steer)
        return float(steer)

    def should_stop_by_traffic(self):
        if self.traffic_light_data is None:
            return False
        if self.traffic_light_data.data != "Red":
            return False

        if self.detection_data is None:
            return True

        for detection in self.detection_data.detections:
            if getattr(detection, "class_name", "") == "traffic_light":
                y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                if y_max < 150:
                    return True
        return False

    # -------------------------
    # Main loop
    # -------------------------
    def timer_callback(self):
        if not self.path_data:
            self.publish_cmd(0.0, 0, 0)
            return

        if self.use_traffic_lidar_stop:
            if self.lidar_data is not None and self.lidar_data.data is True:
                self.publish_cmd(0.0, 0, 0)
                return
            if self.should_stop_by_traffic():
                self.publish_cmd(0.0, 0, 0)
                return

        steer_pp = self.compute_pp_steer_cmd(self.path_data)

        steer_pd = 0.0
        if self.use_pd:
            steer_pd = self.compute_pd_steer_cmd(self.path_data)

        steer_cmd = steer_pp + steer_pd
        steer_cmd = clamp(steer_cmd, -self.max_steer_cmd, self.max_steer_cmd)

        speed = int(self.base_speed - self.steer_speed_gain * abs(steer_cmd))
        speed = int(clamp(speed, self.min_speed, self.max_speed))

        self.publish_cmd(steer_cmd, speed, speed)

    def publish_cmd(self, steering, left_speed, right_speed):
        msg = MotionCommand()
        msg.steering = int(round(steering))
        msg.left_speed = int(left_speed)
        msg.right_speed = int(right_speed)
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = UnitaPurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()