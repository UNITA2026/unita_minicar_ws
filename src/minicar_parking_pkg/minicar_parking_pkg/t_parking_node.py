#!/usr/bin/env python3
import math
from enum import Enum
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# MotionCommand가 있으면 사용 (없으면 Twist로만)
try:
    from interfaces_pkg.msg import MotionCommand
    HAVE_MOTION_COMMAND = True
except Exception:
    HAVE_MOTION_COMMAND = False


class State(Enum):
    CRUISE = 1
    APPROACH = 2
    ENTER_CENTER = 3
    PARKED = 4


@dataclass
class Gap:
    found: bool = False
    mid_angle: float = 0.0
    width_est: float = 0.0
    goal_xy: np.ndarray = None


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class TParkingNode(Node):
    def __init__(self):
        super().__init__("t_parking_node")

        # ========== Parameters ==========
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_topic", "/cmd_vel")

        # auto | motion_command | twist
        self.declare_parameter("publish_mode", "auto")

        # serial_sender가 구독하는 토픽
        self.declare_parameter("motion_cmd_topic", "topic_control_signal")

        # 주차 공간이 진행 방향 기준 오른쪽/왼쪽
        self.declare_parameter("use_side", "right")  # right / left

        # scan preprocess
        self.declare_parameter("range_min", 0.08)
        self.declare_parameter("range_max", 8.0)

        # CRUISE
        self.declare_parameter("cruise_speed", 0.6)
        self.declare_parameter("wall_dist_des", 0.8)
        self.declare_parameter("wall_kp", 1.2)

        # Gap detect
        self.declare_parameter("gap_angle_min_deg", 20.0)
        self.declare_parameter("gap_angle_max_deg", 80.0)
        self.declare_parameter("gap_free_range", 2.5)
        self.declare_parameter("gap_min_width", 0.65)
        self.declare_parameter("gap_goal_dist", 1.2)

        # Approach
        self.declare_parameter("approach_speed", 0.35)
        self.declare_parameter("approach_kp", 2.0)
        self.declare_parameter("reach_dist", 0.7)

        # Enter & center
        self.declare_parameter("enter_speed", 0.25)
        self.declare_parameter("center_kp", 1.6)
        self.declare_parameter("side_window_deg", 25.0)
        self.declare_parameter("left_center_deg", 60.0)
        self.declare_parameter("right_center_deg", -60.0)

        # Stop
        self.declare_parameter("front_window_deg", 10.0)
        self.declare_parameter("stop_dist", 0.35)
        self.declare_parameter("hard_stop_dist", 0.20)

        # ===== MotionCommand 변환 파라미터 (핵심) =====
        # steering: int32, left_speed/right_speed: int32
        self.declare_parameter("steer_max", 30)                 # steering clamp
        self.declare_parameter("w_to_steer_gain", 15.0)         # w(rad/s) * gain -> steering
        self.declare_parameter("speed_max", 300)                # left/right_speed clamp
        self.declare_parameter("v_to_speed_gain", 300.0)        # v(m/s) * gain -> speed
        self.declare_parameter("speed_deadband", 5)             # |speed| < deadband -> 0
        self.declare_parameter("speed_min_abs", 60)             # v != 0인데 너무 작으면 최소 구동값

        # 안전 정지(노이즈 1번으로 PARKED 가는거 방지)
        self.declare_parameter("hard_stop_count", 5)            # 연속 N번 이하일 때만 hard stop

        self.declare_parameter("control_rate_hz", 20.0)

        # ========== IO ==========
        scan_topic = self.get_parameter("scan_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        motion_cmd_topic = self.get_parameter("motion_cmd_topic").value

        self.sub_scan = self.create_subscription(LaserScan, scan_topic, self.cb_scan, 10)
        self.pub_twist = self.create_publisher(Twist, cmd_topic, 10)

        self.pub_motion = None
        if HAVE_MOTION_COMMAND:
            self.pub_motion = self.create_publisher(MotionCommand, motion_cmd_topic, 10)

        hz = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / hz, self.on_timer)

        self.state = State.CRUISE
        self.scan = None
        self.locked_gap = None

        self._hard_stop_hits = 0

        self.get_logger().info(
            f"TParkingNode started. scan={scan_topic}, motion={motion_cmd_topic}, HAVE_MOTION_COMMAND={HAVE_MOTION_COMMAND}"
        )

    def cb_scan(self, msg: LaserScan):
        self.scan = msg

    # ---------- Publishing ----------
    def publish_cmd(self, v: float, w: float):
        """
        v: m/s (양수 전진)
        w: rad/s (우리 내부에서 벽추종/센터링에 쓰는 yaw rate 느낌)
        MotionCommand은 (steering, left_speed, right_speed) int32 이므로 변환해서 보냄.
        """
        mode = str(self.get_parameter("publish_mode").value).lower()

        if mode == "motion_command":
            use_motion = True
        elif mode == "twist":
            use_motion = False
        else:
            use_motion = HAVE_MOTION_COMMAND and (self.pub_motion is not None)

        # ---- MotionCommand로 publish ----
        if use_motion and self.pub_motion is not None:
            msg = MotionCommand()

            steer_max = int(self.get_parameter("steer_max").value)
            w_gain = float(self.get_parameter("w_to_steer_gain").value)

            speed_max = int(self.get_parameter("speed_max").value)
            v_gain = float(self.get_parameter("v_to_speed_gain").value)
            deadband = int(self.get_parameter("speed_deadband").value)
            speed_min_abs = int(self.get_parameter("speed_min_abs").value)

            # steering int32
            steer_cmd = int(round(clamp(w * w_gain, -steer_max, steer_max)))

            # speed int32
            speed_cmd = int(round(clamp(v * v_gain, -speed_max, speed_max)))

            # deadband 처리
            if abs(speed_cmd) < deadband:
                speed_cmd = 0

            # v가 분명히 0이 아닌데 speed_cmd가 0으로 죽는 경우(정지마찰 때문에)
            if v != 0.0 and speed_cmd == 0:
                speed_cmd = int(math.copysign(speed_min_abs, v))

            # 이 플랫폼은 보통 좌우 속도 동일 + steering 으로 조향(“차동 조향” 섞지 않음)
            msg.steering = int(steer_cmd)
            msg.left_speed = int(speed_cmd)
            msg.right_speed = int(speed_cmd)

            self.pub_motion.publish(msg)
            return

        # ---- Twist로 publish (디버그/시뮬용) ----
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_twist.publish(t)

    # ---------- Scan utils ----------
    def _angles_ranges(self):
        s = self.scan
        n = len(s.ranges)
        angles = s.angle_min + np.arange(n) * s.angle_increment
        ranges = np.array(s.ranges, dtype=np.float32)

        prm_rmin = float(self.get_parameter("range_min").value)
        prm_rmax = float(self.get_parameter("range_max").value)

        # 센서 range_max와 파라미터 range_max 중 더 작은 값을 사용 (센서값이 유효할 때)
        sen_rmax = float(getattr(s, "range_max", 0.0))
        use_rmax = min(prm_rmax, sen_rmax) if sen_rmax > 0.0 else prm_rmax

        ranges[~np.isfinite(ranges)] = use_rmax
        ranges = np.clip(ranges, prm_rmin, use_rmax)
        return angles, ranges

    def min_range_window(self, center_rad: float, half_rad: float) -> float:
        angles, ranges = self._angles_ranges()
        c = wrap_pi(center_rad)
        mask = (angles >= (c - half_rad)) & (angles <= (c + half_rad))
        if not np.any(mask):
            return float(self.get_parameter("range_max").value)
        return float(np.min(ranges[mask]))

    # ---------- Behaviors ----------
    def detect_gap(self) -> Gap:
        side = str(self.get_parameter("use_side").value).lower()
        amin = math.radians(float(self.get_parameter("gap_angle_min_deg").value))
        amax = math.radians(float(self.get_parameter("gap_angle_max_deg").value))

        center = (-(amin + amax) / 2.0) if side == "right" else (+(amin + amax) / 2.0)
        half = (amax - amin) / 2.0

        angles, ranges = self._angles_ranges()
        c = wrap_pi(center)
        mask = (angles >= (c - half)) & (angles <= (c + half))
        ang_w = angles[mask]
        ran_w = ranges[mask]
        if ran_w.size < 8:
            return Gap(found=False)

        free_thr = float(self.get_parameter("gap_free_range").value)
        free = ran_w > free_thr
        idx = np.where(free)[0]
        if idx.size == 0:
            return Gap(found=False)

        runs = []
        s0 = idx[0]
        p0 = idx[0]
        for k in idx[1:]:
            if k == p0 + 1:
                p0 = k
            else:
                runs.append((s0, p0))
                s0, p0 = k, k
        runs.append((s0, p0))

        inc = float(self.scan.angle_increment)
        best = None
        best_w = 0.0
        for a, b in runs:
            d_ang = (b - a + 1) * inc
            mean_r = float(np.mean(ran_w[a:b + 1]))
            width = mean_r * d_ang
            if width > best_w:
                best_w = width
                best = (a, b)

        if best is None:
            return Gap(found=False)

        min_width = float(self.get_parameter("gap_min_width").value)
        if best_w < min_width:
            return Gap(found=False)

        a, b = best
        mid = (a + b) // 2
        mid_angle = float(ang_w[mid])

        goal_dist = float(self.get_parameter("gap_goal_dist").value)
        gx = goal_dist * math.cos(mid_angle)
        gy = goal_dist * math.sin(mid_angle)

        return Gap(found=True, mid_angle=mid_angle, width_est=best_w,
                   goal_xy=np.array([gx, gy], dtype=np.float32))

    def wall_follow(self):
        side = str(self.get_parameter("use_side").value).lower()
        amin = math.radians(float(self.get_parameter("gap_angle_min_deg").value))
        amax = math.radians(float(self.get_parameter("gap_angle_max_deg").value))
        center = (-(amin + amax) / 2.0) if side == "right" else (+(amin + amax) / 2.0)
        half = (amax - amin) / 2.0

        d_side = self.min_range_window(center, half)
        d_des = float(self.get_parameter("wall_dist_des").value)
        err = d_des - d_side

        kp = float(self.get_parameter("wall_kp").value)
        w = kp * err
        if side == "left":
            w = -w

        v = float(self.get_parameter("cruise_speed").value)
        return v, clamp(w, -2.0, 2.0)

    def approach_gap(self, gap: Gap):
        goal = gap.goal_xy
        dist = float(np.linalg.norm(goal))
        bearing = math.atan2(float(goal[1]), float(goal[0]))

        kp = float(self.get_parameter("approach_kp").value)
        w = kp * bearing
        v = float(self.get_parameter("approach_speed").value)
        return v, clamp(w, -2.0, 2.0), dist

    def center_in_slot(self):
        win = math.radians(float(self.get_parameter("side_window_deg").value))
        lc = math.radians(float(self.get_parameter("left_center_deg").value))
        rc = math.radians(float(self.get_parameter("right_center_deg").value))

        dl = self.min_range_window(lc, win)
        dr = self.min_range_window(rc, win)

        err = dl - dr
        kp = float(self.get_parameter("center_kp").value)
        w = -kp * err
        v = float(self.get_parameter("enter_speed").value)
        return v, clamp(w, -2.0, 2.0), dl, dr

    def front_distance(self):
        fw = math.radians(float(self.get_parameter("front_window_deg").value))
        return self.min_range_window(0.0, fw)

    # ---------- Main ----------
    def on_timer(self):
        if self.scan is None:
            return

        front = self.front_distance()

        # ---- hard stop: 연속 조건으로만 발동 ----
        hard = float(self.get_parameter("hard_stop_dist").value)
        hard_cnt = int(self.get_parameter("hard_stop_count").value)

        if front <= hard and self.state != State.PARKED:
            self._hard_stop_hits += 1
        else:
            self._hard_stop_hits = 0

        if self._hard_stop_hits >= hard_cnt and self.state != State.PARKED:
            self.get_logger().warn(f"[SAFETY] hard stop front={front:.2f} hits={self._hard_stop_hits}")
            self.publish_cmd(0.0, 0.0)
            self.state = State.PARKED
            return

        if self.state == State.CRUISE:
            gap = self.detect_gap()
            if gap.found:
                self.locked_gap = gap
                self.state = State.APPROACH
                self.get_logger().info(
                    f"[CRUISE] gap found width={gap.width_est:.2f} angle={math.degrees(gap.mid_angle):.1f}"
                )
                self.publish_cmd(0.0, 0.0)
                return

            v, w = self.wall_follow()
            self.publish_cmd(v, w)

        elif self.state == State.APPROACH:
            if self.locked_gap is None:
                self.state = State.CRUISE
                return

            v, w, dist = self.approach_gap(self.locked_gap)
            self.publish_cmd(v, w)

            reach = float(self.get_parameter("reach_dist").value)
            if dist <= reach:
                self.state = State.ENTER_CENTER
                self.get_logger().info("[APPROACH] reached entrance -> ENTER_CENTER")

        elif self.state == State.ENTER_CENTER:
            stop = float(self.get_parameter("stop_dist").value)
            if front <= stop:
                self.publish_cmd(0.0, 0.0)
                self.state = State.PARKED
                self.get_logger().info(f"[PARK] parked. front={front:.2f}")
                return

            v, w, dl, dr = self.center_in_slot()
            self.publish_cmd(v, w)

        else:
            self.publish_cmd(0.0, 0.0)


def main():
    rclpy.init()
    node = TParkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.publish_cmd(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
