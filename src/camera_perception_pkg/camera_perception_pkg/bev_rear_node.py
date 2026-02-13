#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class BirdEyeViewNode(Node):
    """
    Rear-camera oriented Bird-Eye-View (BEV) warp node.

    Sub:
      - /rear_camera/image_raw (sensor_msgs/Image)
      - (optional) /rear_camera/camera_info (sensor_msgs/CameraInfo) for undistort

    Pub:
      - /rear_camera/image_bev
      - /rear_camera/image_bev_debug
    """

    def __init__(self):
        super().__init__('rear_birdeyeview_node')

        # ---------------- Parameters ----------------
        # ✅ Rear camera default topics
        self.declare_parameter('image_topic', '/rear_camera/image_raw')
        self.declare_parameter('camera_info_topic', '/rear_camera/camera_info')

        # ✅ Undistort
        self.declare_parameter('use_undistort', True)

        # If camera_info is missing, use these static intrinsics/distortion as fallback
        self.declare_parameter('use_static_intrinsics', True)

        # User-provided rear camera intrinsics (fallback)
        # K = [[fx, 0, cx],
        #      [0, fy, cy],
        #      [0,  0,  1]]
        self.declare_parameter('static_K', [
            585.710383, 0.0,       356.240979,
            0.0,       586.967092, 232.964844,
            0.0,       0.0,       1.0
        ])
        # D (plumb_bob): [k1, k2, p1, p2, k3]
        self.declare_parameter('static_D', [-0.013602, -0.102496, -0.002990, 0.002395, 0.0])

        # Output topics
        self.declare_parameter('bev_topic', '/rear_camera/image_bev')
        self.declare_parameter('bev_debug_topic', '/rear_camera/image_bev_debug')

        # Output size
        self.declare_parameter('out_w', 640)
        self.declare_parameter('out_h', 640)

        # ✅ src polygon ratios (rear camera default)
        # - pitch(27deg)라서 바닥이 크게 보이고 "유효 바닥 영역"이 조금 아래로 내려오는 경향이 있어
        #   top_y_ratio를 약간 키워(더 아래) 바닥 ROI를 안정적으로 잡는 쪽으로 시작값 설정
        self.declare_parameter('src_top_y_ratio', 0.68)             # (front: 0.62) -> rear: 조금 아래
        self.declare_parameter('src_bottom_y_ratio', 0.98)
        self.declare_parameter('src_top_half_width_ratio', 0.26)
        self.declare_parameter('src_bottom_half_width_ratio', 0.46)
        self.declare_parameter('src_center_x_ratio', 0.50)

        # dst margin
        self.declare_parameter('dst_margin_x', 0.10)

        # ✅ Rear camera may need pre-rotation / flip depending on TF & driver
        # pre_rotate_deg: 0, 90, 180, 270
        # pre_flip_x: flip vertically (up-down)
        # pre_flip_y: flip horizontally (left-right)
        self.declare_parameter('pre_rotate_deg', 0)
        self.declare_parameter('pre_flip_x', False)
        self.declare_parameter('pre_flip_y', False)

        # Debug window
        self.declare_parameter('show_window', True)
        self.declare_parameter('win_name', 'Rear Bird-Eye View')

        # --------------- Load params ---------------
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)

        self.use_undistort = bool(self.get_parameter('use_undistort').value)
        self.use_static_intrinsics = bool(self.get_parameter('use_static_intrinsics').value)

        self.bev_topic = str(self.get_parameter('bev_topic').value)
        self.bev_debug_topic = str(self.get_parameter('bev_debug_topic').value)

        self.out_w = int(self.get_parameter('out_w').value)
        self.out_h = int(self.get_parameter('out_h').value)

        self.pre_rotate_deg = int(self.get_parameter('pre_rotate_deg').value)
        self.pre_flip_x = bool(self.get_parameter('pre_flip_x').value)
        self.pre_flip_y = bool(self.get_parameter('pre_flip_y').value)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.win_name = str(self.get_parameter('win_name').value)

        self.bridge = CvBridge()

        # Homography cache
        self.H = None
        self.last_img_wh = None
        self._last_params_signature = None

        # undistort cache
        self.K = None
        self.D = None
        self._ud_map1 = None
        self._ud_map2 = None
        self._ud_wh = None

        # Load static intrinsics (fallback)
        if self.use_static_intrinsics:
            try:
                K_list = list(self.get_parameter('static_K').value)
                D_list = list(self.get_parameter('static_D').value)
                if len(K_list) == 9 and len(D_list) >= 4:
                    self._static_K = np.array(K_list, dtype=np.float64).reshape(3, 3)
                    self._static_D = np.array(D_list, dtype=np.float64).reshape(-1)
                else:
                    self._static_K = None
                    self._static_D = None
            except Exception:
                self._static_K = None
                self._static_D = None
        else:
            self._static_K = None
            self._static_D = None

        # Publishers
        self.pub_bev = self.create_publisher(Image, self.bev_topic, 10)
        self.pub_bev_dbg = self.create_publisher(Image, self.bev_debug_topic, 10)

        # Subscribers
        self.sub_img = self.create_subscription(
            Image, self.image_topic, self._cb_image, qos_profile_sensor_data
        )
        self.sub_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._cb_info, 10
        )

        if self.show_window:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(
            "Rear BirdEyeViewNode started.\n"
            f"  sub image_topic     : {self.image_topic}\n"
            f"  sub camera_info     : {self.camera_info_topic}\n"
            f"  use_undistort       : {self.use_undistort}\n"
            f"  use_static_intrinsics: {self.use_static_intrinsics}\n"
            f"  pub bev_topic       : {self.bev_topic}\n"
            f"  pub bev_debug_topic : {self.bev_debug_topic}\n"
            f"  out size            : {self.out_w}x{self.out_h}\n"
            f"  pre_rotate_deg      : {self.pre_rotate_deg}\n"
            f"  pre_flip_x/y        : {self.pre_flip_x}/{self.pre_flip_y}\n"
        )

    def _cb_info(self, msg: CameraInfo):
        # camera_info가 들어오면 그걸 우선 사용
        try:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64).reshape(-1)
        except Exception:
            # 무시 (fallback 사용 가능)
            pass

    def _maybe_apply_pre_transform(self, frame: np.ndarray) -> np.ndarray:
        # rotate
        deg = self.pre_rotate_deg % 360
        if deg == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif deg == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif deg == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # flip
        if self.pre_flip_x and self.pre_flip_y:
            frame = cv2.flip(frame, -1)   # both
        elif self.pre_flip_x:
            frame = cv2.flip(frame, 0)    # vertical
        elif self.pre_flip_y:
            frame = cv2.flip(frame, 1)    # horizontal

        return frame

    def _maybe_build_undistort_map(self, w: int, h: int):
        if not self.use_undistort:
            return

        # 우선순위: camera_info(K,D) -> static(K,D) -> undistort 미적용
        K = self.K
        D = self.D
        if (K is None or D is None) and self.use_static_intrinsics:
            K = self._static_K
            D = self._static_D

        if K is None or D is None:
            return

        if self._ud_wh == (w, h) and self._ud_map1 is not None:
            return

        # alpha=0.0: 검은 테두리 최소(대신 조금 크롭됨)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)
        self._ud_map1, self._ud_map2 = cv2.initUndistortRectifyMap(
            K, D, None, newK, (w, h), cv2.CV_16SC2
        )
        self._ud_wh = (w, h)

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def _build_src_dst(self, img_w: int, img_h: int):
        # read params
        top_y_r = float(self.get_parameter('src_top_y_ratio').value)
        bot_y_r = float(self.get_parameter('src_bottom_y_ratio').value)
        top_hw_r = float(self.get_parameter('src_top_half_width_ratio').value)
        bot_hw_r = float(self.get_parameter('src_bottom_half_width_ratio').value)
        cx_r = float(self.get_parameter('src_center_x_ratio').value)
        mx = float(self.get_parameter('dst_margin_x').value)

        # clamp
        top_y_r = self._clamp(top_y_r, 0.0, 1.0)
        bot_y_r = self._clamp(bot_y_r, 0.0, 1.0)
        if bot_y_r <= top_y_r:
            bot_y_r = self._clamp(top_y_r + 0.1, 0.0, 1.0)

        top_hw_r = self._clamp(top_hw_r, 0.01, 0.49)
        bot_hw_r = self._clamp(bot_hw_r, 0.01, 0.49)
        cx_r = self._clamp(cx_r, 0.0, 1.0)

        mx = self._clamp(mx, 0.0, 0.49)

        cx = cx_r * img_w
        top_y = top_y_r * img_h
        bot_y = bot_y_r * img_h

        # src: TL, TR, BR, BL
        src = np.array([
            [cx - top_hw_r * img_w, top_y],
            [cx + top_hw_r * img_w, top_y],
            [cx + bot_hw_r * img_w, bot_y],
            [cx - bot_hw_r * img_w, bot_y],
        ], dtype=np.float32)

        # clip inside image
        src[:, 0] = np.clip(src[:, 0], 0, img_w - 1)
        src[:, 1] = np.clip(src[:, 1], 0, img_h - 1)

        # dst: 좌우 mx만큼 여백
        dst = np.array([
            [mx * self.out_w, 0.0],
            [(1.0 - mx) * self.out_w, 0.0],
            [(1.0 - mx) * self.out_w, float(self.out_h)],
            [mx * self.out_w, float(self.out_h)],
        ], dtype=np.float32)

        signature = (img_w, img_h, top_y_r, bot_y_r, top_hw_r, bot_hw_r, cx_r, mx, self.out_w, self.out_h)
        return src, dst, signature

    def _recompute_h_if_needed(self, img_w: int, img_h: int):
        src, dst, sig = self._build_src_dst(img_w, img_h)

        if self.H is None or self.last_img_wh != (img_w, img_h) or self._last_params_signature != sig:
            self.H = cv2.getPerspectiveTransform(src, dst)
            self.last_img_wh = (img_w, img_h)
            self._last_params_signature = sig
            return True, src
        return False, src

    def _cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # 0) pre rotate/flip for rear camera if needed
        frame = self._maybe_apply_pre_transform(frame)

        h, w = frame.shape[:2]

        # 1) undistort (optional)
        if self.use_undistort:
            self._maybe_build_undistort_map(w, h)
            if self._ud_map1 is not None:
                frame = cv2.remap(frame, self._ud_map1, self._ud_map2, interpolation=cv2.INTER_LINEAR)

        # 2) homography
        _, src_poly = self._recompute_h_if_needed(w, h)
        if self.H is None:
            return

        # 3) warp to BEV
        bev = cv2.warpPerspective(frame, self.H, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR)

        # publish
        bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
        bev_msg.header = msg.header
        self.pub_bev.publish(bev_msg)

        bev_dbg_msg = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
        bev_dbg_msg.header = msg.header
        self.pub_bev_dbg.publish(bev_dbg_msg)

        # debug visualization window
        if self.show_window:
            vis = frame.copy()
            poly_i = src_poly.astype(np.int32).reshape(4, 2)
            cv2.polylines(vis, [poly_i], isClosed=True, color=(0, 255, 0), thickness=2)

            mx = float(self.get_parameter('dst_margin_x').value)
            top_y = float(self.get_parameter('src_top_y_ratio').value)
            bot_y = float(self.get_parameter('src_bottom_y_ratio').value)
            top_hw = float(self.get_parameter('src_top_half_width_ratio').value)
            bot_hw = float(self.get_parameter('src_bottom_half_width_ratio').value)
            cxr = float(self.get_parameter('src_center_x_ratio').value)

            cv2.putText(vis, f"[REAR] undistort={int(self.use_undistort)} staticK={int(self.use_static_intrinsics)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"src top_y={top_y:.2f} bot_y={bot_y:.2f} top_hw={top_hw:.3f} bot_hw={bot_hw:.3f} cx={cxr:.2f} mx={mx:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, f"pre_rotate={self.pre_rotate_deg} flip_x={int(self.pre_flip_x)} flip_y={int(self.pre_flip_y)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            left = cv2.resize(vis, (self.out_w, self.out_h))
            combo = np.hstack([left, bev])
            cv2.imshow(self.win_name, combo)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BirdEyeViewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()