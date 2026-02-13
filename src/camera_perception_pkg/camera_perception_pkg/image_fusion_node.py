#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from std_msgs.msg import Header

from interfaces_pkg.msg import DetectionArray

import tf2_ros


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion -> 3x3 rotation matrix"""
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=np.float64)


def transform_to_T(ts) -> np.ndarray:
    """geometry_msgs/TransformStamped -> 4x4 homogeneous matrix"""
    t = ts.transform.translation
    r = ts.transform.rotation
    R = quat_to_R(r.x, r.y, r.z, r.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return T


# camera_link(x forward, y left, z up) -> camera_optical(x right, y down, z forward)
R_LINK_TO_OPTICAL = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0]
], dtype=np.float64)


def fit_rectangle_pca(
    points_xy: np.ndarray,
    p_lo: float = 1.0,
    p_hi: float = 99.5,
    pad: float = 0.04,
    min_pts_for_percentile: int = 10
) -> Optional[np.ndarray]:
    """
    points_xy: (N,2) in lidar frame
    return corners (4,2) or None
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return None

    mean = pts.mean(axis=0)
    pts_c = pts - mean

    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    v1 = eigvecs[:, 0]
    angle = math.atan2(v1[1], v1[0])

    c, s = math.cos(-angle), math.sin(-angle)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)

    pts_r = (R @ pts_c.T).T

    if pts_r.shape[0] < int(min_pts_for_percentile):
        xmin, ymin = pts_r.min(axis=0)
        xmax, ymax = pts_r.max(axis=0)
    else:
        p_lo = float(np.clip(p_lo, 0.0, 49.9))
        p_hi = float(np.clip(p_hi, 50.1, 100.0))
        xmin, ymin = np.percentile(pts_r, p_lo, axis=0)
        xmax, ymax = np.percentile(pts_r, p_hi, axis=0)

    pad = float(max(0.0, pad))
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    corners_r = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ], dtype=np.float64)

    corners = (R.T @ corners_r.T).T + mean
    return corners


def rect_wh_from_corners(corners: np.ndarray) -> Tuple[float, float]:
    """
    corners (4,2). return (w, h) in meters (edge lengths).
    """
    if corners is None or corners.shape != (4, 2):
        return (float("nan"), float("nan"))
    e0 = np.linalg.norm(corners[1] - corners[0])
    e1 = np.linalg.norm(corners[2] - corners[1])
    w = float(max(e0, e1))
    h = float(min(e0, e1))
    return w, h


class FusionVisualizerNode(Node):
    """
    [수정 포인트]
    - bbox 내부 라이다 점들로 거리 추정 시, BoxLidarMatchNode와 동일하게
      reduce_distance + depth gating을 사용해 배경점 제거
    - 이미지 overlay 텍스트(dist/pts/size)는 라이다 기반 매칭 결과로만 표시
    """

    def __init__(self):
        super().__init__('fusion_visualizer_node')

        # -------------------------
        # Topics
        # -------------------------
        self.declare_parameter('image_topic', '/camera1/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('det_topic', '/detections')

        self.declare_parameter('publish_annotated', False)
        self.declare_parameter('annotated_topic', '/fusion/annotated_image')
        self.declare_parameter('display', True)

        # -------------------------
        # TF frames
        # -------------------------
        self.declare_parameter('camera_frame', 'camera1_link')
        self.declare_parameter('lidar_frame', 'laser')
        self.declare_parameter('tf_timeout_sec', 0.10)
        self.declare_parameter('use_msg_time', True)
        self.declare_parameter('camera_is_optical', True)

        # -------------------------
        # Intrinsics
        # -------------------------
        self.declare_parameter('fx', 585.710383)
        self.declare_parameter('fy', 586.967092)
        self.declare_parameter('cx', 356.240979)
        self.declare_parameter('cy', 232.964844)

        # -------------------------
        # LiDAR filtering / projection
        # -------------------------
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('min_cam_z', 0.1)

        self.declare_parameter('enable_fov_filter', True)
        self.declare_parameter('cam_fov_deg', 55.0)
        self.declare_parameter('fov_center_deg', 183.0)

        self.declare_parameter('point_stride', 2)

        # -------------------------
        # Matching params (BoxLidarMatchNode와 동일 계열)
        # -------------------------
        self.declare_parameter('distance_method', 'p30')  # min / median / pXX
        self.declare_parameter('depth_gate_front', 0.09)
        self.declare_parameter('depth_gate_back', 0.40)
        self.declare_parameter('min_inbox_points', 1)
        self.declare_parameter('min_fit_points', 3)

        self.declare_parameter('rect_percentile_lo', 1.0)
        self.declare_parameter('rect_percentile_hi', 99.5)
        self.declare_parameter('rect_pad', 0.04)
        self.declare_parameter('rect_min_pts_for_percentile', 10)

        # Detection color (원하면 유지)
        self.declare_parameter('person_keyword', 'person')

        # age gating
        self.declare_parameter('max_age_scan', 0.5)
        self.declare_parameter('max_age_det', 0.5)

        # debug
        self.declare_parameter('debug_log', False)

        # -------------------------
        # Load params
        # -------------------------
        self.image_topic = self.get_parameter('image_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.det_topic = self.get_parameter('det_topic').value

        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        self.annotated_topic = self.get_parameter('annotated_topic').value
        self.display = bool(self.get_parameter('display').value)

        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.lidar_frame = str(self.get_parameter('lidar_frame').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.use_msg_time = bool(self.get_parameter('use_msg_time').value)
        self.camera_is_optical = bool(self.get_parameter('camera_is_optical').value)

        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        self.K = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

        self.max_range = float(self.get_parameter('max_range').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.min_cam_z = float(self.get_parameter('min_cam_z').value)

        self.enable_fov_filter = bool(self.get_parameter('enable_fov_filter').value)
        self.fov_deg = float(self.get_parameter('cam_fov_deg').value)
        self.fov_center_rad = math.radians(float(self.get_parameter('fov_center_deg').value))

        self.point_stride = int(self.get_parameter('point_stride').value)

        self.distance_method = str(self.get_parameter('distance_method').value).lower().strip()
        self.depth_gate_front = float(self.get_parameter('depth_gate_front').value)
        self.depth_gate_back = float(self.get_parameter('depth_gate_back').value)
        self.min_inbox_points = int(self.get_parameter('min_inbox_points').value)
        self.min_fit_points = int(self.get_parameter('min_fit_points').value)

        self.rect_p_lo = float(self.get_parameter('rect_percentile_lo').value)
        self.rect_p_hi = float(self.get_parameter('rect_percentile_hi').value)
        self.rect_pad = float(self.get_parameter('rect_pad').value)
        self.rect_min_pts_for_percentile = int(self.get_parameter('rect_min_pts_for_percentile').value)

        self.person_keyword = str(self.get_parameter('person_keyword').value).lower().strip()

        self.max_age_scan = float(self.get_parameter('max_age_scan').value)
        self.max_age_det = float(self.get_parameter('max_age_det').value)

        self.debug_log = bool(self.get_parameter('debug_log').value)

        self.bridge = CvBridge()

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 최신 메시지 버퍼
        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = None

        self.last_det: Optional[DetectionArray] = None
        self.last_det_time = None

        self.last_img_time = None

        # Subscribers
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(DetectionArray, self.det_topic, self.det_cb, 10)

        # Publisher
        self.pub_img = self.create_publisher(Image, self.annotated_topic, 10) if self.publish_annotated else None

        if self.display:
            cv2.namedWindow("Fusion Visualizer", cv2.WINDOW_NORMAL)

        self.create_timer(1.0, self.debug_timer)

        self.get_logger().info(
            "FusionVisualizerNode(TF2) started.\n"
            f"  image_topic={self.image_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  det_topic={self.det_topic}\n"
            f"  camera_frame={self.camera_frame} (TF)\n"
            f"  lidar_frame={self.lidar_frame} (TF)\n"
            f"  camera_is_optical={self.camera_is_optical}\n"
            f"  distance_method={self.distance_method}\n"
            f"  depth_gate=[-{self.depth_gate_front:.2f}, +{self.depth_gate_back:.2f}] m\n"
            f"  rect_p=[{self.rect_p_lo}, {self.rect_p_hi}] pad={self.rect_pad}\n"
        )

    def debug_timer(self):
        now = self.get_clock().now()

        def age(t):
            if t is None:
                return None
            return (now - t).nanoseconds / 1e9

        img_age = age(self.last_img_time)
        scan_age = age(self.last_scan_time)
        det_age = age(self.last_det_time)

        if img_age is None or img_age > 1.0:
            self.get_logger().warn(
                f"[No Image] image_topic='{self.image_topic}' not arriving. scan_age={scan_age}, det_age={det_age}"
            )

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def det_cb(self, msg: DetectionArray):
        self.last_det = msg
        self.last_det_time = self.get_clock().now()

    def _lookup_T_cam_lidar(self, stamp_ros: Time) -> Optional[np.ndarray]:
        """
        return T_cam<-lidar (4x4).
        1) msg stamp로 시도
        2) 실패하면 latest(time=0) fallback
        """
        timeout = Duration(seconds=self.tf_timeout_sec)

        if self.use_msg_time:
            try:
                ts = self.tf_buffer.lookup_transform(
                    self.camera_frame, self.lidar_frame, stamp_ros, timeout=timeout
                )
                return transform_to_T(ts)
            except Exception:
                pass

        try:
            ts = self.tf_buffer.lookup_transform(
                self.camera_frame, self.lidar_frame, Time(), timeout=timeout
            )
            return transform_to_T(ts)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {self.camera_frame} <- {self.lidar_frame} : {e}")
            return None

    @staticmethod
    def _reduce_distance(r: np.ndarray, method: str) -> float:
        r = r[np.isfinite(r)]
        if r.size == 0:
            return float("nan")

        m = (method or "").lower().strip()
        if m == "min":
            return float(np.min(r))
        if m == "median":
            return float(np.median(r))
        if m.startswith("p"):
            try:
                p = float(m[1:])
                return float(np.percentile(r, p))
            except Exception:
                pass
        return float(np.percentile(r, 20))

    def image_cb(self, img_msg: Image):
        self.last_img_time = self.get_clock().now()

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"imgmsg_to_cv2 failed: {e}")
            return

        h, w = img.shape[:2]
        now = self.get_clock().now()

        scan_ok = False
        det_ok = False

        if self.last_scan is not None and self.last_scan_time is not None:
            scan_age = (now - self.last_scan_time).nanoseconds / 1e9
            scan_ok = (scan_age <= self.max_age_scan)

        if self.last_det is not None and self.last_det_time is not None:
            det_age = (now - self.last_det_time).nanoseconds / 1e9
            det_ok = (det_age <= self.max_age_det)

        stamp_ros = Time.from_msg(img_msg.header.stamp)
        T_cam_lidar = self._lookup_T_cam_lidar(stamp_ros)

        # 라이다 점들을 "한 번"만 프로젝션 + (u,v,r,xy) 유지
        uv = np.zeros((0, 2), dtype=np.int32)
        rr = np.zeros((0,), dtype=np.float64)
        pts_xy = np.zeros((0, 2), dtype=np.float64)

        if scan_ok and (T_cam_lidar is not None):
            uv, rr, pts_xy = self.project_scan_to_image_keep_xy(self.last_scan, w, h, T_cam_lidar)

            # 라이다 점 표시(디버그용)
            for i in range(0, uv.shape[0], max(1, self.point_stride)):
                cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), 1, (0, 0, 255), -1)

        if det_ok:
            for det in self.last_det.detections:
                class_name = str(getattr(det, 'class_name', 'Unknown'))
                score = float(getattr(det, 'score', 0.0))

                bbox = det.bbox
                box_cx = float(bbox.center.position.x)
                box_cy = float(bbox.center.position.y)
                bw = float(bbox.size.x)
                bh = float(bbox.size.y)

                x1 = int(box_cx - bw / 2.0)
                y1 = int(box_cy - bh / 2.0)
                x2 = int(box_cx + bw / 2.0)
                y2 = int(box_cy + bh / 2.0)

                x1c = max(0, min(w - 1, x1))
                y1c = max(0, min(h - 1, y1))
                x2c = max(0, min(w - 1, x2))
                y2c = max(0, min(h - 1, y2))

                is_person = (self.person_keyword in class_name.lower())
                color = (0, 0, 255) if is_person else (0, 255, 0)
                cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, 2)

                # -------------------------------
                # [핵심] bbox 내부 라이다 점들로만 dist/size 산출 (depth gating 포함)
                # -------------------------------
                dist_m = None
                pts_cnt = 0
                best_uv = None
                size_w = float("nan")
                size_h = float("nan")

                if scan_ok and (T_cam_lidar is not None) and rr.size > 0:
                    u = uv[:, 0]
                    v = uv[:, 1]
                    inside = (u >= x1c) & (u <= x2c) & (v >= y1c) & (v <= y2c)
                    if np.count_nonzero(inside) >= self.min_inbox_points:
                        rr_in = rr[inside]
                        uv_in = uv[inside]
                        pts_in = pts_xy[inside]

                        # 1) dist0 (게이팅 기준)
                        dist0 = self._reduce_distance(rr_in, self.distance_method)
                        if np.isfinite(dist0):
                            gate_lo = max(self.min_range, dist0 - self.depth_gate_front)
                            gate_hi = dist0 + self.depth_gate_back
                            depth_ok = (rr_in >= gate_lo) & (rr_in <= gate_hi)

                            rr_g = rr_in[depth_ok]
                            uv_g = uv_in[depth_ok]
                            pts_g = pts_in[depth_ok]

                            if rr_g.size >= self.min_inbox_points:
                                # 2) 최종 dist (게이팅 후 다시 reduce)
                                dist_m = float(self._reduce_distance(rr_g, self.distance_method))
                                pts_cnt = int(rr_g.size)

                                # best_uv: dist에 가장 가까운 점
                                idx = int(np.argmin(np.abs(rr_g - dist_m)))
                                best_uv = (int(uv_g[idx, 0]), int(uv_g[idx, 1]))

                                # size: fit rectangle (점 충분할 때만)
                                if pts_cnt >= self.min_fit_points:
                                    corners = fit_rectangle_pca(
                                        pts_g,
                                        p_lo=self.rect_p_lo,
                                        p_hi=self.rect_p_hi,
                                        pad=self.rect_pad,
                                        min_pts_for_percentile=self.rect_min_pts_for_percentile
                                    )
                                    if corners is not None:
                                        size_w, size_h = rect_wh_from_corners(corners)

                                if self.debug_log:
                                    self.get_logger().info(
                                        f"[MATCH] {class_name} score={score:.2f} "
                                        f"dist0={dist0:.2f} -> dist={dist_m:.2f} pts={pts_cnt} "
                                        f"bbox=({x1c},{y1c},{x2c},{y2c})"
                                    )

                # 텍스트 출력 (dist/pts/size는 라이다 기반 매칭 결과)
                if dist_m is None:
                    text = f"{class_name} {score:.2f}  dist:N/A"
                else:
                    if np.isfinite(size_w) and np.isfinite(size_h):
                        text = f"{class_name} {score:.2f}  dist:{dist_m:.2f}m  pts:{pts_cnt}  sz:{size_w:.2f}x{size_h:.2f}"
                    else:
                        text = f"{class_name} {score:.2f}  dist:{dist_m:.2f}m  pts:{pts_cnt}"

                cv2.putText(img, text, (x1c, max(0, y1c - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                if best_uv is not None:
                    cv2.circle(img, best_uv, 4, (255, 255, 255), -1)

        if self.display:
            cv2.imshow("Fusion Visualizer", img)
            cv2.waitKey(1)

        if self.pub_img is not None:
            out_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            out_msg.header = Header()
            out_msg.header.stamp = img_msg.header.stamp
            out_msg.header.frame_id = self.camera_frame
            self.pub_img.publish(out_msg)

    def project_scan_to_image_keep_xy(self, scan_msg: LaserScan, img_w: int, img_h: int, T_cam_lidar: np.ndarray):
        """
        return:
          uv  : (N,2) int32
          rr  : (N,) float64  (range)
          xy  : (N,2) float64 (lidar frame x,y)
        """
        ranges_all = np.asarray(scan_msg.ranges, dtype=np.float64)
        n = ranges_all.shape[0]
        angles = scan_msg.angle_min + np.arange(n, dtype=np.float64) * scan_msg.angle_increment

        finite = np.isfinite(ranges_all)
        valid = finite & (ranges_all >= self.min_range) & (ranges_all <= min(float(scan_msg.range_max), self.max_range))

        if self.enable_fov_filter:
            half_fov = math.radians(self.fov_deg / 2.0)
            angle_diff = np.abs(np.arctan2(np.sin(angles - self.fov_center_rad),
                                           np.cos(angles - self.fov_center_rad)))
            valid = valid & (angle_diff <= half_fov)

        if np.count_nonzero(valid) == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

        r = ranges_all[valid]
        a = angles[valid]

        x = r * np.cos(a)
        y = r * np.sin(a)
        z = np.zeros_like(x)
        ones = np.ones_like(x)

        pts_lidar_h = np.vstack([x, y, z, ones])  # 4xN
        pts_cam_h = T_cam_lidar @ pts_lidar_h
        xyz = pts_cam_h[:3, :]  # 3xN

        if not self.camera_is_optical:
            xyz = (R_LINK_TO_OPTICAL @ xyz)

        front = xyz[2, :] > self.min_cam_z
        xyz = xyz[:, front]
        r = r[front]
        x = x[front]
        y = y[front]

        if xyz.shape[1] == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

        pix = self.K @ xyz
        u = pix[0, :] / pix[2, :]
        v = pix[1, :] / pix[2, :]

        u_i = u.astype(np.int32)
        v_i = v.astype(np.int32)

        inside = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)

        uv = np.stack([u_i[inside], v_i[inside]], axis=1).astype(np.int32)
        rr = r[inside].astype(np.float64)
        xy = np.stack([x[inside], y[inside]], axis=1).astype(np.float64)

        return uv, rr, xy


def main(args=None):
    rclpy.init(args=args)
    node = FusionVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.display:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()