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

from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Point32  # Point32 추가됨 (장애물 정보 전송용)
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from interfaces_pkg.msg import DetectionArray

import tf2_ros


# -------------------------
# TF helpers
# -------------------------
def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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


# -------------------------
# Geometry helpers
# -------------------------
def hsv_to_rgba(h: float, s: float = 0.9, v: float = 0.9, a: float = 1.0) -> ColorRGBA:
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def fit_rectangle_pca(
    points_xy: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 99.0,
    pad: float = 0.04,
    min_pts_for_percentile: int = 10
):
    """
    points_xy: (N,2) in lidar frame
    return corners (4,2)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 2:  # 예외처리 추가
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
    if corners is None or corners.shape != (4, 2):
        return (float("nan"), float("nan"))
    e0 = np.linalg.norm(corners[1] - corners[0])
    e1 = np.linalg.norm(corners[2] - corners[1])
    w = float(max(e0, e1))
    h = float(min(e0, e1))
    return w, h


class BoxLidarMatchNode(Node):
    def __init__(self):
        super().__init__("box_lidar_match_node")

        # Topics (요청하신 대로 수정됨)
        self.declare_parameter("image_topic", "/camera1/image_raw")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("det_topic", "/detections")

        self.declare_parameter("marker_topic", "/box_fusion/matched_rectangles")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("annotated_topic", "/box_fusion/annotated_image")
        self.declare_parameter("display", False)

        # TF frames (시뮬레이션 환경에 맞게 확인 필요)
        self.declare_parameter("camera_frame", "front_camera_link_optical") 
        self.declare_parameter("lidar_frame", "laser")
        self.declare_parameter("tf_timeout_sec", 0.10)
        self.declare_parameter("use_msg_time", True)
        self.declare_parameter("camera_is_optical", True) 

        # Intrinsics (가제보 카메라 설정에 맞춰야 함)
        self.declare_parameter("fx", 585.710383)
        self.declare_parameter("fy", 586.967092)
        self.declare_parameter("cx", 356.240979)
        self.declare_parameter("cy", 232.964844)

        # Age gating
        self.declare_parameter("max_age_scan", 0.5)
        self.declare_parameter("max_age_det", 0.5) #0.5

        # LiDAR filtering
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("min_range", 0.1)

        # (옵션) FOV filter
        self.declare_parameter("enable_fov_filter", False)
        self.declare_parameter("cam_fov_deg", 55.0)
        self.declare_parameter("fov_center_deg", 183.0)

        # Detection filter
        self.declare_parameter("class_keyword", "cone")
        self.declare_parameter("det_score_min", 0.5)

        # Matching / fit thresholds
        self.declare_parameter("min_inbox_points", 1)
        self.declare_parameter("min_fit_points", 3)
        self.declare_parameter("distance_method", "p30")

        # 깊이(거리) 게이팅
        self.declare_parameter("depth_gate_front", 0.09)
        self.declare_parameter("depth_gate_back", 0.40)

        # 퍼센타일 박스
        self.declare_parameter("rect_percentile_lo", 1.0)
        self.declare_parameter("rect_percentile_hi", 99.5)
        self.declare_parameter("rect_pad", 0.15) #0.04
        self.declare_parameter("rect_min_pts_for_percentile", 10)

        # Marker output frame
        self.declare_parameter("marker_frame_id", "laser")

        # Marker styles
        self.declare_parameter("rect_line_width", 0.03)
        self.declare_parameter("text_size", 0.18)
        self.declare_parameter("z_rect", 0.05)
        self.declare_parameter("z_text", 0.25)
        self.declare_parameter("sphere_size", 0.18)

        # Optional clustering
        self.declare_parameter("use_clustering", False)
        self.declare_parameter("cluster_tolerance", 0.20)
        self.declare_parameter("min_cluster_size", 2)
        self.declare_parameter("max_cluster_size", 600)
        self.declare_parameter("merge_wrap", True)

        # Projection
        self.declare_parameter("min_cam_z", 0.01)

        # Debug
        self.declare_parameter("debug_log", True)

        # -------------------------
        # Load params
        # -------------------------
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.det_topic = str(self.get_parameter("det_topic").value)

        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.annotated_topic = str(self.get_parameter("annotated_topic").value)
        self.display = bool(self.get_parameter("display").value)

        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.lidar_frame_param = str(self.get_parameter("lidar_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.use_msg_time = bool(self.get_parameter("use_msg_time").value)
        self.camera_is_optical = bool(self.get_parameter("camera_is_optical").value)

        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)
        self.K = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

        self.max_age_scan = float(self.get_parameter("max_age_scan").value)
        self.max_age_det = float(self.get_parameter("max_age_det").value)

        self.max_range = float(self.get_parameter("max_range").value)
        self.min_range = float(self.get_parameter("min_range").value)

        self.enable_fov_filter = bool(self.get_parameter("enable_fov_filter").value)
        self.fov_deg = float(self.get_parameter("cam_fov_deg").value)
        self.fov_center_rad = math.radians(float(self.get_parameter("fov_center_deg").value))

        self.class_keyword = str(self.get_parameter("class_keyword").value).lower().strip()
        self.det_score_min = float(self.get_parameter("det_score_min").value)

        self.min_inbox_points = int(self.get_parameter("min_inbox_points").value)
        self.min_fit_points = int(self.get_parameter("min_fit_points").value)
        self.distance_method = str(self.get_parameter("distance_method").value).lower().strip()

        self.depth_gate_front = float(self.get_parameter("depth_gate_front").value)
        self.depth_gate_back = float(self.get_parameter("depth_gate_back").value)

        self.rect_p_lo = float(self.get_parameter("rect_percentile_lo").value)
        self.rect_p_hi = float(self.get_parameter("rect_percentile_hi").value)
        self.rect_pad = float(self.get_parameter("rect_pad").value)
        self.rect_min_pts_for_percentile = int(self.get_parameter("rect_min_pts_for_percentile").value)

        self.marker_frame_id = str(self.get_parameter("marker_frame_id").value)

        self.rect_line_width = float(self.get_parameter("rect_line_width").value)
        self.text_size = float(self.get_parameter("text_size").value)
        self.z_rect = float(self.get_parameter("z_rect").value)
        self.z_text = float(self.get_parameter("z_text").value)
        self.sphere_size = float(self.get_parameter("sphere_size").value)

        self.use_clustering = bool(self.get_parameter("use_clustering").value)
        self.cluster_tol = float(self.get_parameter("cluster_tolerance").value)
        self.min_cluster_size = int(self.get_parameter("min_cluster_size").value)
        self.max_cluster_size = int(self.get_parameter("max_cluster_size").value)
        self.merge_wrap = bool(self.get_parameter("merge_wrap").value)

        self.min_cam_z = float(self.get_parameter("min_cam_z").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Buffers
        self.bridge = CvBridge()
        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = None

        self.last_det: Optional[DetectionArray] = None
        self.last_det_time = None

        # ROS interfaces
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(DetectionArray, self.det_topic, self.det_cb, 10)

        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pub_img = self.create_publisher(Image, self.annotated_topic, 10) if self.publish_annotated else None

        # [추가됨] 장애물 정보 퍼블리셔 (제어 노드용)
        # Point32 메시지 사용 (x: 거리(m), y: 이미지 중심 X좌표(pixel), z: 감지여부 플래그(1.0=True))
        self.pub_obstacle = self.create_publisher(Point32, "/lidar_obstacle_info", 10)

        if self.display:
            cv2.namedWindow("Box-LiDAR Fusion (TF2)", cv2.WINDOW_NORMAL)

        self.create_timer(1.0, self.debug_timer)

        self.get_logger().info(
            "BoxLidarMatchNode(TF2) started.\n"
            f"  image_topic={self.image_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  det_topic={self.det_topic}\n"
            f"  camera_frame={self.camera_frame}\n"
            f"  lidar_frame(param)={self.lidar_frame_param}\n"
            f"  distance_method={self.distance_method}\n"
            f"  depth_gate=[-{self.depth_gate_front:.2f}, +{self.depth_gate_back:.2f}] m\n"
        )

    def debug_timer(self):
        now = self.get_clock().now()

        def age(t):
            if t is None:
                return None
            return (now - t).nanoseconds / 1e9

        scan_age = age(self.last_scan_time)
        det_age = age(self.last_det_time)

        if self.last_scan is None:
            self.get_logger().warn(f"[No Scan] '{self.scan_topic}' not arriving")
        elif scan_age is not None and scan_age > 1.0:
            self.get_logger().warn(f"[Stale Scan] age={scan_age:.2f}s frame_id={self.last_scan.header.frame_id}")

        if self.last_det is None:
            self.get_logger().warn(f"[No Detections] '{self.det_topic}' not arriving")
        elif det_age is not None and det_age > 1.0:
            self.get_logger().warn(f"[Stale Detections] age={det_age:.2f}s")

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def det_cb(self, msg: DetectionArray):
        self.last_det = msg
        self.last_det_time = self.get_clock().now()

    def image_cb(self, img_msg: Image):
        now = self.get_clock().now()

        scan_ok = (self.last_scan is not None and self.last_scan_time is not None and
                   ((now - self.last_scan_time).nanoseconds / 1e9 <= self.max_age_scan))
        det_ok = (self.last_det is not None and self.last_det_time is not None and
                  ((now - self.last_det_time).nanoseconds / 1e9 <= self.max_age_det))

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"imgmsg_to_cv2 failed: {e}")
            return

        h, w = img.shape[:2]
        detections = self._extract_box_detections(self.last_det, w, h) if det_ok else []

        # TF lookup (camera <- lidar)
        T_cam_lidar = None
        if scan_ok:
            T_cam_lidar = self._lookup_T_cam_lidar(self.last_scan)
            if T_cam_lidar is None:
                scan_ok = False

        matches = []

        # [추가됨] 가장 가까운 장애물을 찾기 위한 변수 초기화
        closest_obstacle_dist = 999.0
        closest_obstacle_cx = -1.0
        is_obstacle_detected = False

        if scan_ok and detections:
            pts_xy, rr = self._scan_to_points(self.last_scan)  # (N,2), (N,)
            if pts_xy.shape[0] > 0:
                uv, rr_img, pts_xy_kept = self._project_all_points_to_image_tf(
                    pts_xy, rr, T_cam_lidar, w, h
                )

                # bbox별로 내부 점 집계
                for di, d in enumerate(detections):
                    x1, y1, x2, y2 = d["bbox"]
                    u = uv[:, 0]
                    v = uv[:, 1]
                    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                    cnt = int(np.count_nonzero(inside))
                    if cnt < self.min_inbox_points:
                        continue

                    pts_in = pts_xy_kept[inside]
                    rr_in = rr_img[inside]

                    # 1) dist0
                    dist0 = self._reduce_distance(rr_in, self.distance_method)
                    if not np.isfinite(dist0):
                        continue

                    # 2) depth gating
                    gate_lo = max(self.min_range, dist0 - self.depth_gate_front)
                    gate_hi = dist0 + self.depth_gate_back
                    mask_depth = (rr_in >= gate_lo) & (rr_in <= gate_hi)

                    pts_g = pts_in[mask_depth]
                    rr_g = rr_in[mask_depth]
                    cnt2 = int(pts_g.shape[0])
                    if cnt2 < self.min_inbox_points:
                        continue

                    # [수정] 최종 distance는 게이팅 후 재계산한 dist 사용
                    dist = float(self._reduce_distance(rr_g, self.distance_method))

                    # size (가능하면)
                    size_w = float("nan")
                    size_h = float("nan")
                    corners = None
                    if cnt2 >= self.min_fit_points:
                        corners = fit_rectangle_pca(
                            pts_g,
                            p_lo=self.rect_p_lo,
                            p_hi=self.rect_p_hi,
                            pad=self.rect_pad,
                            min_pts_for_percentile=self.rect_min_pts_for_percentile
                        )
                        if corners is not None:
                            size_w, size_h = rect_wh_from_corners(corners)

                    matches.append({
                        "det": d,
                        "idx": di,
                        "pts_in": pts_g,
                        "rr_in": rr_g,
                        "count": cnt2,
                        "distance": dist,   # <= 여기 dist0가 아니라 dist
                        "corners": corners,
                        "size_w": size_w,
                        "size_h": size_h,
                    })

                    # [추가됨] 가장 가까운 장애물 갱신 로직
                    if dist < closest_obstacle_dist:
                        closest_obstacle_dist = dist
                        closest_obstacle_cx = (x1 + x2) / 2.0
                        is_obstacle_detected = True

                    if self.debug_log:
                        self.get_logger().info(
                            f"[DEBUG] det#{di} inbox={cnt}->{cnt2} dist0={dist0:.2f} dist={dist:.2f} "
                            f"bbox={d['bbox']}"
                        )

        # [추가됨] 가장 가까운 장애물 정보 퍼블리시
        obs_msg = Point32()
        if is_obstacle_detected:
            obs_msg.x = float(closest_obstacle_dist)  # 거리 (m)
            obs_msg.y = float(closest_obstacle_cx)    # 이미지 상의 중심 X좌표 (pixel)
            obs_msg.z = 1.0                           # 감지됨
        else:
            obs_msg.x = -1.0
            obs_msg.y = -1.0
            obs_msg.z = 0.0                           # 감지되지 않음

        self.pub_obstacle.publish(obs_msg)

        # RViz markers
        ma = self._make_markers(matches, img_msg.header , detections)
        self.pub_markers.publish(ma)

        # annotated image
        ann = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for m in matches:
            d = m["det"]
            x1, y1, x2, y2 = d["bbox"]
            dist = float(m["distance"])
            cnt = int(m["count"])
            sw = float(m["size_w"])
            sh = float(m["size_h"])

            if np.isfinite(sw) and np.isfinite(sh):
                label = f"{d['class_name']} {d['score']:.2f} dist:{dist:.2f}m pts:{cnt} sz:{sw:.2f}x{sh:.2f}"
            else:
                label = f"{d['class_name']} {d['score']:.2f} dist:{dist:.2f}m pts:{cnt}"

            cv2.putText(ann, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if self.pub_img is not None:
            out = self.bridge.cv2_to_imgmsg(ann, encoding="bgr8")
            out.header = Header()
            out.header.stamp = img_msg.header.stamp
            out.header.frame_id = self.camera_frame
            self.pub_img.publish(out)

        if self.display:
            cv2.imshow("Box-LiDAR Fusion (TF2)", ann)
            cv2.waitKey(1)

    def _lookup_T_cam_lidar(self, scan: LaserScan) -> Optional[np.ndarray]:
        source = self.lidar_frame_param if self.lidar_frame_param else scan.header.frame_id
        if not source:
            source = scan.header.frame_id
        target = self.camera_frame

        if self.use_msg_time:
            t = Time(seconds=int(scan.header.stamp.sec), nanoseconds=int(scan.header.stamp.nanosec))
        else:
            t = Time()

        try:
            ts = self.tf_buffer.lookup_transform(target, source, t, timeout=Duration(seconds=self.tf_timeout_sec))
            return transform_to_T(ts)
        except Exception as e:
            if self.use_msg_time:
                try:
                    ts = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=self.tf_timeout_sec))
                    return transform_to_T(ts)
                except Exception as e2:
                    self.get_logger().warn(f"TF lookup failed: target={target} source={source} (stamp+latest): {e} / {e2}")
                    return None
            self.get_logger().warn(f"TF lookup failed: target={target} source={source}: {e}")
            return None

    def _scan_to_points(self, scan: LaserScan):
        rmin = max(float(scan.range_min), self.min_range)
        rmax = min(float(scan.range_max), self.max_range)

        angles = scan.angle_min + np.arange(len(scan.ranges), dtype=np.float64) * scan.angle_increment
        ranges = np.asarray(scan.ranges, dtype=np.float64)

        finite = np.isfinite(ranges)
        valid = finite & (ranges >= rmin) & (ranges <= rmax)

        if self.enable_fov_filter:
            half_fov = math.radians(self.fov_deg / 2.0)
            ad = np.arctan2(np.sin(angles - self.fov_center_rad), np.cos(angles - self.fov_center_rad))
            valid = valid & (np.abs(ad) <= half_fov)

        if np.count_nonzero(valid) == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        r = ranges[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)

        pts_xy = np.stack([x, y], axis=1).astype(np.float64)
        return pts_xy, r.astype(np.float64)

    def _project_all_points_to_image_tf(
        self,
        pts_xy: np.ndarray,
        rr: np.ndarray,
        T_cam_lidar: np.ndarray,
        img_w: int,
        img_h: int
    ):
        N = pts_xy.shape[0]
        x = pts_xy[:, 0]
        y = pts_xy[:, 1]
        z = np.zeros((N,), dtype=np.float64)
        ones = np.ones((N,), dtype=np.float64)

        pts_lidar_h = np.vstack([x, y, z, ones])      # 4xN
        pts_cam_h = T_cam_lidar @ pts_lidar_h         # 4xN
        xyz = pts_cam_h[:3, :]                        # 3xN

        if not self.camera_is_optical:
            xyz = (R_LINK_TO_OPTICAL @ xyz)

        front = xyz[2, :] > self.min_cam_z
        xyz = xyz[:, front]
        rr_f = rr[front]
        pts_xy_f = pts_xy[front]

        if xyz.shape[1] == 0:
            return (np.zeros((0, 2), dtype=np.int32),
                    np.zeros((0,), dtype=np.float64),
                    np.zeros((0, 2), dtype=np.float64))

        pix = self.K @ xyz
        u = pix[0, :] / pix[2, :]
        v = pix[1, :] / pix[2, :]

        u_i = u.astype(np.int32)
        v_i = v.astype(np.int32)

        inside = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)

        uv = np.stack([u_i[inside], v_i[inside]], axis=1).astype(np.int32)
        rr_img = rr_f[inside].astype(np.float64)
        pts_xy_kept = pts_xy_f[inside].astype(np.float64)

        return uv, rr_img, pts_xy_kept

    def _extract_box_detections(self, det_msg: Optional[DetectionArray], img_w: int, img_h: int) -> List[Dict]:
        if det_msg is None:
            return []

        out: List[Dict] = []
        for det in det_msg.detections:
            class_name = str(getattr(det, "class_name", ""))
            score = float(getattr(det, "score", 0.0))
            if score < self.det_score_min:
                continue
            if self.class_keyword not in class_name.lower():
                continue

            bbox = det.bbox
            cx = float(bbox.center.position.x)
            cy = float(bbox.center.position.y)
            bw = float(bbox.size.x)
            bh = float(bbox.size.y)

            x1 = int(cx - bw / 2.0)
            y1 = int(cy - bh / 2.0)
            x2 = int(cx + bw / 2.0)
            y2 = int(cy + bh / 2.0)

            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2))
            y2 = max(0, min(img_h - 1, y2))

            out.append({"class_name": class_name, "score": score, "bbox": (x1, y1, x2, y2)})
        return out

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

    def _make_markers(self, matches: List[Dict], header: Header, detections: List[Dict] = None) -> MarkerArray:
        """
        RViz 마커 생성 함수
        1. YOLO가 감지한 영역(Frustum)을 하늘색 선으로 표시 (디버깅용)
        2. 라이다와 매칭된 결과를 초록색 박스와 텍스트로 표시
        """
        ma = MarkerArray()

        # ---------------------------------------------------------
        # 0. 기존 마커 싹 지우기 (초기화)
        # ---------------------------------------------------------
        del_all = Marker()
        del_all.action = Marker.DELETEALL
        ma.markers.append(del_all)

        # ---------------------------------------------------------
        # 1. [디버깅] YOLO가 보고 있는 영역(Frustum) 그리기
        #    (매칭 여부와 상관없이 YOLO가 찾은 모든 박스를 표시)
        # ---------------------------------------------------------
        if detections:
            for i, det in enumerate(detections):
                # YOLO 박스 좌표
                x1, y1, x2, y2 = det["bbox"]
                
                # 박스의 4개 모서리 (픽셀 좌표)
                corners_2d = [
                    (x1, y1), (x2, y1), (x2, y2), (x1, y2)
                ]

                # 마커 생성 (선 리스트)
                frustum = Marker()
                frustum.header.frame_id = self.camera_frame  # 카메라 기준 좌표계
                frustum.header.stamp = header.stamp
                frustum.ns = "yolo_frustum"
                frustum.id = i + 2000  # ID가 겹치지 않게 2000번대 사용
                frustum.type = Marker.LINE_LIST
                frustum.action = Marker.ADD
                frustum.scale.x = 0.02  # 선 두께
                frustum.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5)  # 하늘색 (반투명)
                
                # 카메라 원점 (0,0,0)
                p_origin = Point(x=0.0, y=0.0, z=0.0)

                # 4개의 모서리를 향해 뻗어나가는 선 그리기
                for (cx, cy) in corners_2d:
                    # 1) 2D 픽셀(u,v) -> 3D 정규 좌표(x,y,1) 변환 (핀홀 모델 역산)
                    # x = (u - cx) / fx, y = (v - cy) / fy
                    nx = (cx - self.K[0, 2]) / self.K[0, 0]
                    ny = (cy - self.K[1, 2]) / self.K[1, 1]
                    nz = 1.0  # 정면 1m 지점 (Optical Frame 기준 z가 앞쪽)

                    # 2) 5미터 길이로 연장 (시각적으로 잘 보이게)
                    vec = np.array([nx, ny, nz])
                    vec = vec / np.linalg.norm(vec) * 5.0
                    
                    p_end = Point(x=vec[0], y=vec[1], z=vec[2])
                    
                    # 원점 -> 끝점 (선 하나 추가)
                    frustum.points.append(p_origin)
                    frustum.points.append(p_end)

                ma.markers.append(frustum)

        # ---------------------------------------------------------
        # 2. [결과] 매칭된 라이다 박스 & 텍스트 그리기
        # ---------------------------------------------------------
        if not matches:
            return ma

        for i, m in enumerate(matches):
            d = m["det"]
            pts_in = m["pts_in"]
            dist_m = float(m["distance"])
            cnt = int(m["count"])
            corners = m.get("corners", None)
            sw = float(m.get("size_w", float("nan")))
            sh = float(m.get("size_h", float("nan")))

            # 색상 생성 (인덱스별로 다르게)
            color = hsv_to_rgba(i / max(1, len(matches)))

            # 중심점 계산 (텍스트 표시용)
            if pts_in.shape[0] > 0:
                cx = float(np.mean(pts_in[:, 0]))
                cy = float(np.mean(pts_in[:, 1]))
            else:
                cx, cy = 0.0, 0.0

            # (A) 사각형 그리기 (점 3개 이상일 때 PCA 결과 사용)
            if corners is not None and cnt >= self.min_fit_points:
                rect = Marker()
                rect.header.frame_id = self.marker_frame_id  # 보통 'laser'
                rect.header.stamp = header.stamp
                rect.ns = "matched_rectangles"
                rect.id = i
                rect.type = Marker.LINE_STRIP
                rect.action = Marker.ADD
                rect.scale.x = float(self.rect_line_width)
                rect.color = color

                corners_list = corners.tolist()
                # 4개 점을 잇고 다시 처음 점으로 돌아와서 사각형 완성
                for (x, y) in corners_list + [corners_list[0]]:
                    rect.points.append(Point(x=float(x), y=float(y), z=float(self.z_rect)))
                ma.markers.append(rect)

            # (B) 점이 2개뿐이면 선으로 그리기
            elif cnt == 2:
                line = Marker()
                line.header.frame_id = self.marker_frame_id
                line.header.stamp = header.stamp
                line.ns = "matched_lines"
                line.id = i
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = float(self.rect_line_width)
                line.color = color

                p0 = pts_in[0]
                p1 = pts_in[1]
                line.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=float(self.z_rect)))
                line.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=float(self.z_rect)))
                ma.markers.append(line)

            # (C) 점이 1개거나 모양이 안 나오면 구(Sphere)로 표시
            else:
                sp = Marker()
                sp.header.frame_id = self.marker_frame_id
                sp.header.stamp = header.stamp
                sp.ns = "matched_spheres"
                sp.id = i
                sp.type = Marker.SPHERE
                sp.action = Marker.ADD
                sp.pose.position.x = cx
                sp.pose.position.y = cy
                sp.pose.position.z = float(self.z_rect)
                sp.pose.orientation.w = 1.0
                sp.scale.x = float(self.sphere_size)
                sp.scale.y = float(self.sphere_size)
                sp.scale.z = float(self.sphere_size)
                sp.color = color
                ma.markers.append(sp)

            # (D) 정보 텍스트 (거리, 점수 등)
            txt = Marker()
            txt.header.frame_id = self.marker_frame_id
            txt.header.stamp = header.stamp
            txt.ns = "matched_text"
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = cx
            txt.pose.position.y = cy
            txt.pose.position.z = float(self.z_text) # 텍스트는 좀 더 높게
            txt.pose.orientation.w = 1.0
            txt.scale.z = float(self.text_size)
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # 흰색 글씨

            if np.isfinite(sw) and np.isfinite(sh):
                txt.text = f"{d['class_name']} {d['score']:.2f}\ndist:{dist_m:.2f}m\npts:{cnt} sz:{sw:.2f}x{sh:.2f}"
            else:
                txt.text = f"{d['class_name']} {d['score']:.2f}\ndist:{dist_m:.2f}m\npts:{cnt}"
            ma.markers.append(txt)

        return ma

    def destroy_node(self):
        if self.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BoxLidarMatchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()