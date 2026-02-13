import cv2
import numpy as np
from collections import deque
from typing import Deque, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge

from visualization_msgs.msg import Marker

# -------------------- Optional detection msg imports --------------------
_DET_MODE = "none"
try:
    from interfaces_pkg.msg import DetectionArray  # type: ignore
    _DET_MODE = "interfaces_pkg"
except Exception:
    try:
        from vision_msgs.msg import Detection2DArray  # type: ignore
        _DET_MODE = "vision_msgs"
    except Exception:
        _DET_MODE = "none"


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


class ParkingPerceptionCoordinatorNode(Node):
    """
    목표:
      (1) 후방 카메라 원본에서 '고깔 2개' 검출되면 -> 전진 정지(ForwardStop) 판단 토픽 발행
      (2) 고깔 중앙(mid) 기반 align_error 출력 -> 후진 정렬 단계 판단
      (3) 후방 BEV 기반 ParkingLineStopNode에서:
          - /parking/vertical_pair_detected (추가)
          - /parking/lane_center_error_px (추가)
          - /parking/stop_line_detected
        를 받아서 라인 추종 단계 확정 + 최종 stop 판단

    출력:
      - /parking2/cones_detected (Bool)
      - /parking2/forward_stop (Bool)
      - /parking2/cone_mid_x_px (Float32)
      - /parking2/cone_mid_x_norm (Float32)
      - /parking2/align_error_px (Float32)
      - /parking2/align_done (Bool)
      - /parking2/reverse_straight_phase (Bool)
      - /parking2/final_stop (Bool)
      - /parking2/state (String)

    디버그:
      - /parking2/debug/overlay (Image bgr8)
      - /parking2/debug/state_text (Marker)
    """

    S_SEARCH = 0
    S_ALIGN_REV = 1
    S_REVERSE_STRAIGHT = 2
    S_DONE = 3

    def __init__(self):
        super().__init__("parking_perception_coordinator_node")

        # ---------------- Params ----------------
        self.declare_parameter("front_image_topic", "/rear_camera/image_raw")
        self.declare_parameter("det_topic", "/detections")

        self.declare_parameter("cone_class_names", "cone,traffic_cone,pylon,con")
        self.declare_parameter("need_cones", 2)

        self.declare_parameter("cones_vote_window", 8)
        self.declare_parameter("cones_vote_need", 5)

        self.declare_parameter("align_error_thr_px", 18.0)
        self.declare_parameter("align_vote_window", 8)
        self.declare_parameter("align_vote_need", 5)

        self.declare_parameter("target_center_x_ratio", 0.50)

        # stop line inputs from ParkingLineStopNode
        self.declare_parameter("stop_line_topic", "/parking/stop_line_detected")
        self.declare_parameter("stop_dist_topic", "/parking/stop_line_y_from_bottom")

        # Added: vertical pair + lane center error from ParkingLineStopNode
        self.declare_parameter("vertical_pair_topic", "/parking/vertical_pair_detected")
        self.declare_parameter("lane_center_err_topic", "/parking/lane_center_error_px")

        # gating vote for vertical pair
        self.declare_parameter("pair_vote_window", 8)
        self.declare_parameter("pair_vote_need", 5)

        # final stop vote
        self.declare_parameter("final_stop_vote_window", 8)
        self.declare_parameter("final_stop_vote_need", 5)

        # Debug
        self.declare_parameter("pub_debug", True)
        self.declare_parameter("debug_overlay_topic", "/parking2/debug/overlay")
        self.declare_parameter("debug_text_marker_topic", "/parking2/debug/state_text")
        self.declare_parameter("debug_marker_frame", "base_link")
        self.declare_parameter("debug_marker_xyz", [0.0, 0.0, 0.3])

        # ---------------- Load params ----------------
        self.front_image_topic = str(self.get_parameter("front_image_topic").value)
        self.det_topic = str(self.get_parameter("det_topic").value)

        names = str(self.get_parameter("cone_class_names").value)
        self.cone_names = {s.strip().lower() for s in names.split(",") if s.strip()}
        self.need_cones = int(self.get_parameter("need_cones").value)

        self.cones_vote_window = int(self.get_parameter("cones_vote_window").value)
        self.cones_vote_need = int(self.get_parameter("cones_vote_need").value)

        self.align_thr_px = float(self.get_parameter("align_error_thr_px").value)
        self.align_vote_window = int(self.get_parameter("align_vote_window").value)
        self.align_vote_need = int(self.get_parameter("align_vote_need").value)

        self.target_center_x_ratio = float(self.get_parameter("target_center_x_ratio").value)

        self.stop_line_topic = str(self.get_parameter("stop_line_topic").value)
        self.stop_dist_topic = str(self.get_parameter("stop_dist_topic").value)

        self.vertical_pair_topic = str(self.get_parameter("vertical_pair_topic").value)
        self.lane_center_err_topic = str(self.get_parameter("lane_center_err_topic").value)

        self.pair_vote_window = int(self.get_parameter("pair_vote_window").value)
        self.pair_vote_need = int(self.get_parameter("pair_vote_need").value)

        self.final_stop_vote_window = int(self.get_parameter("final_stop_vote_window").value)
        self.final_stop_vote_need = int(self.get_parameter("final_stop_vote_need").value)

        self.pub_debug = bool(self.get_parameter("pub_debug").value)
        self.debug_overlay_topic = str(self.get_parameter("debug_overlay_topic").value)
        self.debug_text_marker_topic = str(self.get_parameter("debug_text_marker_topic").value)

        self.debug_marker_frame = str(self.get_parameter("debug_marker_frame").value)
        xyz = self.get_parameter("debug_marker_xyz").value
        self.debug_marker_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

        # ---------------- ROS ----------------
        self.bridge = CvBridge()

        self.sub_img = self.create_subscription(
            Image, self.front_image_topic, self.cb_front_image, qos_profile_sensor_data
        )

        if _DET_MODE == "interfaces_pkg":
            self.sub_det = self.create_subscription(
                DetectionArray, self.det_topic, self.cb_dets_interfaces, qos_profile_sensor_data
            )
        elif _DET_MODE == "vision_msgs":
            self.sub_det = self.create_subscription(
                Detection2DArray, self.det_topic, self.cb_dets_vision, qos_profile_sensor_data
            )
        else:
            self.sub_det = None
            self.get_logger().warn(
                "No detection msg type available. Install interfaces_pkg/DetectionArray or vision_msgs/Detection2DArray."
            )

        self.sub_stop_line = self.create_subscription(Bool, self.stop_line_topic, self.cb_stop_line, 10)
        self.sub_stop_dist = self.create_subscription(Float32, self.stop_dist_topic, self.cb_stop_dist, 10)

        # Added subscriptions
        self.sub_vpair = self.create_subscription(Bool, self.vertical_pair_topic, self.cb_vertical_pair, 10)
        self.sub_lane_err = self.create_subscription(Float32, self.lane_center_err_topic, self.cb_lane_err, 10)

        # outputs
        self.pub_cones = self.create_publisher(Bool, "/parking2/cones_detected", 10)
        self.pub_forward_stop = self.create_publisher(Bool, "/parking2/forward_stop", 10)

        self.pub_mid_x_px = self.create_publisher(Float32, "/parking2/cone_mid_x_px", 10)
        self.pub_mid_x_norm = self.create_publisher(Float32, "/parking2/cone_mid_x_norm", 10)
        self.pub_align_err = self.create_publisher(Float32, "/parking2/align_error_px", 10)
        self.pub_align_done = self.create_publisher(Bool, "/parking2/align_done", 10)

        self.pub_phase_rev_straight = self.create_publisher(Bool, "/parking2/reverse_straight_phase", 10)
        self.pub_final_stop = self.create_publisher(Bool, "/parking2/final_stop", 10)

        self.pub_state = self.create_publisher(String, "/parking2/state", 10)

        self.pub_overlay = None
        self.pub_text = None
        if self.pub_debug:
            self.pub_overlay = self.create_publisher(Image, self.debug_overlay_topic, 10)
            self.pub_text = self.create_publisher(Marker, self.debug_text_marker_topic, 10)

        # ---------------- Runtime buffers ----------------
        self.last_frame: Optional[np.ndarray] = None
        self.last_header = None
        self.img_wh: Optional[Tuple[int, int]] = None

        self.cone_tops: List[Tuple[float, float]] = []
        self.cone_mid: Optional[Tuple[float, float]] = None

        # votes
        self.cones_votes: Deque[int] = deque(maxlen=max(1, self.cones_vote_window))
        self.align_votes: Deque[int] = deque(maxlen=max(1, self.align_vote_window))
        self.final_stop_votes: Deque[int] = deque(maxlen=max(1, self.final_stop_vote_window))

        # Added: vertical pair vote
        self.pair_votes: Deque[int] = deque(maxlen=max(1, self.pair_vote_window))

        # stop line info
        self.stop_line_detected = False
        self.stop_line_y_from_bottom = float("inf")

        # Added: vertical pair + lane center error
        self.vertical_pair_detected_now = False
        self.lane_center_error_px = float("nan")

        # state
        self.state = self.S_SEARCH

        self.get_logger().info(
            "ParkingPerceptionCoordinatorNode started.\n"
            f"  det_mode={_DET_MODE}\n"
            f"  image_topic={self.front_image_topic}\n"
            f"  det_topic={self.det_topic}\n"
            f"  stop_line_topic={self.stop_line_topic}\n"
            f"  vertical_pair_topic={self.vertical_pair_topic}\n"
            f"  lane_center_err_topic={self.lane_center_err_topic}\n"
            f"  cone_names={sorted(list(self.cone_names))}\n"
        )

    # -------------------- Detection parsing helpers --------------------
    def _is_cone_label(self, label: str) -> bool:
        return bool(label) and (label.strip().lower() in self.cone_names)

    def _pick_two_cones_by_size(self, cones: List[dict]) -> List[dict]:
        if len(cones) <= 2:
            return cones
        cones_sorted = sorted(cones, key=lambda c: (c.get("area", 0.0), c.get("score", 0.0)), reverse=True)
        return cones_sorted[:2]

    def _cone_top_from_bbox(self, bbox_xywh: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x, y, w, h = bbox_xywh
        return (x + 0.5 * w, y)

    # ---- interfaces_pkg DetectionArray ----
    def cb_dets_interfaces(self, msg):
        cones: List[dict] = []
        dets = getattr(msg, "detections", None)
        if dets is None:
            return

        for d in dets:
            label = ""
            for key in ["class_name", "label", "name", "class_id"]:
                if hasattr(d, key):
                    label = str(getattr(d, key))
                    break

            if not self._is_cone_label(label):
                continue

            bbox = None
            if hasattr(d, "bbox"):
                bb = getattr(d, "bbox")
                # NOTE: 원본 코드의 size_x/size_y vs size.x/size.y 혼재 가능
                if hasattr(bb, "center") and hasattr(bb, "size"):
                    cx = float(bb.center.position.x) if hasattr(bb.center, "position") else float(bb.center.x)
                    cy = float(bb.center.position.y) if hasattr(bb.center, "position") else float(bb.center.y)
                    w = float(bb.size.x) if hasattr(bb.size, "x") else float(getattr(bb, "size_x", 0.0))
                    h = float(bb.size.y) if hasattr(bb.size, "y") else float(getattr(bb, "size_y", 0.0))
                    if w > 0 and h > 0:
                        bbox = (cx - 0.5 * w, cy - 0.5 * h, w, h)
                elif all(hasattr(bb, k) for k in ["x", "y", "w", "h"]):
                    bbox = (float(bb.x), float(bb.y), float(bb.w), float(bb.h))
            if bbox is None and all(hasattr(d, k) for k in ["x", "y", "w", "h"]):
                bbox = (float(d.x), float(d.y), float(d.w), float(d.h))
            if bbox is None:
                continue

            score = 0.0
            for key in ["score", "confidence", "prob"]:
                if hasattr(d, key):
                    score = float(getattr(d, key))
                    break

            area = float(bbox[2] * bbox[3])
            cones.append({"label": label, "bbox": bbox, "score": score, "area": area})

        self._update_cones_from_list(cones)

    # ---- vision_msgs Detection2DArray ----
    def cb_dets_vision(self, msg):
        cones: List[dict] = []
        for det in msg.detections:
            best_label = ""
            best_score = -1.0
            for r in det.results:
                cid = str(r.hypothesis.class_id)
                sc = float(r.hypothesis.score)
                if sc > best_score:
                    best_score = sc
                    best_label = cid

            if not self._is_cone_label(best_label):
                continue

            bb = det.bbox
            cx = float(bb.center.x)
            cy = float(bb.center.y)
            w = float(bb.size_x)
            h = float(bb.size_y)
            bbox = (cx - 0.5 * w, cy - 0.5 * h, w, h)
            cones.append({"label": best_label, "bbox": bbox, "score": best_score, "area": w * h})

        self._update_cones_from_list(cones)

    def _update_cones_from_list(self, cones: List[dict]):
        cones2 = self._pick_two_cones_by_size(cones)
        if len(cones2) < self.need_cones:
            self.cone_tops = []
            self.cone_mid = None
            self.cones_votes.append(0)
            return

        tops = [self._cone_top_from_bbox(c["bbox"]) for c in cones2]
        tops = sorted(tops, key=lambda p: p[0])

        mid = ((tops[0][0] + tops[1][0]) * 0.5, (tops[0][1] + tops[1][1]) * 0.5)
        self.cone_tops = tops
        self.cone_mid = mid
        self.cones_votes.append(1)

    # -------------------- Stop line callbacks --------------------
    def cb_stop_line(self, msg: Bool):
        self.stop_line_detected = bool(msg.data)

    def cb_stop_dist(self, msg: Float32):
        self.stop_line_y_from_bottom = float(msg.data)

    # -------------------- Added callbacks --------------------
    def cb_vertical_pair(self, msg: Bool):
        self.vertical_pair_detected_now = bool(msg.data)
        self.pair_votes.append(1 if self.vertical_pair_detected_now else 0)

    def cb_lane_err(self, msg: Float32):
        self.lane_center_error_px = float(msg.data)

    # -------------------- Main image callback --------------------
    def cb_front_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return

        self.last_frame = frame
        self.last_header = msg.header
        H, W = frame.shape[:2]
        self.img_wh = (W, H)

        self._tick_and_publish()

        if self.pub_debug:
            overlay = self._make_overlay(frame.copy())
            omsg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            omsg.header = msg.header
            self.pub_overlay.publish(omsg)

            self._publish_text_marker(self._state_text())

    # -------------------- Voting helpers --------------------
    def _cones_detected_voted(self) -> bool:
        return sum(self.cones_votes) >= self.cones_vote_need

    def _align_done_voted(self) -> bool:
        return sum(self.align_votes) >= self.align_vote_need

    def _pair_detected_voted(self) -> bool:
        return sum(self.pair_votes) >= self.pair_vote_need

    def _final_stop_voted(self) -> bool:
        return sum(self.final_stop_votes) >= self.final_stop_vote_need

    # -------------------- State machine --------------------
    def _tick_and_publish(self):
        W, H = self.img_wh if self.img_wh else (640, 480)
        img_cx = self.target_center_x_ratio * float(W)

        cones_ok = self._cones_detected_voted()
        cones_now = (self.cone_mid is not None and len(self.cone_tops) >= self.need_cones)

        # alignment error (px)
        align_err = 0.0
        mid_x_px = float("nan")
        mid_x_norm = float("nan")

        if cones_now:
            mid_x_px = float(self.cone_mid[0])
            mid_x_norm = float(self.cone_mid[0]) / float(W)
            align_err = float(self.cone_mid[0] - img_cx)
            self.align_votes.append(1 if abs(align_err) <= self.align_thr_px else 0)
        else:
            self.align_votes.append(0)

        align_done = self._align_done_voted()
        pair_ok = self._pair_detected_voted()

        # final stop vote: (선택) 라인추종 단계에서 + pair_ok일 때만 누적
        if self.stop_line_detected and (self.state == self.S_REVERSE_STRAIGHT) and pair_ok:
            self.final_stop_votes.append(1)
        else:
            self.final_stop_votes.append(0)

        final_stop = self._final_stop_voted()

        # ---------------- State transitions ----------------
        if self.state == self.S_SEARCH:
            if cones_ok:
                self.state = self.S_ALIGN_REV

        elif self.state == self.S_ALIGN_REV:
            # ✅ 변경: 정렬 완료 + 세로선 pair 안정 검출일 때만 라인추종 단계(REVERSE_STRAIGHT)로 확정
            if align_done and pair_ok:
                self.state = self.S_REVERSE_STRAIGHT

        elif self.state == self.S_REVERSE_STRAIGHT:
            if final_stop:
                self.state = self.S_DONE

        # ---------------- Publish outputs ----------------
        self.pub_cones.publish(Bool(data=bool(cones_ok)))

        # forward stop: SEARCH 이후 True 유지
        forward_stop = (self.state != self.S_SEARCH)
        self.pub_forward_stop.publish(Bool(data=bool(forward_stop)))

        self.pub_mid_x_px.publish(Float32(data=float(mid_x_px if np.isfinite(mid_x_px) else -1.0)))
        self.pub_mid_x_norm.publish(Float32(data=float(mid_x_norm if np.isfinite(mid_x_norm) else -1.0)))
        self.pub_align_err.publish(Float32(data=float(align_err)))

        self.pub_align_done.publish(Bool(data=bool(self.state >= self.S_REVERSE_STRAIGHT)))

        # ✅ reverse straight phase flag: pair_ok가 True일 때만 “라인 추종 단계”로 인정
        reverse_straight_phase = (self.state == self.S_REVERSE_STRAIGHT) and pair_ok
        self.pub_phase_rev_straight.publish(Bool(data=bool(reverse_straight_phase)))

        self.pub_final_stop.publish(Bool(data=bool(self.state == self.S_DONE)))
        self.pub_state.publish(String(data=self._state_name()))

    def _state_name(self) -> str:
        if self.state == self.S_SEARCH:
            return "SEARCH_CONES"
        if self.state == self.S_ALIGN_REV:
            return "ALIGN_REVERSE"
        if self.state == self.S_REVERSE_STRAIGHT:
            return "REVERSE_STRAIGHT"
        if self.state == self.S_DONE:
            return "DONE"
        return "UNKNOWN"

    # -------------------- Debug overlay --------------------
    def _make_overlay(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        cx = int(self.target_center_x_ratio * W) # 화면의 목표 중앙 (보통 0.5)

        # 1. 화면 중앙선 (흰색)
        cv2.line(img, (cx, 0), (cx, H - 1), (255, 255, 255), 1)

        if self.cone_mid is not None and len(self.cone_tops) >= 2:
            # 2. 감지된 꼬깔들 (노란색 점)
            for i, (x, y) in enumerate(self.cone_tops):
                xi = int(clamp(x, 0, W - 1))
                yi = int(clamp(y, 0, H - 1))
                cv2.circle(img, (xi, yi), 8, (0, 255, 255), -1)
            
            # 3. 꼬깔들의 중점 (빨간색 점)
            mx, my = self.cone_mid
            mxi = int(clamp(mx, 0, W - 1))
            myi = int(clamp(my, 0, H - 1))
            cv2.circle(img, (mxi, myi), 12, (0, 0, 255), -1) 

            # 이 선이 길수록 핸들을 많이 꺾습니다.
            cv2.line(img, (mxi, myi), (cx, myi), (0, 0, 255), 3) 
            
            err_pixel = mxi - cx
            cv2.putText(img, f"Err: {err_pixel}px", (mxi, myi - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img

    def _state_text(self) -> str:
        pair_ok = (sum(self.pair_votes) >= self.pair_vote_need)
        return (
            f"[Parking2]\n"
            f"state={self._state_name()}\n"
            f"cones_vote={sum(self.cones_votes)}/{self.cones_vote_window} need={self.cones_vote_need}\n"
            f"align_vote={sum(self.align_votes)}/{self.align_vote_window} need={self.align_vote_need} thr={self.align_thr_px:.1f}px\n"
            f"vertical_pair={int(pair_ok)} vote={sum(self.pair_votes)}/{self.pair_vote_window} need={self.pair_vote_need}\n"
            f"lane_center_error_px={self.lane_center_error_px:+.1f}\n"
            f"stop_line={int(self.stop_line_detected)} y_from_bottom={self.stop_line_y_from_bottom:.1f}\n"
            f"final_vote={sum(self.final_stop_votes)}/{self.final_stop_vote_window} need={self.final_stop_vote_need}"
        )

    def _publish_text_marker(self, text: str):
        if not self.pub_debug or self.pub_text is None:
            return
        mk = Marker()
        mk.header.frame_id = self.debug_marker_frame
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = "parking2_state"
        mk.id = 0
        mk.type = Marker.TEXT_VIEW_FACING
        mk.action = Marker.ADD
        mk.pose.position.x = self.debug_marker_xyz[0]
        mk.pose.position.y = self.debug_marker_xyz[1]
        mk.pose.position.z = self.debug_marker_xyz[2]
        mk.pose.orientation.w = 1.0
        mk.scale.z = 0.2
        mk.color.a = 1.0
        mk.color.r = 1.0
        mk.color.g = 1.0
        mk.color.b = 1.0
        mk.text = text
        self.pub_text.publish(mk)


def main(args=None):
    rclpy.init(args=args)
    node = ParkingPerceptionCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()