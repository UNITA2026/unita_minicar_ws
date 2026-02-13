#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque
from typing import Deque, Tuple, Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker

class ParkingLineStopNode(Node):
    """
    [최종_v3] 평행 기준 강화 (Strict Parallel)
    - 평행(Vertical Pair) 판단 시 기울기가 거의 수직이어야만 인정
    """

    def __init__(self):
        super().__init__("parking_line_stop_node")

        # ---------------- Params ----------------
        self.declare_parameter("bev_topic", "/rear_camera/image_bev")
        
        # 색상 (흰색/회색)
        self.declare_parameter("hsv_s_max", 30)   
        self.declare_parameter("hsv_v_min", 200)  

        # ROI 
        self.declare_parameter("v_roi_y_min_ratio", 0.0) 
        self.declare_parameter("v_roi_y_max_ratio", 1.0) 
        
        # HoughLinesP
        self.declare_parameter("hough_threshold", 20)      
        self.declare_parameter("hough_min_line_length", 20)
        self.declare_parameter("hough_max_line_gap", 30)   

        # 정지선 조건
        self.declare_parameter("stop_y_from_bottom_px", 100.0) 
        self.declare_parameter("estimated_lane_width_px", 280.0) 

        self.declare_parameter("vote_window", 5)
        self.declare_parameter("vote_need", 3)
        self.declare_parameter("pair_vote_window", 5)
        self.declare_parameter("pair_vote_need", 3)

        self.declare_parameter("pub_debug", True)

        # ---------------- Load params ----------------
        self.bev_topic = str(self.get_parameter("bev_topic").value)
        self.hsv_s_max = int(self.get_parameter("hsv_s_max").value)
        self.hsv_v_min = int(self.get_parameter("hsv_v_min").value)

        self.v_y0_r = float(self.get_parameter("v_roi_y_min_ratio").value)
        self.v_y1_r = float(self.get_parameter("v_roi_y_max_ratio").value)

        self.hough_th = int(self.get_parameter("hough_threshold").value)
        self.min_len = int(self.get_parameter("hough_min_line_length").value)
        self.max_gap = int(self.get_parameter("hough_max_line_gap").value)

        self.stop_y_px = float(self.get_parameter("stop_y_from_bottom_px").value)
        self.est_lane_w = float(self.get_parameter("estimated_lane_width_px").value)

        self.vote_window = int(self.get_parameter("vote_window").value)
        self.vote_need = int(self.get_parameter("vote_need").value)
        self.pair_vote_window = int(self.get_parameter("pair_vote_window").value)
        self.pair_vote_need = int(self.get_parameter("pair_vote_need").value)

        self.pub_debug = bool(self.get_parameter("pub_debug").value)

        # ---------------- ROS ----------------
        self.bridge = CvBridge()
        self.sub_bev = self.create_subscription(Image, self.bev_topic, self.cb_bev, qos_profile_sensor_data)

        self.pub_stop = self.create_publisher(Bool, "/parking/stop_line_detected", 10)
        self.pub_vpair = self.create_publisher(Bool, "/parking/vertical_pair_detected", 10)
        self.pub_lane_err = self.create_publisher(Float32, "/parking/lane_center_error_px", 10)

        if self.pub_debug:
            self.pub_overlay = self.create_publisher(Image, "/parking/debug/overlay", 10)
            self.pub_mask = self.create_publisher(Image, "/parking/debug/mask", 10) 
            self.pub_text = self.create_publisher(Marker, "/parking/debug/state_text", 10)

        self.stop_votes: Deque[int] = deque(maxlen=max(1, self.vote_window))
        self.pair_votes: Deque[int] = deque(maxlen=max(1, self.pair_vote_window))

        self.get_logger().info("ParkingLineStopNode: Strict Parallel Check Mode.")

    def _white_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.hsv_v_min])     
        upper = np.array([180, self.hsv_s_max, 255]) 
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def cb_bev(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        H, W = bgr.shape[:2]
        mask = self._white_mask(bgr)
        overlay = bgr.copy()

        v_y0 = int(H * self.v_y0_r)
        v_y1 = int(H * self.v_y1_r)
        if v_y0 >= v_y1: v_y0, v_y1 = 0, H
        mask_roi = mask[v_y0:v_y1, :]

        lines = cv2.HoughLinesP(
            mask_roi, 1, np.pi/180, 
            threshold=self.hough_th, 
            minLineLength=self.min_len, 
            maxLineGap=self.max_gap
        )

        v_detected = False
        stop_detected = False
        lane_center_err_px = float("nan")
        
        vertical_lines_x = []
        horizontal_lines_y = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                y1 += v_y0
                y2 += v_y0

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                # 모든 선 그리기
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 100, 0), 1)

                # 1. 수직 주차선 (세로가 가로의 5배 이상) -> ✅ [핵심 강화]
                # dy > 5.0 * dx: 약 78도 이상의 급경사만 수직선으로 인정
                # 이렇게 해야 차가 정말 11자로 섰을 때만 vertical_pair가 True가 됨.
                if dy > 5.0 * dx: 
                    mx = (x1 + x2) / 2.0
                    vertical_lines_x.append(mx)
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2) # 진한 초록

                # 2. 정지선 (가로가 세로의 0.5배 이상)
                elif dx > 0.5 * dy:
                    my = (y1 + y2) / 2.0
                    if my > H * 0.6: 
                        horizontal_lines_y.append(my)
                        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨강

        # [A] 정지선 판단
        y_from_bottom = 9999.0
        if len(horizontal_lines_y) > 0:
            avg_y = np.max(horizontal_lines_y)
            y_from_bottom = float(H - avg_y)
            if y_from_bottom <= self.stop_y_px:
                stop_detected = True

        self.stop_votes.append(1 if stop_detected else 0)
        stop_final = (sum(self.stop_votes) >= self.vote_need)

        # [B] 수직 주차선 판단
        img_cx = W / 2.0
        if len(vertical_lines_x) > 0:
            vertical_lines_x.sort()
            
            lx = vertical_lines_x[0]
            rx = vertical_lines_x[-1]
            width = rx - lx

            if 100 < width < 500:
                lane_center = (lx + rx) / 2.0
                lane_center_err_px = lane_center - img_cx
                v_detected = True
                cv2.line(overlay, (int(lx), 0), (int(lx), H), (255, 255, 0), 2)
                cv2.line(overlay, (int(rx), 0), (int(rx), H), (255, 255, 0), 2)
            else:
                avg_x = np.mean(vertical_lines_x)
                if avg_x < img_cx: # L-Only
                    lane_center = avg_x + (self.est_lane_w / 2.0)
                    lane_center_err_px = lane_center - img_cx
                    v_detected = True
                    cv2.putText(overlay, "L-Only", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else: # R-Only
                    lane_center = avg_x - (self.est_lane_w / 2.0)
                    lane_center_err_px = lane_center - img_cx
                    v_detected = True
                    cv2.putText(overlay, "R-Only", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
        self.pair_votes.append(1 if v_detected else 0)
        pair_final = (sum(self.pair_votes) >= self.pair_vote_need)

        # Publish
        self.pub_stop.publish(Bool(data=stop_final))
        self.pub_vpair.publish(Bool(data=pair_final))
        self.pub_lane_err.publish(Float32(data=float(lane_center_err_px if v_detected else 0.0)))

        # Debug
        if self.pub_debug:
            cv2.line(overlay, (int(img_cx), 0), (int(img_cx), H), (255, 255, 255), 1)
            info = f"Stop:{stop_final} y={y_from_bottom:.0f}"
            info2 = f"Line:{pair_final} Err={lane_center_err_px:.1f}"
            cv2.putText(overlay, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(overlay, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
            self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))

def main(args=None):
    rclpy.init(args=args)
    node = ParkingLineStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()