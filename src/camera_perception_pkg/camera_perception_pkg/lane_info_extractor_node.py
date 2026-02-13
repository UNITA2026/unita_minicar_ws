import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray
from .lib import camera_perception_func_lib as CPFL

#---------------Constant Variables---------------
SUB_TOPIC_NAME = "/detections"
SUB_OBSTACLE_TOPIC = "/lidar_obstacle_info"
PUB_TOPIC_NAME = "/yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "/roi_image"
SHOW_IMAGE = True
LANE_WIDTH_PIXEL = 280
AVOIDANCE_TRIGGER_DIST = 1.8
SHIFT_SPEED = 20.0
IMAGE_CENTER_X = 320
LANE_1_FAR_LEFT_THRESHOLD = 180
LANE_2_FAR_RIGHT_THRESHOLD = 460

# [추가] 차선 변경 인식을 위한 임계값 (노이즈 필터링)
# 값이 클수록 안정적이지만 반응이 느려짐 (10~15 추천)
LANE_CHANGE_THRESHOLD_COUNT = 15 
#----------------------------------------------

class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.sub_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_OBSTACLE_TOPIC).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.cv_bridge = CvBridge()
        self.qos_profile = qos_profile_sensor_data
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.obstacle_sub = self.create_subscription(Point32, self.sub_obstacle_topic, self.obstacle_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, 10)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, 10)

        self.current_lane_state = 'lane_2'
        self.current_offset = 0.0
        self.target_offset = 0.0
        self.obstacle_detected = False
        self.obstacle_dist = 999.0
        self.obstacle_pixel_x = -1.0

        # [추가] 상태 변경 카운터 변수
        self.lane_change_counter = 0
        self.potential_next_state = None

        self.get_logger().info("Method B: BBox Overlap Logic with Debouncing Ready.")

    def obstacle_callback(self, msg: Point32):
        if msg.z == 1.0:
            self.obstacle_detected = True
            self.obstacle_dist = msg.x
            self.obstacle_pixel_x = msg.y 
        else:
            self.obstacle_detected = False
            self.obstacle_dist = 999.0
            self.obstacle_pixel_x = -1.0

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0: return

        # 차선 정보 추출 (Localization용)
        lane_1_box = None
        lane_2_box = None
        lane_1_cx, lane_2_cx = -1, -1
        has_lane_1, has_lane_2 = False, False

        for d in detection_msg.detections:
            if d.class_name == 'lane_1':
                lane_1_cx = d.bbox.center.position.x
                lane_1_box = d 
                has_lane_1 = True
            elif d.class_name == 'lane_2':
                lane_2_cx = d.bbox.center.position.x
                lane_2_box = d 
                has_lane_2 = True

        # ---------------------------------------------------
        # [수정] 노이즈 필터링이 적용된 내 차선 판단 로직
        # ---------------------------------------------------
        
        # 1. 이번 프레임에서 감지된 '임시' 상태 판단
        detected_state = self.current_lane_state # 기본값은 유지

        if has_lane_1 and has_lane_2:
            dist_1 = abs(lane_1_cx - IMAGE_CENTER_X)
            dist_2 = abs(lane_2_cx - IMAGE_CENTER_X)
            detected_state = 'lane_1' if dist_1 < dist_2 else 'lane_2'
        elif has_lane_2 and not has_lane_1:
            detected_state = 'lane_1' if lane_2_cx > LANE_2_FAR_RIGHT_THRESHOLD else 'lane_2'
        elif has_lane_1 and not has_lane_2:
            detected_state = 'lane_2' if lane_1_cx < LANE_1_FAR_LEFT_THRESHOLD else 'lane_1'

        # 2. 상태 변경 카운팅 (Debouncing)
        # 감지된 상태가 현재 확정된 상태와 다르면 카운트 증가
        if detected_state != self.current_lane_state:
            # 새로운 상태가 이전 프레임의 '잠재적 변경 상태'와 같으면 카운트 계속 증가
            if detected_state == self.potential_next_state:
                self.lane_change_counter += 1
            else:
                # 튀는 값이 바뀌었으면 카운터 리셋하고 새로운 잠재 상태로 등록
                self.potential_next_state = detected_state
                self.lane_change_counter = 1
            
            # 카운터가 임계치를 넘으면 비로소 상태 변경 승인
            if self.lane_change_counter >= LANE_CHANGE_THRESHOLD_COUNT:
                self.get_logger().info(f"🔄 Lane State Changed: {self.current_lane_state} -> {detected_state}")
                self.current_lane_state = detected_state
                self.lane_change_counter = 0 # 리셋
        else:
            # 감지된 상태가 현재 상태와 같으면 카운터 초기화 (노이즈였다는 뜻)
            self.lane_change_counter = 0
            self.potential_next_state = None

        # ---------------------------------------------------
        # 2. [방식 B] BBox Overlap Check (겹침 확인)
        # ---------------------------------------------------
        self.target_offset = 0.0
        # ★ 중요: tracking_class는 필터링된 self.current_lane_state를 따라감
        tracking_class = self.current_lane_state 

        # 장애물이 어디 있는지 동적으로 판단
        obstacle_in_lane_1 = False
        obstacle_in_lane_2 = False

        if self.obstacle_detected:
            # 1차선 박스 안에 장애물 중심(Pixel X)이 들어가는가?
            if has_lane_1:
                l1_min = lane_1_box.bbox.center.position.x - (lane_1_box.bbox.size.x / 2)
                l1_max = lane_1_box.bbox.center.position.x + (lane_1_box.bbox.size.x / 2)
                if l1_min < self.obstacle_pixel_x < l1_max:
                    obstacle_in_lane_1 = True

            # 2차선 박스 안에 장애물 중심(Pixel X)이 들어가는가?
            if has_lane_2:
                l2_min = lane_2_box.bbox.center.position.x - (lane_2_box.bbox.size.x / 2)
                l2_max = lane_2_box.bbox.center.position.x + (lane_2_box.bbox.size.x / 2)
                if l2_min < self.obstacle_pixel_x < l2_max:
                    obstacle_in_lane_2 = True

            # (만약 박스가 안 잡혔다면 픽셀 기준으로 대체)
            if not has_lane_1 and not has_lane_2:
                if self.obstacle_pixel_x < 320: obstacle_in_lane_1 = True
                else: obstacle_in_lane_2 = True

        # 전략 수립 (이제 self.current_lane_state가 안정적이므로 튀지 않음)
        if self.obstacle_detected and self.obstacle_dist < AVOIDANCE_TRIGGER_DIST:
            if self.current_lane_state == 'lane_2':
                # 내가 2차선인데 2차선에 장애물이 '확실히' 있다 -> 피함
                if obstacle_in_lane_2:
                    self.target_offset = -LANE_WIDTH_PIXEL
                    self.get_logger().warn(f"🚧 Obs Inside Lane 2 Box -> Dodge LEFT")
                else:
                    self.target_offset = 0.0

            elif self.current_lane_state == 'lane_1':
                # 내가 1차선인데 1차선에 장애물이 '확실히' 있다 -> 피함
                if obstacle_in_lane_1:
                    self.target_offset = LANE_WIDTH_PIXEL
                    self.get_logger().warn(f"🚧 Obs Inside Lane 1 Box -> Dodge RIGHT")
                else:
                    self.target_offset = 0.0

        # ... (이하 로직 동일) ...
        final_tracking_class = tracking_class
        final_offset_modifier = 0.0

        if tracking_class == 'lane_1':
            if has_lane_1: final_tracking_class = 'lane_1'; final_offset_modifier = 0.0
            elif has_lane_2: final_tracking_class = 'lane_2'; final_offset_modifier = -LANE_WIDTH_PIXEL
        elif tracking_class == 'lane_2':
            if has_lane_2: final_tracking_class = 'lane_2'; final_offset_modifier = 0.0
            elif has_lane_1: final_tracking_class = 'lane_1'; final_offset_modifier = LANE_WIDTH_PIXEL

        real_target_offset = self.target_offset + final_offset_modifier

        if self.current_offset < real_target_offset: self.current_offset = min(self.current_offset + SHIFT_SPEED, real_target_offset)
        elif self.current_offset > real_target_offset: self.current_offset = max(self.current_offset - SHIFT_SPEED, real_target_offset)

        try:
            edge_image = CPFL.draw_edges(detection_msg, cls_name=final_tracking_class, color=255)
            (h, w) = (edge_image.shape[0], edge_image.shape[1])
            # [사용자 설정값 적용됨]
            dst_mat = [[round(w * 0.2), round(h * 0.0)], [round(w * 0.8), round(h * 0.0)], [round(w * 0.8), h], [round(w * 0.2), h]]
            src_mat = [[154, 298], [486, 298], [614, 470], [26, 470]]

            bird_image_raw = CPFL.bird_convert(edge_image, srcmat=src_mat, dstmat=dst_mat)
            bird_image = cv2.convertScaleAbs(bird_image_raw)
            roi_image = CPFL.roi_rectangle_below(bird_image, cutting_idx=300)

            if self.show_image:
                debug_img = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
                # 디버깅 정보 추가 (카운터 표시)
                cv2.putText(debug_img, f"State: {self.current_lane_state} (cnt:{self.lane_change_counter})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if self.obstacle_detected:
                     obs_info = "L1" if obstacle_in_lane_1 else ("L2" if obstacle_in_lane_2 else "None")
                     color = (0,0,255) if (self.current_lane_state=='lane_1' and obstacle_in_lane_1) or (self.current_lane_state=='lane_2' and obstacle_in_lane_2) else (200,200,200)
                     cv2.putText(debug_img, f"Obs In: {obs_info}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow('Overlap-Based Logic', debug_img)
                cv2.waitKey(1)
        except Exception: return

        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
        target_points = []
        for target_point_y in range(5, 155, 30):
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, detection_thickness=10, road_gradient=grad, lane_width=300)
            if target_point_x != -1:
                final_x = target_point_x + self.current_offset
                final_x = max(0, min(640, final_x))
            else: final_x = -1
            tp = TargetPoint(); tp.target_x = round(final_x); tp.target_y = round(target_point_y); target_points.append(tp)

        lane = LaneInfo(); lane.slope = grad; lane.target_points = target_points
        self.publisher.publish(lane)
        try: self.roi_image_publisher.publish(self.cv_bridge.cv2_to_imgmsg(cv2.convertScaleAbs(roi_image), encoding="mono8"))
        except: pass

def main(args=None):
    rclpy.init(args=args); node = Yolov8InfoExtractor()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); cv2.destroyAllWindows(); rclpy.shutdown()
if __name__ == '__main__': main()