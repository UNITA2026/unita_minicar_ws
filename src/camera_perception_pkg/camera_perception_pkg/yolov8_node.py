# Copyright (C) 2023  Miguel Ángel González Santamarta
# Modified for Dual Model Inference (Lane + Cone)

from typing import List, Dict

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
from torch import cuda

from sensor_msgs.msg import Image
from interfaces_pkg.msg import Point2D
from interfaces_pkg.msg import BoundingBox2D
from interfaces_pkg.msg import Mask
from interfaces_pkg.msg import KeyPoint2D
from interfaces_pkg.msg import KeyPoint2DArray
from interfaces_pkg.msg import Detection
from interfaces_pkg.msg import DetectionArray

from std_srvs.srv import SetBool


class Yolov8Node(LifecycleNode):

    def __init__(self, **kwargs) -> None:
        super().__init__("yolov8_node", **kwargs)
        
        #---------------Variable Setting---------------
        # 차선 인식 모델
        self.declare_parameter("model_lane", "lane.pt")
        # 꼬깔(장애물) 인식 모델 (추가됨)
        self.declare_parameter("model_cone", "best_cone.pt")
        
        # 추론 하드웨어 선택 (cpu / gpu) 
        self.declare_parameter("device", "cpu")
        #----------------------------------------------
        
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.RELIABLE)

        self.get_logger().info('Yolov8Node created')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')

        self.model_lane_path = self.get_parameter(
            "model_lane").get_parameter_value().string_value
        
        self.model_cone_path = self.get_parameter(
            "model_cone").get_parameter_value().string_value

        self.device = self.get_parameter(
            "device").get_parameter_value().string_value

        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value

        self.reliability = self.get_parameter(
            "image_reliability").get_parameter_value().integer_value

        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._pub = self.create_lifecycle_publisher(
            DetectionArray, "detections", 10)
        self._srv = self.create_service(
            SetBool, "enable", self.enable_cb
        )
        self.cv_bridge = CvBridge()

        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')

        try:
            # 두 모델 모두 로딩
            self.get_logger().info(f"Loading Lane Model: {self.model_lane_path}")
            self.yolo_lane = YOLO(self.model_lane_path)
            
            self.get_logger().info(f"Loading Cone Model: {self.model_cone_path}")
            self.yolo_cone = YOLO(self.model_cone_path)

        except FileNotFoundError as e:
            self.get_logger().error(f"Error: Model file not found! {e}")
            return TransitionCallbackReturn.FAILURE
        except Exception as e:
            self.get_logger().error(f"Error while loading models: {str(e)}")
            return TransitionCallbackReturn.FAILURE

        # subs
        self._sub = self.create_subscription(
            Image,
            "camera1/image_raw",
            self.image_cb,
            self.image_qos_profile
        )

        super().on_activate(state)

        return TransitionCallbackReturn.SUCCESS


    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating {self.get_name()}')

        if hasattr(self, 'yolo_lane'): del self.yolo_lane
        if hasattr(self, 'yolo_cone'): del self.yolo_cone

        if 'cuda' in self.device:
            self.get_logger().info("Clearing CUDA cache")
            cuda.empty_cache()

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        self.destroy_publisher(self._pub)
        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    # -------------------------------------------------------------------------
    # Helper Functions for Parsing (Modified to accept class names dynamically)
    # -------------------------------------------------------------------------

    def parse_hypothesis(self, results: Results, class_names: Dict) -> List[Dict]:
        hypothesis_list = []
        box_data: Boxes
        for box_data in results.boxes:
            cls_id = int(box_data.cls)
            hypothesis = {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, f"unknown_{cls_id}"), 
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)
        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list = []
        box_data: Boxes
        for box_data in results.boxes:
            msg = BoundingBox2D()
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])
            boxes_list.append(msg)
        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        masks_list = []
        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p
        mask: Masks
        for mask in results.masks:
            msg = Mask()
            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]
            masks_list.append(msg)
        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        keypoints_list = []
        points: Keypoints
        for points in results.keypoints:
            msg_array = KeyPoint2DArray()
            if points.conf is None:
                continue
            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self.threshold:
                    msg = KeyPoint2D()
                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)
                    msg_array.data.append(msg)
            keypoints_list.append(msg_array)
        return keypoints_list

    def process_model_results(self, results_list, detections_msg: DetectionArray):
        """
        모델의 추론 결과를 파싱하여 detections_msg에 추가하는 함수
        """
        results: Results = results_list[0].cpu()
        class_names = results.names 

        hypothesis = []
        boxes = []
        masks = []
        keypoints = []

        if results.boxes:
            hypothesis = self.parse_hypothesis(results, class_names)
            boxes = self.parse_boxes(results)

        if results.masks:
            masks = self.parse_masks(results)

        if results.keypoints:
            keypoints = self.parse_keypoints(results)

        # Append to main message
        for i in range(len(results)):
            aux_msg = Detection()

            if results.boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

            if results.masks:
                aux_msg.mask = masks[i]

            if results.keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

    def image_cb(self, msg: Image) -> None:
        
        if self.enable:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            detections_msg = DetectionArray()
            detections_msg.header = msg.header

            results_lane = self.yolo_lane.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device,
                # imgsz=320
            )
            self.process_model_results(results_lane, detections_msg)

            results_cone = self.yolo_cone.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device,
                imgsz=320
            )
            self.process_model_results(results_cone, detections_msg)

            self._pub.publish(detections_msg)


def main():
    rclpy.init()
    node = Yolov8Node()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()