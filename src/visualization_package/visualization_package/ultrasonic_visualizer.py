import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray 

class UltrasonicVisualizer(Node):
    def __init__(self):
        super().__init__('ultrasonic_visualizer_node')

        # ==========================================================
        # 1. 설정값 정의
        # ==========================================================
        self.frame_ids = [
            'ultrasonic1_link',
            'ultrasonic2_link',
            'ultrasonic3_link',
            'ultrasonic4_link',
            'ultrasonic5_link',
            'ultrasonic6_link'
        ]

        # ==========================================================
        # 2. Publisher 생성 (MarkerArray)
        # ==========================================================
        self.marker_publisher = self.create_publisher(MarkerArray, 'ultrasonic/markers', 10)
            
        # ==========================================================
        # 3. Subscriber 생성
        # ==========================================================
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ultrasonic', 
            self.listener_callback,
            10
        )
        self.get_logger().info('Ultrasonic Visualizer Started')

    def listener_callback(self, msg):
        if len(msg.data) < 6:
            return

        current_time = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for i in range(6):
            dist = float(msg.data[i])
            frame_id = self.frame_ids[i]

            # ------------------------------------------------------
            # Marker 시각화 로직
            # ------------------------------------------------------
            
            # 1. 색상 결정
            color = ColorRGBA(a=1.0)
            if dist < 0.5:
                color.r, color.g, color.b = 1.0, 0.0, 0.0 # Red
            elif dist < 1.5:
                color.r, color.g, color.b = 1.0, 1.0, 0.0 # Yellow
            else:
                color.r, color.g, color.b = 0.0, 1.0, 0.0 # Green

            # 2. 구(Sphere) 마커
            sphere_marker = Marker()
            sphere_marker.header.frame_id = frame_id
            sphere_marker.header.stamp = current_time
            sphere_marker.ns = "obstacle"
            sphere_marker.id = i 
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            
            sphere_marker.scale.x = 0.05
            sphere_marker.scale.y = 0.05
            sphere_marker.scale.z = 0.05
            
            sphere_marker.pose.position.x = dist 
            sphere_marker.pose.position.y = 0.0
            sphere_marker.pose.position.z = 0.0
            sphere_marker.pose.orientation.w = 1.0
            
            sphere_marker.color = color
            sphere_marker.lifetime.nanosec = 200000000
            
            marker_array.markers.append(sphere_marker)

            # 3. 텍스트(Text) 마커
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = current_time
            text_marker.ns = "distance_text"
            text_marker.id = i + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.scale.z = 0.1 
            
            text_marker.pose.position.x = dist
            text_marker.pose.position.y = 0.0
            text_marker.pose.position.z = 0.3 
            
            text_marker.text = f"{dist:.2f}m"
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.lifetime.nanosec = 200000000
            
            marker_array.markers.append(text_marker)

        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()