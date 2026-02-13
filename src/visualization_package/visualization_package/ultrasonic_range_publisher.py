import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Range
import math

class UltrasonicRangePublisher(Node):
    def __init__(self):
        super().__init__('ultrasonic_range_publisher_node')

        # ==========================================================
        # 1. 설정값 정의
        # ==========================================================
        self.min_range = 0.02  # 2cm
        self.max_range = 4.0   # 4m
        self.fov = math.radians(15) 
        
        self.frame_ids = [
            'ultrasonic1_link',
            'ultrasonic2_link',
            'ultrasonic3_link',
            'ultrasonic4_link',
            'ultrasonic5_link',
            'ultrasonic6_link'
        ]

        # ==========================================================
        # 2. Publisher 생성 (Range 6개)
        # ==========================================================
        self.range_publishers = []
        for i in range(1, 7):
            topic_name = f'ultrasonic{i}/range'
            pub = self.create_publisher(Range, topic_name, 10)
            self.range_publishers.append(pub)
            
        # ==========================================================
        # 3. Subscriber 생성
        # ==========================================================
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ultrasonic', 
            self.listener_callback,
            10
        )
        self.get_logger().info('Ultrasonic Range Publisher Started')

    def listener_callback(self, msg):
        if len(msg.data) < 6:
            self.get_logger().warn(f'데이터 부족! 기대값: 6, 실제값: {len(msg.data)}')
            return

        current_time = self.get_clock().now().to_msg()
        
        for i in range(6):
            dist = float(msg.data[i])
            frame_id = self.frame_ids[i]

            # Range 메시지 발행
            range_msg = Range()
            range_msg.header.stamp = current_time
            range_msg.header.frame_id = frame_id
            range_msg.radiation_type = Range.ULTRASOUND
            range_msg.field_of_view = self.fov
            range_msg.min_range = self.min_range
            range_msg.max_range = self.max_range
            range_msg.range = dist
            
            self.range_publishers[i].publish(range_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicRangePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()