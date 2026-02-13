import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    description_pkg_share = get_package_share_directory('minicar_description')

    urdf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_pkg_share, 'launch', 'display.launch.py')
        ),
    )


    workspace_model_path = os.path.join(os.getcwd(), 'best_cone.pt')
    print(f"\n[Real-Car] Loading YOLO model from: {workspace_model_path}\n")

    usb_cam_share = get_package_share_directory('usb_cam')
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(usb_cam_share, 'launch', 'camera2.launch.py')
        ),
    )

    rplidar_pkg_dir = get_package_share_directory('rplidar_ros')
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_pkg_dir, 'launch', 'rplidar_c1_launch.py')
        ),
    )

    yolo_node = Node(
        package='camera_perception_pkg',
        executable='yolov8_node',
        name='yolov8_node',
        output='screen',
        parameters=[{
                'enable_lane': False,   # 차선 인식 끄기
                'enablecone': True,    # 꼬깔 인식 켜기
        }],
        remappings=[
                ('camera1/image_raw', '/rear_camera/image_raw') 
        ]
    )

    yolo_visualizer_node = Node(
        package='debug_pkg',
        executable='yolov8_visualizer_node',
        name='yolov8_visualizer_node',
        output='screen',
        remappings=[
                ('camera1/image_raw', '/rear_camera/image_raw') 
        ]
    )

    bev_rear_node = Node(
        package='camera_perception_pkg',
        executable='bev_rear_node',
        name='bev_rear_node',
        output='screen'
    )

    parking_perception_node = Node(
        package='camera_perception_pkg',
        executable='parking_perception_node',
        name='parking_perception_node',
        output='screen'
    )

    parking_line_stop_node = Node(
        package='decision_making_pkg',
        executable='parking_line_stop_node',
        name='parking_line_stop_node',
        output='screen'
    )

    parking_planner_node = Node(
        package='decision_making_pkg',
        executable='parking_planner_node',
        name='parking_planner_node',
        output='screen'
    )

    serial_sender_node = Node(
            package='serial_communication_pkg',
            executable='serial_sender_node',
            name='serial_sender_node',
            output='screen'
        )


    return LaunchDescription([
        urdf_launch,
        camera_launch,
        lidar_launch,

        TimerAction(
            period=3.0,
            actions=[
                yolo_node,
                yolo_visualizer_node,
                bev_rear_node,
                parking_perception_node,
                parking_line_stop_node,
                parking_planner_node,
                serial_sender_node,
            ]
        )
    ])