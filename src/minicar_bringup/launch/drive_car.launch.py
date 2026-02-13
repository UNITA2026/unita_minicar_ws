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

    workspace_model_path = os.path.join(os.getcwd(), 'lane.pt')
    print(f"\n[Real-Car] Loading YOLO model from: {workspace_model_path}\n")

    usb_cam_share = get_package_share_directory('usb_cam')
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(usb_cam_share, 'launch', 'camera.launch.py')
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
    )

    yolo_visualizer_node = Node(
        package='debug_pkg',
        executable='yolov8_visualizer_node',
        name='yolov8_visualizer_node',
        output='screen',
    )

    box_lidar_match_node = Node(
        package='camera_perception_pkg',
        executable='box_lidar_match_node',
        name='box_lidar_match_node',
        output='screen'
    )

    # image_fusion_node = Node(
    #         package='camera_perception_pkg',
    #         executable='image_fusion_node',
    #         name='image_fusion_node',
    #         output='screen'
    #     )

    lane_info_node = Node(
        package='camera_perception_pkg',
        executable='lane_info_extractor_node',
        name='lane_info_extractor_node',
        output='screen'
    )

    path_planner_node = Node(
        package='decision_making_pkg',
        executable='path_planner_node',
        name='path_planner_node',
        output='screen'
    )

    motion_planner_node = Node(
        package='decision_making_pkg',
        executable='motion_planner_node',
        name='motion_planner_node',
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
                box_lidar_match_node,
                # image_fusion_node,
                lane_info_node,
                path_planner_node,
                motion_planner_node,
                serial_sender_node,
            ]
        )
    ])