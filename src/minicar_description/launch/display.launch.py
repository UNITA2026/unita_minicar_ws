import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    package_name = 'minicar_description'
    urdf_file_name = 'unita_minicar.urdf' 
    rviz_file_name = 'minicar.rviz'

    pkg_path = get_package_share_directory(package_name)
    xacro_file = os.path.join(pkg_path, 'urdf', urdf_file_name)
    rviz_config_file = os.path.join(pkg_path, 'rviz', rviz_file_name)

    robot_description = Command(['xacro ', xacro_file])
    robot_description = ParameterValue(
        Command(['cat ', xacro_file]), # xacro 파일이면 'xacro ' 사용
        value_type=str
    )
    
    # Robot State Publisher: 로봇의 관절 상태를 받아 TF(좌표)를 계산
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # Joint State Publisher GUI: 관절을 움직이는 슬라이더 창 띄우기
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    # RViz2: 시각화 도구 실행
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    # visualization_package 노드 추가
    # ultrasonic_range_publisher_node = Node(
    #     package='visualization_package',
    #     executable='ultrasonic_range_publisher',
    #     name='ultrasonic_range_publisher',
    #     output='screen'
    # )

    # ultrasonic_visualizer_node = Node(
    #     package='visualization_package',
    #     executable='ultrasonic_visualizer',
    #     name='ultrasonic_visualizer',
    #     output='screen'
    # )

    # 5. 실행할 노드들을 리스트로 반환
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        # ultrasonic_range_publisher_node,
        # ultrasonic_visualizer_node,
        rviz_node
    ])