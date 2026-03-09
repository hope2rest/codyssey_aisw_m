"""
Fleet 관리 시스템 Launch 파일.

다중 로봇 네비게이션 + Fleet 관리자 + 트래픽 매니저.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(pkg_dir, 'config')
    launch_dir = os.path.join(pkg_dir, 'launch')

    fleet_config = os.path.join(config_dir, 'fleet_config.yaml')
    num_robots = LaunchConfiguration('num_robots', default='5')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 로봇 네임스페이스 목록
    robot_namespaces = ['amr_01', 'amr_02', 'amr_03', 'amr_04', 'amr_05']

    launch_items = [
        DeclareLaunchArgument('num_robots', default_value='5'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
    ]

    # 각 로봇에 대해 네비게이션 스택 실행
    for ns in robot_namespaces:
        nav_group = GroupAction([
            PushRosNamespace(ns),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_dir, 'navigation.launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'namespace': ns,
                }.items(),
            ),
        ])
        launch_items.append(nav_group)

    # Fleet 관리자 노드 (사용자 정의)
    launch_items.append(
        Node(
            package='amr_fleet_system',
            executable='fleet_manager',
            name='fleet_manager',
            output='screen',
            parameters=[fleet_config, {'use_sim_time': use_sim_time}],
        )
    )

    return LaunchDescription(launch_items)
