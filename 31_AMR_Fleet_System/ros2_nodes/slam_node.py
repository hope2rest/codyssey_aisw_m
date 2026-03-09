"""slam_node.py - 2D LiDAR SLAM ROS2 노드.

/scan (LaserScan) + /odom (Odometry) 을 구독하여
ICP 스캔 매칭 기반 SLAM을 수행하고
/map (OccupancyGrid), /slam_pose (PoseStamped) 를 발행한다.
"""

import sys
import os
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.slam import SLAM2D


class SlamNode:
    """2D LiDAR SLAM 노드.

    LiDAR 스캔과 오도메트리를 입력받아 ICP 기반 스캔 매칭으로
    로봇 위치를 추정하고 점유 격자 지도를 생성한다.

    Subscriptions:
        /scan (LaserScan): 2D LiDAR 스캔 데이터
        /odom (Odometry): 휠 오도메트리

    Publications:
        /map (OccupancyGrid): 점유 격자 지도
        /slam_pose (PoseStamped): SLAM 추정 위치
    """

    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[SlamNode] rclpy 미설치 - 독립 실행 모드")
            self.slam = SLAM2D(map_size=(1200, 800), resolution=0.05)
            return

        self.node = rclpy.create_node('slam_node')

        # 파라미터
        self.node.declare_parameter('map_resolution', 0.05)
        self.node.declare_parameter('map_width', 1200)
        self.node.declare_parameter('map_height', 800)
        self.node.declare_parameter('loop_closure_interval', 50)

        resolution = self.node.get_parameter('map_resolution').value
        width = self.node.get_parameter('map_width').value
        height = self.node.get_parameter('map_height').value
        self.loop_interval = self.node.get_parameter('loop_closure_interval').value

        self.slam = SLAM2D(map_size=(width, height), resolution=resolution)
        self.scan_count = 0
        self.prev_odom = None

        # QoS 설정
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # 구독
        self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, sensor_qos)
        self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, sensor_qos)

        # 발행
        self.map_pub = self.node.create_publisher(
            OccupancyGrid, '/map', map_qos)
        self.pose_pub = self.node.create_publisher(
            PoseStamped, '/slam_pose', 10)

        # 주기적 지도 발행 (1Hz)
        self.node.create_timer(1.0, self._publish_map)

        self.node.get_logger().info('SLAM 노드 초기화 완료')

    def _scan_callback(self, msg):
        """LiDAR 스캔 데이터를 극좌표→직교좌표로 변환 후 SLAM 처리."""
        angles = np.arange(
            msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # 유효 범위 필터링
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid[:len(angles)]]
        ranges = ranges[valid[:len(ranges)]]

        # 직교좌표 변환
        points = np.column_stack([
            ranges * np.cos(angles),
            ranges * np.sin(angles)
        ])

        # 오도메트리 델타 계산
        odom_delta = None
        if self.prev_odom is not None and hasattr(self, '_current_odom'):
            dx = self._current_odom[0] - self.prev_odom[0]
            dy = self._current_odom[1] - self.prev_odom[1]
            dtheta = self._current_odom[2] - self.prev_odom[2]
            odom_delta = np.array([dx, dy, dtheta])
            self.prev_odom = self._current_odom.copy()

        # SLAM 처리
        pose = self.slam.process_scan(points, odom_delta)
        self.scan_count += 1

        # Loop Closure 주기적 실행
        if self.scan_count % self.loop_interval == 0:
            self.slam.run_loop_closure()

        # 위치 발행
        self._publish_pose(pose)

    def _odom_callback(self, msg):
        """오도메트리 데이터 저장."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._current_odom = np.array([p.x, p.y, yaw])
        if self.prev_odom is None:
            self.prev_odom = self._current_odom.copy()

    def _publish_pose(self, pose):
        """SLAM 추정 위치를 PoseStamped로 발행."""
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = 0.0
        yaw = float(pose[2])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(msg)

    def _publish_map(self):
        """점유 격자 지도를 OccupancyGrid로 발행."""
        grid = self.slam.get_map()
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.info.resolution = self.slam.map.resolution
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]
        msg.info.origin.position.x = float(self.slam.map.origin[0])
        msg.info.origin.position.y = float(self.slam.map.origin[1])
        # -1(unknown)→-1, 0(free)→0, 100(occupied)→100
        msg.data = grid.flatten().astype(np.int8).tolist()
        self.map_pub.publish(msg)

    def spin(self):
        """노드 실행."""
        if ROS2_AVAILABLE:
            rclpy.spin(self.node)


def main():
    if ROS2_AVAILABLE:
        rclpy.init()
        node = SlamNode()
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.shutdown()
    else:
        print("[SlamNode] ROS2 미설치 - demo 모드로 main.py를 사용하세요.")


if __name__ == '__main__':
    main()
