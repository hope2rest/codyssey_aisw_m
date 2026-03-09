"""navigation_node.py - 자율 주행 네비게이션 ROS2 노드.

A* 전역 경로 계획, DWA 로컬 플래너, Pure Pursuit 경로 추종을 통합하여
목표 지점까지 자율 주행을 수행한다.

Subscriptions:
    /goal_pose (PoseStamped): 목표 위치
    /map (OccupancyGrid): 점유 격자 지도
    /odom (Odometry): 로봇 위치/속도

Publications:
    /cmd_vel (Twist): 속도 명령
    /planned_path (Path): 계획된 전역 경로
    /local_trajectory (Path): DWA 로컬 궤적
"""

import sys
import os
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist, PoseStamped, Point
    from nav_msgs.msg import Odometry, OccupancyGrid, Path
    from sensor_msgs.msg import LaserScan
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.astar import AStarPlanner
from core.dwa import DWAPlanner
from core.pure_pursuit import PurePursuitController
from core.costmap import Costmap2D


class NavigationNode:
    """자율 주행 네비게이션 노드.

    전역 경로 계획(A*) → 로컬 경로 계획(DWA) → 경로 추종(Pure Pursuit)
    파이프라인을 통합하여 목표 지점까지 안전하게 주행한다.
    """

    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[NavigationNode] rclpy 미설치 - 독립 실행 모드")
            return

        self.node = rclpy.create_node('navigation_node')

        # 파라미터 선언
        self.node.declare_parameter('max_linear_vel', 1.0)
        self.node.declare_parameter('max_angular_vel', 1.5)
        self.node.declare_parameter('max_accel', 1.0)
        self.node.declare_parameter('robot_radius', 0.3)
        self.node.declare_parameter('goal_tolerance', 0.2)
        self.node.declare_parameter('replan_interval', 2.0)
        self.node.declare_parameter('lookahead_distance', 0.8)
        self.node.declare_parameter('costmap_resolution', 0.05)
        self.node.declare_parameter('inflation_radius', 0.5)

        # 파라미터 로드
        self.max_v = self.node.get_parameter('max_linear_vel').value
        self.max_w = self.node.get_parameter('max_angular_vel').value
        self.max_accel = self.node.get_parameter('max_accel').value
        self.robot_radius = self.node.get_parameter('robot_radius').value
        self.goal_tol = self.node.get_parameter('goal_tolerance').value
        self.lookahead = self.node.get_parameter('lookahead_distance').value

        # 핵심 알고리즘 모듈
        self.astar = AStarPlanner()
        self.dwa = DWAPlanner(
            max_speed=self.max_v,
            max_yaw_rate=self.max_w,
            max_accel=self.max_accel,
            robot_radius=self.robot_radius
        )
        self.pure_pursuit = PurePursuitController(
            lookahead_distance=self.lookahead
        )

        # 상태 변수
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.current_vel = np.array([0.0, 0.0])
        self.goal_pose = None
        self.global_path = None
        self.costmap = None
        self.obstacles = []
        self.is_navigating = False

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=10)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        # 구독
        self.node.create_subscription(
            PoseStamped, '/goal_pose', self._goal_callback, 10)
        self.node.create_subscription(
            OccupancyGrid, '/map', self._map_callback, map_qos)
        self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, sensor_qos)
        self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, sensor_qos)

        # 발행
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.node.create_publisher(Path, '/planned_path', 10)

        # 제어 루프 (20Hz)
        self.node.create_timer(0.05, self._control_loop)

        self.node.get_logger().info('네비게이션 노드 초기화 완료')

    def _goal_callback(self, msg):
        """목표 위치 수신 시 전역 경로 계획을 시작."""
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        q = msg.pose.orientation
        gyaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.goal_pose = np.array([gx, gy, gyaw])
        self.node.get_logger().info(
            f'목표 수신: ({gx:.2f}, {gy:.2f}, {math.degrees(gyaw):.1f}°)')
        self._plan_global_path()

    def _map_callback(self, msg):
        """지도 수신 시 Costmap 업데이트."""
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        data = np.array(msg.data).reshape((height, width))
        self.costmap = Costmap2D(
            width=width, height=height, resolution=resolution)
        self.costmap.static_layer = data
        self.costmap.update()
        self.node.get_logger().info(
            f'지도 수신: {width}x{height}, 해상도 {resolution}m')

    def _odom_callback(self, msg):
        """오도메트리에서 현재 위치/속도 추출."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.current_pose = np.array([p.x, p.y, yaw])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z])

    def _scan_callback(self, msg):
        """LiDAR 스캔에서 장애물 좌표 추출."""
        angles = np.arange(
            msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid[:len(angles)]]
        ranges = ranges[valid[:len(ranges)]]

        cos_a = np.cos(angles + self.current_pose[2])
        sin_a = np.sin(angles + self.current_pose[2])
        self.obstacles = np.column_stack([
            self.current_pose[0] + ranges * cos_a,
            self.current_pose[1] + ranges * sin_a
        ]).tolist()

    def _plan_global_path(self):
        """A* 전역 경로 계획."""
        if self.goal_pose is None or self.costmap is None:
            return
        try:
            start = tuple(self.current_pose[:2])
            goal = tuple(self.goal_pose[:2])
            path = self.astar.plan(
                self.costmap.get_costmap(), start, goal,
                self.costmap.resolution)
            self.global_path = path
            self.is_navigating = True
            self._publish_path(path)
            self.node.get_logger().info(
                f'전역 경로 계획 완료: {len(path)} waypoints')
        except Exception as e:
            self.node.get_logger().error(f'경로 계획 실패: {e}')

    def _control_loop(self):
        """주 제어 루프: DWA → Pure Pursuit → cmd_vel 발행."""
        if not self.is_navigating or self.goal_pose is None:
            return

        # 목표 도달 확인
        dist = np.linalg.norm(
            self.current_pose[:2] - self.goal_pose[:2])
        if dist < self.goal_tol:
            self._stop()
            self.is_navigating = False
            self.node.get_logger().info('목표 도달!')
            return

        # DWA 로컬 플래닝
        state = np.array([
            self.current_pose[0], self.current_pose[1],
            self.current_pose[2],
            self.current_vel[0], self.current_vel[1]])

        v, w = self.dwa.plan(
            state, self.goal_pose[:2], self.obstacles)

        # 속도 명령 발행
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def _stop(self):
        """로봇 정지 명령."""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def _publish_path(self, path):
        """전역 경로를 Path 메시지로 발행."""
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for x, y in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def spin(self):
        if ROS2_AVAILABLE:
            rclpy.spin(self.node)


def main():
    if ROS2_AVAILABLE:
        rclpy.init()
        node = NavigationNode()
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.shutdown()
    else:
        print("[NavigationNode] ROS2 미설치 - demo 모드로 main.py를 사용하세요.")


if __name__ == '__main__':
    main()
