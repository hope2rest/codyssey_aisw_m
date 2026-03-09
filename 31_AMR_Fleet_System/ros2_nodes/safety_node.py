"""safety_node.py - 안전 시스템 ROS2 노드.

LiDAR 데이터를 모니터링하여 장애물 접근 시 단계적으로 대응한다.
Warning Zone(1.0m) 진입 시 감속, Critical Zone(0.3m) 진입 시 긴급 정지.
E-Stop 서비스를 제공한다.

Subscriptions:
    /scan (LaserScan): LiDAR 스캔 데이터
    /odom (Odometry): 현재 속도 정보

Publications:
    /cmd_vel (Twist): 속도 명령 오버라이드 (긴급 정지 시)
    /safety/status (String): 안전 상태 (JSON)

Services:
    /safety/e_stop (Trigger): 비상 정지
    /safety/release (Trigger): 비상 정지 해제
"""

import sys
import os
import json
import time
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class SafetyZone:
    """안전 구역 정의."""
    SAFE = 'safe'
    WARNING = 'warning'
    CRITICAL = 'critical'
    EMERGENCY = 'emergency'


class SafetyNode:
    """안전 시스템 노드.

    LiDAR 기반 장애물 거리를 실시간 모니터링하여
    다단계 안전 대응을 수행한다.

    Safety Zones:
        - Safe (> 1.0m): 정상 주행
        - Warning (0.3m ~ 1.0m): 감속 주행 (50%)
        - Critical (< 0.3m): 긴급 정지
        - Emergency: E-Stop 활성화 (수동 해제 필요)
    """

    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[SafetyNode] rclpy 미설치 - 독립 실행 모드")
            return

        self.node = rclpy.create_node('safety_node')

        # 파라미터
        self.node.declare_parameter('warning_distance', 1.0)
        self.node.declare_parameter('critical_distance', 0.3)
        self.node.declare_parameter('slowdown_factor', 0.5)
        self.node.declare_parameter('check_angle_range', 120.0)
        self.node.declare_parameter('sensor_timeout', 1.0)

        self.warning_dist = self.node.get_parameter(
            'warning_distance').value
        self.critical_dist = self.node.get_parameter(
            'critical_distance').value
        self.slowdown = self.node.get_parameter('slowdown_factor').value
        self.check_angle = math.radians(
            self.node.get_parameter('check_angle_range').value / 2.0)
        self.sensor_timeout = self.node.get_parameter(
            'sensor_timeout').value

        # 상태
        self.current_zone = SafetyZone.SAFE
        self.e_stop_active = False
        self.min_distance = float('inf')
        self.min_distance_angle = 0.0
        self.last_scan_time = time.time()
        self.original_cmd_vel = None
        self.current_speed = 0.0
        self.obstacle_count = 0

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=10)

        # 구독
        self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, sensor_qos)
        self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, sensor_qos)

        # 발행
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.node.create_publisher(
            String, '/safety/status', 10)

        # 서비스
        self.node.create_service(
            Trigger, '/safety/e_stop', self._e_stop_callback)
        self.node.create_service(
            Trigger, '/safety/release', self._release_callback)

        # 안전 체크 루프 (20Hz)
        self.node.create_timer(0.05, self._safety_check)
        # 상태 발행 (2Hz)
        self.node.create_timer(0.5, self._publish_status)

        self.node.get_logger().info(
            f'안전 노드 초기화 완료 '
            f'(Warning: {self.warning_dist}m, '
            f'Critical: {self.critical_dist}m)')

    def _scan_callback(self, msg):
        """LiDAR 스캔에서 전방 최소 거리 계산."""
        self.last_scan_time = time.time()
        angles = np.arange(
            msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # 전방 영역 필터 (-check_angle ~ +check_angle)
        front_mask = np.abs(angles) < self.check_angle
        valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        mask = front_mask[:len(valid_mask)] & valid_mask[:len(front_mask)]

        if np.any(mask):
            filtered_ranges = ranges[:len(mask)][mask]
            filtered_angles = angles[:len(mask)][mask]
            min_idx = np.argmin(filtered_ranges)
            self.min_distance = float(filtered_ranges[min_idx])
            self.min_distance_angle = float(filtered_angles[min_idx])
            self.obstacle_count = int(
                np.sum(filtered_ranges < self.warning_dist))
        else:
            self.min_distance = float('inf')
            self.obstacle_count = 0

    def _odom_callback(self, msg):
        """현재 속도 추적."""
        self.current_speed = math.sqrt(
            msg.twist.twist.linear.x ** 2 +
            msg.twist.twist.linear.y ** 2)

    def _safety_check(self):
        """주기적 안전 상태 판별 및 대응."""
        # E-Stop 활성 시 무조건 정지
        if self.e_stop_active:
            self.current_zone = SafetyZone.EMERGENCY
            self._emergency_stop()
            return

        # 센서 타임아웃 확인
        if time.time() - self.last_scan_time > self.sensor_timeout:
            self.node.get_logger().warn('LiDAR 센서 타임아웃 - 안전 정지')
            self._emergency_stop()
            self.current_zone = SafetyZone.CRITICAL
            return

        # 거리 기반 구역 판별
        prev_zone = self.current_zone

        if self.min_distance < self.critical_dist:
            self.current_zone = SafetyZone.CRITICAL
            self._emergency_stop()
            if prev_zone != SafetyZone.CRITICAL:
                self.node.get_logger().warn(
                    f'긴급 정지! 장애물 거리: {self.min_distance:.3f}m')

        elif self.min_distance < self.warning_dist:
            self.current_zone = SafetyZone.WARNING
            if prev_zone == SafetyZone.SAFE:
                self.node.get_logger().info(
                    f'경고 구역 진입. 장애물 거리: {self.min_distance:.3f}m')

        else:
            self.current_zone = SafetyZone.SAFE

    def _emergency_stop(self):
        """긴급 정지 명령 발행."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def _e_stop_callback(self, request, response):
        """E-Stop 서비스: 비상 정지 활성화."""
        self.e_stop_active = True
        self._emergency_stop()
        response.success = True
        response.message = '비상 정지 활성화'
        self.node.get_logger().error('E-STOP 활성화!')
        return response

    def _release_callback(self, request, response):
        """E-Stop 해제 서비스."""
        if self.min_distance > self.critical_dist:
            self.e_stop_active = False
            self.current_zone = SafetyZone.SAFE
            response.success = True
            response.message = '비상 정지 해제'
            self.node.get_logger().info('E-STOP 해제')
        else:
            response.success = False
            response.message = (
                f'장애물 근접 ({self.min_distance:.3f}m) - 해제 불가')
        return response

    def _publish_status(self):
        """안전 상태 발행."""
        status = {
            'zone': self.current_zone,
            'min_distance': round(self.min_distance, 3),
            'min_distance_angle_deg': round(
                math.degrees(self.min_distance_angle), 1),
            'e_stop_active': self.e_stop_active,
            'current_speed': round(self.current_speed, 3),
            'obstacles_in_warning': self.obstacle_count,
            'slowdown_factor': (
                self.slowdown
                if self.current_zone == SafetyZone.WARNING else 1.0),
            'timestamp': time.time()
        }
        msg = String()
        msg.data = json.dumps(status, ensure_ascii=False)
        self.status_pub.publish(msg)

    def get_speed_limit(self):
        """현재 안전 구역에 따른 속도 제한 비율 반환.

        Returns:
            float: 1.0(정상), slowdown_factor(경고), 0.0(정지)
        """
        if self.current_zone in (SafetyZone.CRITICAL, SafetyZone.EMERGENCY):
            return 0.0
        elif self.current_zone == SafetyZone.WARNING:
            return self.slowdown
        return 1.0

    def spin(self):
        if ROS2_AVAILABLE:
            rclpy.spin(self.node)


def main():
    if ROS2_AVAILABLE:
        rclpy.init()
        node = SafetyNode()
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.shutdown()
    else:
        print("[SafetyNode] ROS2 미설치 - demo 모드로 main.py를 사용하세요.")


if __name__ == '__main__':
    main()
