"""fleet_manager_node.py - 다중 AMR Fleet 관리 ROS2 노드.

5대의 AMR 로봇을 중앙 집중식으로 관리한다.
Hungarian 알고리즘으로 작업을 최적 할당하고,
교통 관리 및 교착 탐지/해소를 수행한다.

Subscriptions:
    /amr_XX/status (String): 각 로봇 상태 (JSON)
    /task_request (String): 작업 요청 (JSON)

Publications:
    /amr_XX/goal (PoseStamped): 각 로봇 목표 위치
    /fleet/status (String): Fleet 전체 상태 (JSON)
    /fleet/kpi (String): KPI 지표 (JSON)
    /fleet/alerts (String): 이상 상황 알림
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
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fleet.task_allocator import TaskAllocator
from fleet.traffic_manager import TrafficManager
from fleet.monitor import FleetMonitor


class FleetManagerNode:
    """다중 AMR Fleet 관리 노드.

    5대의 AMR을 동시 운용하며 다음 기능을 수행한다:
    - Hungarian 알고리즘 기반 최적 작업 할당
    - 경로 충돌 예측 및 교통 관리
    - Wait-For Graph 기반 교착 탐지 및 해소
    - 실시간 KPI 모니터링 및 알림
    """

    NUM_ROBOTS = 5

    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[FleetManagerNode] rclpy 미설치 - 독립 실행 모드")
            return

        self.node = rclpy.create_node('fleet_manager_node')

        # 파라미터
        self.node.declare_parameter('num_robots', 5)
        self.node.declare_parameter('allocation_method', 'hungarian')
        self.node.declare_parameter('deadlock_timeout', 10.0)
        self.node.declare_parameter('kpi_publish_rate', 1.0)
        self.node.declare_parameter('conflict_check_rate', 5.0)

        self.num_robots = self.node.get_parameter('num_robots').value
        alloc_method = self.node.get_parameter('allocation_method').value

        # Fleet 관리 모듈 초기화
        self.allocator = TaskAllocator(method=alloc_method)
        self.traffic_mgr = TrafficManager(
            deadlock_timeout=self.node.get_parameter(
                'deadlock_timeout').value)
        self.monitor = FleetMonitor(num_robots=self.num_robots)

        # 로봇 상태 관리
        self.robot_ids = [f'amr_{i+1:02d}' for i in range(self.num_robots)]
        self.robot_states = {}
        self.pending_tasks = []
        self.active_tasks = {}

        for rid in self.robot_ids:
            self.robot_states[rid] = {
                'position': [0.0, 0.0, 0.0],
                'velocity': [0.0, 0.0],
                'battery': 100.0,
                'state': 'idle',
                'current_task': None,
                'last_update': time.time()
            }

        # 구독: 각 로봇 상태
        for rid in self.robot_ids:
            self.node.create_subscription(
                String, f'/{rid}/status',
                lambda msg, r=rid: self._robot_status_callback(r, msg), 10)

        # 구독: 작업 요청
        self.node.create_subscription(
            String, '/task_request', self._task_request_callback, 10)

        # 발행: 각 로봇 목표
        self.goal_pubs = {}
        for rid in self.robot_ids:
            self.goal_pubs[rid] = self.node.create_publisher(
                PoseStamped, f'/{rid}/goal', 10)

        # 발행: Fleet 상태/KPI/알림
        self.status_pub = self.node.create_publisher(
            String, '/fleet/status', 10)
        self.kpi_pub = self.node.create_publisher(
            String, '/fleet/kpi', 10)
        self.alert_pub = self.node.create_publisher(
            String, '/fleet/alerts', 10)

        # 주기적 처리
        kpi_rate = self.node.get_parameter('kpi_publish_rate').value
        conflict_rate = self.node.get_parameter('conflict_check_rate').value

        self.node.create_timer(1.0 / kpi_rate, self._publish_kpi)
        self.node.create_timer(
            1.0 / conflict_rate, self._check_conflicts)
        self.node.create_timer(2.0, self._allocate_pending_tasks)

        self.node.get_logger().info(
            f'Fleet Manager 초기화 완료: {self.num_robots}대 AMR, '
            f'할당 방법={alloc_method}')

    def _robot_status_callback(self, robot_id, msg):
        """개별 로봇 상태 업데이트."""
        try:
            status = json.loads(msg.data)
            self.robot_states[robot_id].update(status)
            self.robot_states[robot_id]['last_update'] = time.time()
            self.monitor.update_robot_status(robot_id, status)
        except json.JSONDecodeError:
            self.node.get_logger().warn(
                f'{robot_id} 상태 파싱 실패')

    def _task_request_callback(self, msg):
        """작업 요청 수신 및 대기열 추가."""
        try:
            task = json.loads(msg.data)
            task['status'] = 'pending'
            task['received_at'] = time.time()
            self.pending_tasks.append(task)
            self.node.get_logger().info(
                f"작업 수신: {task.get('id', 'unknown')} "
                f"({task.get('from', '?')} → {task.get('to', '?')})")
        except json.JSONDecodeError:
            self.node.get_logger().warn('작업 요청 파싱 실패')

    def _allocate_pending_tasks(self):
        """대기 중인 작업을 유휴 로봇에 할당."""
        if not self.pending_tasks:
            return

        # 유휴 로봇 확인
        idle_robots = [
            rid for rid, state in self.robot_states.items()
            if state['state'] == 'idle'
        ]
        if not idle_robots:
            return

        # Hungarian 할당
        tasks_to_assign = self.pending_tasks[:len(idle_robots)]
        robot_positions = {
            rid: self.robot_states[rid]['position'][:2]
            for rid in idle_robots
        }
        task_positions = {
            t.get('id', str(i)): t.get('from_position', [0, 0])
            for i, t in enumerate(tasks_to_assign)
        }

        assignments = self.allocator.allocate(
            robot_positions, task_positions)

        # 할당 결과 적용
        for robot_id, task_id in assignments.items():
            task_idx = None
            for i, t in enumerate(tasks_to_assign):
                if t.get('id', str(i)) == task_id:
                    task_idx = i
                    break

            if task_idx is not None:
                task = tasks_to_assign[task_idx]
                self._assign_task(robot_id, task)
                self.pending_tasks.remove(task)

    def _assign_task(self, robot_id, task):
        """로봇에 작업 할당 및 목표 발행."""
        self.robot_states[robot_id]['state'] = 'navigating'
        self.robot_states[robot_id]['current_task'] = task.get('id')
        self.active_tasks[robot_id] = task

        # 목표 위치 발행
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.node.get_clock().now().to_msg()
        pos = task.get('to_position', [0, 0])
        goal.pose.position.x = float(pos[0])
        goal.pose.position.y = float(pos[1])
        goal.pose.orientation.w = 1.0
        self.goal_pubs[robot_id].publish(goal)

        self.node.get_logger().info(
            f"작업 할당: {robot_id} → {task.get('id', 'unknown')}")

    def _check_conflicts(self):
        """경로 충돌 예측 및 교착 탐지."""
        # 교착 탐지
        deadlocks = self.traffic_mgr.detect_deadlocks(self.robot_states)
        if deadlocks:
            for cycle in deadlocks:
                alert = {
                    'type': 'deadlock',
                    'robots': cycle,
                    'timestamp': time.time()
                }
                msg = String()
                msg.data = json.dumps(alert, ensure_ascii=False)
                self.alert_pub.publish(msg)
                self.node.get_logger().warn(
                    f'교착 감지: {cycle}')

                # 교착 해소
                self.traffic_mgr.resolve_deadlock(
                    cycle, self.robot_states)

        # 경로 충돌 확인
        conflicts = self.traffic_mgr.predict_conflicts(self.robot_states)
        for conflict in conflicts:
            self.traffic_mgr.resolve_conflict(
                conflict, self.robot_states)

    def _publish_kpi(self):
        """Fleet KPI 지표 발행."""
        kpi = self.monitor.compute_kpi()

        # Fleet 상태
        status = {
            'robots': {
                rid: {
                    'state': s['state'],
                    'battery': s['battery'],
                    'position': s['position'],
                    'current_task': s['current_task']
                }
                for rid, s in self.robot_states.items()
            },
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks)
        }

        status_msg = String()
        status_msg.data = json.dumps(status, ensure_ascii=False)
        self.status_pub.publish(status_msg)

        kpi_msg = String()
        kpi_msg.data = json.dumps(kpi, ensure_ascii=False)
        self.kpi_pub.publish(kpi_msg)

    def spin(self):
        if ROS2_AVAILABLE:
            rclpy.spin(self.node)


def main():
    if ROS2_AVAILABLE:
        rclpy.init()
        node = FleetManagerNode()
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.shutdown()
    else:
        print("[FleetManagerNode] ROS2 미설치 - demo 모드로 main.py를 사용하세요.")


if __name__ == '__main__':
    main()
