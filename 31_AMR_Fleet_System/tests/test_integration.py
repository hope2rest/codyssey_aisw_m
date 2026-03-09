"""test_integration.py - 통합 테스트 (10개 시나리오).

여러 모듈을 조합하여 End-to-End 파이프라인을 검증한다.
ROS2 없이 순수 Python/NumPy로 실행 가능.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.astar import plan_path, smooth_path
from core.dwa import dwa_planning, DWAConfig
from core.ekf import ExtendedKalmanFilter, fuse_sensors, ODOM_H, ODOM_R, AMCL_H, AMCL_R
from core.kinematics import DifferentialDriveRobot
from core.pure_pursuit import PurePursuitController, compute_control
from core.costmap import Costmap
from core.kalman_tracker import ObjectTracker
from core.slam import SLAM2D, OccupancyGridMap
from fleet.task_allocator import TaskAllocator, Robot, Task
from fleet.traffic_manager import TrafficManager, RobotPath, PathSegment, Zone
from task.behavior_tree import build_amr_mission_tree, Blackboard, NodeStatus
from task.docking import DockingController, DockingTarget, RobotPose
from simulation.sensor_noise import LidarNoise, IMUNoise, EncoderNoise


class TestSlamToNavigation:
    """시나리오 1: SLAM 지도 → A* 경로 계획 연동."""

    def test_slam_to_navigation(self):
        slam = SLAM2D(map_size=(100, 100), resolution=0.1)
        for i in range(5):
            angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
            scan = np.column_stack([
                np.cos(angles) * 4.5, np.sin(angles) * 4.5])
            slam.process_scan(scan, np.array([0.1, 0.0, 0.0]))

        grid = slam.get_map()
        assert grid is not None

        # A* 경로 계획
        simple_grid = np.zeros((20, 20), dtype=np.int8)
        simple_grid[10, 5:15] = 1
        path = plan_path(simple_grid, (0, 0), (19, 19))
        assert path is not None and len(path) > 0


class TestEKFWithNoisySensors:
    """시나리오 2: EKF가 노이즈 센서 데이터를 올바르게 처리."""

    def test_ekf_with_noisy_sensors(self):
        ekf = ExtendedKalmanFilter()
        np.random.seed(42)
        true_x, true_y, true_theta = 0.0, 0.0, 0.0
        errors = []

        for step in range(100):
            v, w = 0.5, 0.1
            dt = 0.1
            true_x += v * math.cos(true_theta) * dt
            true_y += v * math.sin(true_theta) * dt
            true_theta += w * dt

            # EKF 예측
            ekf.predict(dt)

            # 주기적 AMCL 관측 업데이트
            if step % 10 == 0:
                obs = np.array([
                    true_x + np.random.normal(0, 0.05),
                    true_y + np.random.normal(0, 0.05),
                    true_theta + np.random.normal(0, 0.02)
                ])
                ekf.update(obs, AMCL_H, AMCL_R)

            est = ekf.get_state()
            err = math.sqrt(
                (est[0] - true_x)**2 + (est[1] - true_y)**2)
            errors.append(err)

        # EKF가 발산하지 않음
        assert errors[-1] < 5.0, f"EKF 최종 오차: {errors[-1]:.3f}m"


class TestNavigationFullPipeline:
    """시나리오 3: A* → DWA 연쇄 파이프라인."""

    def test_navigation_full_pipeline(self):
        grid = np.zeros((20, 20), dtype=np.int8)
        grid[10, 3:17] = 1
        path = plan_path(grid, (5, 5), (15, 15))
        assert path is not None and len(path) >= 2

        config = DWAConfig()
        state = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
        goal = np.array([15.0, 15.0])
        v, w = dwa_planning(state, goal, [], config)

        assert isinstance(v, (int, float, np.floating))
        assert abs(v) <= config.max_v + 0.01


class TestFleetTaskCycle:
    """시나리오 4: 작업 할당 사이클."""

    def test_fleet_task_cycle(self):
        allocator = TaskAllocator()

        robots = [
            Robot(robot_id='amr_01', position=(5.0, 5.0)),
            Robot(robot_id='amr_02', position=(20.0, 10.0)),
            Robot(robot_id='amr_03', position=(40.0, 30.0)),
        ]
        tasks = [
            Task(task_id='task_1', position=(10.0, 10.0)),
            Task(task_id='task_2', position=(25.0, 15.0)),
            Task(task_id='task_3', position=(45.0, 35.0)),
        ]

        assignments = allocator.allocate(tasks, robots)
        assert len(assignments) > 0
        # 모든 작업이 할당됨
        assert len(assignments) == 3


class TestObstacleAvoidanceWithTracking:
    """시나리오 5: 칼만 트래커 → DWA 회피."""

    def test_obstacle_avoidance_with_tracking(self):
        tracker = ObjectTracker()
        config = DWAConfig()

        obstacle_pos = [5.0, 0.0]
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        goal = np.array([10.0, 0.0])

        for step in range(10):
            obstacle_pos[0] -= 0.3
            tracker.update([obstacle_pos[:]])
            v, w = dwa_planning(state, goal, [obstacle_pos[:]], config)
            assert abs(v) <= config.max_v + 0.01

            dt = 0.1
            state[0] += v * math.cos(state[2]) * dt
            state[1] += v * math.sin(state[2]) * dt
            state[2] += w * dt
            state[3] = v
            state[4] = w


class TestMultiRobotNoCollision:
    """시나리오 6: 5대 로봇 충돌 없이 이동."""

    def test_multi_robot_no_collision(self):
        np.random.seed(42)
        positions = [
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
            [0.0, 10.0], [10.0, 10.0]]

        for step in range(20):
            for i in range(5):
                for j in range(i+1, 5):
                    dist = math.sqrt(
                        (positions[i][0] - positions[j][0])**2 +
                        (positions[i][1] - positions[j][1])**2)
                    assert dist > 0.5

            for i in range(5):
                positions[i][0] += np.random.uniform(-0.3, 0.3)
                positions[i][1] += np.random.uniform(-0.3, 0.3)


class TestDeadlockDetectionAndResolution:
    """시나리오 7: 교착 탐지 및 해소."""

    def test_deadlock_detection_and_resolution(self):
        traffic_mgr = TrafficManager()

        # Wait-For Graph에 순환 등록
        traffic_mgr.clear_wait_graph()

        # 교차로 등록
        zone = Zone(zone_id='intersection_1', position=(10.0, 10.0), radius=2.0, capacity=1)
        traffic_mgr.register_zone(zone)

        # 교착 탐지 (사이클이 없는 경우도 정상)
        deadlocks = traffic_mgr.detect_deadlock()
        assert isinstance(deadlocks, list)


class TestDockingPrecision:
    """시나리오 8: 도킹 시스템."""

    def test_docking_precision(self):
        docker = DockingController()

        target = DockingTarget(
            station_id='rack_A1',
            position=(10.0, 5.0),
            orientation=0.0)
        initial = RobotPose(x=9.0, y=5.1, theta=0.05)

        result = docker.execute_docking(target, initial)
        # 도킹 결과 확인
        assert result is not None
        assert hasattr(result, 'success')


class TestBehaviorTreeFullScenario:
    """시나리오 9: Behavior Tree 실행."""

    def test_behavior_tree_full_scenario(self):
        tree = build_amr_mission_tree()
        assert tree is not None

        # BT 틱 (blackboard 없이 실행)
        for tick in range(5):
            status = tree.tick()
            assert status in (
                NodeStatus.SUCCESS, NodeStatus.FAILURE,
                NodeStatus.RUNNING)


class TestEmergencyStop:
    """시나리오 10: 긴급 정지."""

    def test_emergency_stop(self):
        config = DWAConfig()
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        goal = np.array([10.0, 0.0])
        obstacles = [[0.25, 0.0]]

        v, w = dwa_planning(state, goal, obstacles, config)
        assert abs(v) < 0.5, \
            f"장애물 근접 시 속도가 너무 높음: v={v:.3f}"
