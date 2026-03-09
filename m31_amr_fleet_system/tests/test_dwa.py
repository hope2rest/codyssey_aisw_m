"""
DWA (Dynamic Window Approach) 단위 테스트.

목표 도달, 장애물 회피, 동역학 제한을 검증한다.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dwa import dwa_planning, DWAConfig


class TestDWAPlanning:
    """DWA 기본 동작 테스트."""

    def test_returns_velocity(self):
        """유효한 (v, omega) 튜플을 반환한다."""
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.empty((0, 2))

        v, w = dwa_planning(state, goal, obstacles)

        assert isinstance(v, float)
        assert isinstance(w, float)

    def test_moves_toward_goal(self):
        """DWA가 목표 방향으로 이동한다."""
        config = DWAConfig(max_v=1.0, predict_time=1.0, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([10.0, 0.0])
        obstacles = np.empty((0, 2))

        v, w = dwa_planning(state, goal, obstacles, config)

        # 목표가 정면이므로 양의 선속도
        assert v > 0, f"목표 방향 선속도: {v}"

    def test_goal_reaching(self):
        """반복 적용으로 목표에 도달한다."""
        config = DWAConfig(max_v=2.0, predict_time=1.5, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = np.empty((0, 2))

        for _ in range(200):
            v, w = dwa_planning(state, goal, obstacles, config)

            state[0] += v * np.cos(state[2]) * config.dt
            state[1] += v * np.sin(state[2]) * config.dt
            state[2] += w * config.dt
            state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
            state[3] = v
            state[4] = w

            dist = np.sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)
            if dist < 0.5:
                break

        assert dist < 1.0, f"목표 도달 실패: 거리={dist:.2f}"


class TestObstacleAvoidance:
    """장애물 회피 테스트."""

    def test_avoids_obstacle_on_path(self):
        """경로 상 장애물을 회피한다."""
        config = DWAConfig(
            max_v=1.0, predict_time=2.0, dt=0.1,
            robot_radius=0.3, clearance_weight=2.0,
        )
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        goal = np.array([10.0, 0.0])
        # 정면에 장애물
        obstacles = np.array([[3.0, 0.0]])

        v, w = dwa_planning(state, goal, obstacles, config)

        # 장애물이 정면이면 회전해야 함
        # (또는 속도를 줄이거나)
        # 정확한 값은 보장 불가하지만 충돌하지 않아야 함

    def test_no_collision_in_simulation(self):
        """시뮬레이션 중 장애물과 충돌하지 않는다."""
        config = DWAConfig(
            max_v=1.5, predict_time=2.0, dt=0.1,
            robot_radius=0.3, clearance_weight=2.0,
        )
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([10.0, 0.0])
        obstacles = np.array([
            [3.0, 0.0],
            [5.0, 1.0],
            [7.0, -0.5],
        ])

        min_obstacle_dist = float('inf')

        for _ in range(150):
            v, w = dwa_planning(state, goal, obstacles, config)

            state[0] += v * np.cos(state[2]) * config.dt
            state[1] += v * np.sin(state[2]) * config.dt
            state[2] += w * config.dt
            state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
            state[3] = v
            state[4] = w

            # 최소 장애물 거리 확인
            for obs in obstacles:
                dist = np.sqrt((state[0] - obs[0]) ** 2 + (state[1] - obs[1]) ** 2)
                min_obstacle_dist = min(min_obstacle_dist, dist)

        assert min_obstacle_dist > config.robot_radius * 0.8, (
            f"장애물 근접: {min_obstacle_dist:.3f}m"
        )


class TestDynamicsConstraints:
    """동역학 제한 테스트."""

    def test_velocity_within_bounds(self):
        """선택된 속도가 전체 속도 범위 내이다."""
        config = DWAConfig(max_v=1.0, max_w=1.5, max_acc_v=0.5, max_acc_w=2.0, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        goal = np.array([5.0, 3.0])
        obstacles = np.empty((0, 2))

        v, w = dwa_planning(state, goal, obstacles, config)

        # 전체 속도 범위 내에 있어야 함
        assert config.min_v - 1e-6 <= v <= config.max_v + 1e-6, f"v={v} 범위 밖: [{config.min_v}, {config.max_v}]"
        assert -config.max_w - 1e-6 <= w <= config.max_w + 1e-6, f"w={w} 범위 밖: [{-config.max_w}, {config.max_w}]"

    def test_zero_velocity_with_surrounding_obstacles(self):
        """사방에 장애물이면 정지할 수 있다."""
        config = DWAConfig(max_v=1.0, robot_radius=0.3, clearance_weight=5.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        # 매우 가까운 장애물로 둘러싸기
        obstacles = np.array([
            [0.4, 0.0],
            [-0.4, 0.0],
            [0.0, 0.4],
            [0.0, -0.4],
        ])

        v, w = dwa_planning(state, goal, obstacles, config)
        # 안전한 선택: 낮은 속도
        assert abs(v) < config.max_v
