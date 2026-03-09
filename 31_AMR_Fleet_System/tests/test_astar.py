"""
A* 경로 계획 단위 테스트.

최단 경로 검증, 장애물 회피, 에지 케이스를 검증한다.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.astar import plan_path


class TestPathFinding:
    """경로 탐색 기본 테스트."""

    def test_straight_path(self):
        """장애물 없는 그리드에서 경로를 찾는다."""
        grid = np.zeros((10, 10))
        path = plan_path(grid, (0, 0), (9, 9), smooth=False)

        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_shortest_path_no_obstacle(self):
        """장애물 없을 때 최단 경로 길이 검증."""
        grid = np.zeros((10, 10))
        path = plan_path(grid, (0, 0), (9, 9), smooth=False)

        # 대각선 이동 가능 시 최단: max(dy, dx) = 9
        # A*의 옥타일 거리에서 대각선 9칸이면 경로 길이 10 (시작 포함)
        assert len(path) == 10, f"최단 경로 길이: {len(path)}"

    def test_path_avoids_obstacles(self):
        """장애물을 피해 경로를 탐색한다."""
        grid = np.zeros((10, 10))
        # 중앙에 벽 배치 (완전 차단은 아님)
        grid[5, 1:9] = 1

        path = plan_path(grid, (0, 0), (9, 9), smooth=False)
        assert len(path) > 0

        # 경로가 장애물 위를 지나지 않아야 함
        for r, c in path:
            assert grid[r, c] < 0.5, f"경로가 장애물 위: ({r}, {c})"

    def test_no_path_exists(self):
        """경로가 없을 때 빈 리스트를 반환한다."""
        grid = np.zeros((10, 10))
        # 완전 차단
        grid[5, :] = 1

        path = plan_path(grid, (0, 0), (9, 9), smooth=False)
        assert path == []

    def test_start_equals_goal(self):
        """시작과 목표가 같으면 단일 점 경로."""
        grid = np.zeros((10, 10))
        path = plan_path(grid, (5, 5), (5, 5), smooth=False)
        assert len(path) == 1
        assert path[0] == (5, 5)

    def test_adjacent_goal(self):
        """인접한 목표까지의 경로."""
        grid = np.zeros((10, 10))
        path = plan_path(grid, (5, 5), (5, 6), smooth=False)
        assert len(path) == 2


class TestPathValidity:
    """경로 유효성 테스트."""

    def test_path_continuity(self):
        """경로의 연속성: 인접 셀만 이동."""
        grid = np.zeros((20, 20))
        grid[10, 3:17] = 1
        path = plan_path(grid, (0, 0), (19, 19), smooth=False)

        for i in range(1, len(path)):
            dr = abs(path[i][0] - path[i - 1][0])
            dc = abs(path[i][1] - path[i - 1][1])
            assert dr <= 1 and dc <= 1, (
                f"불연속 경로: {path[i-1]} -> {path[i]}"
            )

    def test_path_on_large_grid(self):
        """큰 그리드에서도 경로를 찾는다."""
        grid = np.zeros((100, 100))
        # 산발적 장애물
        np.random.seed(42)
        for _ in range(200):
            r, c = np.random.randint(1, 99, 2)
            grid[r, c] = 1

        # 시작/끝이 자유 공간인지 확인
        grid[0, 0] = 0
        grid[99, 99] = 0

        path = plan_path(grid, (0, 0), (99, 99), smooth=False)
        # 경로가 있을 수도 없을 수도 있지만 에러 없이 실행
        assert isinstance(path, list)


class TestEdgeCases:
    """에지 케이스 테스트."""

    def test_start_on_obstacle_raises(self):
        """시작이 장애물 위이면 ValueError."""
        grid = np.zeros((10, 10))
        grid[0, 0] = 1
        with pytest.raises(ValueError):
            plan_path(grid, (0, 0), (9, 9))

    def test_goal_on_obstacle_raises(self):
        """목표가 장애물 위이면 ValueError."""
        grid = np.zeros((10, 10))
        grid[9, 9] = 1
        with pytest.raises(ValueError):
            plan_path(grid, (0, 0), (9, 9))

    def test_out_of_bounds_raises(self):
        """범위 밖 좌표는 ValueError."""
        grid = np.zeros((10, 10))
        with pytest.raises(ValueError):
            plan_path(grid, (-1, 0), (9, 9))
        with pytest.raises(ValueError):
            plan_path(grid, (0, 0), (10, 10))

    def test_smooth_path(self):
        """평활화된 경로도 유효하다."""
        grid = np.zeros((20, 20))
        grid[10, 3:17] = 1

        path = plan_path(grid, (0, 0), (19, 19), smooth=True)
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (19, 19)

        # 평활화된 경로가 장애물 위를 지나지 않아야 함
        for r, c in path:
            assert grid[r, c] < 0.5
