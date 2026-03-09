"""
Fleet 관리 단위 테스트.

Hungarian 할당 최적성, 교착 탐지를 검증한다.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fleet.task_allocator import TaskAllocator, Task, Robot
from fleet.traffic_manager import TrafficManager, RobotPath, PathSegment, Zone


class TestHungarianAlgorithm:
    """Hungarian 알고리즘 최적성 테스트."""

    def test_simple_assignment(self):
        """간단한 2x2 비용 행렬 할당."""
        cost = np.array([
            [1.0, 3.0],
            [2.0, 1.0],
        ])
        allocator = TaskAllocator()
        result = allocator.hungarian_algorithm(cost)

        # 최적 할당: (0,0)=1 + (1,1)=1 = 2
        assigned = dict(result)
        assert assigned[0] == 0 and assigned[1] == 1, f"할당 결과: {result}"

    def test_optimal_cost(self):
        """3x3 비용 행렬에서 최적 비용 검증."""
        cost = np.array([
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ])
        allocator = TaskAllocator()
        result = allocator.hungarian_algorithm(cost)

        total_cost = sum(cost[r, c] for r, c in result)
        # 최적: (0,1)=1 + (1,0)=2 + (2,2)=2 = 5
        assert abs(total_cost - 5.0) < 1e-6, f"총 비용: {total_cost}"

    def test_rectangular_matrix(self):
        """비정방 행렬 (로봇 < 작업)."""
        cost = np.array([
            [1.0, 5.0, 3.0],
            [2.0, 4.0, 1.0],
        ])
        allocator = TaskAllocator()
        result = allocator.hungarian_algorithm(cost)

        assert len(result) == 2
        robots = [r for r, _ in result]
        tasks = [t for _, t in result]
        assert len(set(robots)) == 2
        assert len(set(tasks)) == 2

    def test_all_same_cost(self):
        """모든 비용이 동일한 경우."""
        cost = np.ones((3, 3)) * 5.0
        allocator = TaskAllocator()
        result = allocator.hungarian_algorithm(cost)

        assert len(result) == 3
        total_cost = sum(cost[r, c] for r, c in result)
        assert abs(total_cost - 15.0) < 1e-6

    def test_large_matrix(self):
        """10x10 매트릭스에서도 정상 동작."""
        np.random.seed(42)
        cost = np.random.rand(10, 10) * 100
        allocator = TaskAllocator()
        result = allocator.hungarian_algorithm(cost)

        # 최소한 절반 이상 할당되어야 함
        assert len(result) >= 5, f"할당 수: {len(result)}"
        robots = [r for r, _ in result]
        tasks = [t for _, t in result]
        # 로봇/작업 중복 없음
        assert len(set(robots)) == len(robots)
        assert len(set(tasks)) == len(tasks)


class TestTaskAllocation:
    """작업 할당 통합 테스트."""

    def test_allocate_basic(self):
        """기본 작업 할당이 유효한 결과를 반환한다."""
        allocator = TaskAllocator()
        tasks = [
            Task("T1", (10.0, 5.0), priority=5),
            Task("T2", (3.0, 8.0), priority=3),
        ]
        robots = [
            Robot("R1", (0.0, 0.0)),
            Robot("R2", (5.0, 5.0)),
        ]
        assignments = allocator.allocate(tasks, robots)

        assert len(assignments) > 0
        robot_ids = {a.robot_id for a in assignments}
        task_ids = {a.task_id for a in assignments}
        assert robot_ids.issubset({"R1", "R2"})
        assert task_ids.issubset({"T1", "T2"})

    def test_allocate_empty(self):
        """작업이나 로봇이 없으면 빈 리스트."""
        allocator = TaskAllocator()
        assert allocator.allocate([], [Robot("R1", (0, 0))]) == []
        assert allocator.allocate([Task("T1", (1, 1))], []) == []

    def test_unavailable_robot_skipped(self):
        """비가용 로봇은 할당에서 제외된다."""
        allocator = TaskAllocator()
        tasks = [Task("T1", (5.0, 5.0))]
        robots = [
            Robot("R1", (0.0, 0.0), available=False),
            Robot("R2", (1.0, 1.0), available=True),
        ]
        assignments = allocator.allocate(tasks, robots)

        if assignments:
            assert assignments[0].robot_id == "R2"


class TestConflictPrediction:
    """경로 충돌 예측 테스트."""

    def test_predict_conflict_at_intersection(self):
        """교차점에서 충돌을 탐지한다."""
        tm = TrafficManager(safety_distance=0.8)

        path1 = RobotPath(
            "R1",
            segments=[
                PathSegment((0, 5), 0, 2),
                PathSegment((5, 5), 4, 6),
            ],
            priority=2,
        )
        path2 = RobotPath(
            "R2",
            segments=[
                PathSegment((5, 0), 0, 2),
                PathSegment((5, 5), 4, 6),  # 같은 위치 + 같은 시간
            ],
            priority=1,
        )

        conflicts = tm.predict_conflicts([path1, path2])
        # 같은 위치(5,5) + 같은 시간(4~6) -> 충돌 탐지
        assert len(conflicts) > 0

    def test_no_conflict_different_time(self):
        """시간이 다르면 충돌 없음."""
        tm = TrafficManager(safety_distance=0.8)

        path1 = RobotPath(
            "R1",
            segments=[PathSegment((5, 5), 0, 2)],
        )
        path2 = RobotPath(
            "R2",
            segments=[PathSegment((5, 5), 5, 7)],  # 시간이 겹치지 않음
        )

        conflicts = tm.predict_conflicts([path1, path2])
        assert len(conflicts) == 0


class TestDeadlockDetection:
    """교착 탐지 테스트."""

    def test_cycle_detection(self):
        """Wait-For Graph에서 사이클을 탐지한다."""
        tm = TrafficManager()

        # A -> B -> C -> A 사이클 설정
        tm._wait_for_graph['R1'].add('R2')
        tm._wait_for_graph['R2'].add('R3')
        tm._wait_for_graph['R3'].add('R1')

        cycles = tm.detect_deadlock()
        assert len(cycles) > 0

    def test_no_cycle(self):
        """사이클이 없으면 교착 없음."""
        tm = TrafficManager()

        tm._wait_for_graph['R1'].add('R2')
        tm._wait_for_graph['R3'].add('R2')
        # R2는 아무도 기다리지 않음 -> 사이클 없음

        cycles = tm.detect_deadlock()
        assert len(cycles) == 0

    def test_deadlock_resolution(self):
        """교착 해소 시 양보 로봇이 결정된다."""
        tm = TrafficManager()

        # 경로 등록 (우선순위 포함)
        tm.register_path(RobotPath("R1", [], priority=2))
        tm.register_path(RobotPath("R2", [], priority=1))
        tm.register_path(RobotPath("R3", [], priority=3))

        tm._wait_for_graph['R1'].add('R2')
        tm._wait_for_graph['R2'].add('R3')
        tm._wait_for_graph['R3'].add('R1')

        cycles = tm.detect_deadlock()
        yielded = tm.resolve_deadlock(cycles)

        assert len(yielded) > 0
        # 가장 낮은 우선순위(R2, priority=1)가 양보
        assert 'R2' in yielded


class TestZoneManagement:
    """교차로/병목 구간 관리 테스트."""

    def test_zone_entry_and_release(self):
        """구간 진입 및 해제."""
        tm = TrafficManager()
        tm.register_zone(Zone("Z1", (5.0, 5.0), capacity=1))

        # 첫 번째 로봇 진입 허용
        assert tm.request_zone_entry("Z1", "R1") is True
        # 두 번째 로봇 대기
        assert tm.request_zone_entry("Z1", "R2") is False

        # 첫 번째 로봇 이탈
        tm.release_zone("Z1", "R1")

        # 두 번째 로봇 자동 진입
        zone = tm._zones["Z1"]
        assert "R2" in zone.current_occupants

    def test_zone_capacity(self):
        """구간 용량 초과 시 대기."""
        tm = TrafficManager()
        tm.register_zone(Zone("Z1", (5.0, 5.0), capacity=2))

        assert tm.request_zone_entry("Z1", "R1") is True
        assert tm.request_zone_entry("Z1", "R2") is True
        assert tm.request_zone_entry("Z1", "R3") is False  # 용량 초과
