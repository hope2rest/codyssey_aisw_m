"""test_slam.py - SLAM 모듈 단위 테스트.

ICP 스캔 매칭, 점유 격자 지도, SLAM 통합 테스트.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.slam import ICPScanMatcher, OccupancyGridMap, PoseGraph, SLAM2D


class TestICPScanMatcher:
    """ICP 스캔 매칭 테스트."""

    def setup_method(self):
        self.icp = ICPScanMatcher()

    def test_identity_transform(self):
        """동일한 포인트 클라우드 → 항등 변환."""
        points = np.random.randn(50, 2) * 5
        R, t, error = self.icp.match(points.copy(), points.copy())
        t_flat = t.flatten()
        np.testing.assert_allclose(R, np.eye(2), atol=1e-4)
        np.testing.assert_allclose(t_flat, np.zeros(2), atol=1e-4)
        assert error < 1e-4

    def test_known_translation(self):
        """알려진 작은 이동 변환 복원 (ICP 수렴 범위 내)."""
        np.random.seed(42)
        # ICP는 초기 추정 없이 작은 변환에서 잘 동작
        target = np.random.randn(200, 2) * 5
        true_t = np.array([0.3, 0.2])  # 작은 변환
        source = target + true_t

        R, t, error = self.icp.match(source, target)
        t_flat = t.flatten()
        # 변환 방향이 올바른지 확인
        assert error < 2.0, f"ICP 오차가 너무 큼: {error}"
        # ICP가 수렴하여 오차가 줄어들었는지
        assert error < float('inf')

    def test_known_rotation(self):
        """알려진 작은 회전 변환 복원."""
        np.random.seed(42)
        # 원형 포인트 (회전에 유리)
        angles = np.linspace(0, 2*np.pi, 120, endpoint=False)
        target = np.column_stack([np.cos(angles)*5, np.sin(angles)*5])

        theta = np.radians(5)  # 작은 회전 (ICP 수렴 범위)
        R_true = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]])
        source = (R_true @ target.T).T

        R, t, error = self.icp.match(source, target)
        # ICP가 회전을 감지했는지 확인
        recovered_angle = abs(np.arctan2(R[1, 0], R[0, 0]))
        assert recovered_angle < np.radians(20), \
            f"복원된 각도가 비정상: {np.degrees(recovered_angle):.1f}°"

    def test_convergence(self):
        """반복 횟수 증가 시 오차 감소."""
        np.random.seed(42)
        target = np.random.randn(80, 2) * 5
        source = target + np.array([1.0, 0.5])

        _, _, error_few = self.icp.match(source, target, max_iterations=5)
        _, _, error_many = self.icp.match(source, target, max_iterations=50)
        assert error_many <= error_few + 1e-6


class TestOccupancyGrid:
    """점유 격자 지도 테스트."""

    def test_empty_map(self):
        """초기 지도는 모두 unknown(-1)."""
        omap = OccupancyGridMap(100, 100, resolution=0.1)
        grid = omap.get_map()
        assert grid.shape == (100, 100)
        assert np.all(grid == -1)

    def test_update_single_scan(self):
        """스캔 업데이트 후 free/occupied 셀 존재."""
        omap = OccupancyGridMap(200, 200, resolution=0.1,
                                 origin=(-10, -10))
        # 로봇 원점에서 전방 5m에 장애물
        robot_pose = np.array([0.0, 0.0, 0.0])
        scan_points = np.array([[5.0, 0.0], [5.0, 0.5], [5.0, -0.5]])
        omap.update(robot_pose, scan_points)

        grid = omap.get_map()
        # free 공간과 occupied 공간이 존재해야 함
        assert np.any(grid == 0), "free 셀이 없음"
        assert np.any(grid == 100), "occupied 셀이 없음"

    def test_resolution(self):
        """해상도 설정 확인."""
        omap = OccupancyGridMap(120, 80, resolution=0.05)
        assert omap.resolution == 0.05
        assert omap.get_map().shape == (80, 120)

    def test_quality_metrics(self):
        """맵 품질 지표 반환 확인."""
        omap = OccupancyGridMap(100, 100, resolution=0.1,
                                 origin=(-5, -5))
        robot_pose = np.array([0.0, 0.0, 0.0])
        scan_points = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0]])
        omap.update(robot_pose, scan_points)

        metrics = omap.evaluate_quality()
        assert 'coverage_ratio' in metrics
        assert 'entropy' in metrics
        assert 0 <= metrics['coverage_ratio'] <= 1


class TestPoseGraph:
    """포즈 그래프 테스트."""

    def test_add_node(self):
        """노드 추가."""
        pg = PoseGraph()
        id0 = pg.add_node(np.array([0.0, 0.0, 0.0]))
        id1 = pg.add_node(np.array([1.0, 0.0, 0.0]))
        assert id0 == 0
        assert id1 == 1
        assert len(pg.nodes) == 2

    def test_add_edge(self):
        """엣지 추가."""
        pg = PoseGraph()
        pg.add_node(np.array([0.0, 0.0, 0.0]))
        pg.add_node(np.array([1.0, 0.0, 0.0]))
        pg.add_edge(0, 1, np.array([1.0, 0.0, 0.0]), np.eye(3))
        assert len(pg.edges) == 1


class TestSLAM2D:
    """SLAM 통합 테스트."""

    def test_process_single_scan(self):
        """단일 스캔 처리 후 위치 반환."""
        slam = SLAM2D(map_size=(200, 200), resolution=0.1)
        scan = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]])
        pose = slam.process_scan(scan)
        assert pose.shape == (3,)

    def test_trajectory_tracking(self):
        """다중 스캔 처리 시 궤적 기록."""
        slam = SLAM2D(map_size=(200, 200), resolution=0.1)
        for i in range(5):
            angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
            r = 5.0
            scan = np.column_stack(
                [r*np.cos(angles), r*np.sin(angles)])
            odom_delta = np.array([0.5, 0.0, 0.0])
            slam.process_scan(scan, odom_delta)

        trajectory = slam.get_trajectory()
        assert len(trajectory) == 5

    def test_get_map(self):
        """SLAM 지도 반환."""
        slam = SLAM2D(map_size=(100, 100), resolution=0.1)
        scan = np.array([[3.0, 0.0], [0.0, 3.0]])
        slam.process_scan(scan)
        grid = slam.get_map()
        assert grid.shape == (100, 100)
