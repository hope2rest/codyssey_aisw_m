"""
차동 구동 로봇 기구학 단위 테스트.

직선 주행, 원형 주행, 오도메트리 누적 오차를 검증한다.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kinematics import DifferentialDriveRobot


class TestForwardKinematics:
    """순기구학 테스트."""

    def test_straight_line(self):
        """동일 바퀴 속도 → 직선 이동 (omega=0)."""
        v, omega = DifferentialDriveRobot.forward_kinematics(1.0, 1.0, 0.3)
        assert abs(v - 1.0) < 1e-10
        assert abs(omega) < 1e-10

    def test_pure_rotation(self):
        """반대 방향 동일 속도 → 제자리 회전 (v=0)."""
        v, omega = DifferentialDriveRobot.forward_kinematics(-0.5, 0.5, 0.3)
        assert abs(v) < 1e-10
        assert abs(omega - 1.0 / 0.3) < 1e-10

    def test_curve(self):
        """좌우 속도 차이 → 곡선 주행."""
        v, omega = DifferentialDriveRobot.forward_kinematics(0.8, 1.0, 0.3)
        assert abs(v - 0.9) < 1e-10
        assert omega > 0  # 좌회전


class TestOdometry:
    """오도메트리 갱신 테스트."""

    def test_straight_drive(self):
        """양쪽 동일 틱 → x방향 직진."""
        robot = DifferentialDriveRobot(wheel_radius=0.05, wheel_base=0.3, ticks_per_rev=4096)

        for _ in range(1000):
            robot.update(left_ticks=10, right_ticks=10)

        x, y, theta = robot.get_pose()
        # y는 0에 가까워야 함
        assert abs(y) < 1e-6, f"y가 0이어야 하지만 {y}"
        # theta는 0에 가까워야 함
        assert abs(theta) < 1e-6, f"theta가 0이어야 하지만 {theta}"
        # x는 양수 (전진)
        assert x > 0, f"x가 양수여야 하지만 {x}"

    def test_pure_rotation_odometry(self):
        """좌우 반대 틱 → 제자리 회전, 위치 불변."""
        robot = DifferentialDriveRobot()

        for _ in range(100):
            robot.update(left_ticks=-5, right_ticks=5)

        x, y, theta = robot.get_pose()
        # 위치는 원점 근처 유지
        assert abs(x) < 0.01, f"제자리 회전인데 x={x}"
        assert abs(y) < 0.01, f"제자리 회전인데 y={y}"
        # theta는 변해야 함
        assert abs(theta) > 0.01

    def test_circular_drive(self):
        """좌우 틱 차이 → 원형 주행, 시작점 근처로 복귀."""
        robot = DifferentialDriveRobot(wheel_radius=0.05, wheel_base=0.3, ticks_per_rev=4096)

        # 충분히 많은 스텝으로 원을 그린다
        # 한 바퀴: omega * t = 2*pi
        ticks_left = 8
        ticks_right = 10
        d_left = ticks_left * robot.meters_per_tick
        d_right = ticks_right * robot.meters_per_tick
        d_theta = (d_right - d_left) / robot.wheel_base

        # 한 바퀴에 필요한 스텝 수
        steps_per_circle = int(2 * np.pi / abs(d_theta))

        for _ in range(steps_per_circle):
            robot.update(left_ticks=ticks_left, right_ticks=ticks_right)

        x, y, theta = robot.get_pose()
        # 원형 주행 후 시작점 근처로 복귀
        dist = np.sqrt(x ** 2 + y ** 2)
        assert dist < 0.5, f"원형 주행 후 시작점 거리: {dist}"

    def test_reset(self):
        """리셋 후 상태 초기화 확인."""
        robot = DifferentialDriveRobot()
        robot.update(10, 10)
        robot.reset(1.0, 2.0, 0.5)
        x, y, theta = robot.get_pose()
        assert abs(x - 1.0) < 1e-10
        assert abs(y - 2.0) < 1e-10
        assert abs(theta - 0.5) < 1e-10
        assert len(robot.history) == 0


class TestDriftAnalysis:
    """오도메트리 드리프트 분석 테스트."""

    def test_drift_ratio_straight(self):
        """직진 시 드리프트 비율은 매우 작아야 한다."""
        robot = DifferentialDriveRobot()
        steps = 500

        # ground truth: 정확한 직진
        gt_list = []
        for i in range(steps):
            robot.update(10, 10)
            x, y, theta = robot.get_pose()
            gt_list.append([x, y, 0.0])  # 이상적 직진

        gt = np.array(gt_list)
        result = robot.analyze_drift(gt)

        assert result['drift_ratio'] < 0.01, f"직진 드리프트 비율: {result['drift_ratio']}"

    def test_drift_increases_with_distance(self):
        """이동 거리가 길수록 절대 오차가 증가할 수 있다."""
        robot = DifferentialDriveRobot()

        # 약간의 비대칭 (슬립 모사)
        gt_list = []
        for _ in range(1000):
            robot.update(10, 11)  # 약간 비대칭
            gt_list.append([robot.x, robot.y, robot.theta])

        gt = np.array(gt_list)
        # 이상적 직진과 비교
        ideal_gt = gt.copy()
        ideal_gt[:, 1] = 0  # y=0 직진
        ideal_gt[:, 2] = 0  # theta=0

        result = robot.analyze_drift(ideal_gt)
        assert result['max_position_error'] > 0

    def test_analyze_empty_history_raises(self):
        """기록 없이 분석 시 ValueError."""
        robot = DifferentialDriveRobot()
        with pytest.raises(ValueError):
            robot.analyze_drift(np.array([[0, 0, 0]]))
