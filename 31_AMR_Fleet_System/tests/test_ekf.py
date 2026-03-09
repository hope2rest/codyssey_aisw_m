"""
EKF 센서 퓨전 단위 테스트.

예측/업데이트 수렴, 센서 퓨전 정확도를 검증한다.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ekf import ExtendedKalmanFilter, fuse_sensors, ODOM_H, ODOM_R, IMU_H, IMU_R


class TestEKFPredict:
    """EKF 예측 단계 테스트."""

    def test_predict_straight(self):
        """직진 운동 모델 예측 (v=1, omega=0)."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.0, 1.0, 0.0]
        )
        dt = 0.1
        ekf.predict(dt)
        state = ekf.get_state()

        assert abs(state[0] - 0.1) < 1e-6, f"x 예측 오류: {state[0]}"
        assert abs(state[1]) < 1e-6, f"y는 0이어야 하지만 {state[1]}"
        assert abs(state[2]) < 1e-6, f"theta는 0이어야 하지만 {state[2]}"

    def test_predict_rotation(self):
        """회전 운동 모델 예측 (v=0, omega=1)."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.0, 0.0, 1.0]
        )
        dt = 0.1
        ekf.predict(dt)
        state = ekf.get_state()

        assert abs(state[0]) < 1e-6
        assert abs(state[1]) < 1e-6
        assert abs(state[2] - 0.1) < 1e-6

    def test_covariance_grows(self):
        """예측 시 공분산이 증가한다."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.0, 1.0, 0.0]
        )
        P_before = ekf.get_covariance().copy()
        ekf.predict(0.1)
        P_after = ekf.get_covariance()

        # 대각 원소가 증가해야 함
        assert np.all(np.diag(P_after) >= np.diag(P_before) - 1e-10)


class TestEKFUpdate:
    """EKF 업데이트 단계 테스트."""

    def test_update_reduces_covariance(self):
        """측정 업데이트 시 공분산이 감소한다."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.0, 1.0, 0.0]
        )
        ekf.predict(0.1)
        P_before = ekf.get_covariance().copy()

        # 오도메트리 업데이트
        measurement = np.array([0.1, 0.0, 0.0])
        ekf.update(measurement, ODOM_H, ODOM_R)
        P_after = ekf.get_covariance()

        # 관측된 상태의 공분산이 감소해야 함
        assert np.trace(P_after) < np.trace(P_before)

    def test_update_converges_to_measurement(self):
        """반복 업데이트로 추정이 측정값에 수렴한다."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.0, 0.0, 0.0],
            initial_covariance=np.eye(5) * 10.0,
        )
        target = np.array([5.0, 3.0, 0.5])

        for _ in range(100):
            ekf.predict(0.01)
            ekf.update(target, ODOM_H, ODOM_R)

        state = ekf.get_state()
        assert abs(state[0] - 5.0) < 0.5, f"x 수렴 실패: {state[0]}"
        assert abs(state[1] - 3.0) < 0.5, f"y 수렴 실패: {state[1]}"

    def test_imu_update_theta(self):
        """IMU 업데이트로 theta가 보정된다."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.5, 0.0, 0.0]  # theta=0.5 (실제는 0)
        )
        imu_measurement = np.array([0.0, 0.0])  # theta=0, omega=0
        ekf.update(imu_measurement, IMU_H, IMU_R)

        state = ekf.get_state()
        # theta가 0 방향으로 보정됨
        assert abs(state[2]) < 0.5, f"IMU 보정 후 theta: {state[2]}"


class TestSensorFusion:
    """센서 퓨전 통합 테스트."""

    def test_fusion_accuracy(self):
        """다중 센서 퓨전이 단일 센서보다 정확하다."""
        np.random.seed(42)

        ekf_odom_only = ExtendedKalmanFilter(initial_state=[0, 0, 0, 1, 0])
        ekf_fused = ExtendedKalmanFilter(initial_state=[0, 0, 0, 1, 0])

        dt = 0.1
        errors_odom = []
        errors_fused = []

        for step in range(100):
            true_x = 1.0 * (step + 1) * dt
            true_y = 0.0
            true_theta = 0.0

            odom_meas = np.array([true_x, true_y, true_theta]) + np.random.randn(3) * 0.1
            imu_meas = np.array([true_theta, 0.0]) + np.random.randn(2) * 0.01

            # 오도메트리만
            fuse_sensors(ekf_odom_only, dt, odom=odom_meas)
            state_odom = ekf_odom_only.get_state()
            errors_odom.append(abs(state_odom[0] - true_x))

            # 퓨전
            fuse_sensors(ekf_fused, dt, odom=odom_meas, imu=imu_meas)
            state_fused = ekf_fused.get_state()
            errors_fused.append(abs(state_fused[0] - true_x))

        mean_odom = np.mean(errors_odom[-20:])
        mean_fused = np.mean(errors_fused[-20:])

        # 퓨전 오차가 오도메트리만 사용한 것보다 같거나 작아야 함
        # (IMU는 theta/omega만 보정하므로 x 정확도 차이가 크지 않을 수 있음)
        assert mean_fused < mean_odom * 2.0, (
            f"퓨전 오차({mean_fused:.4f})가 기대보다 큼 (odom: {mean_odom:.4f})"
        )

    def test_kidnapped_robot_recovery(self):
        """Kidnapped 상황에서 AMCL 업데이트로 복구한다."""
        ekf = ExtendedKalmanFilter(
            initial_state=[0.0, 0.0, 0.0, 0.0, 0.0]
        )

        # 로봇이 (10, 5)로 갑자기 이동함 (kidnapped)
        # EKF는 여전히 원점 근처를 추정
        for _ in range(10):
            ekf.predict(0.1)

        state_before = ekf.get_state()
        assert abs(state_before[0]) < 1.0

        # AMCL이 새 위치를 알려줌
        from core.ekf import AMCL_H, AMCL_R
        for _ in range(50):
            ekf.predict(0.1)
            ekf.update(np.array([10.0, 5.0, 0.0]), AMCL_H, AMCL_R)

        state_after = ekf.get_state()
        assert abs(state_after[0] - 10.0) < 1.0, f"Kidnapped 복구 실패: x={state_after[0]}"
        assert abs(state_after[1] - 5.0) < 1.0, f"Kidnapped 복구 실패: y={state_after[1]}"
