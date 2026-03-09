"""
확장 칼만 필터(Extended Kalman Filter) 기반 센서 퓨전 모듈.

상태 벡터 [x, y, theta, v, omega]에 대한 EKF를 직접 구현하며,
Wheel Odometry, IMU, AMCL 센서 데이터를 융합하여 로봇의 자세를 추정한다.
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    확장 칼만 필터 클래스.

    5차원 상태 벡터 [x, y, theta, v, omega]를 추정한다.

    매개변수
    --------
    initial_state : np.ndarray, shape (5,)
        초기 상태 벡터 [x, y, theta, v, omega].
    initial_covariance : np.ndarray, shape (5, 5), optional
        초기 공분산 행렬. None이면 단위 행렬 사용.
    process_noise : np.ndarray, shape (5, 5), optional
        프로세스 잡음 공분산 Q. None이면 기본값 사용.
    """

    def __init__(
        self,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        process_noise: np.ndarray | None = None,
    ):
        # 상태 벡터: [x, y, theta, v, omega]
        self.x = (
            np.array(initial_state, dtype=np.float64)
            if initial_state is not None
            else np.zeros(5)
        )

        # 공분산 행렬 P
        self.P = (
            np.array(initial_covariance, dtype=np.float64)
            if initial_covariance is not None
            else np.eye(5) * 0.1
        )

        # 프로세스 잡음 Q
        if process_noise is not None:
            self.Q = np.array(process_noise, dtype=np.float64)
        else:
            self.Q = np.diag([0.01, 0.01, 0.005, 0.05, 0.05])

    # ------------------------------------------------------------------
    # 예측 단계
    # ------------------------------------------------------------------
    def predict(self, dt: float):
        """
        EKF 예측 단계: 운동 모델로 상태를 전파한다.

        비선형 운동 모델 (등속 원호 모델):
            x'     = x + v * cos(theta) * dt
            y'     = y + v * sin(theta) * dt
            theta' = theta + omega * dt
            v'     = v
            omega' = omega

        매개변수
        --------
        dt : float
            시간 간격 (초).

        반환
        ----
        np.ndarray
            예측된 상태 벡터.
        """
        x, y, theta, v, omega = self.x

        # 상태 전이 (비선형)
        x_pred = x + v * np.cos(theta) * dt
        y_pred = y + v * np.sin(theta) * dt
        theta_pred = theta + omega * dt
        theta_pred = np.arctan2(np.sin(theta_pred), np.cos(theta_pred))
        v_pred = v
        omega_pred = omega

        self.x = np.array([x_pred, y_pred, theta_pred, v_pred, omega_pred])

        # 야코비안 F = d(f)/d(state)
        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt

        # 공분산 전파
        self.P = F @ self.P @ F.T + self.Q * dt

        return self.x.copy()

    # ------------------------------------------------------------------
    # 갱신 단계
    # ------------------------------------------------------------------
    def update(self, measurement: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        EKF 갱신 단계: 관측값으로 상태를 보정한다.

        매개변수
        --------
        measurement : np.ndarray
            관측 벡터 z.
        H : np.ndarray
            관측 모델 야코비안 행렬 (m x 5).
        R : np.ndarray
            관측 잡음 공분산 행렬 (m x m).

        반환
        ----
        np.ndarray
            갱신된 상태 벡터.
        """
        z = np.asarray(measurement, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        # 혁신 (innovation)
        y = z - H @ self.x

        # theta 관련 혁신의 각도 정규화
        # H 행렬에서 theta(인덱스 2)에 매핑되는 관측 차원을 찾아 정규화
        for i in range(H.shape[0]):
            if H[i, 2] != 0.0 and np.sum(np.abs(H[i])) == np.abs(H[i, 2]):
                y[i] = np.arctan2(np.sin(y[i]), np.cos(y[i]))

        # 칼만 이득
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 상태 갱신
        self.x = self.x + K @ y
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        # 공분산 갱신 — Joseph form (수치 안정성)
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x.copy()

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------
    def get_state(self):
        """현재 상태 벡터 [x, y, theta, v, omega]를 반환한다."""
        return self.x.copy()

    def get_covariance(self):
        """현재 공분산 행렬을 반환한다."""
        return self.P.copy()


# ======================================================================
# 센서별 공분산 및 관측 모델 설정
# ======================================================================

# Wheel Odometry: [x, y, theta] 관측
ODOM_H = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=np.float64)

ODOM_R = np.diag([0.05, 0.05, 0.02])  # 오도메트리 잡음 공분산

# IMU: [theta, omega] 관측
IMU_H = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
], dtype=np.float64)

IMU_R = np.diag([0.01, 0.005])  # IMU 잡음 공분산

# AMCL: [x, y, theta] 관측 (전역 위치 추정)
AMCL_H = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=np.float64)

AMCL_R = np.diag([0.02, 0.02, 0.01])  # AMCL 잡음 공분산 (오도메트리보다 정밀)


def fuse_sensors(
    ekf: ExtendedKalmanFilter,
    dt: float,
    odom: np.ndarray | None = None,
    imu: np.ndarray | None = None,
    amcl: np.ndarray | None = None,
) -> np.ndarray:
    """
    센서 퓨전 함수: 다중 센서 데이터를 EKF로 융합하여 자세를 추정한다.

    예측 단계 후 가용 센서 데이터로 순차 갱신을 수행한다.

    매개변수
    --------
    ekf : ExtendedKalmanFilter
        EKF 인스턴스.
    dt : float
        시간 간격 (초).
    odom : np.ndarray, shape (3,), optional
        Wheel Odometry 관측 [x, y, theta].
    imu : np.ndarray, shape (2,), optional
        IMU 관측 [theta, omega].
    amcl : np.ndarray, shape (3,), optional
        AMCL 관측 [x, y, theta].

    반환
    ----
    np.ndarray
        융합된 추정 자세 [x, y, theta, v, omega].
    """
    # 예측 단계
    ekf.predict(dt)

    # 가용 센서로 순차 갱신
    if odom is not None:
        ekf.update(np.asarray(odom), ODOM_H, ODOM_R)

    if imu is not None:
        ekf.update(np.asarray(imu), IMU_H, IMU_R)

    if amcl is not None:
        ekf.update(np.asarray(amcl), AMCL_H, AMCL_R)

    return ekf.get_state()


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== EKF 센서 퓨전 테스트 ===")

    ekf = ExtendedKalmanFilter(
        initial_state=[0.0, 0.0, 0.0, 1.0, 0.0]
    )

    dt = 0.1
    np.random.seed(42)

    for step in range(50):
        # 실제 로봇이 직진한다고 가정
        true_x = 1.0 * (step + 1) * dt
        true_y = 0.0
        true_theta = 0.0

        # 노이즈가 포함된 센서 관측 생성
        odom_meas = np.array([true_x, true_y, true_theta]) + np.random.randn(3) * 0.05
        imu_meas = np.array([true_theta, 0.0]) + np.random.randn(2) * 0.01

        state = fuse_sensors(ekf, dt, odom=odom_meas, imu=imu_meas)

        if (step + 1) % 10 == 0:
            print(
                f"  스텝 {step+1:3d}: "
                f"추정=({state[0]:.3f}, {state[1]:.3f}, θ={state[2]:.3f}), "
                f"실제=({true_x:.3f}, {true_y:.3f})"
            )

    print("최종 공분산 대각 성분:", np.diag(ekf.get_covariance()).round(6))
