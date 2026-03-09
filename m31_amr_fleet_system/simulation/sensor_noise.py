"""
센서 노이즈 모델.

LiDAR, IMU, Wheel Encoder, Depth Camera 각각의 센서 특성에 맞는
노이즈 모델을 제공한다. 시뮬레이션 환경에서 센서 데이터를 현실적으로
모사하기 위해 사용한다.
"""

import numpy as np


class LidarNoise:
    """
    LiDAR 거리 측정 노이즈 모델.

    가우시안 노이즈(sigma=0.03m)를 거리 측정값에 추가한다.
    최대 측정 거리를 초과하면 inf로 처리한다.

    매개변수
    --------
    sigma : float
        거리 측정 표준편차 (m). 기본값 0.03.
    max_range : float
        최대 측정 거리 (m). 기본값 12.0.
    min_range : float
        최소 측정 거리 (m). 기본값 0.1.
    miss_rate : float
        측정 누락 확률 (0~1). 기본값 0.01.
    """

    def __init__(
        self,
        sigma: float = 0.03,
        max_range: float = 12.0,
        min_range: float = 0.1,
        miss_rate: float = 0.01,
    ):
        self.sigma = sigma
        self.max_range = max_range
        self.min_range = min_range
        self.miss_rate = miss_rate

    def apply(self, ranges: np.ndarray) -> np.ndarray:
        """
        LiDAR 거리 배열에 노이즈를 적용한다.

        매개변수
        --------
        ranges : np.ndarray
            이상적인 거리 측정값 배열 (m).

        반환
        ----
        np.ndarray
            노이즈가 적용된 거리 배열.
        """
        noisy = ranges.copy().astype(np.float64)

        # 가우시안 노이즈 추가
        noise = np.random.normal(0, self.sigma, size=noisy.shape)
        noisy += noise

        # 측정 누락 (inf 처리)
        miss_mask = np.random.random(size=noisy.shape) < self.miss_rate
        noisy[miss_mask] = np.inf

        # 범위 제한
        noisy[noisy < self.min_range] = self.min_range
        noisy[noisy > self.max_range] = np.inf

        return noisy


class IMUNoise:
    """
    IMU 센서 노이즈 모델.

    가속도계와 자이로스코프에 바이어스 + 가우시안 노이즈를 추가한다.

    매개변수
    --------
    accel_sigma : float
        가속도 노이즈 표준편차 (m/s^2). 기본값 0.02.
    accel_bias : float
        가속도 바이어스 (m/s^2). 기본값 0.01.
    gyro_sigma : float
        각속도 노이즈 표준편차 (rad/s). 기본값 0.005.
    gyro_bias : float
        각속도 바이어스 (rad/s). 기본값 0.001.
    bias_drift_rate : float
        바이어스 드리프트 비율 (1/s). 기본값 0.0001.
    """

    def __init__(
        self,
        accel_sigma: float = 0.02,
        accel_bias: float = 0.01,
        gyro_sigma: float = 0.005,
        gyro_bias: float = 0.001,
        bias_drift_rate: float = 0.0001,
    ):
        self.accel_sigma = accel_sigma
        self.accel_bias = accel_bias
        self.gyro_sigma = gyro_sigma
        self.gyro_bias = gyro_bias
        self.bias_drift_rate = bias_drift_rate

        # 현재 바이어스 상태 (드리프트 적용)
        self._current_accel_bias = np.array([accel_bias, accel_bias, accel_bias])
        self._current_gyro_bias = np.array([gyro_bias, gyro_bias, gyro_bias])

    def apply_accel(self, accel: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        가속도 측정값에 노이즈를 적용한다.

        매개변수
        --------
        accel : np.ndarray, shape (3,)
            이상적인 가속도 [ax, ay, az] (m/s^2).
        dt : float
            시간 간격 (초).

        반환
        ----
        np.ndarray, shape (3,)
            노이즈가 적용된 가속도.
        """
        # 바이어스 드리프트
        self._current_accel_bias += (
            np.random.normal(0, self.bias_drift_rate, size=3) * dt
        )

        noise = np.random.normal(0, self.accel_sigma, size=3)
        return accel + self._current_accel_bias + noise

    def apply_gyro(self, gyro: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        각속도 측정값에 노이즈를 적용한다.

        매개변수
        --------
        gyro : np.ndarray, shape (3,)
            이상적인 각속도 [wx, wy, wz] (rad/s).
        dt : float
            시간 간격 (초).

        반환
        ----
        np.ndarray, shape (3,)
            노이즈가 적용된 각속도.
        """
        self._current_gyro_bias += (
            np.random.normal(0, self.bias_drift_rate, size=3) * dt
        )

        noise = np.random.normal(0, self.gyro_sigma, size=3)
        return gyro + self._current_gyro_bias + noise

    def reset_bias(self):
        """바이어스를 초기값으로 리셋한다."""
        self._current_accel_bias = np.array(
            [self.accel_bias, self.accel_bias, self.accel_bias]
        )
        self._current_gyro_bias = np.array(
            [self.gyro_bias, self.gyro_bias, self.gyro_bias]
        )


class EncoderNoise:
    """
    Wheel Encoder 슬립 노이즈 모델.

    바퀴 미끄러짐(슬립)에 의한 인코더 측정 오차를 모사한다.

    매개변수
    --------
    slip_ratio : float
        슬립 비율 (0~1). 기본값 0.02 (2%).
    sigma : float
        틱 변화량에 대한 가우시안 노이즈 표준편차 (ticks). 기본값 1.0.
    """

    def __init__(self, slip_ratio: float = 0.02, sigma: float = 1.0):
        self.slip_ratio = slip_ratio
        self.sigma = sigma

    def apply(self, left_ticks: int, right_ticks: int) -> tuple:
        """
        인코더 틱에 슬립 노이즈를 적용한다.

        매개변수
        --------
        left_ticks : int
            이상적인 왼쪽 바퀴 틱 변화량.
        right_ticks : int
            이상적인 오른쪽 바퀴 틱 변화량.

        반환
        ----
        tuple[int, int]
            노이즈가 적용된 (left_ticks, right_ticks).
        """
        # 슬립: 랜덤하게 틱이 줄어듦
        left_slip = 1.0 - np.random.uniform(0, self.slip_ratio)
        right_slip = 1.0 - np.random.uniform(0, self.slip_ratio)

        # 가우시안 노이즈
        left_noise = np.random.normal(0, self.sigma)
        right_noise = np.random.normal(0, self.sigma)

        noisy_left = int(round(left_ticks * left_slip + left_noise))
        noisy_right = int(round(right_ticks * right_slip + right_noise))

        return noisy_left, noisy_right


class DepthNoise:
    """
    Depth Camera 깊이 측정 노이즈 모델.

    거리에 비례하는 노이즈와 양자화 노이즈를 적용한다.

    매개변수
    --------
    base_sigma : float
        기본 깊이 노이즈 표준편차 (m). 기본값 0.01.
    distance_factor : float
        거리 비례 노이즈 계수. 기본값 0.005 (거리 1m당 5mm 추가).
    max_depth : float
        최대 측정 깊이 (m). 기본값 5.0.
    min_depth : float
        최소 측정 깊이 (m). 기본값 0.3.
    quantization : float
        깊이 양자화 단위 (m). 기본값 0.001.
    """

    def __init__(
        self,
        base_sigma: float = 0.01,
        distance_factor: float = 0.005,
        max_depth: float = 5.0,
        min_depth: float = 0.3,
        quantization: float = 0.001,
    ):
        self.base_sigma = base_sigma
        self.distance_factor = distance_factor
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.quantization = quantization

    def apply(self, depth_image: np.ndarray) -> np.ndarray:
        """
        깊이 이미지에 노이즈를 적용한다.

        매개변수
        --------
        depth_image : np.ndarray
            이상적인 깊이 이미지 (m 단위).

        반환
        ----
        np.ndarray
            노이즈가 적용된 깊이 이미지.
        """
        noisy = depth_image.copy().astype(np.float64)

        # 거리 비례 노이즈
        sigma_map = self.base_sigma + self.distance_factor * noisy
        noise = np.random.normal(0, 1, size=noisy.shape) * sigma_map
        noisy += noise

        # 양자화
        noisy = np.round(noisy / self.quantization) * self.quantization

        # 범위 제한
        noisy[noisy < self.min_depth] = 0.0  # 너무 가까우면 0 (측정 불가)
        noisy[noisy > self.max_depth] = 0.0  # 너무 멀면 0 (측정 불가)

        return noisy


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== 센서 노이즈 모델 테스트 ===\n")

    # LiDAR
    lidar = LidarNoise()
    ideal_ranges = np.array([1.0, 3.0, 5.0, 10.0, 15.0])
    noisy_ranges = lidar.apply(ideal_ranges)
    print("LiDAR 노이즈:")
    for i, r in enumerate(ideal_ranges):
        print(f"  이상적: {r:.2f}m -> 노이즈: {noisy_ranges[i]:.4f}m")

    # IMU
    imu = IMUNoise()
    ideal_accel = np.array([0.0, 0.0, 9.81])
    ideal_gyro = np.array([0.0, 0.0, 0.1])
    noisy_accel = imu.apply_accel(ideal_accel)
    noisy_gyro = imu.apply_gyro(ideal_gyro)
    print(f"\nIMU 노이즈:")
    print(f"  가속도: {ideal_accel} -> {noisy_accel}")
    print(f"  각속도: {ideal_gyro} -> {noisy_gyro}")

    # Encoder
    encoder = EncoderNoise()
    noisy_l, noisy_r = encoder.apply(100, 100)
    print(f"\nEncoder 노이즈:")
    print(f"  이상적: (100, 100) -> 노이즈: ({noisy_l}, {noisy_r})")

    # Depth Camera
    depth = DepthNoise()
    ideal_depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    noisy_depth = depth.apply(ideal_depth)
    print(f"\nDepth Camera 노이즈:")
    print(f"  이상적:\n{ideal_depth}")
    print(f"  노이즈:\n{noisy_depth}")
