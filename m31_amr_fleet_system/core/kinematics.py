"""
차동 구동 로봇 기구학 및 오도메트리 모듈.

차동 구동(Differential Drive) 로봇의 순기구학(Forward Kinematics)을 직접 구현하며,
Wheel Encoder 틱 값으로부터 누적 위치(x, y)와 방향각(theta)을 계산한다.
오도메트리 드리프트 분석 기능도 포함한다.
"""

import numpy as np


class DifferentialDriveRobot:
    """
    차동 구동 로봇 순기구학 및 오도메트리 클래스.

    바퀴 인코더 틱 값을 이용하여 로봇의 위치와 자세를 누적 추정한다.

    매개변수
    --------
    wheel_radius : float
        바퀴 반지름 (m). 기본값 0.05m.
    wheel_base : float
        좌우 바퀴 간 거리 (m). 기본값 0.3m.
    ticks_per_rev : int
        바퀴 1회전당 인코더 틱 수. 기본값 4096.
    x0 : float
        초기 x 좌표 (m).
    y0 : float
        초기 y 좌표 (m).
    theta0 : float
        초기 방향각 (rad).
    """

    def __init__(
        self,
        wheel_radius: float = 0.05,
        wheel_base: float = 0.3,
        ticks_per_rev: int = 4096,
        x0: float = 0.0,
        y0: float = 0.0,
        theta0: float = 0.0,
    ):
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.ticks_per_rev = ticks_per_rev

        # 틱 하나당 이동 거리 (m/tick)
        self.meters_per_tick = (2.0 * np.pi * wheel_radius) / ticks_per_rev

        # 로봇 상태: [x, y, theta]
        self.x = x0
        self.y = y0
        self.theta = theta0

        # 오도메트리 누적 기록 (드리프트 분석용)
        self.history: list[dict] = []
        self._total_distance = 0.0

    # ------------------------------------------------------------------
    # 순기구학
    # ------------------------------------------------------------------
    @staticmethod
    def forward_kinematics(v_left: float, v_right: float, wheel_base: float):
        """
        차동 구동 순기구학: 좌/우 바퀴 속도로부터 선속도(v)와 각속도(omega)를 계산한다.

        매개변수
        --------
        v_left : float
            왼쪽 바퀴 선속도 (m/s).
        v_right : float
            오른쪽 바퀴 선속도 (m/s).
        wheel_base : float
            좌우 바퀴 간 거리 (m).

        반환
        ----
        v : float
            로봇 중심 선속도 (m/s).
        omega : float
            로봇 각속도 (rad/s).
        """
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / wheel_base
        return v, omega

    # ------------------------------------------------------------------
    # 오도메트리 갱신
    # ------------------------------------------------------------------
    def update(self, left_ticks: int, right_ticks: int):
        """
        인코더 틱 변화량으로 오도메트리를 갱신한다.

        매개변수
        --------
        left_ticks : int
            왼쪽 바퀴 틱 변화량 (현재 주기의 delta ticks).
        right_ticks : int
            오른쪽 바퀴 틱 변화량.

        반환
        ----
        tuple[float, float, float]
            갱신된 (x, y, theta).
        """
        # 각 바퀴의 이동 거리
        d_left = left_ticks * self.meters_per_tick
        d_right = right_ticks * self.meters_per_tick

        # 로봇 중심 이동 거리 및 회전각
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_base

        # 위치 갱신 — 작은 d_theta에서도 정확하도록 중간각 사용
        mid_theta = self.theta + d_theta / 2.0
        self.x += d_center * np.cos(mid_theta)
        self.y += d_center * np.sin(mid_theta)
        self.theta += d_theta

        # theta를 [-pi, pi] 범위로 정규화
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # 누적 거리
        self._total_distance += abs(d_center)

        # 기록 저장
        self.history.append(
            {
                "x": self.x,
                "y": self.y,
                "theta": self.theta,
                "d_left": d_left,
                "d_right": d_right,
                "total_distance": self._total_distance,
            }
        )

        return self.x, self.y, self.theta

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------
    def get_pose(self):
        """현재 로봇 자세 (x, y, theta)를 반환한다."""
        return self.x, self.y, self.theta

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """오도메트리를 초기화한다."""
        self.x = x
        self.y = y
        self.theta = theta
        self._total_distance = 0.0
        self.history.clear()

    # ------------------------------------------------------------------
    # 드리프트 분석
    # ------------------------------------------------------------------
    def analyze_drift(self, ground_truth: np.ndarray) -> dict:
        """
        오도메트리 드리프트를 분석한다.

        오도메트리로 추정한 경로와 ground truth를 비교하여
        위치 오차, 방향 오차, 누적 드리프트 비율을 계산한다.

        매개변수
        --------
        ground_truth : np.ndarray, shape (N, 3)
            실제 위치 배열. 각 행은 [x, y, theta].

        반환
        ----
        dict
            - 'position_errors': 각 스텝별 위치 오차 (m)
            - 'heading_errors': 각 스텝별 방향 오차 (rad)
            - 'mean_position_error': 평균 위치 오차 (m)
            - 'max_position_error': 최대 위치 오차 (m)
            - 'drift_ratio': 총 이동 거리 대비 최종 위치 오차 비율
        """
        if len(self.history) == 0:
            raise ValueError("오도메트리 기록이 비어 있습니다. update()를 먼저 호출하세요.")

        odom_poses = np.array([[h["x"], h["y"], h["theta"]] for h in self.history])
        n = min(len(odom_poses), len(ground_truth))
        odom_poses = odom_poses[:n]
        gt = ground_truth[:n]

        # 위치 오차 (유클리드 거리)
        pos_errors = np.sqrt(
            (odom_poses[:, 0] - gt[:, 0]) ** 2
            + (odom_poses[:, 1] - gt[:, 1]) ** 2
        )

        # 방향 오차 ([-pi, pi] 범위)
        heading_errors = odom_poses[:, 2] - gt[:, 2]
        heading_errors = np.arctan2(np.sin(heading_errors), np.cos(heading_errors))

        # 드리프트 비율
        total_dist = self._total_distance if self._total_distance > 0 else 1e-9
        drift_ratio = pos_errors[-1] / total_dist

        return {
            "position_errors": pos_errors,
            "heading_errors": heading_errors,
            "mean_position_error": float(np.mean(pos_errors)),
            "max_position_error": float(np.max(pos_errors)),
            "drift_ratio": float(drift_ratio),
        }


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    robot = DifferentialDriveRobot()
    print("=== 차동 구동 로봇 오도메트리 테스트 ===")

    # 직진 — 양쪽 바퀴 동일 틱
    for _ in range(100):
        robot.update(left_ticks=10, right_ticks=10)
    print(f"직진 100스텝 후 위치: x={robot.x:.4f}, y={robot.y:.4f}, theta={robot.theta:.4f}")

    robot.reset()

    # 제자리 회전 — 좌우 반대 방향
    for _ in range(100):
        robot.update(left_ticks=-5, right_ticks=5)
    print(f"제자리 회전 100스텝 후: x={robot.x:.4f}, y={robot.y:.4f}, theta={robot.theta:.4f} rad")

    robot.reset()

    # 원호 주행
    for _ in range(200):
        robot.update(left_ticks=8, right_ticks=10)
    print(f"원호 주행 200스텝 후: x={robot.x:.4f}, y={robot.y:.4f}, theta={robot.theta:.4f} rad")
