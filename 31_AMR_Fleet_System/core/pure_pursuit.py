"""
Pure Pursuit 경로 추종 컨트롤러 모듈.

Look-ahead 거리를 속도에 비례하여 적응적으로 조정하며,
주어진 경로를 추종하기 위한 조향 제어 입력을 계산한다.
Cross Track Error (횡방향 오차) 계산 기능을 포함한다.
"""

import numpy as np


class PurePursuitController:
    """
    Pure Pursuit 경로 추종 컨트롤러.

    매개변수
    --------
    min_lookahead : float
        최소 Look-ahead 거리 (m). 기본값 0.5.
    max_lookahead : float
        최대 Look-ahead 거리 (m). 기본값 3.0.
    lookahead_gain : float
        속도에 대한 Look-ahead 거리 비례 계수. 기본값 1.0.
        look-ahead = gain * velocity (min/max로 클리핑).
    wheel_base : float
        로봇 축간 거리 (m). 기본값 0.3.
    goal_tolerance : float
        목표 도달 판정 거리 (m). 기본값 0.2.
    """

    def __init__(
        self,
        min_lookahead: float = 0.5,
        max_lookahead: float = 3.0,
        lookahead_gain: float = 1.0,
        wheel_base: float = 0.3,
        goal_tolerance: float = 0.2,
    ):
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.lookahead_gain = lookahead_gain
        self.wheel_base = wheel_base
        self.goal_tolerance = goal_tolerance

        # 경로 상의 가장 가까운 점 인덱스 추적
        self._nearest_idx = 0

    def reset(self):
        """내부 상태를 초기화한다."""
        self._nearest_idx = 0

    def _find_nearest_point(
        self, path: np.ndarray, position: np.ndarray
    ) -> int:
        """
        경로 상에서 현재 위치에 가장 가까운 점의 인덱스를 찾는다.
        이전에 지나친 점으로 후퇴하지 않도록 _nearest_idx부터 탐색한다.

        매개변수
        --------
        path : np.ndarray, shape (N, 2)
            경로 점 배열.
        position : np.ndarray, shape (2,)
            현재 위치 [x, y].

        반환
        ----
        int
            가장 가까운 점의 인덱스.
        """
        search_from = self._nearest_idx
        dists = np.sqrt(np.sum((path[search_from:] - position) ** 2, axis=1))
        nearest = search_from + int(np.argmin(dists))
        self._nearest_idx = nearest
        return nearest

    def _find_lookahead_point(
        self, path: np.ndarray, position: np.ndarray, lookahead_dist: float
    ) -> np.ndarray:
        """
        Look-ahead 거리만큼 전방에 있는 경로 점을 찾는다.

        경로 세그먼트와 원의 교점을 기하학적으로 계산한다.

        매개변수
        --------
        path : np.ndarray, shape (N, 2)
            경로 점 배열.
        position : np.ndarray, shape (2,)
            현재 위치 [x, y].
        lookahead_dist : float
            Look-ahead 거리 (m).

        반환
        ----
        np.ndarray, shape (2,)
            Look-ahead 목표 점 [x, y].
        """
        # 가장 가까운 점부터 탐색
        for i in range(self._nearest_idx, len(path) - 1):
            seg_start = path[i]
            seg_end = path[i + 1]

            # 세그먼트와 look-ahead 원의 교점 계산
            d = seg_end - seg_start
            f = seg_start - position

            a = np.dot(d, d)
            b = 2.0 * np.dot(f, d)
            c = np.dot(f, f) - lookahead_dist ** 2

            discriminant = b ** 2 - 4.0 * a * c

            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # t가 [0, 1] 범위에 있는 교점 선택 (더 먼 점 우선)
            if 0.0 <= t2 <= 1.0:
                return seg_start + t2 * d
            if 0.0 <= t1 <= 1.0:
                return seg_start + t1 * d

        # 교점을 찾지 못하면 경로 끝점 반환
        return path[-1].copy()

    def compute(
        self,
        current_pose: np.ndarray,
        path: np.ndarray,
        velocity: float,
    ) -> tuple[float, float]:
        """
        Pure Pursuit 제어 입력을 계산한다.

        매개변수
        --------
        current_pose : np.ndarray, shape (3,)
            현재 로봇 자세 [x, y, theta].
        path : np.ndarray, shape (N, 2)
            추종할 경로 점 배열 [[x, y], ...].
        velocity : float
            현재 선속도 (m/s).

        반환
        ----
        tuple[float, float]
            (target_velocity, steering_angle).
            - target_velocity: 목표 선속도 (m/s). 목표 근처에서 감속.
            - steering_angle: 조향각 (rad).
        """
        position = current_pose[:2]
        theta = current_pose[2]

        # 목표 도달 확인
        dist_to_goal = np.sqrt(np.sum((path[-1] - position) ** 2))
        if dist_to_goal < self.goal_tolerance:
            return 0.0, 0.0

        # 적응형 look-ahead 거리
        lookahead_dist = np.clip(
            self.lookahead_gain * abs(velocity),
            self.min_lookahead,
            self.max_lookahead,
        )

        # 가장 가까운 점 갱신
        self._find_nearest_point(path, position)

        # Look-ahead 목표점 탐색
        target = self._find_lookahead_point(path, position, lookahead_dist)

        # 로봇 좌표계에서의 look-ahead 점 변환
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        local_x = dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        # 곡률 계산: kappa = 2 * y_local / L^2
        L_sq = local_x ** 2 + local_y ** 2
        if L_sq < 1e-6:
            return velocity, 0.0

        curvature = 2.0 * local_y / L_sq

        # 조향각 = atan(kappa * wheel_base)
        steering = np.arctan(curvature * self.wheel_base)

        # 목표 근처 감속
        target_v = velocity
        if dist_to_goal < 2.0:
            target_v = max(0.1, velocity * (dist_to_goal / 2.0))

        return target_v, steering


def compute_control(
    current_pose: np.ndarray,
    path: np.ndarray,
    velocity: float,
    **kwargs,
) -> tuple[float, float]:
    """
    Pure Pursuit 제어 입력을 계산하는 편의 함수.

    매개변수
    --------
    current_pose : np.ndarray, shape (3,)
        현재 로봇 자세 [x, y, theta].
    path : np.ndarray, shape (N, 2)
        추종할 경로.
    velocity : float
        현재 선속도 (m/s).
    **kwargs
        PurePursuitController 생성자에 전달할 추가 인자.

    반환
    ----
    tuple[float, float]
        (target_velocity, steering_angle).
    """
    controller = PurePursuitController(**kwargs)
    return controller.compute(current_pose, np.asarray(path), velocity)


def compute_cross_track_error(
    position: np.ndarray, path: np.ndarray
) -> tuple[float, int]:
    """
    Cross Track Error (횡방향 오차)를 계산한다.

    현재 위치에서 가장 가까운 경로 세그먼트까지의 수직 거리를 구한다.

    매개변수
    --------
    position : np.ndarray, shape (2,)
        현재 위치 [x, y].
    path : np.ndarray, shape (N, 2)
        경로 점 배열.

    반환
    ----
    tuple[float, int]
        (cross_track_error, nearest_segment_index).
        - cross_track_error: 횡방향 오차 (m). 양수=경로 왼쪽, 음수=경로 오른쪽.
        - nearest_segment_index: 가장 가까운 세그먼트의 시작 인덱스.
    """
    position = np.asarray(position, dtype=np.float64)
    path = np.asarray(path, dtype=np.float64)

    min_dist = float("inf")
    min_idx = 0
    signed_error = 0.0

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)

        if seg_len_sq < 1e-12:
            dist = np.sqrt(np.sum((position - p1) ** 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                signed_error = dist
            continue

        # 세그먼트 위의 최근접점 파라미터 t
        t = np.clip(np.dot(position - p1, seg) / seg_len_sq, 0.0, 1.0)
        proj = p1 + t * seg
        diff = position - proj
        dist = np.sqrt(np.dot(diff, diff))

        if dist < min_dist:
            min_dist = dist
            min_idx = i
            # 부호 판정: 세그먼트 방향 기준 외적
            cross = seg[0] * diff[1] - seg[1] * diff[0]
            signed_error = dist if cross >= 0 else -dist

    return signed_error, min_idx


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== Pure Pursuit 경로 추종 테스트 ===")

    # 원형 경로 생성
    t = np.linspace(0, 2 * np.pi, 100)
    radius = 5.0
    path = np.column_stack([radius * np.cos(t), radius * np.sin(t)])

    controller = PurePursuitController(wheel_base=0.3, lookahead_gain=0.8)

    # 초기 위치 (원 위에서 시작)
    pose = np.array([5.0, 0.0, np.pi / 2])
    velocity = 1.0
    dt = 0.05

    print(f"원형 경로 (반지름={radius}m) 추종 시작")

    for step in range(200):
        target_v, steering = controller.compute(pose, path, velocity)

        # 간단한 자전거 모델로 상태 갱신
        pose[0] += target_v * np.cos(pose[2]) * dt
        pose[1] += target_v * np.sin(pose[2]) * dt
        pose[2] += target_v * np.tan(steering) / controller.wheel_base * dt
        pose[2] = np.arctan2(np.sin(pose[2]), np.cos(pose[2]))

        cte, _ = compute_cross_track_error(pose[:2], path)

        if step % 40 == 0:
            print(
                f"  스텝 {step:3d}: "
                f"위치=({pose[0]:.2f}, {pose[1]:.2f}), "
                f"조향={np.degrees(steering):.1f}°, "
                f"CTE={cte:.3f}m"
            )
