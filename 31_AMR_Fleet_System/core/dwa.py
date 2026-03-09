"""
Dynamic Window Approach (DWA) 로컬 경로 계획 모듈.

속도 공간에서 실현 가능한 속도 쌍 (v, omega)을 샘플링하고,
각각에 대해 궤적을 시뮬레이션하여 비용 함수를 기반으로
최적의 제어 입력을 선택한다.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DWAConfig:
    """
    DWA 설정 파라미터 클래스.

    속성
    ----
    max_v : float
        최대 선속도 (m/s).
    min_v : float
        최소 선속도 (m/s). 보통 0 또는 음수(후진 허용 시).
    max_w : float
        최대 각속도 (rad/s).
    max_acc_v : float
        최대 선가속도 (m/s²).
    max_acc_w : float
        최대 각가속도 (rad/s²).
    v_resolution : float
        선속도 샘플링 해상도 (m/s).
    w_resolution : float
        각속도 샘플링 해상도 (rad/s).
    predict_time : float
        궤적 시뮬레이션 시간 (초).
    dt : float
        시뮬레이션 시간 간격 (초).
    heading_weight : float
        목표 방향 비용 가중치.
    clearance_weight : float
        장애물 회피 비용 가중치.
    velocity_weight : float
        속도 비용 가중치.
    robot_radius : float
        로봇 반경 (m), 충돌 판정용.
    """

    max_v: float = 2.0
    min_v: float = 0.0
    max_w: float = 1.5
    max_acc_v: float = 1.0
    max_acc_w: float = 2.0
    v_resolution: float = 0.05
    w_resolution: float = 0.05
    predict_time: float = 2.0
    dt: float = 0.1
    heading_weight: float = 1.0
    clearance_weight: float = 1.5
    velocity_weight: float = 1.0
    robot_radius: float = 0.3


def _compute_dynamic_window(
    state: np.ndarray, config: DWAConfig
) -> tuple[float, float, float, float]:
    """
    동적 윈도우를 계산한다.

    현재 속도와 가속 제한으로부터 다음 제어 주기에서
    도달 가능한 속도 범위를 반환한다.

    매개변수
    --------
    state : np.ndarray, shape (5,)
        [x, y, theta, v, omega].
    config : DWAConfig
        DWA 설정.

    반환
    ----
    tuple[float, float, float, float]
        (v_min, v_max, w_min, w_max).
    """
    v = state[3]
    w = state[4]

    # 동역학 제한 윈도우
    v_min = max(config.min_v, v - config.max_acc_v * config.dt)
    v_max = min(config.max_v, v + config.max_acc_v * config.dt)
    w_min = max(-config.max_w, w - config.max_acc_w * config.dt)
    w_max = min(config.max_w, w + config.max_acc_w * config.dt)

    return v_min, v_max, w_min, w_max


def _simulate_trajectory(
    state: np.ndarray, v: float, w: float, config: DWAConfig
) -> np.ndarray:
    """
    주어진 (v, w)로 궤적을 시뮬레이션한다.

    매개변수
    --------
    state : np.ndarray
        현재 상태 [x, y, theta, v, omega].
    v, w : float
        선속도, 각속도.
    config : DWAConfig
        설정.

    반환
    ----
    np.ndarray, shape (N, 5)
        시뮬레이션된 궤적.
    """
    trajectory = [state.copy()]
    s = state.copy()

    steps = int(config.predict_time / config.dt)
    for _ in range(steps):
        s[0] += v * np.cos(s[2]) * config.dt
        s[1] += v * np.sin(s[2]) * config.dt
        s[2] += w * config.dt
        s[2] = np.arctan2(np.sin(s[2]), np.cos(s[2]))
        s[3] = v
        s[4] = w
        trajectory.append(s.copy())

    return np.array(trajectory)


def _heading_score(trajectory: np.ndarray, goal: np.ndarray) -> float:
    """
    목표 방향 점수: 궤적 끝점에서 목표까지의 방향과 로봇 방향의 일치도.

    값이 클수록 목표를 잘 향하고 있음을 나타낸다.
    """
    last = trajectory[-1]
    angle_to_goal = np.arctan2(goal[1] - last[1], goal[0] - last[0])
    heading_diff = angle_to_goal - last[2]
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    return np.pi - abs(heading_diff)  # [0, pi] 범위, 클수록 좋음


def _clearance_score(
    trajectory: np.ndarray, obstacles: np.ndarray, robot_radius: float
) -> float:
    """
    장애물 회피 점수: 궤적 상의 모든 점에서 가장 가까운 장애물까지의 최소 거리.

    충돌 시 음의 무한대를 반환한다.
    """
    if len(obstacles) == 0:
        return float("inf")

    min_dist = float("inf")
    for point in trajectory:
        diffs = obstacles[:, :2] - point[:2]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        d = np.min(dists)
        if d <= robot_radius:
            return float("-inf")  # 충돌
        min_dist = min(min_dist, d)

    return min_dist


def _velocity_score(v: float, config: DWAConfig) -> float:
    """
    속도 점수: 빠른 속도를 선호한다.
    """
    return abs(v)


def dwa_planning(
    state: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    config: DWAConfig | None = None,
) -> tuple[float, float]:
    """
    DWA 기반 로컬 경로 계획을 수행한다.

    속도 공간을 샘플링하고 궤적을 시뮬레이션하여
    최적의 (v, omega) 제어 입력을 반환한다.

    매개변수
    --------
    state : np.ndarray, shape (5,)
        현재 로봇 상태 [x, y, theta, v, omega].
    goal : np.ndarray, shape (2,)
        목표 위치 [x, y].
    obstacles : np.ndarray, shape (N, 2)
        장애물 위치 배열. 각 행은 [x, y].
        장애물이 없으면 빈 배열 np.empty((0, 2)).
    config : DWAConfig, optional
        DWA 설정. None이면 기본값 사용.

    반환
    ----
    tuple[float, float]
        최적 제어 입력 (v, omega).
    """
    if config is None:
        config = DWAConfig()

    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    obstacles = np.asarray(obstacles, dtype=np.float64)

    # 동적 윈도우 계산
    v_min, v_max, w_min, w_max = _compute_dynamic_window(state, config)

    best_v, best_w = 0.0, 0.0
    best_score = float("-inf")
    best_trajectory = None

    # 속도 공간 샘플링
    v_range = np.arange(v_min, v_max + config.v_resolution, config.v_resolution)
    w_range = np.arange(w_min, w_max + config.w_resolution, config.w_resolution)

    for v in v_range:
        for w in w_range:
            # 궤적 시뮬레이션
            traj = _simulate_trajectory(state, v, w, config)

            # 비용 함수 계산
            heading = _heading_score(traj, goal)
            clearance = _clearance_score(traj, obstacles, config.robot_radius)
            velocity = _velocity_score(v, config)

            # 충돌 궤적 제외
            if clearance == float("-inf"):
                continue

            # clearance를 유한한 값으로 클리핑 (정규화용)
            clearance = min(clearance, 5.0)

            # 종합 점수
            score = (
                config.heading_weight * heading
                + config.clearance_weight * clearance
                + config.velocity_weight * velocity
            )

            if score > best_score:
                best_score = score
                best_v = v
                best_w = w
                best_trajectory = traj

    return best_v, best_w


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== DWA 경로 계획 테스트 ===")

    config = DWAConfig()
    state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])  # 초기 상태
    goal = np.array([10.0, 5.0])  # 목표

    # 장애물 배치
    obstacles = np.array([
        [3.0, 1.0],
        [5.0, 3.0],
        [7.0, 2.0],
        [4.0, 4.0],
    ])

    print(f"시작 상태: {state[:3]}")
    print(f"목표: {goal}")
    print(f"장애물 수: {len(obstacles)}")

    # 10스텝 시뮬레이션
    for step in range(30):
        v, w = dwa_planning(state, goal, obstacles, config)

        # 상태 갱신
        state[0] += v * np.cos(state[2]) * config.dt
        state[1] += v * np.sin(state[2]) * config.dt
        state[2] += w * config.dt
        state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
        state[3] = v
        state[4] = w

        dist = np.sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)
        if step % 5 == 0:
            print(
                f"  스텝 {step:3d}: "
                f"위치=({state[0]:.2f}, {state[1]:.2f}), "
                f"v={v:.2f}, w={w:.2f}, "
                f"목표거리={dist:.2f}"
            )
        if dist < 0.5:
            print(f"  목표 도달! 스텝 {step}")
            break
