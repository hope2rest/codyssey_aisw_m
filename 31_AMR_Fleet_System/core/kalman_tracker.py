"""
칼만 필터 기반 동적 장애물 추적 모듈.

등속 직선 운동 모델의 칼만 필터로 다중 객체를 추적한다.
새로운 검출과 기존 트랙 사이의 매칭(헝가리안 알고리즘 대신 탐욕적 매칭 사용),
ID 할당, 미래 위치 예측, Time-To-Collision (TTC) 계산 기능을 제공한다.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrackedObject:
    """
    추적 중인 객체 정보.

    속성
    ----
    obj_id : int
        고유 추적 ID.
    state : np.ndarray, shape (4,)
        상태 벡터 [x, y, vx, vy].
    P : np.ndarray, shape (4, 4)
        상태 공분산 행렬.
    age : int
        추적 유지 프레임 수.
    hits : int
        관측 매칭 성공 횟수.
    misses : int
        연속 관측 실패 횟수.
    """

    obj_id: int = 0
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4))
    age: int = 0
    hits: int = 0
    misses: int = 0


class ObjectTracker:
    """
    칼만 필터 기반 다중 객체 추적기.

    등속 직선 운동(Constant Velocity) 모델을 사용하며,
    새로운 검출에 대해 ID를 할당하고 기존 트랙과 매칭한다.

    매개변수
    --------
    dt : float
        추적 시간 간격 (초). 기본값 0.1.
    process_noise_std : float
        프로세스 잡음 표준편차. 기본값 0.5.
    measurement_noise_std : float
        관측 잡음 표준편차. 기본값 0.3.
    max_misses : int
        트랙 삭제까지 허용되는 연속 미관측 프레임 수. 기본값 5.
    match_threshold : float
        매칭 거리 임계값 (m). 기본값 2.0.
    min_hits : int
        트랙을 확정(confirmed)으로 판단하기 위한 최소 매칭 횟수. 기본값 3.
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.3,
        max_misses: int = 5,
        match_threshold: float = 2.0,
        min_hits: int = 3,
    ):
        self.dt = dt
        self.max_misses = max_misses
        self.match_threshold = match_threshold
        self.min_hits = min_hits

        # 상태 전이 행렬 F (등속 모델)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # 관측 행렬 H: [x, y] 관측
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # 프로세스 잡음 Q
        q = process_noise_std ** 2
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=np.float64) * q

        # 관측 잡음 R
        r = measurement_noise_std ** 2
        self.R = np.eye(2) * r

        # 트랙 관리
        self._tracks: list[TrackedObject] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    # 칼만 필터 예측/갱신
    # ------------------------------------------------------------------
    def _predict_track(self, track: TrackedObject):
        """트랙 상태를 한 스텝 예측한다."""
        track.state = self.F @ track.state
        track.P = self.F @ track.P @ self.F.T + self.Q

    def _update_track(self, track: TrackedObject, measurement: np.ndarray):
        """관측값으로 트랙 상태를 갱신한다."""
        z = measurement
        y = z - self.H @ track.state  # 혁신
        S = self.H @ track.P @ self.H.T + self.R  # 혁신 공분산
        K = track.P @ self.H.T @ np.linalg.inv(S)  # 칼만 이득

        track.state = track.state + K @ y
        I_KH = np.eye(4) - K @ self.H
        track.P = I_KH @ track.P @ I_KH.T + K @ self.R @ K.T

    # ------------------------------------------------------------------
    # 매칭
    # ------------------------------------------------------------------
    def _compute_cost_matrix(
        self, detections: np.ndarray
    ) -> np.ndarray:
        """
        트랙-검출 간 유클리드 거리 비용 행렬을 계산한다.

        반환
        ----
        np.ndarray, shape (num_tracks, num_detections)
        """
        n_tracks = len(self._tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), float("inf"))

        for i, track in enumerate(self._tracks):
            pred_pos = track.state[:2]
            for j in range(n_dets):
                dist = np.sqrt(np.sum((pred_pos - detections[j]) ** 2))
                cost[i, j] = dist

        return cost

    def _greedy_match(
        self, cost_matrix: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        탐욕적 매칭: 비용이 가장 작은 쌍부터 매칭한다.

        반환
        ----
        tuple
            (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        n_tracks, n_dets = cost_matrix.shape
        matched_pairs: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        # 모든 (track, det) 쌍을 비용 순 정렬
        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if cost_matrix[i, j] <= self.match_threshold:
                    pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()

        for _, i, j in pairs:
            if i not in used_tracks and j not in used_dets:
                matched_pairs.append((i, j))
                used_tracks.add(i)
                used_dets.add(j)

        unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
        unmatched_dets = [j for j in range(n_dets) if j not in used_dets]

        return matched_pairs, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------
    # 메인 업데이트
    # ------------------------------------------------------------------
    def update(self, detections: np.ndarray) -> list[dict]:
        """
        검출 결과로 트래커를 갱신한다.

        매개변수
        --------
        detections : np.ndarray, shape (M, 2)
            현재 프레임의 검출 위치 배열. 각 행은 [x, y].
            검출이 없으면 빈 배열.

        반환
        ----
        list[dict]
            확정(confirmed) 트랙 목록. 각 딕셔너리는:
            - 'id': 추적 ID
            - 'position': [x, y] 위치
            - 'velocity': [vx, vy] 속도
            - 'age': 추적 프레임 수
        """
        detections = np.asarray(detections, dtype=np.float64)
        if detections.ndim == 1 and len(detections) > 0:
            detections = detections.reshape(-1, 2)

        # 1) 모든 트랙 예측
        for track in self._tracks:
            self._predict_track(track)

        # 2) 매칭
        if len(self._tracks) > 0 and len(detections) > 0:
            cost = self._compute_cost_matrix(detections)
            matched, unmatched_tracks, unmatched_dets = self._greedy_match(cost)
        else:
            matched = []
            unmatched_tracks = list(range(len(self._tracks)))
            unmatched_dets = list(range(len(detections)))

        # 3) 매칭된 트랙 갱신
        for track_idx, det_idx in matched:
            track = self._tracks[track_idx]
            self._update_track(track, detections[det_idx])
            track.hits += 1
            track.misses = 0
            track.age += 1

        # 4) 매칭되지 않은 트랙: miss 증가
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].misses += 1
            self._tracks[track_idx].age += 1

        # 5) 매칭되지 않은 검출: 새 트랙 생성
        for det_idx in unmatched_dets:
            new_state = np.array([
                detections[det_idx][0],
                detections[det_idx][1],
                0.0,  # vx 초기값
                0.0,  # vy 초기값
            ])
            new_track = TrackedObject(
                obj_id=self._next_id,
                state=new_state,
                P=np.eye(4) * 1.0,
                age=1,
                hits=1,
                misses=0,
            )
            self._tracks.append(new_track)
            self._next_id += 1

        # 6) 오래된 트랙 제거
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        # 7) 확정 트랙만 반환
        results = []
        for track in self._tracks:
            if track.hits >= self.min_hits:
                results.append({
                    "id": track.obj_id,
                    "position": track.state[:2].copy(),
                    "velocity": track.state[2:].copy(),
                    "age": track.age,
                })

        return results

    # ------------------------------------------------------------------
    # 미래 위치 예측
    # ------------------------------------------------------------------
    def predict_future(
        self, obj_id: int, horizon: float, dt: float | None = None
    ) -> np.ndarray:
        """
        특정 객체의 미래 위치를 예측한다.

        등속 직선 운동 모델로 horizon 초 후까지의 위치를 계산한다.

        매개변수
        --------
        obj_id : int
            추적 ID.
        horizon : float
            예측 시간 (초).
        dt : float, optional
            예측 시간 간격. None이면 self.dt 사용.

        반환
        ----
        np.ndarray, shape (N, 2)
            예측된 미래 위치 배열 [[x, y], ...].

        예외
        ----
        ValueError
            해당 ID의 트랙이 존재하지 않을 때.
        """
        if dt is None:
            dt = self.dt

        track = None
        for t in self._tracks:
            if t.obj_id == obj_id:
                track = t
                break

        if track is None:
            raise ValueError(f"추적 ID {obj_id}를 찾을 수 없습니다.")

        steps = int(np.ceil(horizon / dt))
        positions = np.zeros((steps + 1, 2))
        state = track.state.copy()

        F_pred = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        positions[0] = state[:2]
        for i in range(1, steps + 1):
            state = F_pred @ state
            positions[i] = state[:2]

        return positions

    # ------------------------------------------------------------------
    # TTC (Time-To-Collision) 계산
    # ------------------------------------------------------------------
    def compute_ttc(
        self,
        robot_position: np.ndarray,
        robot_velocity: np.ndarray,
        robot_radius: float = 0.3,
        obstacle_radius: float = 0.3,
    ) -> list[dict]:
        """
        각 추적 객체에 대한 Time-To-Collision을 계산한다.

        로봇과 장애물이 각각 등속 직선 운동을 한다고 가정하여
        최소 접근 시각(closest point of approach)을 구한다.

        매개변수
        --------
        robot_position : np.ndarray, shape (2,)
            로봇 위치 [x, y].
        robot_velocity : np.ndarray, shape (2,)
            로봇 속도 [vx, vy].
        robot_radius : float
            로봇 반경 (m).
        obstacle_radius : float
            장애물 반경 (m).

        반환
        ----
        list[dict]
            각 추적 객체에 대한 TTC 정보:
            - 'id': 추적 ID
            - 'ttc': Time-To-Collision (초). 충돌 없으면 inf.
            - 'min_distance': 최소 접근 거리 (m).
        """
        robot_pos = np.asarray(robot_position, dtype=np.float64)
        robot_vel = np.asarray(robot_velocity, dtype=np.float64)
        collision_dist = robot_radius + obstacle_radius

        results = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue

            # 상대 위치/속도
            rel_pos = track.state[:2] - robot_pos
            rel_vel = track.state[2:] - robot_vel

            # 2차 방정식: |rel_pos + rel_vel * t|^2 = collision_dist^2
            a = np.dot(rel_vel, rel_vel)
            b = 2.0 * np.dot(rel_pos, rel_vel)
            c = np.dot(rel_pos, rel_pos) - collision_dist ** 2

            ttc = float("inf")
            min_dist = np.sqrt(np.dot(rel_pos, rel_pos))

            if a > 1e-9:
                disc = b ** 2 - 4.0 * a * c
                if disc >= 0:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-b - sqrt_disc) / (2.0 * a)
                    t2 = (-b + sqrt_disc) / (2.0 * a)
                    # 가장 빠른 양의 충돌 시각
                    if t1 > 0:
                        ttc = t1
                    elif t2 > 0:
                        ttc = t2

                # 최소 접근 거리 (CPA)
                t_cpa = -b / (2.0 * a)
                if t_cpa > 0:
                    cpa_pos = rel_pos + rel_vel * t_cpa
                    min_dist = min(min_dist, np.sqrt(np.dot(cpa_pos, cpa_pos)))

            results.append({
                "id": track.obj_id,
                "ttc": ttc,
                "min_distance": float(min_dist),
            })

        return results

    def get_all_tracks(self) -> list[TrackedObject]:
        """모든 트랙 목록을 반환한다 (디버깅용)."""
        return list(self._tracks)


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== 칼만 필터 다중 객체 추적 테스트 ===")

    tracker = ObjectTracker(dt=0.1, min_hits=2)
    np.random.seed(42)

    # 두 객체 시뮬레이션: 하나는 오른쪽, 하나는 위쪽으로 이동
    for step in range(20):
        t = step * 0.1
        det1 = np.array([1.0 * t + np.random.randn() * 0.1, 0.0 + np.random.randn() * 0.1])
        det2 = np.array([0.0 + np.random.randn() * 0.1, 0.5 * t + np.random.randn() * 0.1])
        detections = np.array([det1, det2])

        tracked = tracker.update(detections)

        if step % 5 == 0:
            print(f"\n  프레임 {step}:")
            for obj in tracked:
                print(
                    f"    ID={obj['id']}: "
                    f"위치=({obj['position'][0]:.2f}, {obj['position'][1]:.2f}), "
                    f"속도=({obj['velocity'][0]:.2f}, {obj['velocity'][1]:.2f})"
                )

    # TTC 계산
    print("\n  TTC 계산 (로봇 위치=(0,0), 속도=(1,0)):")
    ttc_results = tracker.compute_ttc(
        robot_position=np.array([0.0, 0.0]),
        robot_velocity=np.array([1.0, 0.0]),
    )
    for r in ttc_results:
        print(f"    ID={r['id']}: TTC={r['ttc']:.2f}s, 최소거리={r['min_distance']:.2f}m")
