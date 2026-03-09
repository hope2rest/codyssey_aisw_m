"""
2D LiDAR SLAM (Simultaneous Localization and Mapping) 모듈.

ICP(Iterative Closest Point) 스캔 매칭, 포즈 그래프 최적화, 루프 클로저 검출,
Occupancy Grid 맵 생성 기능을 직접 구현한다.
NumPy만 사용하여 ROS2 의존 없이 독립 실행이 가능하다.
"""

import numpy as np

__all__ = [
    "ICPScanMatcher",
    "PoseGraph",
    "OccupancyGridMap",
    "SLAM2D",
]


# ======================================================================
# ICP 스캔 매칭
# ======================================================================

class ICPScanMatcher:
    """
    SVD 기반 Point-to-Point ICP 스캔 매칭 클래스.

    2D LiDAR 스캔 간의 상대 변환(회전, 병진)을 추정한다.
    최근접 이웃 탐색에 유클리드 거리 기반 브루트 포스를 사용한다.
    """

    @staticmethod
    def _find_correspondences(
        source: np.ndarray, target: np.ndarray, max_dist: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        최근접 이웃 대응점을 탐색한다.

        매개변수
        --------
        source : np.ndarray, shape (N, 2)
            소스 포인트 클라우드.
        target : np.ndarray, shape (M, 2)
            타겟 포인트 클라우드.
        max_dist : float
            대응점으로 인정할 최대 거리. 기본값 2.0m.

        반환
        ----
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (매칭된 소스 인덱스, 매칭된 타겟 인덱스, 거리 배열).
        """
        # 브루트 포스 최근접 이웃 (N x M 거리 행렬)
        # 메모리 효율을 위해 청크 처리
        n = source.shape[0]
        chunk_size = 500
        src_idx_list = []
        tgt_idx_list = []
        dist_list = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            src_chunk = source[start:end]  # (C, 2)
            # 거리 행렬: (C, M)
            diff = src_chunk[:, np.newaxis, :] - target[np.newaxis, :, :]  # (C, M, 2)
            dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (C, M)
            min_idx = np.argmin(dists, axis=1)  # (C,)
            min_dist = dists[np.arange(end - start), min_idx]  # (C,)

            mask = min_dist < max_dist
            local_src = np.arange(start, end)[mask]
            src_idx_list.append(local_src)
            tgt_idx_list.append(min_idx[mask])
            dist_list.append(min_dist[mask])

        if len(src_idx_list) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([])

        return (
            np.concatenate(src_idx_list),
            np.concatenate(tgt_idx_list),
            np.concatenate(dist_list),
        )

    @staticmethod
    def match(
        source_points: np.ndarray,
        target_points: np.ndarray,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_dist: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        ICP 알고리즘으로 소스→타겟 변환을 추정한다.

        SVD 기반 Point-to-Point ICP를 사용하며, 수렴할 때까지
        반복적으로 대응점 탐색과 변환 추정을 수행한다.

        매개변수
        --------
        source_points : np.ndarray, shape (N, 2)
            소스 포인트 클라우드 (현재 스캔).
        target_points : np.ndarray, shape (M, 2)
            타겟 포인트 클라우드 (이전 스캔 또는 맵).
        max_iterations : int
            최대 반복 횟수. 기본값 50.
        tolerance : float
            수렴 판단 오차 허용값. 기본값 1e-6.
        max_correspondence_dist : float
            대응점 최대 거리. 기본값 2.0m.

        반환
        ----
        tuple[np.ndarray, np.ndarray, float]
            (R, t, error) — 2×2 회전 행렬, 2×1 병진 벡터, 최종 평균 오차.
        """
        source = np.array(source_points, dtype=np.float64)
        target = np.array(target_points, dtype=np.float64)

        if source.shape[0] < 3 or target.shape[0] < 3:
            return np.eye(2), np.zeros((2, 1)), float("inf")

        # 누적 변환
        R_total = np.eye(2)
        t_total = np.zeros((2, 1))
        prev_error = float("inf")

        transformed = source.copy()

        for _iter in range(max_iterations):
            # 1) 대응점 탐색
            src_idx, tgt_idx, dists = ICPScanMatcher._find_correspondences(
                transformed, target, max_correspondence_dist
            )

            if len(src_idx) < 3:
                break

            matched_src = transformed[src_idx]
            matched_tgt = target[tgt_idx]

            # 2) 중심 계산
            centroid_src = np.mean(matched_src, axis=0)
            centroid_tgt = np.mean(matched_tgt, axis=0)

            # 3) 중심 제거
            src_centered = matched_src - centroid_src
            tgt_centered = matched_tgt - centroid_tgt

            # 4) SVD 기반 최적 회전/병진 계산
            W = src_centered.T @ tgt_centered  # (2, 2)
            U, _, Vt = np.linalg.svd(W)

            # 반사(reflection) 보정
            d = np.linalg.det(Vt.T @ U.T)
            S = np.eye(2)
            if d < 0:
                S[1, 1] = -1

            R_step = Vt.T @ S @ U.T
            t_step = (centroid_tgt - R_step @ centroid_src).reshape(2, 1)

            # 5) 변환 적용
            transformed = (R_step @ transformed.T).T + t_step.T

            # 누적
            R_total = R_step @ R_total
            t_total = R_step @ t_total + t_step

            # 6) 수렴 판정
            mean_error = np.mean(dists[src_idx < len(dists)] if len(dists) > 0 else [0])
            mean_error = float(np.mean(dists))

            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        return R_total, t_total, prev_error


# ======================================================================
# 포즈 그래프
# ======================================================================

class PoseGraph:
    """
    포즈 그래프 기반 SLAM 백엔드 클래스.

    노드(로봇 자세)와 에지(상대 변환 제약)를 관리하며,
    Gauss-Newton 방식의 그래프 최적화와 루프 클로저 검출을 수행한다.

    매개변수
    --------
    scan_match_threshold : float
        루프 클로저 ICP 매칭 오차 기준값. 기본값 0.5.
    min_loop_distance : int
        루프 클로저 검출 시 최소 노드 간격. 기본값 10.
    """

    def __init__(
        self,
        scan_match_threshold: float = 0.5,
        min_loop_distance: int = 10,
    ):
        self.nodes: list[np.ndarray] = []          # 각 노드의 자세 [x, y, theta]
        self.scans: list[np.ndarray | None] = []   # 각 노드에 연결된 스캔 데이터
        self.edges: list[dict] = []                # 에지 목록

        self.scan_match_threshold = scan_match_threshold
        self.min_loop_distance = min_loop_distance
        self._icp = ICPScanMatcher()

    def add_node(self, pose: np.ndarray, scan: np.ndarray | None = None) -> int:
        """
        포즈 그래프에 노드를 추가한다.

        매개변수
        --------
        pose : np.ndarray, shape (3,)
            로봇 자세 [x, y, theta].
        scan : np.ndarray, shape (N, 2), optional
            해당 자세에서 취득한 스캔 데이터.

        반환
        ----
        int
            추가된 노드 ID.
        """
        node_id = len(self.nodes)
        self.nodes.append(np.array(pose, dtype=np.float64))
        self.scans.append(scan)
        return node_id

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information_matrix: np.ndarray | None = None,
    ):
        """
        두 노드 사이에 에지(상대 자세 제약)를 추가한다.

        매개변수
        --------
        from_id : int
            시작 노드 ID.
        to_id : int
            끝 노드 ID.
        relative_pose : np.ndarray, shape (3,)
            상대 자세 [dx, dy, dtheta].
        information_matrix : np.ndarray, shape (3, 3), optional
            정보 행렬 (공분산의 역행렬). None이면 단위 행렬 사용.
        """
        if information_matrix is None:
            information_matrix = np.eye(3)

        self.edges.append({
            "from": from_id,
            "to": to_id,
            "measurement": np.array(relative_pose, dtype=np.float64),
            "information": np.array(information_matrix, dtype=np.float64),
        })

    @staticmethod
    def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
        """자세 [x, y, theta]를 3×3 동차 변환 행렬로 변환한다."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.array([
            [c, -s, pose[0]],
            [s,  c, pose[1]],
            [0,  0, 1],
        ])

    @staticmethod
    def _transform_to_pose(T: np.ndarray) -> np.ndarray:
        """3×3 동차 변환 행렬을 자세 [x, y, theta]로 변환한다."""
        return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])

    def optimize(self, iterations: int = 10) -> list[np.ndarray]:
        """
        Gauss-Newton 방식으로 포즈 그래프를 최적화한다.

        첫 번째 노드를 고정(앵커)하고 나머지 노드의 자세를 조정하여
        모든 에지 제약의 오차를 최소화한다.

        매개변수
        --------
        iterations : int
            최적화 반복 횟수. 기본값 10.

        반환
        ----
        list[np.ndarray]
            최적화된 자세 목록.
        """
        if len(self.nodes) < 2 or len(self.edges) == 0:
            return [p.copy() for p in self.nodes]

        n = len(self.nodes)

        for _ in range(iterations):
            # 선형 시스템 구축: H * dx = -b
            H = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)

            for edge in self.edges:
                i = edge["from"]
                j = edge["to"]
                z_ij = edge["measurement"]
                omega = edge["information"]

                xi = self.nodes[i]
                xj = self.nodes[j]

                # 예측 상대 자세
                Ti = self._pose_to_transform(xi)
                Tj = self._pose_to_transform(xj)
                Ti_inv = np.linalg.inv(Ti)
                T_ij = Ti_inv @ Tj
                z_pred = self._transform_to_pose(T_ij)

                # 오차
                e = z_pred - z_ij
                e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))

                # 야코비안 (수치 근사 대신 해석적 야코비안 사용)
                ci, si = np.cos(xi[2]), np.sin(xi[2])
                dx = xj[0] - xi[0]
                dy = xj[1] - xi[1]

                # d(e)/d(xi)
                A = np.array([
                    [-ci, -si, -si * dx + ci * dy],
                    [ si, -ci,  ci * dx + si * dy],
                    [  0,   0,                 -1],
                ])

                # d(e)/d(xj)
                B = np.array([
                    [ ci, si, 0],
                    [-si, ci, 0],
                    [  0,  0, 1],
                ])

                # 헤시안과 그래디언트에 누적
                bi = 3 * i
                bj = 3 * j

                H[bi:bi+3, bi:bi+3] += A.T @ omega @ A
                H[bi:bi+3, bj:bj+3] += A.T @ omega @ B
                H[bj:bj+3, bi:bi+3] += B.T @ omega @ A
                H[bj:bj+3, bj:bj+3] += B.T @ omega @ B

                b[bi:bi+3] += A.T @ omega @ e
                b[bj:bj+3] += B.T @ omega @ e

            # 첫 번째 노드 고정 (앵커)
            H[0:3, 0:3] += np.eye(3) * 1e6

            # 선형 시스템 풀기
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break

            # 자세 갱신
            for k in range(n):
                self.nodes[k] += dx[3*k:3*k+3]
                self.nodes[k][2] = np.arctan2(
                    np.sin(self.nodes[k][2]), np.cos(self.nodes[k][2])
                )

            # 수렴 판정
            if np.linalg.norm(dx) < 1e-6:
                break

        return [p.copy() for p in self.nodes]

    def detect_loop_closure(
        self,
        current_scan: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[int, np.ndarray] | None:
        """
        현재 스캔과 과거 스캔을 비교하여 루프 클로저를 검출한다.

        현재 노드에서 일정 거리 이상 떨어진 과거 노드들과
        ICP 매칭을 수행하여 오차가 기준값 이하인 경우를 탐지한다.

        매개변수
        --------
        current_scan : np.ndarray, shape (N, 2)
            현재 LiDAR 스캔 포인트.
        threshold : float, optional
            ICP 매칭 오차 기준값. None이면 인스턴스 기본값 사용.

        반환
        ----
        tuple[int, np.ndarray] | None
            (매칭된 노드 ID, 상대 자세 [dx, dy, dtheta]) 또는 None.
        """
        if threshold is None:
            threshold = self.scan_match_threshold

        n = len(self.nodes)
        if n < self.min_loop_distance + 1:
            return None

        current_pose = self.nodes[-1]
        best_match = None
        best_error = threshold

        # 최근 노드를 제외한 과거 노드 탐색
        for k in range(n - self.min_loop_distance):
            if self.scans[k] is None or self.scans[k].shape[0] < 10:
                continue

            # 거리 기반 사전 필터 — 현재 위치와 가까운 과거 노드만 검사
            dist = np.linalg.norm(current_pose[:2] - self.nodes[k][:2])
            if dist > 5.0:  # 5m 이상 떨어진 노드는 스킵
                continue

            # 스캔을 글로벌 좌표로 변환 후 비교
            R, t, error = self._icp.match(
                current_scan, self.scans[k],
                max_iterations=30, tolerance=1e-5,
            )

            if error < best_error:
                best_error = error
                # 상대 자세 계산
                angle = np.arctan2(R[1, 0], R[0, 0])
                relative_pose = np.array([t[0, 0], t[1, 0], angle])
                best_match = (k, relative_pose)

        return best_match


# ======================================================================
# Occupancy Grid 맵
# ======================================================================

class OccupancyGridMap:
    """
    로그 오즈(log-odds) 기반 Occupancy Grid 맵 클래스.

    레이 캐스팅으로 자유 공간과 점유 공간을 갱신하며,
    맵 품질 평가 메트릭을 제공한다.

    매개변수
    --------
    width : int
        맵 격자 너비 (셀 수).
    height : int
        맵 격자 높이 (셀 수).
    resolution : float
        격자 해상도 (m/셀). 기본값 0.05.
    origin : tuple[float, float]
        맵 원점의 월드 좌표 (m). 기본값 (0, 0).
    """

    # 로그 오즈 파라미터
    _L_OCC = 0.85     # 점유 관측 시 로그 오즈 증가량
    _L_FREE = -0.40   # 자유 관측 시 로그 오즈 감소량
    _L_PRIOR = 0.0    # 사전 확률 (로그 오즈)
    _L_MAX = 5.0      # 로그 오즈 상한
    _L_MIN = -5.0     # 로그 오즈 하한

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float = 0.05,
        origin: tuple[float, float] = (0, 0),
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin, dtype=np.float64)

        # 로그 오즈 격자 (초기값 = 사전 확률)
        self._log_odds = np.full((height, width), self._L_PRIOR, dtype=np.float64)

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """월드 좌표를 격자 인덱스로 변환한다."""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        """격자 인덱스가 범위 내인지 확인한다."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """
        Bresenham 직선 알고리즘으로 두 격자 셀 사이의 경로를 생성한다.

        반환
        ----
        list[tuple[int, int]]
            (gx, gy) 격자 인덱스 목록 (끝점 제외).
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if x0 == x1 and y0 == y1:
                break
            cells.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells

    def update(self, robot_pose: np.ndarray, scan_points: np.ndarray):
        """
        레이 캐스팅 기반으로 Occupancy Grid를 갱신한다.

        로봇 위치에서 각 스캔 포인트까지 직선을 그어
        경로상의 셀은 자유 공간으로, 끝점은 점유 공간으로 갱신한다.

        매개변수
        --------
        robot_pose : np.ndarray, shape (3,)
            로봇 자세 [x, y, theta] (월드 좌표).
        scan_points : np.ndarray, shape (N, 2)
            스캔 포인트 (로봇 로컬 좌표).
        """
        x, y, theta = robot_pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # 로봇 위치의 격자 인덱스
        rx, ry = self._world_to_grid(x, y)

        for pt in scan_points:
            # 로컬 → 월드 좌표 변환
            wx = x + cos_t * pt[0] - sin_t * pt[1]
            wy = y + sin_t * pt[0] + cos_t * pt[1]

            # 끝점 격자 인덱스
            ex, ey = self._world_to_grid(wx, wy)

            # 레이 캐스팅: 자유 공간 갱신
            free_cells = self._bresenham(rx, ry, ex, ey)
            for gx, gy in free_cells:
                if self._in_bounds(gx, gy):
                    self._log_odds[gy, gx] = np.clip(
                        self._log_odds[gy, gx] + self._L_FREE,
                        self._L_MIN, self._L_MAX,
                    )

            # 끝점: 점유 공간 갱신
            if self._in_bounds(ex, ey):
                self._log_odds[ey, ex] = np.clip(
                    self._log_odds[ey, ex] + self._L_OCC,
                    self._L_MIN, self._L_MAX,
                )

    def get_map(self) -> np.ndarray:
        """
        현재 맵을 ROS 표준 Occupancy Grid 형식으로 반환한다.

        반환
        ----
        np.ndarray, shape (height, width), dtype int8
            -1=미탐색, 0=자유 공간, 100=점유.
        """
        grid = np.full((self.height, self.width), -1, dtype=np.int8)

        # 탐색된 셀 판별 (사전 확률과 다르면 탐색됨)
        explored = np.abs(self._log_odds - self._L_PRIOR) > 0.01

        # 확률 변환: p = 1 - 1/(1 + exp(L))
        prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))

        # 자유 공간 (확률 < 0.4)
        free_mask = explored & (prob < 0.4)
        grid[free_mask] = 0

        # 점유 공간 (확률 > 0.6)
        occ_mask = explored & (prob > 0.6)
        grid[occ_mask] = 100

        return grid

    def save(self, filepath: str):
        """
        맵 데이터를 .npz 파일로 저장한다.

        매개변수
        --------
        filepath : str
            저장 경로 (.npz).
        """
        np.savez_compressed(
            filepath,
            log_odds=self._log_odds,
            resolution=self.resolution,
            origin=self.origin,
            width=self.width,
            height=self.height,
        )

    def load(self, filepath: str):
        """
        .npz 파일에서 맵 데이터를 로드한다.

        매개변수
        --------
        filepath : str
            로드 경로 (.npz).
        """
        data = np.load(filepath)
        self._log_odds = data["log_odds"]
        self.resolution = float(data["resolution"])
        self.origin = data["origin"]
        self.width = int(data["width"])
        self.height = int(data["height"])

    def evaluate_quality(self) -> dict:
        """
        맵 품질 평가 메트릭을 계산한다.

        반환
        ----
        dict
            - coverage_ratio: 탐색된 셀 비율 (0~1).
            - entropy: 맵 전체 엔트로피 (낮을수록 확실).
            - edge_sharpness: 점유/자유 경계의 선명도 (높을수록 선명).
        """
        total_cells = self.height * self.width

        # 확률 변환
        prob = 1.0 - 1.0 / (1.0 + np.exp(self._log_odds))
        explored = np.abs(self._log_odds - self._L_PRIOR) > 0.01

        # 1) 커버리지 비율
        coverage_ratio = float(np.sum(explored)) / total_cells

        # 2) 엔트로피 (탐색된 셀만)
        if np.any(explored):
            p = prob[explored]
            p = np.clip(p, 1e-10, 1 - 1e-10)
            entropy_vals = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            entropy = float(np.mean(entropy_vals))
        else:
            entropy = 1.0  # 최대 불확실성

        # 3) 경계 선명도 (Sobel 기반 에지 강도)
        grid = self.get_map().astype(np.float64)
        grid[grid < 0] = 50  # 미탐색은 중간값으로 처리

        # 간단한 Sobel 필터
        if grid.shape[0] > 2 and grid.shape[1] > 2:
            gx = grid[:, 2:] - grid[:, :-2]
            gy = grid[2:, :] - grid[:-2, :]
            # 크기 맞추기
            min_h = min(gx.shape[0], gy.shape[0])
            min_w = min(gx.shape[1], gy.shape[1])
            gx = gx[:min_h, :min_w]
            gy = gy[:min_h, :min_w]
            gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

            if np.max(gradient_mag) > 0:
                edge_sharpness = float(np.mean(gradient_mag[gradient_mag > 10]))
                if np.isnan(edge_sharpness):
                    edge_sharpness = 0.0
            else:
                edge_sharpness = 0.0
        else:
            edge_sharpness = 0.0

        return {
            "coverage_ratio": round(coverage_ratio, 4),
            "entropy": round(entropy, 4),
            "edge_sharpness": round(edge_sharpness, 4),
        }


# ======================================================================
# SLAM2D 오케스트레이터
# ======================================================================

class SLAM2D:
    """
    2D LiDAR SLAM 오케스트레이터 클래스.

    ICP 스캔 매칭, 포즈 그래프 최적화, Occupancy Grid 맵 생성을
    통합하여 실시간 SLAM을 수행한다.

    매개변수
    --------
    map_size : tuple[int, int]
        맵 격자 크기 (width, height). 기본값 (1200, 800) → 60m × 40m.
    resolution : float
        맵 격자 해상도 (m/셀). 기본값 0.05.
    origin : tuple[float, float]
        맵 원점 월드 좌표. 기본값 (-10, -10).
    loop_closure_interval : int
        루프 클로저 검출 주기 (스캔 수). 기본값 20.
    """

    def __init__(
        self,
        map_size: tuple[int, int] = (1200, 800),
        resolution: float = 0.05,
        origin: tuple[float, float] = (-10.0, -10.0),
        loop_closure_interval: int = 20,
    ):
        self.resolution = resolution
        self.map_size = map_size

        # 핵심 구성 요소
        self._icp = ICPScanMatcher()
        self._pose_graph = PoseGraph()
        self._grid_map = OccupancyGridMap(
            width=map_size[0],
            height=map_size[1],
            resolution=resolution,
            origin=origin,
        )

        # 상태
        self._current_pose = np.array([0.0, 0.0, 0.0])
        self._prev_scan: np.ndarray | None = None
        self._scan_count = 0
        self._loop_closure_interval = loop_closure_interval

        # 첫 번째 노드 추가
        self._pose_graph.add_node(self._current_pose.copy())

    def process_scan(
        self,
        scan_points: np.ndarray,
        odometry_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        새로운 LiDAR 스캔을 처리하여 자세를 추정하고 맵을 갱신한다.

        매개변수
        --------
        scan_points : np.ndarray, shape (N, 2)
            현재 LiDAR 스캔 포인트 (로봇 로컬 좌표).
        odometry_delta : np.ndarray, shape (3,), optional
            오도메트리 상대 변위 [dx, dy, dtheta].
            None이면 ICP 결과만 사용.

        반환
        ----
        np.ndarray, shape (3,)
            현재 추정 자세 [x, y, theta].
        """
        scan_points = np.array(scan_points, dtype=np.float64)

        if self._prev_scan is not None and scan_points.shape[0] >= 3:
            # ICP 스캔 매칭
            R, t, error = self._icp.match(scan_points, self._prev_scan)
            angle = np.arctan2(R[1, 0], R[0, 0])

            # 오도메트리가 있으면 가중 평균
            if odometry_delta is not None:
                odom = np.array(odometry_delta, dtype=np.float64)
                icp_delta = np.array([t[0, 0], t[1, 0], angle])
                # ICP가 신뢰할 수 있으면 ICP 가중치 높게
                w_icp = 0.7 if error < 0.5 else 0.3
                delta = w_icp * icp_delta + (1 - w_icp) * odom
            else:
                delta = np.array([t[0, 0], t[1, 0], angle])

            # 자세 갱신 (글로벌 좌표계)
            cos_t = np.cos(self._current_pose[2])
            sin_t = np.sin(self._current_pose[2])
            dx_global = cos_t * delta[0] - sin_t * delta[1]
            dy_global = sin_t * delta[0] + cos_t * delta[1]

            self._current_pose[0] += dx_global
            self._current_pose[1] += dy_global
            self._current_pose[2] += delta[2]
            self._current_pose[2] = np.arctan2(
                np.sin(self._current_pose[2]),
                np.cos(self._current_pose[2]),
            )

            # 포즈 그래프에 노드/에지 추가
            node_id = self._pose_graph.add_node(
                self._current_pose.copy(), scan_points
            )
            self._pose_graph.add_edge(
                node_id - 1, node_id, delta,
            )

        elif self._prev_scan is None:
            # 첫 번째 스캔: 초기 노드에 스캔 데이터 연결
            self._pose_graph.scans[0] = scan_points

            if odometry_delta is not None:
                odom = np.array(odometry_delta, dtype=np.float64)
                self._current_pose += odom

        # 맵 갱신
        self._grid_map.update(self._current_pose, scan_points)

        # 이전 스캔 저장
        self._prev_scan = scan_points.copy()
        self._scan_count += 1

        # 주기적 루프 클로저 검출
        if self._scan_count % self._loop_closure_interval == 0:
            self.run_loop_closure()

        return self._current_pose.copy()

    def run_loop_closure(self):
        """
        루프 클로저 검출 및 그래프 최적화를 실행한다.

        루프 클로저가 검출되면 해당 에지를 포즈 그래프에 추가하고
        그래프 최적화를 수행하여 누적 오차를 보정한다.
        """
        if self._prev_scan is None:
            return

        result = self._pose_graph.detect_loop_closure(self._prev_scan)
        if result is not None:
            matched_id, relative_pose = result
            current_id = len(self._pose_graph.nodes) - 1

            # 루프 클로저 에지 추가 (높은 정보 가중치)
            info = np.eye(3) * 10.0
            self._pose_graph.add_edge(
                current_id, matched_id, relative_pose, info
            )

            # 그래프 최적화 실행
            optimized = self._pose_graph.optimize(iterations=15)

            # 최적화 결과로 현재 자세 갱신
            if len(optimized) > 0:
                self._current_pose = optimized[-1].copy()

    def get_map(self) -> np.ndarray:
        """현재 Occupancy Grid 맵을 반환한다."""
        return self._grid_map.get_map()

    def get_raw_map(self) -> OccupancyGridMap:
        """OccupancyGridMap 인스턴스를 반환한다."""
        return self._grid_map

    def get_trajectory(self) -> list[np.ndarray]:
        """포즈 그래프의 전체 궤적(자세 목록)을 반환한다."""
        return [p.copy() for p in self._pose_graph.nodes]

    def get_pose(self) -> np.ndarray:
        """현재 추정 자세 [x, y, theta]를 반환한다."""
        return self._current_pose.copy()


# ======================================================================
# 독립 실행 데모
# ======================================================================
if __name__ == "__main__":
    print("=== 2D LiDAR SLAM 데모 ===\n")

    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1) ICP 스캔 매칭 테스트
    # ------------------------------------------------------------------
    print("[1] ICP 스캔 매칭 테스트")

    # 가상 스캔 생성: 원형 벽
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    radius = 5.0
    target_scan = np.column_stack([
        radius * np.cos(angles) + np.random.randn(180) * 0.02,
        radius * np.sin(angles) + np.random.randn(180) * 0.02,
    ])

    # 알려진 변환 적용 (0.3m 이동, 5도 회전)
    true_angle = np.radians(5)
    true_t = np.array([0.3, 0.1])
    R_true = np.array([
        [np.cos(true_angle), -np.sin(true_angle)],
        [np.sin(true_angle),  np.cos(true_angle)],
    ])
    source_scan = (R_true @ target_scan.T).T + true_t + np.random.randn(180, 2) * 0.02

    icp = ICPScanMatcher()
    R_est, t_est, err = icp.match(source_scan, target_scan)
    angle_est = np.degrees(np.arctan2(R_est[1, 0], R_est[0, 0]))

    print(f"  실제 변환: tx={true_t[0]:.3f}, ty={true_t[1]:.3f}, "
          f"θ={np.degrees(true_angle):.1f}°")
    print(f"  추정 변환: tx={t_est[0,0]:.3f}, ty={t_est[1,0]:.3f}, "
          f"θ={angle_est:.1f}°")
    print(f"  매칭 오차: {err:.6f}\n")

    # ------------------------------------------------------------------
    # 2) SLAM 시스템 테스트: 사각형 경로 주행
    # ------------------------------------------------------------------
    print("[2] SLAM 시스템 테스트 - 사각형 경로 주행")

    slam = SLAM2D(
        map_size=(400, 400),
        resolution=0.05,
        origin=(-5.0, -5.0),
        loop_closure_interval=50,
    )

    def generate_lidar_scan(
        robot_x: float, robot_y: float, robot_theta: float,
        walls: list[tuple[float, float, float, float]],
        num_beams: int = 180,
        max_range: float = 8.0,
    ) -> np.ndarray:
        """
        가상 LiDAR 스캔을 생성한다.

        벽(선분) 목록에 대해 레이 캐스팅을 수행하여
        로봇 로컬 좌표계의 포인트 클라우드를 반환한다.
        """
        beam_angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
        points = []

        for ba in beam_angles:
            global_angle = robot_theta + ba
            dx = np.cos(global_angle)
            dy = np.sin(global_angle)

            min_dist = max_range
            for wx0, wy0, wx1, wy1 in walls:
                # 레이-선분 교차 판정
                ex, ey = wx1 - wx0, wy1 - wy0
                denom = dx * ey - dy * ex
                if abs(denom) < 1e-12:
                    continue
                t_ray = ((wx0 - robot_x) * ey - (wy0 - robot_y) * ex) / denom
                t_seg = ((wx0 - robot_x) * dy - (wy0 - robot_y) * dx) / denom

                if 0 < t_ray < min_dist and 0 <= t_seg <= 1:
                    min_dist = t_ray

            if min_dist < max_range:
                # 로컬 좌표
                lx = min_dist * np.cos(ba) + np.random.randn() * 0.01
                ly = min_dist * np.sin(ba) + np.random.randn() * 0.01
                points.append([lx, ly])

        return np.array(points) if len(points) > 0 else np.zeros((0, 2))

    # 창고 벽 정의 (10m × 8m 사각형)
    warehouse_walls = [
        (0, 0, 10, 0),    # 하단 벽
        (10, 0, 10, 8),   # 우측 벽
        (10, 8, 0, 8),    # 상단 벽
        (0, 8, 0, 0),     # 좌측 벽
        # 내부 장애물
        (3, 3, 3, 5),
        (3, 5, 5, 5),
        (7, 2, 7, 6),
    ]

    # 사각형 경로 웨이포인트
    waypoints = [
        (2, 2, 0),
        (8, 2, 0),
        (8, 6, np.pi / 2),
        (2, 6, np.pi),
        (2, 2, -np.pi / 2),  # 루프 클로저 지점
    ]

    # 경로 보간 및 스캔 처리
    all_poses = []
    steps_per_segment = 20

    for seg in range(len(waypoints) - 1):
        x0, y0, t0 = waypoints[seg]
        x1, y1, t1 = waypoints[seg + 1]

        for s in range(steps_per_segment):
            alpha = s / steps_per_segment
            rx = x0 + alpha * (x1 - x0)
            ry = y0 + alpha * (y1 - y0)
            rt = t0 + alpha * (t1 - t0)

            scan = generate_lidar_scan(rx, ry, rt, warehouse_walls, num_beams=90)
            if scan.shape[0] < 5:
                continue

            pose = slam.process_scan(scan)
            all_poses.append(pose.copy())

    trajectory = slam.get_trajectory()
    print(f"  처리된 스캔 수: {slam._scan_count}")
    print(f"  포즈 그래프 노드 수: {len(trajectory)}")
    print(f"  최종 추정 자세: ({pose[0]:.3f}, {pose[1]:.3f}, θ={np.degrees(pose[2]):.1f}°)")

    # ------------------------------------------------------------------
    # 3) 맵 품질 평가
    # ------------------------------------------------------------------
    print("\n[3] 맵 품질 평가")
    quality = slam.get_raw_map().evaluate_quality()
    for metric, value in quality.items():
        print(f"  {metric}: {value}")

    grid = slam.get_map()
    free_cells = int(np.sum(grid == 0))
    occ_cells = int(np.sum(grid == 100))
    unknown_cells = int(np.sum(grid == -1))
    print(f"\n  맵 통계: 자유={free_cells}, 점유={occ_cells}, 미탐색={unknown_cells}")

    # ------------------------------------------------------------------
    # 4) 포즈 그래프 최적화 테스트
    # ------------------------------------------------------------------
    print("\n[4] 포즈 그래프 최적화 테스트")
    pg = PoseGraph()
    pg.add_node(np.array([0.0, 0.0, 0.0]))
    pg.add_node(np.array([1.0, 0.0, 0.0]))
    pg.add_node(np.array([1.0, 1.0, np.pi / 2]))
    pg.add_node(np.array([0.0, 1.0, np.pi]))

    pg.add_edge(0, 1, np.array([1.0, 0.0, 0.0]))
    pg.add_edge(1, 2, np.array([0.0, 1.0, np.pi / 2]))
    pg.add_edge(2, 3, np.array([-1.0, 0.0, np.pi / 2]))
    # 루프 클로저 에지
    pg.add_edge(3, 0, np.array([0.0, -1.0, np.pi / 2]), np.eye(3) * 5.0)

    optimized = pg.optimize(iterations=20)
    print("  최적화 전 → 후:")
    for k, (orig, opt) in enumerate(zip(
        [np.array([0,0,0]), np.array([1,0,0]),
         np.array([1,1,np.pi/2]), np.array([0,1,np.pi])],
        optimized
    )):
        print(f"    노드 {k}: ({orig[0]:.2f}, {orig[1]:.2f}) → "
              f"({opt[0]:.3f}, {opt[1]:.3f})")

    print("\n=== SLAM 데모 완료 ===")
