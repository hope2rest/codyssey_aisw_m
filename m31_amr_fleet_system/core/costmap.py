"""
2D Costmap 생성 모듈.

Static Layer (정적 점유 격자), Inflation Layer (거리 기반 비용 감쇠),
Obstacle Layer (실시간 센서 데이터)를 결합하여 로봇 경로 계획에 사용할
비용맵을 생성한다.
"""

import numpy as np
from scipy import ndimage


class Costmap:
    """
    2D 비용맵 클래스.

    여러 레이어(Static, Inflation, Obstacle)를 결합하여
    최종 비용 격자를 생성한다.

    매개변수
    --------
    width : int
        맵 가로 크기 (셀 수).
    height : int
        맵 세로 크기 (셀 수).
    resolution : float
        셀 하나의 크기 (m/cell). 기본값 0.05.
    origin : tuple[float, float]
        맵 원점의 월드 좌표 (x, y). 기본값 (0.0, 0.0).
    inflation_radius : float
        Inflation 반경 (m). 기본값 0.5.
    inscribed_radius : float
        로봇 내접 반경 (m). 이 반경 안은 치명적 비용. 기본값 0.2.
    cost_scaling_factor : float
        Inflation 비용 감쇠 계수. 기본값 5.0.
    """

    # 비용 상수
    FREE_SPACE = 0
    INSCRIBED_COST = 253
    LETHAL_COST = 254
    NO_INFORMATION = 255

    def __init__(
        self,
        width: int = 200,
        height: int = 200,
        resolution: float = 0.05,
        origin: tuple[float, float] = (0.0, 0.0),
        inflation_radius: float = 0.5,
        inscribed_radius: float = 0.2,
        cost_scaling_factor: float = 5.0,
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        self.inflation_radius = inflation_radius
        self.inscribed_radius = inscribed_radius
        self.cost_scaling_factor = cost_scaling_factor

        # 레이어별 격자
        self._static_layer = np.zeros((height, width), dtype=np.float64)
        self._obstacle_layer = np.zeros((height, width), dtype=np.float64)
        self._inflation_layer = np.zeros((height, width), dtype=np.float64)

        # 최종 비용 격자
        self.cost_grid = np.zeros((height, width), dtype=np.float64)

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """
        월드 좌표를 격자 인덱스로 변환한다.

        매개변수
        --------
        wx, wy : float
            월드 좌표 (m).

        반환
        ----
        tuple[int, int]
            격자 인덱스 (row, col).
        """
        col = int((wx - self.origin[0]) / self.resolution)
        row = int((wy - self.origin[1]) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """
        격자 인덱스를 월드 좌표로 변환한다.

        반환
        ----
        tuple[float, float]
            월드 좌표 (x, y).
        """
        wx = col * self.resolution + self.origin[0]
        wy = row * self.resolution + self.origin[1]
        return wx, wy

    # ------------------------------------------------------------------
    # Static Layer
    # ------------------------------------------------------------------
    def set_static_map(self, occupancy_grid: np.ndarray):
        """
        정적 점유 격자를 설정한다.

        매개변수
        --------
        occupancy_grid : np.ndarray, shape (height, width)
            점유 격자. 0=자유, 1=점유(장애물).
        """
        assert occupancy_grid.shape == (self.height, self.width), (
            f"격자 크기 불일치: 기대 ({self.height}, {self.width}), "
            f"입력 {occupancy_grid.shape}"
        )
        self._static_layer = np.where(
            occupancy_grid > 0.5, self.LETHAL_COST, self.FREE_SPACE
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Obstacle Layer
    # ------------------------------------------------------------------
    def update_obstacles(self, obstacles: np.ndarray):
        """
        실시간 센서 데이터로 장애물 레이어를 갱신한다.

        매개변수
        --------
        obstacles : np.ndarray, shape (N, 2)
            장애물 월드 좌표 배열. 각 행은 [x, y].
        """
        self._obstacle_layer[:] = self.FREE_SPACE

        if len(obstacles) == 0:
            return

        obstacles = np.asarray(obstacles, dtype=np.float64)
        for obs in obstacles:
            row, col = self.world_to_grid(obs[0], obs[1])
            if 0 <= row < self.height and 0 <= col < self.width:
                self._obstacle_layer[row, col] = self.LETHAL_COST

    # ------------------------------------------------------------------
    # Inflation Layer
    # ------------------------------------------------------------------
    def _compute_inflation(self, combined: np.ndarray) -> np.ndarray:
        """
        치명적 셀 주변에 거리 기반 감쇠 비용을 계산한다.

        inscribed_radius 이내: INSCRIBED_COST
        inscribed ~ inflation_radius: 지수적 감쇠

        매개변수
        --------
        combined : np.ndarray
            Static + Obstacle 레이어 합산.

        반환
        ----
        np.ndarray
            Inflation이 적용된 비용 격자.
        """
        lethal_mask = combined >= self.LETHAL_COST

        if not np.any(lethal_mask):
            return np.zeros_like(combined)

        # 장애물로부터의 유클리드 거리 (셀 단위)
        dist_cells = ndimage.distance_transform_edt(~lethal_mask)
        dist_meters = dist_cells * self.resolution

        inflation = np.zeros_like(combined)

        # 치명적 셀 유지
        inflation[lethal_mask] = self.LETHAL_COST

        # 내접 반경 이내: 치명적에 가까운 비용
        inscribed_mask = (dist_meters > 0) & (dist_meters <= self.inscribed_radius)
        inflation[inscribed_mask] = self.INSCRIBED_COST

        # 내접 ~ 팽창 반경: 지수적 감쇠
        decay_mask = (dist_meters > self.inscribed_radius) & (
            dist_meters <= self.inflation_radius
        )
        if np.any(decay_mask):
            # cost = INSCRIBED_COST * exp(-factor * (dist - inscribed_radius))
            decay_dist = dist_meters[decay_mask] - self.inscribed_radius
            inflation[decay_mask] = self.INSCRIBED_COST * np.exp(
                -self.cost_scaling_factor * decay_dist
            )

        return inflation

    # ------------------------------------------------------------------
    # 통합 갱신
    # ------------------------------------------------------------------
    def update(self, obstacles: np.ndarray | None = None) -> np.ndarray:
        """
        비용맵을 갱신한다.

        Static Layer + Obstacle Layer를 합산하고 Inflation을 적용하여
        최종 비용 격자를 생성한다.

        매개변수
        --------
        obstacles : np.ndarray, shape (N, 2), optional
            실시간 장애물 좌표. None이면 장애물 레이어를 갱신하지 않는다.

        반환
        ----
        np.ndarray, shape (height, width)
            갱신된 비용 격자. 값 범위 [0, 254].
        """
        if obstacles is not None:
            self.update_obstacles(obstacles)

        # Static + Obstacle 합산 (최대값 기준)
        combined = np.maximum(self._static_layer, self._obstacle_layer)

        # Inflation 적용
        self._inflation_layer = self._compute_inflation(combined)

        # 최종 비용 = max(combined, inflation)
        self.cost_grid = np.maximum(combined, self._inflation_layer)

        return self.cost_grid.copy()

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------
    def is_free(self, row: int, col: int, threshold: float = 1.0) -> bool:
        """해당 셀이 자유 공간인지 판단한다."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return self.cost_grid[row, col] < threshold

    def get_cost(self, row: int, col: int) -> float:
        """해당 셀의 비용을 반환한다."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return float(self.LETHAL_COST)
        return float(self.cost_grid[row, col])

    def get_cost_grid(self) -> np.ndarray:
        """비용 격자 사본을 반환한다."""
        return self.cost_grid.copy()


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== 2D Costmap 테스트 ===")

    # 50x50 비용맵 (해상도 0.1m → 5m x 5m)
    cmap = Costmap(
        width=50,
        height=50,
        resolution=0.1,
        origin=(0.0, 0.0),
        inflation_radius=0.5,
        inscribed_radius=0.15,
        cost_scaling_factor=5.0,
    )

    # 정적 맵 설정 (벽)
    static = np.zeros((50, 50))
    static[20, 10:40] = 1  # 수평 벽
    static[10:40, 25] = 1  # 수직 벽
    cmap.set_static_map(static)

    # 동적 장애물
    obstacles = np.array([
        [1.0, 1.0],
        [3.0, 3.0],
        [2.5, 1.5],
    ])

    cost_grid = cmap.update(obstacles)

    print(f"맵 크기: {cmap.height} x {cmap.width}")
    print(f"해상도: {cmap.resolution}m/cell")
    print(f"최대 비용: {cost_grid.max():.1f}")
    print(f"치명적 셀 수: {np.sum(cost_grid >= Costmap.LETHAL_COST)}")
    print(f"Inflation 셀 수: {np.sum((cost_grid > 0) & (cost_grid < Costmap.LETHAL_COST))}")
    print(f"자유 공간 셀 수: {np.sum(cost_grid == 0)}")

    # 좌표 변환 테스트
    row, col = cmap.world_to_grid(2.0, 3.0)
    wx, wy = cmap.grid_to_world(row, col)
    print(f"\n좌표 변환: 월드(2.0, 3.0) → 격자({row}, {col}) → 월드({wx}, {wy})")
    print(f"해당 셀 비용: {cmap.get_cost(row, col):.1f}")
