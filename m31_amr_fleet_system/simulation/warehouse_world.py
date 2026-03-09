"""
60m x 40m 물류센터 시뮬레이션 환경.

Occupancy Grid 기반으로 선반, 통로, 입고/출고 구역, 충전 스테이션을 배치하고,
동적 장애물(작업자, 지게차)을 생성하여 시뮬레이션한다.
"""

import numpy as np


class DynamicObstacle:
    """
    동적 장애물 (작업자 또는 지게차).

    매개변수
    --------
    x, y : float
        초기 위치 (m).
    speed : float
        이동 속도 (m/s), 0.3~1.5 범위.
    pattern : str
        이동 패턴 ('linear', 'curve', 'random').
    obstacle_type : str
        장애물 유형 ('worker', 'forklift').
    radius : float
        충돌 반경 (m).
    """

    def __init__(
        self,
        x: float,
        y: float,
        speed: float = 0.8,
        pattern: str = 'linear',
        obstacle_type: str = 'worker',
        radius: float = 0.4,
    ):
        self.x = x
        self.y = y
        self.speed = np.clip(speed, 0.3, 1.5)
        self.pattern = pattern
        self.obstacle_type = obstacle_type
        self.radius = radius

        # 이동 방향 (rad)
        self.heading = np.random.uniform(0, 2 * np.pi)
        # 곡선 이동 시 각속도
        self.angular_velocity = np.random.uniform(-0.3, 0.3)
        # 무작위 이동 시 방향 전환 카운터
        self._random_timer = 0.0
        self._random_interval = np.random.uniform(1.0, 4.0)

    def step(self, dt: float, world_width: float, world_height: float):
        """
        dt(초) 만큼 장애물을 이동시킨다.

        벽에 닿으면 반사하여 방향을 바꾼다.
        """
        if self.pattern == 'linear':
            pass  # heading 유지
        elif self.pattern == 'curve':
            self.heading += self.angular_velocity * dt
        elif self.pattern == 'random':
            self._random_timer += dt
            if self._random_timer >= self._random_interval:
                self.heading = np.random.uniform(0, 2 * np.pi)
                self._random_timer = 0.0
                self._random_interval = np.random.uniform(1.0, 4.0)

        dx = self.speed * np.cos(self.heading) * dt
        dy = self.speed * np.sin(self.heading) * dt

        new_x = self.x + dx
        new_y = self.y + dy

        # 벽 반사
        if new_x < self.radius or new_x > world_width - self.radius:
            self.heading = np.pi - self.heading
            new_x = np.clip(new_x, self.radius, world_width - self.radius)
        if new_y < self.radius or new_y > world_height - self.radius:
            self.heading = -self.heading
            new_y = np.clip(new_y, self.radius, world_height - self.radius)

        self.x = new_x
        self.y = new_y

    def get_position(self) -> tuple:
        """현재 위치를 (x, y) 튜플로 반환한다."""
        return self.x, self.y


class WarehouseWorld:
    """
    60m x 40m 물류센터 시뮬레이션 환경.

    Occupancy Grid는 0.1m 해상도의 numpy 배열(600 x 400)로 관리한다.
    - 0: 빈 공간 (통로)
    - 1: 정적 장애물 (선반, 벽, 기둥)
    - 2: 입고 구역
    - 3: 출고 구역
    - 4: 충전 스테이션

    매개변수
    --------
    width : float
        환경 가로 길이 (m). 기본값 60.0.
    height : float
        환경 세로 길이 (m). 기본값 40.0.
    resolution : float
        그리드 해상도 (m/cell). 기본값 0.1.
    """

    # 선반 열 이름과 개수
    SHELF_COLUMNS = ['A', 'B', 'C']
    SHELVES_PER_COLUMN = 7

    def __init__(
        self,
        width: float = 60.0,
        height: float = 40.0,
        resolution: float = 0.1,
    ):
        self.width = width
        self.height = height
        self.resolution = resolution

        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)

        # 그리드 초기화 (모두 빈 공간)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        # 동적 장애물 리스트
        self.dynamic_obstacles: list[DynamicObstacle] = []

        # 선반 위치 기록 (열_이름, 번호) -> (x_start, y_start, x_end, y_end)
        self.shelf_positions: dict[str, tuple] = {}

        # 충전 스테이션 위치
        self.charging_stations: list[tuple] = []

        # 환경 빌드
        self._build_walls()
        self._build_pillars()
        self._build_shelves()
        self._build_zones()
        self._build_charging_stations()

    # ------------------------------------------------------------------
    # 환경 구성
    # ------------------------------------------------------------------
    def _fill_rect(self, x_start: float, y_start: float, x_end: float, y_end: float, value: int):
        """사각 영역을 value 로 채운다. 좌표는 m 단위."""
        c0 = max(0, int(x_start / self.resolution))
        c1 = min(self.grid_width, int(x_end / self.resolution))
        r0 = max(0, int(y_start / self.resolution))
        r1 = min(self.grid_height, int(y_end / self.resolution))
        self.grid[r0:r1, c0:c1] = value

    def _build_walls(self):
        """외벽을 생성한다."""
        wall_thickness = 0.3  # m
        # 상단 벽
        self._fill_rect(0, 0, self.width, wall_thickness, 1)
        # 하단 벽
        self._fill_rect(0, self.height - wall_thickness, self.width, self.height, 1)
        # 좌측 벽
        self._fill_rect(0, 0, wall_thickness, self.height, 1)
        # 우측 벽
        self._fill_rect(self.width - wall_thickness, 0, self.width, self.height, 1)

    def _build_pillars(self):
        """내부 기둥을 생성한다 (4개, 중앙 영역)."""
        pillar_size = 0.4  # m
        pillar_positions = [
            (15.0, 10.0),
            (15.0, 30.0),
            (45.0, 10.0),
            (45.0, 30.0),
        ]
        for px, py in pillar_positions:
            self._fill_rect(
                px - pillar_size / 2, py - pillar_size / 2,
                px + pillar_size / 2, py + pillar_size / 2,
                1,
            )

    def _build_shelves(self):
        """
        선반 구역을 배치한다.

        A, B, C 열 x 7개 = 총 21개 선반.
        각 선반: 4m x 1.2m, 열 간 간격 8m, 선반 간 간격 4m.
        """
        column_x_starts = {'A': 8.0, 'B': 24.0, 'C': 40.0}
        shelf_width = 4.0   # m (x 방향)
        shelf_depth = 1.2   # m (y 방향)
        y_start_base = 4.0  # 첫 선반 y 시작
        y_gap = 4.0         # 선반 간 y 간격

        for col_name, x_start in column_x_starts.items():
            for i in range(self.SHELVES_PER_COLUMN):
                y_start = y_start_base + i * (shelf_depth + y_gap)
                x_end = x_start + shelf_width
                y_end = y_start + shelf_depth

                key = f"{col_name}{i + 1}"
                self.shelf_positions[key] = (x_start, y_start, x_end, y_end)
                self._fill_rect(x_start, y_start, x_end, y_end, 1)

    def _build_zones(self):
        """입고 구역과 출고 구역을 배치한다."""
        # 입고 구역: 좌측 하단
        self._fill_rect(1.0, 35.0, 6.0, 39.0, 2)
        # 출고 구역: 우측 하단
        self._fill_rect(54.0, 35.0, 59.0, 39.0, 3)

    def _build_charging_stations(self):
        """충전 스테이션을 배치한다 (5대분)."""
        stations = [
            (2.0, 2.0),
            (4.0, 2.0),
            (6.0, 2.0),
            (8.0, 2.0),
            (10.0, 2.0),
        ]
        for sx, sy in stations:
            self.charging_stations.append((sx, sy))
            self._fill_rect(sx - 0.3, sy - 0.3, sx + 0.3, sy + 0.3, 4)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    def get_occupancy_grid(self) -> np.ndarray:
        """
        현재 occupancy grid (정적 + 동적 장애물 반영)를 반환한다.

        반환
        ----
        np.ndarray, shape (grid_height, grid_width)
            각 셀 값: 0=빈공간, 1=장애물, 2=입고, 3=출고, 4=충전.
        """
        grid = self.grid.copy()

        # 동적 장애물을 그리드에 반영
        for obs in self.dynamic_obstacles:
            cx = int(obs.x / self.resolution)
            cy = int(obs.y / self.resolution)
            r_cells = int(obs.radius / self.resolution)
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    if dx * dx + dy * dy <= r_cells * r_cells:
                        gx = cx + dx
                        gy = cy + dy
                        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                            grid[gy, gx] = 1

        return grid

    def spawn_dynamic_obstacles(
        self,
        num_workers: int = 3,
        num_forklifts: int = 2,
    ) -> list[DynamicObstacle]:
        """
        동적 장애물을 생성하여 환경에 추가한다.

        작업자는 통로에, 지게차는 넓은 통로에 배치한다.

        매개변수
        --------
        num_workers : int
            작업자 수.
        num_forklifts : int
            지게차 수.

        반환
        ----
        list[DynamicObstacle]
            새로 생성된 장애물 리스트.
        """
        new_obstacles = []
        patterns = ['linear', 'curve', 'random']

        for _ in range(num_workers):
            x, y = self._find_free_position()
            speed = np.random.uniform(0.3, 1.0)
            pattern = patterns[np.random.randint(0, len(patterns))]
            obs = DynamicObstacle(
                x=x, y=y, speed=speed, pattern=pattern,
                obstacle_type='worker', radius=0.3,
            )
            new_obstacles.append(obs)

        for _ in range(num_forklifts):
            x, y = self._find_free_position()
            speed = np.random.uniform(0.5, 1.5)
            pattern = patterns[np.random.randint(0, len(patterns))]
            obs = DynamicObstacle(
                x=x, y=y, speed=speed, pattern=pattern,
                obstacle_type='forklift', radius=0.6,
            )
            new_obstacles.append(obs)

        self.dynamic_obstacles.extend(new_obstacles)
        return new_obstacles

    def step(self, dt: float):
        """
        시뮬레이션을 dt(초) 만큼 전진시킨다.

        모든 동적 장애물을 이동시킨다.

        매개변수
        --------
        dt : float
            시뮬레이션 시간 간격 (초).
        """
        for obs in self.dynamic_obstacles:
            obs.step(dt, self.width, self.height)

    def is_free(self, x: float, y: float) -> bool:
        """(x, y) 위치가 빈 공간인지 확인한다."""
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return self.grid[gy, gx] == 0
        return False

    def world_to_grid(self, x: float, y: float) -> tuple:
        """월드 좌표를 그리드 좌표로 변환한다."""
        return int(x / self.resolution), int(y / self.resolution)

    def grid_to_world(self, gx: int, gy: int) -> tuple:
        """그리드 좌표를 월드 좌표 중심으로 변환한다."""
        return (gx + 0.5) * self.resolution, (gy + 0.5) * self.resolution

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------
    def _find_free_position(self) -> tuple:
        """통로 위의 무작위 빈 위치를 찾는다."""
        margin = 1.0
        for _ in range(1000):
            x = np.random.uniform(margin, self.width - margin)
            y = np.random.uniform(margin, self.height - margin)
            if self.is_free(x, y):
                return x, y
        # 실패 시 기본 위치
        return self.width / 2, self.height / 2


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    world = WarehouseWorld()
    grid = world.get_occupancy_grid()
    print(f"그리드 크기: {grid.shape}")
    print(f"장애물 셀 수: {np.sum(grid == 1)}")
    print(f"빈 공간 셀 수: {np.sum(grid == 0)}")
    print(f"입고 구역 셀 수: {np.sum(grid == 2)}")
    print(f"출고 구역 셀 수: {np.sum(grid == 3)}")
    print(f"충전 스테이션 셀 수: {np.sum(grid == 4)}")
    print(f"선반 수: {len(world.shelf_positions)}")
    print(f"충전 스테이션 수: {len(world.charging_stations)}")

    # 동적 장애물 생성 및 시뮬레이션
    obstacles = world.spawn_dynamic_obstacles(num_workers=3, num_forklifts=2)
    print(f"\n동적 장애물 {len(obstacles)}개 생성")
    for i, obs in enumerate(obstacles):
        print(f"  [{i}] {obs.obstacle_type}: ({obs.x:.1f}, {obs.y:.1f}), "
              f"speed={obs.speed:.2f}m/s, pattern={obs.pattern}")

    # 5초 시뮬레이션
    for t in range(50):
        world.step(0.1)
    print("\n5초 시뮬레이션 후 위치:")
    for i, obs in enumerate(world.dynamic_obstacles):
        print(f"  [{i}] ({obs.x:.1f}, {obs.y:.1f})")
