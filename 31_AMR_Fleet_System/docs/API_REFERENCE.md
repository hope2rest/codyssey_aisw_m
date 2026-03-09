# AMR Fleet System API 레퍼런스

## 목차

1. [Core 모듈](#1-core-모듈)
2. [Perception 모듈](#2-perception-모듈)
3. [Fleet 모듈](#3-fleet-모듈)
4. [Task 모듈](#4-task-모듈)
5. [Simulation 모듈](#5-simulation-모듈)
6. [ROS2 노드 인터페이스](#6-ros2-노드-인터페이스)
7. [설정 파일 레퍼런스](#7-설정-파일-레퍼런스)

---

## 1. Core 모듈

### 1.1 A* 경로 계획 (`core/astar.py`)

#### `plan_path()`

A* 알고리즘으로 2D Occupancy Grid 상의 최단 경로를 탐색한다.

```python
def plan_path(
    grid: np.ndarray,            # (H, W) Occupancy Grid. 0=자유, 1=장애물
    start: tuple[int, int],      # 시작 좌표 (row, col)
    goal: tuple[int, int],       # 목표 좌표 (row, col)
    obstacle_threshold: float = 0.5,   # 장애물 판단 기준값
    smooth: bool = True,               # 경로 평활화 적용 여부
    smooth_weight_data: float = 0.5,   # 원본 경로 유지 가중치
    smooth_weight_smooth: float = 0.3, # 매끄러움 가중치
    smooth_tolerance: float = 1e-4,    # 평활화 수렴 허용 오차
) -> list[tuple[int, int]]       # 경로 좌표 리스트. 빈 리스트=경로 없음
```

**예외**: `ValueError` - 시작/목표 좌표가 범위 밖이거나 장애물 위에 있을 때

#### `smooth_path()`

Gradient Descent 기반 경로 평활화.

```python
def smooth_path(
    path: list[tuple[int, int]],
    grid: np.ndarray,
    obstacle_threshold: float = 0.5,
    weight_data: float = 0.5,
    weight_smooth: float = 0.3,
    tolerance: float = 1e-4,
) -> list[tuple[int, int]]
```

**사용 예시**:

```python
from core.astar import plan_path

grid = np.zeros((20, 20))
grid[5, 3:15] = 1  # 장애물 벽

path = plan_path(grid, (0, 0), (19, 19), smooth=True)
print(f"경로 길이: {len(path)}")
```

---

### 1.2 DWA 로컬 경로 계획 (`core/dwa.py`)

#### `DWAConfig`

DWA 설정 데이터 클래스.

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `max_v` | 2.0 | 최대 선속도 (m/s) |
| `min_v` | 0.0 | 최소 선속도 (m/s) |
| `max_w` | 1.5 | 최대 각속도 (rad/s) |
| `max_acc_v` | 1.0 | 최대 선가속도 (m/s^2) |
| `max_acc_w` | 2.0 | 최대 각가속도 (rad/s^2) |
| `v_resolution` | 0.05 | 선속도 샘플링 해상도 |
| `w_resolution` | 0.05 | 각속도 샘플링 해상도 |
| `predict_time` | 2.0 | 궤적 시뮬레이션 시간 (초) |
| `dt` | 0.1 | 시뮬레이션 시간 간격 (초) |
| `heading_weight` | 1.0 | 목표 방향 비용 가중치 |
| `clearance_weight` | 1.5 | 장애물 회피 비용 가중치 |
| `velocity_weight` | 1.0 | 속도 비용 가중치 |
| `robot_radius` | 0.3 | 로봇 반경 (m) |

#### `dwa_planning()`

```python
def dwa_planning(
    state: np.ndarray,           # (5,) [x, y, theta, v, omega]
    goal: np.ndarray,            # (2,) [x, y] 목표 위치
    obstacles: np.ndarray,       # (N, 2) 장애물 위치 배열
    config: DWAConfig | None = None,
) -> tuple[float, float]        # (v, omega) 최적 제어 입력
```

**사용 예시**:

```python
from core.dwa import dwa_planning, DWAConfig

config = DWAConfig(max_v=1.0, predict_time=1.5)
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
goal = np.array([5.0, 3.0])
obstacles = np.array([[2.0, 1.0], [3.0, 2.0]])

v, w = dwa_planning(state, goal, obstacles, config)
```

---

### 1.3 EKF 센서 퓨전 (`core/ekf.py`)

#### `ExtendedKalmanFilter` 클래스

5차원 상태 벡터 `[x, y, theta, v, omega]`를 추정하는 EKF.

```python
class ExtendedKalmanFilter:
    def __init__(
        self,
        initial_state: np.ndarray | None = None,      # (5,) 초기 상태
        initial_covariance: np.ndarray | None = None,  # (5,5) 초기 P. None → 0.1*I
        process_noise: np.ndarray | None = None,       # (5,5) Q. None → 기본값
    )

    def predict(self, dt: float) -> np.ndarray          # 예측 단계
    def update(self, measurement, H, R) -> np.ndarray   # 갱신 단계
    def get_state(self) -> np.ndarray                    # 현재 상태 반환
    def get_covariance(self) -> np.ndarray               # 현재 P 반환
```

#### 센서별 관측 모델

| 센서 | 관측 행렬 (H) | 잡음 공분산 (R) |
|------|-------------|----------------|
| Wheel Odometry | `ODOM_H` (3x5): [x, y, theta] 관측 | `ODOM_R = diag([0.05, 0.05, 0.02])` |
| IMU | `IMU_H` (2x5): [theta, omega] 관측 | `IMU_R = diag([0.01, 0.005])` |
| AMCL | `AMCL_H` (3x5): [x, y, theta] 관측 | `AMCL_R = diag([0.02, 0.02, 0.01])` |

#### `fuse_sensors()`

```python
def fuse_sensors(
    ekf: ExtendedKalmanFilter,
    dt: float,
    odom: np.ndarray | None = None,   # (3,) [x, y, theta]
    imu: np.ndarray | None = None,    # (2,) [theta, omega]
    amcl: np.ndarray | None = None,   # (3,) [x, y, theta]
) -> np.ndarray                        # (5,) 융합된 상태
```

---

### 1.4 Costmap (`core/costmap.py`)

#### `Costmap` 클래스

```python
class Costmap:
    # 비용 상수
    FREE_SPACE = 0
    INSCRIBED_COST = 253
    LETHAL_COST = 254
    NO_INFORMATION = 255

    def __init__(
        self,
        width: int = 200,
        height: int = 200,
        resolution: float = 0.05,        # m/cell
        origin: tuple[float, float] = (0.0, 0.0),
        inflation_radius: float = 0.5,   # m
        inscribed_radius: float = 0.2,   # m
        cost_scaling_factor: float = 5.0,
    )

    def world_to_grid(self, wx, wy) -> tuple[int, int]
    def grid_to_world(self, row, col) -> tuple[float, float]
    def set_static_map(self, occupancy_grid: np.ndarray)
    def update_obstacles(self, obstacles: np.ndarray)  # (N, 2)
    def update(self, obstacles=None) -> np.ndarray     # 전체 비용맵 갱신
    def is_free(self, row, col, threshold=1.0) -> bool
    def get_cost(self, row, col) -> float
    def get_cost_grid(self) -> np.ndarray
```

---

### 1.5 차동 구동 기구학 (`core/kinematics.py`)

#### `DifferentialDriveRobot` 클래스

```python
class DifferentialDriveRobot:
    def __init__(
        self,
        wheel_radius: float = 0.05,    # m
        wheel_base: float = 0.3,       # m
        ticks_per_rev: int = 4096,
        x0: float = 0.0,
        y0: float = 0.0,
        theta0: float = 0.0,
    )

    @staticmethod
    def forward_kinematics(v_left, v_right, wheel_base) -> (v, omega)
    def update(self, left_ticks: int, right_ticks: int) -> (x, y, theta)
    def get_pose(self) -> (x, y, theta)
    def reset(self, x=0.0, y=0.0, theta=0.0)
    def analyze_drift(self, ground_truth: np.ndarray) -> dict
    # 반환: {'position_errors', 'heading_errors', 'mean_position_error',
    #         'max_position_error', 'drift_ratio'}
```

---

### 1.6 PID 제어기 (`core/pid_controller.py`)

#### `PIDController` 클래스

```python
class PIDController:
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_min: float = -inf,
        output_max: float = inf,
        anti_windup_limit: float | None = None,
    )

    def compute(self, setpoint, measured, dt) -> float
    def reset(self)
    def set_gains(self, kp, ki, kd)
```

#### `TrapezoidalProfile` 클래스

```python
class TrapezoidalProfile:
    def __init__(
        self,
        max_velocity: float = 2.0,        # m/s 또는 rad/s
        max_acceleration: float = 1.0,     # m/s^2
        max_deceleration: float | None = None,  # None → max_acceleration
    )

    def generate(self, distance, dt=0.01) -> dict
    # 반환: {'time', 'velocity', 'position', 'acceleration', 'phase'}

    def get_velocity_at_time(self, distance, t) -> float
```

---

### 1.7 Pure Pursuit (`core/pure_pursuit.py`)

#### `PurePursuitController` 클래스

```python
class PurePursuitController:
    def __init__(
        self,
        min_lookahead: float = 0.5,    # m
        max_lookahead: float = 3.0,    # m
        lookahead_gain: float = 1.0,
        wheel_base: float = 0.3,      # m
        goal_tolerance: float = 0.2,   # m
    )

    def compute(
        self,
        current_pose: np.ndarray,  # (3,) [x, y, theta]
        path: np.ndarray,          # (N, 2) [[x, y], ...]
        velocity: float,           # 현재 선속도
    ) -> tuple[float, float]       # (target_velocity, steering_angle)

    def reset(self)
```

#### `compute_cross_track_error()`

```python
def compute_cross_track_error(
    position: np.ndarray,  # (2,) [x, y]
    path: np.ndarray,      # (N, 2)
) -> tuple[float, int]    # (CTE_signed, nearest_segment_index)
```

---

### 1.8 칼만 트래커 (`core/kalman_tracker.py`)

#### `ObjectTracker` 클래스

```python
class ObjectTracker:
    def __init__(
        self,
        dt: float = 0.1,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.3,
        max_misses: int = 5,
        match_threshold: float = 2.0,   # m
        min_hits: int = 3,
    )

    def update(self, detections: np.ndarray) -> list[dict]
    # detections: (M, 2) 검출 위치
    # 반환: [{'id', 'position', 'velocity', 'age'}, ...]

    def predict_future(self, obj_id, horizon, dt=None) -> np.ndarray
    # 반환: (N, 2) 미래 위치 배열

    def compute_ttc(
        self,
        robot_position: np.ndarray,  # (2,)
        robot_velocity: np.ndarray,  # (2,)
        robot_radius: float = 0.3,
        obstacle_radius: float = 0.3,
    ) -> list[dict]
    # 반환: [{'id', 'ttc', 'min_distance'}, ...]

    def get_all_tracks(self) -> list[TrackedObject]
```

---

## 2. Perception 모듈

### 2.1 LiDAR 처리 (`perception/lidar_processor.py`)

#### `LidarProcessor` 클래스

```python
class LidarProcessor:
    def __init__(
        self,
        min_range: float = 0.2,   # m
        max_range: float = 50.0,  # m
        min_height: float = -0.5, # m
        max_height: float = 3.0,  # m
    )

    def filter_by_range(self, cloud: PointCloud) -> PointCloud
    def filter_by_angle(self, cloud, min_angle_deg=-180, max_angle_deg=180) -> PointCloud
    def remove_outliers(self, cloud, k_neighbors=20, std_ratio=2.0) -> PointCloud
    def voxel_grid_downsample(self, cloud, voxel_size=0.1) -> PointCloud
    def dbscan_cluster(self, cloud, eps=0.5, min_samples=5) -> list[Cluster]
    def process(self, cloud, fov_min_deg=-120, fov_max_deg=120,
                voxel_size=0.1, cluster_eps=0.5,
                cluster_min_samples=5) -> list[Cluster]
```

#### 데이터 클래스

```python
@dataclass
class PointCloud:
    points: np.ndarray           # (N, 3) [x, y, z]
    intensities: np.ndarray | None  # (N,) 선택적

@dataclass
class Cluster:
    cluster_id: int
    points: np.ndarray           # (M, 3)
    centroid: np.ndarray         # (3,) 자동 계산
    bbox_min: np.ndarray         # (3,) 자동 계산
    bbox_max: np.ndarray         # (3,) 자동 계산
```

---

### 2.2 YOLO 객체 인식 (`perception/yolo_detector.py`)

#### `YoloDetector` 클래스

```python
class YoloDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        device: str | None = None,       # 'cuda' | 'cpu' | None(자동)
        class_mapping: dict | None = None,
    )

    def detect(self, image: np.ndarray) -> list[Detection]
    # image: (H, W, 3) BGR/RGB 이미지

    def detect_specific_class(self, image, target_class) -> list[Detection]

    @property
    def is_simulation(self) -> bool
```

**AMR 클래스 매핑**: COCO 클래스 → `Box`, `Person`, `Sign`

```python
@dataclass
class Detection:
    class_name: str        # 'Box' | 'Person' | 'Sign'
    confidence: float      # 0.0 ~ 1.0
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: tuple[float, float]              # (cx, cy)
```

---

### 2.3 카메라 투영 (`perception/camera_projection.py`)

#### `CameraProjector` 클래스

```python
class CameraProjector:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics | None = None,
    )

    def pixel_to_3d(self, u, v, depth) -> np.ndarray       # → (3,) 카메라 좌표
    def project_3d_to_pixel(self, point_3d) -> np.ndarray   # → (2,) 픽셀 좌표
    def camera_to_map(self, point_cam) -> np.ndarray        # → (3,) Map 좌표
    def map_to_camera(self, point_map) -> np.ndarray        # → (3,) 카메라 좌표
    def pixel_to_map(self, u, v, depth) -> np.ndarray       # → (3,) Map 좌표 (원스텝)
    def is_in_image(self, u, v) -> bool
```

```python
@dataclass
class CameraIntrinsics:
    fx: float       # x축 초점 거리 (px)
    fy: float       # y축 초점 거리 (px)
    cx: float       # x축 주점 (px)
    cy: float       # y축 주점 (px)
    width: int = 640
    height: int = 480

    @property
    def matrix(self) -> np.ndarray      # 3x3 K 행렬
    @property
    def inv_matrix(self) -> np.ndarray  # K^-1

@dataclass
class CameraExtrinsics:
    rotation: np.ndarray        # 3x3 회전 행렬
    translation: np.ndarray     # 3x1 이동 벡터

    @property
    def transform_matrix(self) -> np.ndarray      # 4x4 T
    @property
    def inv_transform_matrix(self) -> np.ndarray   # T^-1

    @classmethod
    def from_euler_angles(cls, roll, pitch, yaw, tx, ty, tz) -> CameraExtrinsics
```

---

## 3. Fleet 모듈

### 3.1 작업 할당 (`fleet/task_allocator.py`)

#### `TaskAllocator` 클래스

```python
class TaskAllocator:
    def __init__(
        self,
        distance_weight: float = 1.0,
        load_weight: float = 2.0,
        priority_weight: float = 1.5,
        deadline_weight: float = 3.0,
    )

    def compute_cost_matrix(self, tasks, robots) -> np.ndarray  # (n_robots, n_tasks)
    def hungarian_algorithm(self, cost_matrix) -> list[tuple[int, int]]
    def allocate(self, tasks, robots) -> list[Assignment]
```

```python
@dataclass
class Task:
    task_id: str
    position: tuple[float, float]
    priority: int = 3           # 1(최저)~5(최고)
    deadline: float | None = None  # Unix timestamp
    estimated_duration: float = 60.0  # 초
    task_type: str = "delivery"  # 'pickup' | 'delivery' | 'inspect'

@dataclass
class Robot:
    robot_id: str
    position: tuple[float, float]
    speed: float = 1.0          # m/s
    current_load: int = 0
    max_load: int = 5
    battery_level: float = 100.0
    available: bool = True

@dataclass
class Assignment:
    robot_id: str
    task_id: str
    cost: float
    estimated_arrival: float    # 초
```

---

### 3.2 교통 관리 (`fleet/traffic_manager.py`)

#### `TrafficManager` 클래스

```python
class TrafficManager:
    def __init__(
        self,
        safety_distance: float = 0.5,   # m
        time_resolution: float = 0.5,   # 초
    )

    # 경로/구간 관리
    def register_zone(self, zone: Zone)
    def register_path(self, path: RobotPath)

    # 충돌 관리
    def predict_conflicts(self, paths: list[RobotPath]) -> list[Conflict]
    def resolve_conflicts(self, conflicts, paths) -> list[Conflict]

    # 교차로 관리
    def request_zone_entry(self, zone_id, robot_id, priority=1) -> bool
    def release_zone(self, zone_id, robot_id)

    # 교착 관리
    def detect_deadlock(self) -> list[list[str]]      # 사이클 리스트
    def resolve_deadlock(self, cycles) -> list[str]    # 양보한 로봇 ID
    def get_zone_status(self) -> dict[str, dict]
    def clear_wait_graph(self)
```

```python
class ConflictType(Enum):
    HEAD_ON = auto()    # 정면 충돌
    CROSSING = auto()   # 교차 충돌
    REAR_END = auto()   # 추돌
    MERGE = auto()      # 합류 충돌

@dataclass
class Conflict:
    robot_a: str
    robot_b: str
    conflict_type: ConflictType
    position: tuple[float, float]
    time: float
    resolved: bool = False
    resolution: str = ""

@dataclass
class Zone:
    zone_id: str
    position: tuple[float, float]
    radius: float = 1.0
    capacity: int = 1
    current_occupants: list[str]
    queue: list[str]
```

---

### 3.3 Fleet 모니터링 (`fleet/monitor.py`)

#### `FleetMonitor` 클래스

```python
class FleetMonitor:
    def __init__(
        self,
        battery_warning_threshold: float = 30.0,
        battery_critical_threshold: float = 10.0,
        max_alerts: int = 100,
        task_history_size: int = 500,
    )

    # 로봇 관리
    def register_robot(self, robot_id, position=(0.0, 0.0))
    def update_state(self, robot_id, position=None, status=None,
                     battery_level=None, current_task=None, speed=None)

    # 작업/알림
    def record_task(self, task_id, robot_id, start_time, end_time, success=True)
    def report_deadlock(self, robot_ids: list[str])

    # KPI/대시보드
    def compute_kpi(self, time_window=3600.0) -> KPI
    def print_dashboard(self) -> str

    # 조회
    def get_robot_state(self, robot_id) -> RobotState | None
    def get_all_states(self) -> dict[str, RobotState]
    def get_alerts(self, level=None) -> list[Alert]
    def acknowledge_alert(self, alert_id) -> bool

    @property
    def robot_count(self) -> int
```

```python
class RobotStatus(Enum):
    IDLE = auto()
    MOVING = auto()
    WORKING = auto()
    CHARGING = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()

class AlertLevel(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()

@dataclass
class KPI:
    throughput: float         # tasks/hour
    avg_task_time: float      # 초
    robot_utilization: float  # 0.0 ~ 1.0
    total_completed: int
    total_failed: int
    success_rate: float       # 0.0 ~ 1.0
    avg_battery: float        # %
    fleet_availability: float # 0.0 ~ 1.0
```

---

## 4. Task 모듈

### 4.1 Behavior Tree (`task/behavior_tree.py`)

#### 노드 타입

| 노드 | 클래스 | 역할 |
|------|--------|------|
| Action | `ActionNode` | 동작 수행 (콜백 함수 실행) |
| Condition | `ConditionNode` | 조건 검사 (bool 반환) |
| Sequence | `SequenceNode` | 순서 실행 (AND 로직) |
| Selector | `SelectorNode` | 선택 실행 (OR 로직) |
| Parallel | `ParallelNode` | 병렬 실행 |
| Decorator | `DecoratorNode` | 결과 변형 (inverter, repeat, retry, force_success, force_failure) |

```python
class NodeStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()

class Blackboard:
    def get(self, key, default=None) -> Any
    def set(self, key, value)
    def has(self, key) -> bool
    def clear(self)

# AMR 미션 트리 구성 (20개 노드)
def build_amr_mission_tree(blackboard=None) -> TreeNode
def count_nodes(node: TreeNode) -> int
```

**사용 예시**:

```python
from task.behavior_tree import build_amr_mission_tree, Blackboard, NodeStatus

bb = Blackboard()
bb.set("battery_level", 80)
bb.set("target_position", (10.0, 5.0))

root = build_amr_mission_tree(bb)

for tick_num in range(10):
    status = root.tick()
    if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
        break
```

---

### 4.2 도킹 제어 (`task/docking.py`)

#### `DockingController` 클래스

```python
class DockingController:
    def __init__(
        self,
        position_tolerance: float = 0.02,   # m (2cm)
        angle_tolerance_deg: float = 1.0,    # 도
        max_attempts: int = 3,
        approach_speed: float = 0.3,         # m/s
        final_speed: float = 0.05,           # m/s
        kp_linear: float = 1.0,
        kd_linear: float = 0.1,
        kp_angular: float = 2.0,
        kd_angular: float = 0.2,
        dt: float = 0.05,                    # 초
    )

    def detect_aruco_marker(self, robot_pose, target) -> MarkerDetection
    def compute_control(self, marker, phase) -> tuple[float, float]  # (v, w)
    def update_pose(self, pose, v, w) -> RobotPose
    def check_docking_precision(self, robot_pose, target) -> tuple[float, float]
    # 반환: (위치오차_m, 각도오차_도)

    def execute_docking(self, target, initial_pose) -> DockingResult

    @property
    def state(self) -> DockingState
```

```python
class DockingState(Enum):
    IDLE = auto()
    APPROACHING = auto()
    ALIGNING = auto()
    FINAL_APPROACH = auto()
    DOCKED = auto()
    FAILED = auto()

@dataclass
class DockingResult:
    success: bool
    position_error: float   # m
    angle_error: float      # 도
    attempts: int
    final_state: DockingState
```

---

## 5. Simulation 모듈

### 5.1 물류센터 환경 (`simulation/warehouse_world.py`)

#### `WarehouseWorld` 클래스

```python
class WarehouseWorld:
    def __init__(
        self,
        width: float = 60.0,      # m
        height: float = 40.0,     # m
        resolution: float = 0.1,  # m/cell → 600x400 격자
    )

    # 공개 API
    def get_occupancy_grid(self) -> np.ndarray  # (400, 600) int8
    # 값: 0=빈공간, 1=장애물, 2=입고, 3=출고, 4=충전

    def spawn_dynamic_obstacles(
        self, num_workers=3, num_forklifts=2
    ) -> list[DynamicObstacle]

    def step(self, dt: float)              # 시뮬레이션 전진
    def is_free(self, x, y) -> bool
    def world_to_grid(self, x, y) -> tuple
    def grid_to_world(self, gx, gy) -> tuple

    # 속성
    shelf_positions: dict[str, tuple]      # 21개 선반 (A1~C7)
    charging_stations: list[tuple]          # 5개 충전 스테이션
    dynamic_obstacles: list[DynamicObstacle]
```

### 5.2 센서 노이즈 (`simulation/sensor_noise.py`)

| 클래스 | 파라미터 | 적용 대상 |
|--------|---------|----------|
| `LidarNoise` | `sigma=0.03m`, `miss_rate=0.01` | 거리 배열 |
| `IMUNoise` | `accel_sigma=0.02`, `gyro_sigma=0.005`, `bias_drift_rate=0.0001` | 가속도/각속도 |
| `EncoderNoise` | `slip_ratio=0.02`, `sigma=1.0 tick` | 인코더 틱 |
| `DepthNoise` | `base_sigma=0.01m`, `distance_factor=0.005` | 깊이 이미지 |

---

## 6. ROS2 노드 인터페이스

### 6.1 Launch 파일

| Launch 파일 | 용도 | 실행 노드 |
|------------|------|----------|
| `launch/simulation.launch.py` | Gazebo 시뮬레이션 | `robot_state_publisher`, `joint_state_publisher` |
| `launch/navigation.launch.py` | 단일 로봇 네비게이션 | `ekf_node`, `planner_server`, `controller_server`, `behavior_server`, `lifecycle_manager` |
| `launch/fleet.launch.py` | 다중 로봇 Fleet | 5x navigation + `fleet_manager` |

### 6.2 ROS2 토픽

| 토픽 | 메시지 타입 | 발행자 | 구독자 |
|------|-----------|--------|--------|
| `/amr/scan` | `sensor_msgs/LaserScan` | Gazebo LiDAR 플러그인 | obstacle_layer, lidar_processor |
| `/amr/odom` | `nav_msgs/Odometry` | Gazebo diff_drive 플러그인 | EKF, AMCL |
| `/amr/imu` | `sensor_msgs/Imu` | Gazebo IMU 플러그인 | EKF |
| `/amr/depth_camera/*` | `sensor_msgs/Image`, `PointCloud2` | Gazebo 카메라 플러그인 | yolo_detector, camera_projector |
| `/amr/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | AMCL | EKF |
| `/amr/cmd_vel` | `geometry_msgs/Twist` | controller_server | Gazebo diff_drive |

### 6.3 Launch 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `use_sim_time` | `true` | 시뮬레이션 시간 사용 |
| `num_robots` | `1` (simulation), `5` (fleet) | 로봇 수 |
| `namespace` | `amr` | ROS 네임스페이스 |

---

## 7. 설정 파일 레퍼런스

### 7.1 설정 파일 목록

| 파일 | 설명 |
|------|------|
| `config/robot_params.yaml` | 로봇 물리 파라미터 (치수, 속도, 배터리 등) |
| `config/nav2_params.yaml` | Nav2 네비게이션 파라미터 (Costmap, DWA, A* 등) |
| `config/ekf_params.yaml` | EKF 센서 퓨전 설정 (robot_localization 형식) |
| `config/fleet_config.yaml` | Fleet 관리 설정 (로봇 구성, 작업 할당, 교통 관리) |

### 7.2 주요 설정 값 요약

```yaml
# robot_params.yaml - 핵심 파라미터
robot:
  wheel_radius: 0.05      # m
  wheel_base: 0.3          # m
  ticks_per_rev: 4096
  max_velocity: 1.0        # m/s
  max_acceleration: 0.5    # m/s^2
  max_angular_velocity: 1.5  # rad/s
  battery:
    low_threshold: 20.0    # %
    critical_threshold: 10.0  # %

# nav2_params.yaml - 핵심 파라미터
local_costmap:
  resolution: 0.05         # m/cell
  inflation_radius: 0.55   # m
controller_server:
  max_vel_x: 1.0           # m/s
  max_vel_theta: 1.5       # rad/s
  sim_time: 1.7            # 초
planner_server:
  use_astar: true
  allow_diagonal: true
  weight: 1.2              # 휴리스틱 가중치

# fleet_config.yaml - 핵심 파라미터
fleet:
  num_robots: 5
  task_allocation:
    algorithm: hungarian
    max_tasks_per_robot: 5
  traffic:
    safety_distance: 1.0   # m
    deadlock:
      detection_timeout: 30.0  # 초
```

### 7.3 CLI 사용법

```bash
# 데모 모드 (ROS2 불필요)
python main.py demo

# 테스트 실행
python main.py test

# ROS2 모드 (Docker 환경 권장)
python main.py simulation
python main.py navigation
python main.py fleet
```
