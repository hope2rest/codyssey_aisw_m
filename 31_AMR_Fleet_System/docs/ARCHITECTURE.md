# AMR Fleet System 아키텍처 문서

## 목차

1. [4-Layer 아키텍처 개요](#1-4-layer-아키텍처-개요)
2. [컴포넌트 다이어그램](#2-컴포넌트-다이어그램)
3. [시퀀스 다이어그램](#3-시퀀스-다이어그램)
4. [데이터 흐름](#4-데이터-흐름)
5. [TF 트리 구조](#5-tf-트리-구조)

---

## 1. 4-Layer 아키텍처 개요

본 시스템은 센서 입력부터 Fleet 관리까지를 4개 계층으로 구분한다.

```
+=========================================================================+
||                    Layer 4: Fleet Management                           ||
||  +-------------------+  +-------------------+  +-------------------+  ||
||  | Task Allocator    |  | Traffic Manager   |  | Fleet Monitor     |  ||
||  | (Hungarian Alg.)  |  | (Deadlock Detect) |  | (KPI, Dashboard)  |  ||
||  +-------------------+  +-------------------+  +-------------------+  ||
+=========================================================================+
         |                        |                        |
         v                        v                        v
+=========================================================================+
||                    Layer 3: Motion Control                              ||
||  +------------+  +-----------+  +-------------+  +------------------+ ||
||  | A* Planner |  | DWA Local |  | Pure Pursuit |  | PID Controller  | ||
||  | (Global)   |  | (Local)   |  | (Path Track) |  | (Speed Profile) | ||
||  +------------+  +-----------+  +-------------+  +------------------+ ||
||  +------------+  +------------------+  +----------------------------+ ||
||  | Costmap    |  | Behavior Tree    |  | Docking Controller         | ||
||  | (2D Grid)  |  | (Mission Logic)  |  | (ArUco, PD, State Machine)| ||
||  +------------+  +------------------+  +----------------------------+ ||
+=========================================================================+
         |                        |                        |
         v                        v                        v
+=========================================================================+
||                    Layer 2: Perception                                  ||
||  +------------------+  +------------------+  +----------------------+ ||
||  | LiDAR Processor  |  | YOLO Detector    |  | Camera Projector     | ||
||  | (DBSCAN, SOR,    |  | (YOLOv8, Box/    |  | (Pinhole Model,     | ||
||  |  Voxel Grid)     |  |  Person/Sign)    |  |  2D-3D-Map 변환)     | ||
||  +------------------+  +------------------+  +----------------------+ ||
||  +------------------+  +-------------------------------------------+  ||
||  | Kalman Tracker   |  | Sensor Fusion (EKF)                       |  ||
||  | (Multi-Object,   |  | [x, y, theta, v, omega]                   |  ||
||  |  TTC, Predict)   |  | Odom + IMU + AMCL                        |  ||
||  +------------------+  +-------------------------------------------+  ||
+=========================================================================+
         |                        |                        |
         v                        v                        v
+=========================================================================+
||                    Layer 1: Sensor & Robot                              ||
||  +-------------+  +----------+  +---------+  +----+  +--------------+ ||
||  | LiDAR       |  | Depth    |  | IMU     |  | Wheel Encoders     | ||
||  | (360, 12m)  |  | Camera   |  | (100Hz) |  | (4096 ticks/rev)   | ||
||  +-------------+  +----------+  +---------+  +--------------------+ ||
||  +------------------+  +-------------------------------------------+  ||
||  | Diff. Drive      |  | Sensor Noise Models                      |  ||
||  | Kinematics       |  | (LiDAR, IMU, Encoder, Depth)             |  ||
||  +------------------+  +-------------------------------------------+  ||
+=========================================================================+
```

### 계층별 역할

| 계층 | 역할 | 주요 모듈 | 실행 주기 |
|------|------|----------|----------|
| **Layer 1: Sensor & Robot** | 하드웨어 추상화, 센서 데이터 수집, 기구학 계산 | `kinematics.py`, `sensor_noise.py`, URDF | 50~100 Hz |
| **Layer 2: Perception** | 환경 인식, 위치 추정, 객체 추적 | `lidar_processor.py`, `yolo_detector.py`, `camera_projection.py`, `ekf.py`, `kalman_tracker.py` | 10~50 Hz |
| **Layer 3: Motion Control** | 경로 계획, 경로 추종, 미션 실행 | `astar.py`, `dwa.py`, `pure_pursuit.py`, `pid_controller.py`, `costmap.py`, `behavior_tree.py`, `docking.py` | 10~20 Hz |
| **Layer 4: Fleet Management** | 다중 로봇 조율, 작업 할당, 교통 관리 | `task_allocator.py`, `traffic_manager.py`, `monitor.py` | 1~5 Hz |

---

## 2. 컴포넌트 다이어그램

### 2.1 전체 노드 및 토픽 구성

```
+---------------------------+         +---------------------------+
|     Gazebo Simulation     |         |     Sensor Drivers        |
|  (warehouse_world.py)     |         |  (URDF Gazebo Plugins)    |
+---------------------------+         +---------------------------+
   |          |          |               |         |         |
   | /amr/scan | /amr/depth_camera     | /amr/odom | /amr/imu
   |          | /amr/camera/image_raw  |         |         |
   v          v                        v         v         v
+------------------------------------------------------------------+
|                    Perception Nodes                                |
|                                                                    |
|  +-----------------+    +------------------+    +---------------+  |
|  | lidar_processor |    | yolo_detector    |    | camera_proj   |  |
|  |  /amr/scan ─────|──> |  /amr/camera ────|──> |  pixel_to_map |  |
|  |  └→ /obstacles  |    |  └→ /detections  |    |  └→ /det_3d   |  |
|  +-----------------+    +------------------+    +---------------+  |
|          |                       |                      |          |
|          v                       v                      v          |
|  +-----------------+    +------------------+                       |
|  | kalman_tracker  |    | EKF Sensor       |                       |
|  |  /obstacles ────|    | Fusion           |                       |
|  |  └→ /tracked    |    |  /amr/odom ──────|                       |
|  |  └→ /predicted  |    |  /amr/imu ───────|                       |
|  +-----------------+    |  /amr/amcl_pose ─|                       |
|                         |  └→ /amr/pose    |                       |
|                         +------------------+                       |
+------------------------------------------------------------------+
            |                    |                    |
            v                    v                    v
+------------------------------------------------------------------+
|                    Motion Control Nodes                             |
|                                                                    |
|  +-----------+  +-----------+  +-------------+  +--------------+  |
|  | costmap   |  | A* planner|  | DWA local   |  | pure_pursuit |  |
|  | /obstacles|  | /costmap ─|  | /goal ──────|  | /path ───────|  |
|  | └→/costmap|  | └→ /path  |  | └→ /cmd_vel |  | └→ /cmd_vel  |  |
|  +-----------+  +-----------+  +-------------+  +--------------+  |
|                                                                    |
|  +------------------+    +-------------------+                     |
|  | behavior_tree    |    | docking_controller|                     |
|  | (mission logic)  |    | /dock_target ─────|                     |
|  | └→ /mission_stat |    | └→ /cmd_vel       |                     |
|  +------------------+    | └→ /dock_status   |                     |
|                          +-------------------+                     |
+------------------------------------------------------------------+
            |                    |                    |
            v                    v                    v
+------------------------------------------------------------------+
|                    Fleet Management Nodes                           |
|                                                                    |
|  +-------------------+  +-------------------+  +----------------+  |
|  | task_allocator    |  | traffic_manager   |  | fleet_monitor  |  |
|  | /task_requests ───|  | /robot_paths ─────|  | /robot_states ─|  |
|  | /robot_states ────|  | └→ /zone_status   |  | └→ /kpi        |  |
|  | └→ /assignments   |  | └→ /conflicts     |  | └→ /alerts     |  |
|  +-------------------+  | └→ /deadlocks     |  | └→ /dashboard  |  |
|                         +-------------------+  +----------------+  |
+------------------------------------------------------------------+
```

### 2.2 모듈 의존성

```
fleet/
  task_allocator.py ──── numpy
  traffic_manager.py ─── numpy
  monitor.py ─────────── numpy

core/
  astar.py ───────────── numpy
  costmap.py ─────────── numpy, scipy (ndimage)
  dwa.py ─────────────── numpy
  ekf.py ─────────────── numpy
  kalman_tracker.py ──── numpy
  kinematics.py ──────── numpy
  pid_controller.py ──── numpy
  pure_pursuit.py ────── numpy

perception/
  lidar_processor.py ─── numpy
  yolo_detector.py ───── numpy, ultralytics (선택적)
  camera_projection.py ─ numpy

task/
  behavior_tree.py ───── (표준 라이브러리만)
  docking.py ─────────── numpy

simulation/
  warehouse_world.py ─── numpy
  sensor_noise.py ────── numpy
```

---

## 3. 시퀀스 다이어그램

### 3.1 시나리오 1: 작업 할당 → 네비게이션 → 도킹 → 복귀

```
    FleetMonitor   TaskAllocator   BehaviorTree   A*Planner   DWA/PurePursuit   DockingCtrl   Robot
        |               |              |              |              |              |          |
        |  작업 요청 수신  |              |              |              |              |          |
        |<──────────────|              |              |              |              |          |
        |               |              |              |              |              |          |
        | 로봇 상태 조회   |              |              |              |              |          |
        |──────────────>|              |              |              |              |          |
        |               |              |              |              |              |          |
        |               | 비용 행렬 계산  |              |              |              |          |
        |               |──────┐       |              |              |              |          |
        |               |      | Hungarian            |              |              |          |
        |               |<─────┘ Algorithm            |              |              |          |
        |               |              |              |              |              |          |
        |  할당 결과      |   작업 할당    |              |              |              |          |
        |<──────────────|──────────── >|              |              |              |          |
        |               |              |              |              |              |          |
        | 상태: MOVING   |              | [Sequence]   |              |              |          |
        |<──────────────|──────────── ─|─ 배터리 확인   |              |              |          |
        |               |              |─ 작업 대기    |              |              |          |
        |               |              |              |              |              |          |
        |               |              | 경로 계획 요청  |              |              |          |
        |               |              |─────────────>| plan_path()  |              |          |
        |               |              |              |──────┐       |              |          |
        |               |              |              |      | A* 탐색 |              |          |
        |               |              |              |      | + 평활화 |              |          |
        |               |              |              |<─────┘       |              |          |
        |               |              |  경로 반환    |              |              |          |
        |               |              |<─────────────|              |              |          |
        |               |              |              |              |              |          |
        |               |              | [이동 루프]    |              |              |          |
        |               |              |──────────────|──────────── >| dwa_planning()|          |
        |               |              |              |              |──────┐       |          |
        |               |              |              |              |      | 궤적 평가|          |
        |               |              |              |              |<─────┘       |          |
        |               |              |              |              | (v, omega)   |          |
        |               |              |              |              |─────────────>| cmd_vel  |
        |               |              |              |              |              |──────── >|
        |               |              |              |              |              |          |
        | 위치 업데이트   |              | 목표 근접 확인 |              |              |          |
        |<──────────────|──────────── ─|──────────────|──────────── ─|              |          |
        |               |              |              |              |              |          |
        |               |              | [도킹 시작]   |              |              |          |
        |               |              |─────────────────────────── >| execute_     |          |
        | 상태: WORKING  |              |              |              | docking()    |          |
        |<──────────────|              |              |              |──────┐       |          |
        |               |              |              |              |      | ArUco   |          |
        |               |              |              |              |      | 인식     |          |
        |               |              |              |              |      | PD 제어  |          |
        |               |              |              |              |<─────┘       |          |
        |               |              |  도킹 완료    |              | DOCKED       |          |
        |               |              |<─────────────|──────────── ─|              |          |
        |               |              |              |              |              |          |
        |               |              | [화물 처리]   |              |              |          |
        |               |              |──────┐       |              |              |          |
        |               |              |<─────┘       |              |              |          |
        |               |              |              |              |              |          |
        |               |              | [홈 복귀]     | plan_path()  |              |          |
        |               |              |─────────────>| (홈 좌표)     |              |          |
        |               |              |              |──────────── >| 이동 실행     |          |
        |               |              |              |              |─────────────>|          |
        |               |              |              |              |              |          |
        | 작업 완료 기록  |              | SUCCESS      |              |              |          |
        |<──────────────|──────────── ─|              |              |              |          |
        | 상태: IDLE     |              |              |              |              |          |
        |               |              |              |              |              |          |
```

### 3.2 시나리오 2: 동적 장애물 탐지 → 예측 → 회피

```
    LiDAR     LidarProcessor   KalmanTracker   DWA    CostMap    Robot
      |            |                |            |        |         |
      | raw scan   |                |            |        |         |
      |───────────>|                |            |        |         |
      |            |                |            |        |         |
      |            | [파이프라인]     |            |        |         |
      |            | 1. 거리 필터    |            |        |         |
      |            | 2. 각도 필터    |            |        |         |
      |            | 3. SOR 아웃라이어|            |        |         |
      |            | 4. Voxel 다운샘플|            |        |         |
      |            | 5. DBSCAN 클러스터|           |        |         |
      |            |                |            |        |         |
      |            | 클러스터 중심    |            |        |         |
      |            | (장애물 위치)    |            |        |         |
      |            |───────────────>|            |        |         |
      |            |                |            |        |         |
      |            |                | [추적 갱신]  |        |         |
      |            |                | 1. 전 트랙 예측       |         |
      |            |                |    F @ state         |         |
      |            |                | 2. 비용 행렬 계산     |         |
      |            |                |    (유클리드 거리)     |         |
      |            |                | 3. 탐욕적 매칭        |         |
      |            |                | 4. 매칭 트랙 갱신     |         |
      |            |                |    (칼만 갱신)        |         |
      |            |                | 5. 미매칭 → 새 트랙   |         |
      |            |                | 6. 오래된 트랙 삭제   |         |
      |            |                |            |        |         |
      |            |                | 확정 트랙 반환        |         |
      |            |                | (id, pos, vel)      |         |
      |            |                |            |        |         |
      |            |                | [미래 예측]  |        |         |
      |            |                | predict_future()    |         |
      |            |                | (등속 직선 모델)      |         |
      |            |                |────────── >|        |         |
      |            |                |            |        |         |
      |            |                | [TTC 계산]  |        |         |
      |            |                | compute_ttc()       |         |
      |            |                |     ┌──────────────>|         |
      |            |                |     | 장애물 위치    |         |
      |            |                |     | update_obstacles()      |
      |            |                |     |        |──────┐         |
      |            |                |     |        | Inflation      |
      |            |                |     |        | 재계산          |
      |            |                |     |        |<─────┘         |
      |            |                |     |        |                |
      |            |                |     |        | 갱신된 비용맵   |
      |            |                |     |        |────── >|       |
      |            |                |            |        |         |
      |            |                |            | [DWA 실행]       |
      |            |                |            | 1. Dynamic Window|
      |            |                |            | 2. 궤적 시뮬레이션|
      |            |                |            | 3. 충돌 궤적 제외 |
      |            |                |            |    (clearance    |
      |            |                |            |     < robot_rad) |
      |            |                |            | 4. 최적 (v, w)   |
      |            |                |            |────────────────>|
      |            |                |            |        |  cmd_vel|
      |            |                |            |        |         |
```

### 3.3 시나리오 3: 교착(Deadlock) 탐지 → 해소

```
    Robot_A     Robot_B     Robot_C     TrafficMgr     FleetMonitor
       |           |           |            |               |
       | 경로 등록  |           |            |               |
       |──────────────────────────────────>|               |
       |           | 경로 등록  |            |               |
       |           |──────────────────────>|               |
       |           |           | 경로 등록   |               |
       |           |           |───────── >|               |
       |           |           |            |               |
       |           |           |  [충돌 예측]|               |
       |           |           |  predict_conflicts()       |
       |           |           |            |──────┐        |
       |           |           |            |      | 시간-공간|
       |           |           |            |      | 충돌 검사|
       |           |           |            |<─────┘        |
       |           |           |            |               |
       |           |           |  [충돌 해소]|               |
       |           |           |  resolve_conflicts()       |
       |           |           |            |               |
       | 양보(대기) |           |            |               |
       |<─────────────────────────────────|               |
       |           |           |            |               |
       | Zone 진입 요청         |            |               |
       |──────────────────────────────────>|               |
       |           | Zone 진입 요청          |               |
       |           |──────────────────────>|               |
       |           |           |            |               |
       |           |           |            | [Wait-For Graph 갱신]
       |           |           |            | A → B (A가 B를 대기)
       |           |           |            | B → C (B가 C를 대기)
       |           |           |            | C → A (C가 A를 대기)
       |           |           |            |               |
       |           |           |            | [교착 탐지]    |
       |           |           |            | detect_deadlock()
       |           |           |            |──────┐        |
       |           |           |            |      | DFS    |
       |           |           |            |      | 사이클  |
       |           |           |            |      | 탐지   |
       |           |           |            |<─────┘        |
       |           |           |            |               |
       |           |           |            | 사이클 발견:    |
       |           |           |            | A → B → C → A |
       |           |           |            |               |
       |           |           |            | [교착 해소]    |
       |           |           |            | resolve_deadlock()
       |           |           |            |               |
       |           |           |            | 최저 우선순위   |
       |           |           |            | 로봇 선택      |
       |           |           |            | (예: Robot_C)  |
       |           |           |            |               |
       |           |           | Wait-For   |               |
       |           |           | 관계 해제   |               |
       |           |           |<──────────|               |
       |           |           |            |               |
       |           |           | Zone 대기열 |               |
       |           |           | 에서 제거   |               |
       |           |           |<──────────|               |
       |           |           |            |               |
       |           |           |            | 교착 보고      |
       |           |           |            |──────────────>|
       |           |           |            |               |
       |           |           |            |     [CRITICAL 알림]
       |           |           |            |     report_deadlock()
       |           |           |            |               |
       | Zone 진입 허용         |            |               |
       |<─────────────────────────────────|               |
       | 이동 재개  |           |            |               |
       |──────────>|           |            |               |
       |           | Zone 진입 허용          |               |
       |           |<─────────────────────|               |
       |           | 이동 재개  |            |               |
       |           |──────────>|            |               |
       |           |           |            |               |
```

---

## 4. 데이터 흐름

### 4.1 센서 데이터 → 제어 명령 파이프라인

```
[센서 입력]                [처리 단계]              [출력]

Wheel Encoder ──────> Kinematics ─────> Odometry [x, y, theta]
(L/R ticks)           (DifferentialDriveRobot)       │
                                                      ├──> EKF ──> Fused Pose
IMU ────────────────> (theta, omega) ─────────────────┤    [x, y, theta, v, omega]
(accel, gyro)                                         │         │
                                                      │         │
AMCL ───────────────> (x, y, theta) ──────────────────┘         │
(particle filter)                                               │
                                                                v
LiDAR Scan ──────────> LidarProcessor ──> Clusters ──> KalmanTracker
(360 beams, 12m)       (Filter → SOR     (centroid,    (tracked objects,
                        → Voxel → DBSCAN)  bbox)        velocity, TTC)
                                                            │
                                                            v
Camera Image ────────> YoloDetector ───> Detections ──> CameraProjector
(640x480, RGB)         (Box/Person/Sign,  (class,       (pixel → 3D camera
                        confidence)        bbox)         → Map coordinates)
                                                            │
                                                            v
                                              ┌─────────────┴──────────────┐
                                              v                            v
                                          Costmap                    Obstacle Array
                                          (Static + Obstacle         (for DWA)
                                           + Inflation)
                                              │                            │
                                              v                            v
                                          A* Planner ──> Global Path ──> DWA ──> (v, omega)
                                                                           │
                                                           Pure Pursuit ──>├──> cmd_vel
                                                           (path tracking) │
                                                                           v
                                                                     PID Controller
                                                                     + Trapezoidal
                                                                       Profile
                                                                           │
                                                                           v
                                                                     Motor Command
                                                                     (left_vel,
                                                                      right_vel)
```

### 4.2 Fleet 데이터 흐름

```
[외부 입력]              [Fleet 처리]              [출력]

작업 요청 ──────────> TaskAllocator              할당 결과
(task_id, position,   ├── compute_cost_matrix()  (robot_id → task_id,
 priority, deadline)  │   (거리+부하+우선순위      cost, ETA)
                      │    +마감 가중합)               │
로봇 상태 ──────────>├── hungarian_algorithm()        │
(robot_id, position,  │   (Kuhn-Munkres 최적 매칭)     │
 speed, load,         └── allocate()                   │
 battery, available)       │                           │
                           v                           v
                     TrafficManager               FleetMonitor
                     ├── predict_conflicts()      ├── update_state()
                     │   (시간-공간 충돌 검사)      │   (위치, 배터리, 상태)
로봇 경로 ──────────>├── resolve_conflicts()      ├── record_task()
(segments with       │   (우선순위 양보)           │   (완료 기록)
 time_enter/exit)    ├── request_zone_entry()     ├── _check_anomalies()
                     │   (교차로 진입 관리)        │   (긴급정지, 오류, 배터리)
                     ├── detect_deadlock()        ├── compute_kpi()
                     │   (DFS 사이클 탐지)        │   (throughput, utilization,
                     └── resolve_deadlock()       │    success_rate, battery)
                         (최저 우선순위 양보)       └── print_dashboard()
                                                      (텍스트 기반 콘솔 출력)
```

---

## 5. TF 트리 구조

### 5.1 좌표 프레임 트리

```
                    map
                     │
                     │ (AMCL이 발행)
                     │ 위치 추정 기반 변환
                     │
                    odom
                     │
                     │ (Odometry가 발행)
                     │ 연속적 변환 (드리프트 누적)
                     │
                base_footprint
                     │
                     │ (fixed, z=0 투영)
                     │
                 base_link
                /    |    \        \
               /     |     \        \
              /      |      \        \
    left_wheel  right_wheel  front_caster  rear_caster
             |       |
        lidar_link  camera_link    imu_link
                        |
                 camera_optical_frame
```

### 5.2 좌표 프레임 설명

| 프레임 | 설명 | 부모 | 변환 유형 |
|--------|------|------|----------|
| **map** | 전역 고정 프레임. 맵 원점 기준의 절대 좌표계. AMCL/EKF가 odom→map 변환을 발행하여 드리프트를 보정 | (월드 기준) | 고정 |
| **odom** | 오도메트리 원점 프레임. 로봇 시작 위치 기준. 연속적이지만 드리프트가 누적됨 | map | 비연속 (AMCL 보정 시 점프) |
| **base_footprint** | 로봇 바닥면 투영 프레임. z=0 (지면). 2D 네비게이션에서 기준 프레임으로 사용 | odom | 연속 (Odometry) |
| **base_link** | 로봇 본체 중심 프레임. 모든 센서/바퀴가 이 프레임에 고정 연결 | base_footprint | fixed |
| **left_wheel** | 왼쪽 구동 바퀴. base_link에서 y=+0.15m 위치. continuous joint (회전) | base_link | continuous |
| **right_wheel** | 오른쪽 구동 바퀴. base_link에서 y=-0.15m 위치. continuous joint (회전) | base_link | continuous |
| **front_caster** | 전방 캐스터. base_link에서 x=+0.25m 위치. 자유 회전 | base_link | fixed |
| **rear_caster** | 후방 캐스터. base_link에서 x=-0.25m 위치. 자유 회전 | base_link | fixed |
| **lidar_link** | LiDAR 센서 프레임. base_link에서 x=+0.15m, z=+0.2m 위치. 360도 스캔 기준 | base_link | fixed |
| **camera_link** | Depth Camera 프레임. base_link에서 x=+0.18m, z=+0.25m 위치 | base_link | fixed |
| **camera_optical_frame** | 카메라 광학 프레임. camera_link에서 회전 (z-forward → x-right, y-down 변환). ROS 카메라 관례 준수 | camera_link | fixed (rpy=-90, 0, -90) |
| **imu_link** | IMU 센서 프레임. base_link에서 z=+0.1m 위치. 로봇 본체 중앙 | base_link | fixed |

### 5.3 주요 좌표 변환 관계

```
[map → odom]
    AMCL이 발행. 파티클 필터 기반 전역 위치 추정.
    드리프트 보정 역할. 이 변환이 점프하면 kidnapped robot 상황.
    EKF가 robot_localization으로 이 변환을 부드럽게 보간.

[odom → base_footprint]
    Wheel Odometry (차동 구동 기구학)가 발행.
    x' = x + d_center * cos(mid_theta)
    y' = y + d_center * sin(mid_theta)
    theta' = theta + d_theta
    연속적이지만 시간이 지남에 따라 실제 위치와 벌어짐.

[base_link → sensor_frames]
    URDF에 정의된 고정 변환.
    robot_state_publisher 노드가 발행.
    센서 데이터를 base_link 기준으로 변환하는 데 사용.

[camera_link → camera_optical_frame]
    ROS 카메라 관례: optical frame은 z-forward, x-right, y-down.
    회전: rpy = (-pi/2, 0, -pi/2)
    이미지 처리 결과를 로봇 좌표계로 변환할 때 필수.
```

### 5.4 비용맵 좌표 변환

```
[월드 좌표 (m)] ←──→ [격자 좌표 (cell)]

world_to_grid(wx, wy):
    col = (wx - origin_x) / resolution
    row = (wy - origin_y) / resolution
    return (row, col)

grid_to_world(row, col):
    wx = col * resolution + origin_x
    wy = row * resolution + origin_y
    return (wx, wy)

설정값:
    Costmap:     resolution = 0.05 m/cell, 200x200 cells = 10m x 10m
    Warehouse:   resolution = 0.10 m/cell, 600x400 cells = 60m x 40m
    Local Costmap: resolution = 0.05, 6m x 6m (rolling window)
    Global Costmap: resolution = 0.10, 전체 맵
```
