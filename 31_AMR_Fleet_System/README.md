# 31. 지능형 물류 AMR 시스템 구축

ROS2 Humble 기반 다중 AMR(Autonomous Mobile Robot) 물류센터 운용 시스템.
차동 구동 로봇 5대가 60m x 40m 물류센터에서 자율 주행, 피킹, 운반 작업을 수행한다.

> **320시간 커리큘럼** 전 과정을 코드로 구현 — SLAM, Navigation, Perception, Fleet 관리, 웹 대시보드까지 포함.

---

## 시스템 아키텍처

```
+------------------------------------------------------------------+
|                     Web Dashboard (:8080)                          |
|  (실시간 모니터링 / KPI / 작업 큐 / 창고 맵 시각화)               |
+------------------------------------------------------------------+
|                        Fleet Manager                              |
|  (Hungarian 할당 / 교착 탐지 / 교통 관리 / Safety 모니터링)       |
+-----+--------+--------+--------+--------+-----------------------+
      |        |        |        |        |
  +---v---+ +--v---+ +-v----+ +-v----+ +-v----+
  |AMR_01 | |AMR_02| |AMR_03| |AMR_04| |AMR_05|
  +---+---+ +--+---+ +--+---+ +--+---+ +--+---+
      |        |        |        |        |
      v        v        v        v        v
+------------------------------------------------------------------+
|                     Navigation Stack                              |
|  +--------+  +-------+  +-------+  +--------+  +----------+     |
|  | SLAM   |  | EKF   |  | A*    |  | DWA    |  | Pure     |     |
|  | (ICP + |  | Sensor|  | Global|  | Local  |  | Pursuit  |     |
|  | Pose   |  | Fusion|  | Plan  |  | Plan   |  | Tracking |     |
|  | Graph) |  |       |  |       |  |        |  |          |     |
|  +--------+  +-------+  +-------+  +--------+  +----------+     |
+------------------------------------------------------------------+
|                     Perception                                    |
|  +----------+  +-----------+  +------------------+               |
|  | LiDAR    |  | Depth     |  | YOLO Object      |               |
|  | 처리     |  | Camera    |  | Detection (v8n)  |               |
|  +----------+  +-----------+  +------------------+               |
+------------------------------------------------------------------+
|                     Simulation                                    |
|  +------------------+  +-------------------+                     |
|  | Warehouse World  |  | Sensor Noise      |                     |
|  | (60m x 40m)      |  | (LiDAR/IMU/Enc/  |                     |
|  | 선반/통로/충전    |  |  Depth)           |                     |
|  +------------------+  +-------------------+                     |
+------------------------------------------------------------------+
```

---

## TF 트리 구조

```
map
 └── odom                    (EKF: odom + IMU 퓨전)
      └── base_footprint
           └── base_link     (로봇 본체: 600x400x300mm)
                ├── left_wheel
                ├── right_wheel
                ├── front_caster
                ├── rear_caster
                ├── lidar_link       (x:0.15, z:0.2)
                ├── camera_link      (x:0.18, z:0.25)
                │    └── camera_optical_frame
                └── imu_link         (z:0.1)
```

---

## 실행 방법

### Demo 모드 (ROS2 불필요)

```bash
pip install numpy scipy

# 핵심 알고리즘 데모 (기구학, A*, DWA, EKF, SLAM, Fleet 할당)
python main.py demo

# 전체 테스트 (75개)
python main.py test
# 또는
pytest tests/ -v
```

### 웹 대시보드

```bash
# 외부 의존성 없이 순수 Python으로 실행
python dashboard/app.py

# 브라우저에서 http://localhost:8888 접속
# 5대 로봇 실시간 시뮬레이션, KPI, 작업 큐, 이상 알림 모니터링
```

### Docker 환경 (전체 시스템)

```bash
docker-compose build
docker-compose up simulation     # Gazebo 시뮬레이션
docker-compose up navigation     # 네비게이션
docker-compose up amr_fleet      # Fleet 전체 실행
```

### ROS2 직접 실행

```bash
ros2 launch amr_fleet_system simulation.launch.py
ros2 launch amr_fleet_system navigation.launch.py
ros2 launch amr_fleet_system fleet.launch.py num_robots:=5
```

---

## 파일 구조

```
31_AMR_Fleet_System/
├── main.py                              # CLI 진입점 (demo/test 모드)
├── Dockerfile                           # ROS2 Humble 기반 Docker
├── docker-compose.yml                   # GPU 가속 + 볼륨 마운트
│
├── core/                                # 핵심 알고리즘 (NumPy 직접 구현)
│   ├── kinematics.py                    #   차동 구동 순기구학 + 오도메트리
│   ├── ekf.py                           #   EKF 센서 퓨전 (odom + IMU + AMCL)
│   ├── astar.py                         #   A* 경로 계획 (8방향 + 평활화)
│   ├── dwa.py                           #   DWA 로컬 경로 계획
│   ├── pure_pursuit.py                  #   Pure Pursuit 경로 추종
│   ├── pid_controller.py                #   PID 속도 제어
│   ├── kalman_tracker.py                #   칼만 필터 기반 객체 추적
│   ├── costmap.py                       #   2D 비용맵 (Static + Inflation + Obstacle)
│   └── slam.py                          #   ★ SLAM (ICP + PoseGraph + OccupancyGrid)
│
├── fleet/                               # Fleet 관리
│   ├── task_allocator.py                #   작업 할당 (Hungarian/Greedy/Auction)
│   ├── traffic_manager.py               #   교통 관리 + 교착 탐지 (Wait-For Graph)
│   └── monitor.py                       #   실시간 모니터링
│
├── perception/                          # 인지 모듈
│   ├── lidar_processor.py               #   LiDAR 포인트 클라우드 처리
│   ├── yolo_detector.py                 #   YOLOv8 객체 인식
│   └── camera_projection.py             #   Pinhole 카메라 2D→3D 투영 변환
│
├── task/                                # 작업 관리
│   ├── behavior_tree.py                 #   Behavior Tree 행동 계획
│   └── docking.py                       #   도킹 제어
│
├── simulation/                          # 시뮬레이션 환경
│   ├── warehouse_world.py               #   60m x 40m 물류센터 (occupancy grid)
│   └── sensor_noise.py                  #   센서 노이즈 모델 (LiDAR/IMU/Encoder/Depth)
│
├── ros2_nodes/                          # ★ ROS2 노드 (rclpy 기반)
│   ├── slam_node.py                     #   SLAM 노드 (/scan → /map, /slam_pose)
│   ├── navigation_node.py               #   내비게이션 노드 (A*→DWA→Pure Pursuit)
│   ├── perception_node.py               #   인지 노드 (YOLOv8 + Depth → 3D 마커)
│   ├── fleet_manager_node.py            #   Fleet 관리 노드 (할당 + 교착 탐지)
│   └── safety_node.py                   #   안전 노드 (E-Stop + Safety Zone)
│
├── dashboard/                           # ★ 웹 모니터링 대시보드 (한국어 UI)
│   └── app.py                           #   순수 Python HTTP 서버 (외부 의존성 없음)
│                                        #   로봇 상태 카드, KPI 6종, 이상 알림, 작업 큐
│
├── robot_description/                   # 로봇 모델
│   └── amr_robot.urdf.xacro            #   차동 구동 AMR URDF (xacro 매크로)
│
├── config/                              # 설정 파일
│   ├── robot_params.yaml                #   로봇 물리 파라미터
│   ├── ekf_params.yaml                  #   EKF 센서 퓨전 공분산 설정
│   ├── nav2_params.yaml                 #   Nav2 costmap/DWA/A* 파라미터
│   └── fleet_config.yaml                #   Fleet 5대 구성 + 할당/교통 설정
│
├── launch/                              # ROS2 Launch 파일
│   ├── simulation.launch.py             #   Gazebo + URDF 스폰
│   ├── navigation.launch.py             #   Nav2 + EKF
│   └── fleet.launch.py                  #   다중 로봇 네비게이션 + Fleet 관리
│
├── tests/                               # ★ 테스트 (75개 전체 통과)
│   ├── test_kinematics.py               #   기구학: 직선/원형/드리프트 (6개)
│   ├── test_ekf.py                      #   EKF: 예측/업데이트/퓨전/kidnapped (10개)
│   ├── test_astar.py                    #   A*: 최단경로/장애물/에지케이스 (9개)
│   ├── test_dwa.py                      #   DWA: 목표도달/장애물회피/동역학제한 (9개)
│   ├── test_fleet.py                    #   Fleet: Hungarian 최적성/교착 탐지 (8개)
│   ├── test_slam.py                     #   ★ SLAM: ICP/OccupancyGrid/PoseGraph (13개)
│   └── test_integration.py              #   ★ 통합 테스트: E2E 파이프라인 (10개)
│
├── docs/                                # ★ 기술 문서
│   ├── ARCHITECTURE.md                  #   4계층 아키텍처, 시퀀스 다이어그램
│   ├── ALGORITHM_COMPARISON.md          #   A* vs NavFn vs Smac, DWA vs TEB, EKF 분석
│   ├── API_REFERENCE.md                 #   전 모듈 API + ROS2 노드 인터페이스
│   └── SETUP_GUIDE.md                   #   Docker/ROS2/Demo 설치 가이드
│
├── README.md
└── EVALUATION.md                        # 성능 평가 보고서
```

---

## 핵심 구현 사항

### SLAM (Simultaneous Localization and Mapping)

- **ICP 스캔 매칭**: SVD 기반 최근접점 매칭으로 연속 LiDAR 스캔 정합
- **Pose Graph 최적화**: Gauss-Newton 반복법으로 루프 클로저 보정
- **점유 격자 지도**: Log-odds 확률 모델 + Bresenham ray casting
- `core/slam.py` → `ICPScanMatcher`, `PoseGraph`, `OccupancyGridMap`, `SLAM2D`

### 차동 구동 순기구학

- v = (v_right + v_left) / 2, omega = (v_right - v_left) / L
- 중간각(mid_theta) 방식으로 오도메트리 누적 오차 최소화

### EKF 센서 퓨전

- 5차원 상태 벡터 [x, y, theta, v, omega]
- Joseph form 공분산 업데이트 (수치 안정성)
- odom + IMU + AMCL 순차 갱신

### A* 경로 계획

- 옥타일(Octile) 휴리스틱으로 8방향 탐색
- 코너 절단 방지 (대각선 이동 시 인접 셀 통과 확인)
- Gradient Descent 기반 경로 평활화

### DWA 로컬 경로 계획

- 동적 윈도우 + 궤적 시뮬레이션 + 3가지 비용 함수
- heading(목표 방향) + clearance(장애물 회피) + velocity(속도)

### Hungarian 작업 할당

- O(n^3) 최적 할당, 비정방 행렬 지원
- Greedy, Auction 알고리즘도 제공

### 웹 모니터링 대시보드

- 순수 Python HTTP 서버 (외부 라이브러리 불필요), 한국어 UI
- 로봇별 상태 카드 (상태, 속도, 배터리 바, 현재 작업)
- KPI 6종: 시간당 처리량, 로봇 가동률, 평균 작업 시간, 교착 상태, 완료/대기 작업
- 이상 알림 패널: 배터리 부족/위험, 도킹 상태 실시간 경고
- 창고 맵: 로봇 위치 + 이동 궤적 + 목표 점선 표시
- 작업 큐 테이블, 처리 추이 차트 (탭 전환)
- 0.75초 간격 자동 갱신, 50건 완료 시 자동 리셋 사이클

### ROS2 노드

| 노드 | 구독 토픽 | 발행 토픽 | 기능 |
|------|----------|----------|------|
| `slam_node` | `/scan`, `/odom` | `/map`, `/slam_pose` | ICP SLAM + 지도 생성 |
| `navigation_node` | `/goal_pose`, `/map`, `/odom`, `/scan` | `/cmd_vel`, `/planned_path` | A*→DWA→Pure Pursuit |
| `perception_node` | `/camera/image_raw`, `/camera/depth` | `/detections`, `/detection_markers` | YOLOv8 + 3D 투영 |
| `fleet_manager_node` | `/amr_XX/status` | `/amr_XX/goal` | Hungarian 할당 + 교착 탐지 |
| `safety_node` | `/scan` | `/cmd_vel`, `/safety/status` | E-Stop + Safety Zone |

---

## 테스트

총 **75개** 테스트, 전체 통과 (4.05초):

```bash
pytest tests/ -v
```

| 테스트 파일 | 테스트 수 | 범위 |
|------------|----------|------|
| `test_kinematics.py` | 6 | 직선/원형 주행, 드리프트, 오도메트리 |
| `test_ekf.py` | 10 | 예측/업데이트/퓨전, kidnapped 복구 |
| `test_astar.py` | 9 | 최단경로, 장애물 회피, 경로 평활화 |
| `test_dwa.py` | 9 | 목표 도달, 장애물 회피, 동역학 제한 |
| `test_fleet.py` | 8 | Hungarian 최적성, 교착 탐지, Auction |
| `test_slam.py` | 13 | ICP 정합, 점유 격자, PoseGraph, SLAM 통합 |
| `test_integration.py` | 10 | SLAM→Nav, EKF+노이즈, Fleet 사이클, 긴급 정지 |

---

## 기술 문서

- **[아키텍처 설계](docs/ARCHITECTURE.md)** — 4계층 구조, 컴포넌트/시퀀스 다이어그램, TF 트리
- **[알고리즘 비교](docs/ALGORITHM_COMPARISON.md)** — A* vs NavFn vs Smac, DWA vs TEB, EKF 성능 분석
- **[API 레퍼런스](docs/API_REFERENCE.md)** — 전 모듈 함수/클래스 API, ROS2 토픽/서비스 명세
- **[설치 가이드](docs/SETUP_GUIDE.md)** — Docker, ROS2 Humble, Demo 모드 설치 방법

---

## 환경 요구사항

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10 (Costmap의 distance_transform_edt)
- pytest (테스트)
- ROS2 Humble (시뮬레이션/네비게이션 모드, 선택)
- Docker + NVIDIA Container Toolkit (GPU 가속, 선택)

---

## 학습 성과

이 프로젝트를 완료하면 다음을 얻을 수 있습니다:

1. **SLAM 구현 능력** — ICP, PoseGraph, OccupancyGrid를 NumPy로 직접 구현
2. **자율 주행 파이프라인** — EKF → A* → DWA → Pure Pursuit 전 과정 이해
3. **다중 로봇 관리** — Hungarian 할당, 교착 탐지, 교통 관리 실무 경험
4. **ROS2 시스템 설계** — 노드/토픽/서비스 아키텍처, TF 트리, Launch 파일
5. **웹 대시보드 개발** — 실시간 모니터링 UI 구축 (순수 Python)
6. **75개 테스트** — 단위/통합 테스트로 검증된 신뢰할 수 있는 코드베이스
