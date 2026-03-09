# 31. AMR Fleet System (자율 이동 로봇 Fleet 관리)

ROS2 Humble 기반 다중 AMR(Autonomous Mobile Robot) 물류센터 운용 시스템. 차동 구동 로봇 5대가 60m x 40m 물류센터에서 자율 주행, 피킹, 운반 작업을 수행한다.

---

## 시스템 아키텍처

```
+------------------------------------------------------------------+
|                        Fleet Manager                              |
|  (Hungarian 할당 / 교착 탐지 / 교통 관리 / 모니터링)             |
+-----+--------+--------+--------+--------+-----------------------+
      |        |        |        |        |
  +---v---+ +--v---+ +-v----+ +-v----+ +-v----+
  |AMR_01 | |AMR_02| |AMR_03| |AMR_04| |AMR_05|
  +---+---+ +--+---+ +--+---+ +--+---+ +--+---+
      |        |        |        |        |
      v        v        v        v        v
+------------------------------------------------------------------+
|                     Navigation Stack                              |
|  +-------------+  +-----------+  +------------+  +----------+   |
|  | Localization|  |  Path     |  |  Local     |  | Recovery |   |
|  | (EKF Fusion)|  |  Planning |  |  Planning  |  | Behavior |   |
|  | odom+IMU+   |  |  (A*)    |  |  (DWA)     |  | (Spin/   |   |
|  | AMCL        |  |          |  |            |  |  Backup)  |   |
|  +------+------+  +----+-----+  +-----+------+  +-----+----+   |
|         |              |              |                |          |
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

# 핵심 알고리즘 데모 (기구학, A*, DWA, EKF, Fleet 할당, 시뮬레이션)
python main.py demo

# 단위 테스트
python main.py test
```

### Docker 환경 (전체 시스템)

```bash
# 이미지 빌드
docker-compose build

# 시뮬레이션 실행
docker-compose up simulation

# 네비게이션 실행
docker-compose up navigation

# Fleet 전체 실행
docker-compose up amr_fleet
```

### ROS2 직접 실행

```bash
# 시뮬레이션
ros2 launch amr_fleet_system simulation.launch.py

# 네비게이션
ros2 launch amr_fleet_system navigation.launch.py

# Fleet 관리
ros2 launch amr_fleet_system fleet.launch.py num_robots:=5
```

---

## 파일 구조

```
31_AMR_Fleet_System/
├── main.py                              # CLI 진입점 (5개 모드)
├── Dockerfile                           # ROS2 Humble 기반 Docker
├── docker-compose.yml                   # GPU 가속 + 볼륨 마운트
├── core/                                # 핵심 알고리즘 (NumPy 직접 구현)
│   ├── kinematics.py                    #   차동 구동 순기구학 + 오도메트리
│   ├── ekf.py                           #   EKF 센서 퓨전 (odom + IMU + AMCL)
│   ├── astar.py                         #   A* 경로 계획 (8방향 + 평활화)
│   ├── dwa.py                           #   DWA 로컬 경로 계획
│   ├── pure_pursuit.py                  #   Pure Pursuit 경로 추종
│   ├── pid_controller.py                #   PID 속도 제어
│   ├── kalman_tracker.py                #   칼만 필터 기반 객체 추적
│   └── costmap.py                       #   2D 비용맵 (Static + Inflation + Obstacle)
├── fleet/                               # Fleet 관리
│   ├── task_allocator.py                #   작업 할당 (Hungarian/Greedy/Auction)
│   ├── traffic_manager.py               #   교통 관리 + 교착 탐지
│   └── monitor.py                       #   실시간 모니터링
├── perception/                          # 인지 모듈
│   ├── lidar_processor.py               #   LiDAR 포인트 클라우드 처리
│   ├── yolo_detector.py                 #   YOLOv8 객체 인식
│   └── camera_projection.py             #   카메라 투영 변환
├── task/                                # 작업 관리
│   ├── behavior_tree.py                 #   Behavior Tree 행동 계획
│   └── docking.py                       #   도킹 제어
├── simulation/                          # 시뮬레이션 환경
│   ├── warehouse_world.py               #   60m x 40m 물류센터 (occupancy grid)
│   └── sensor_noise.py                  #   센서 노이즈 모델 (LiDAR/IMU/Encoder/Depth)
├── robot_description/                   # 로봇 모델
│   └── amr_robot.urdf.xacro            #   차동 구동 AMR URDF (xacro 매크로)
├── config/                              # 설정 파일
│   ├── robot_params.yaml                #   로봇 물리 파라미터
│   ├── ekf_params.yaml                  #   EKF 센서 퓨전 공분산 설정
│   ├── nav2_params.yaml                 #   Nav2 costmap/DWA/A* 파라미터
│   └── fleet_config.yaml                #   Fleet 5대 구성 + 할당/교통 설정
├── launch/                              # ROS2 Launch 파일
│   ├── simulation.launch.py             #   Gazebo + URDF 스폰
│   ├── navigation.launch.py             #   Nav2 + EKF
│   └── fleet.launch.py                  #   다중 로봇 네비게이션 + Fleet 관리
├── tests/                               # 단위 테스트
│   ├── test_kinematics.py               #   기구학: 직선/원형/드리프트
│   ├── test_ekf.py                      #   EKF: 예측/업데이트/퓨전/kidnapped
│   ├── test_astar.py                    #   A*: 최단경로/장애물 회피/에지케이스
│   ├── test_dwa.py                      #   DWA: 목표도달/장애물회피/동역학제한
│   └── test_fleet.py                    #   Fleet: Hungarian 최적성/교착 탐지
├── README.md
└── EVALUATION.md                        # 성능 평가 보고서
```

---

## 핵심 구현 사항

**차동 구동 순기구학 (직접 구현)**
- v = (v_right + v_left) / 2, omega = (v_right - v_left) / L
- 중간각(mid_theta) 방식으로 오도메트리 누적 오차 최소화

**EKF 센서 퓨전 (직접 구현)**
- 5차원 상태 벡터 [x, y, theta, v, omega]
- Joseph form 공분산 업데이트 (수치 안정성)
- odom + IMU + AMCL 순차 갱신

**A* 경로 계획 (직접 구현)**
- 옥타일(Octile) 휴리스틱으로 8방향 탐색
- 코너 절단 방지 (대각선 이동 시 인접 셀 통과 확인)
- Gradient Descent 기반 경로 평활화

**DWA 로컬 경로 계획 (직접 구현)**
- 동적 윈도우 + 궤적 시뮬레이션 + 3가지 비용 함수
- heading(목표 방향) + clearance(장애물 회피) + velocity(속도)

**Hungarian 작업 할당 (직접 구현)**
- O(n^3) 최적 할당, 비정방 행렬 지원
- Greedy, Auction 알고리즘도 제공

---

## 환경 요구사항

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10 (Costmap의 distance_transform_edt)
- ROS2 Humble (시뮬레이션/네비게이션 모드)
- Docker + NVIDIA Container Toolkit (GPU 가속)
- pytest (단위 테스트)
