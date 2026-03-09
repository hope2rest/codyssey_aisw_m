# AMR Fleet System 설치 및 실행 가이드

> **버전**: 1.0
> **대상 환경**: Ubuntu 22.04 LTS / ROS2 Humble
> **최종 업데이트**: 2026-03

---

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [Docker 기반 설치 (권장)](#2-docker-기반-설치-권장)
3. [수동 설치](#3-수동-설치)
4. [실행 모드별 가이드](#4-실행-모드별-가이드)
5. [데모 모드 (ROS2 불필요)](#5-데모-모드-ros2-불필요)
6. [설정 파일 안내](#6-설정-파일-안내)
7. [문제 해결](#7-문제-해결)
8. [개발 환경 구성](#8-개발-환경-구성)

---

## 1. 시스템 요구사항

### 하드웨어 최소 사양

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| CPU | 4코어 (x86_64) | 8코어 이상 |
| RAM | 8 GB | 16 GB 이상 |
| 디스크 | 30 GB 여유 공간 | 50 GB SSD |
| GPU | - | NVIDIA GPU (CUDA 지원) |

> **참고**: GPU는 YOLOv8 객체 감지 및 Gazebo 시뮬레이션 가속에 사용됩니다. GPU가 없어도 시뮬레이션 폴백 모드로 동작합니다.

### 소프트웨어 의존성

| 소프트웨어 | 버전 | 비고 |
|-----------|------|------|
| Ubuntu | 22.04 LTS (Jammy) | 필수 |
| ROS2 | Humble Hawksbill | 시뮬레이션/네비게이션 모드 필수 |
| Python | 3.10+ | Ubuntu 22.04 기본 포함 |
| Docker | 20.10+ | Docker 설치 시 |
| Docker Compose | 2.0+ | Docker 설치 시 |
| NVIDIA Driver | 515+ | GPU 사용 시 |
| nvidia-docker2 | 최신 | GPU + Docker 사용 시 |

### Python 패키지 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| numpy | >= 1.24 | 수치 연산, 행렬 계산 |
| scipy | >= 1.10 | 최적화 (Hungarian), 거리 변환 |
| ultralytics | >= 8.0 | YOLOv8 객체 감지 |
| pyyaml | >= 6.0 | YAML 설정 파일 파싱 |
| matplotlib | >= 3.7 | 시각화 (선택) |

---

## 2. Docker 기반 설치 (권장)

Docker를 사용하면 ROS2, Nav2, Gazebo 등 모든 의존성이 자동으로 설치됩니다.

### 2.1 사전 준비

#### Docker 설치

```bash
# Docker 공식 GPG 키 추가
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# 현재 사용자를 docker 그룹에 추가 (로그아웃 후 재로그인 필요)
sudo usermod -aG docker $USER
```

#### NVIDIA Container Toolkit (GPU 사용 시)

```bash
# NVIDIA Container Toolkit 저장소 추가
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.2 이미지 빌드

```bash
cd 31_AMR_Fleet_System

# Docker 이미지 빌드
docker compose build
```

빌드 과정에서 설치되는 항목:

```
+-------------------------------------------------------+
|  ros:humble (베이스 이미지)                              |
|  ├── python3-pip                                       |
|  ├── python3-colcon-common-extensions                  |
|  ├── ros-humble-navigation2                            |
|  ├── ros-humble-nav2-bringup                           |
|  ├── ros-humble-robot-localization                     |
|  ├── ros-humble-gazebo-ros-pkgs                        |
|  ├── ros-humble-xacro                                  |
|  ├── ros-humble-joint-state-publisher                  |
|  ├── ros-humble-robot-state-publisher                  |
|  ├── numpy >= 1.24                                     |
|  ├── scipy >= 1.10                                     |
|  ├── ultralytics >= 8.0                                |
|  ├── pyyaml >= 6.0                                     |
|  └── matplotlib >= 3.7                                 |
+-------------------------------------------------------+
```

### 2.3 서비스 실행

#### 전체 시스템 실행

```bash
# Fleet 관리 시스템 (amr_fleet + simulation + navigation)
docker compose up
```

#### 개별 서비스 실행

```bash
# 시뮬레이션만 실행
docker compose up simulation

# 네비게이션만 실행 (simulation 서비스에 의존)
docker compose up navigation

# Fleet 관리 시스템 (인터랙티브 모드)
docker compose run amr_fleet bash
```

#### 컨테이너 내부 접속

```bash
# 실행 중인 컨테이너에 접속
docker exec -it amr_fleet_system bash

# 컨테이너 내부에서 ROS2 명령 실행
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 topic list
```

### 2.4 Docker 환경 변수

| 환경 변수 | 값 | 설명 |
|----------|-----|------|
| `ROS_DOMAIN_ID` | 42 | ROS2 도메인 ID (서비스 간 통신 격리) |
| `DISPLAY` | `${DISPLAY}` | GUI 표시를 위한 X11 디스플레이 |
| `GAZEBO_MODEL_PATH` | `/ros2_ws/src/...` | Gazebo 모델 검색 경로 |
| `NVIDIA_VISIBLE_DEVICES` | all | GPU 장치 매핑 |

### 2.5 GUI 표시 설정 (X11 포워딩)

Docker 컨테이너에서 Gazebo/RViz GUI를 표시하려면:

```bash
# 호스트에서 X11 접근 허용
xhost +local:docker

# docker compose 실행
docker compose up simulation
```

---

## 3. 수동 설치

Docker 없이 직접 설치하는 방법입니다.

### 3.1 ROS2 Humble 설치

```bash
# 로케일 설정
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS2 저장소 추가
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ROS2 Humble 데스크톱 설치
sudo apt update
sudo apt install -y ros-humble-desktop
```

### 3.2 Nav2 및 추가 패키지 설치

```bash
sudo apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-robot-localization \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher
```

### 3.3 Python 의존성 설치

```bash
pip3 install numpy>=1.24 scipy>=1.10 ultralytics>=8.0 pyyaml>=6.0 matplotlib>=3.7
```

### 3.4 워크스페이스 빌드

```bash
# 워크스페이스 생성
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 프로젝트 복사 또는 심볼릭 링크
ln -s /path/to/31_AMR_Fleet_System ~/ros2_ws/src/amr_fleet_system

# 빌드
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# 환경 설정 (매 터미널마다 실행 또는 .bashrc에 추가)
source ~/ros2_ws/install/setup.bash
```

### 3.5 환경 변수 영구 설정

```bash
# ~/.bashrc에 추가
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=~/ros2_ws/src/amr_fleet_system/robot_description:$GAZEBO_MODEL_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 실행 모드별 가이드

AMR Fleet System은 5가지 실행 모드를 지원합니다.

```
+-----------------------------------------------------------+
|                    실행 모드 구조                             |
+-----------------------------------------------------------+
|                                                           |
|   main.py ─┬─ demo        ROS2 불필요, 알고리즘 시연        |
|             ├─ test        pytest 단위 테스트                |
|             ├─ simulation  Gazebo 시뮬레이션 (ROS2 필요)     |
|             ├─ navigation  네비게이션 스택 (ROS2 필요)        |
|             └─ fleet       Fleet 관리 (ROS2 필요)           |
|                                                           |
+-----------------------------------------------------------+
```

### 4.1 시뮬레이션 모드

Gazebo 환경에서 로봇 모델을 시뮬레이션합니다.

```bash
# 방법 1: Docker
docker compose up simulation

# 방법 2: 수동 실행
ros2 launch amr_fleet_system simulation.launch.py

# 방법 3: CLI
python3 main.py simulation
```

**실행되는 노드:**

| 노드 | 역할 |
|------|------|
| `robot_state_publisher` | URDF → TF 퍼블리시 |
| `joint_state_publisher` | 조인트 상태 퍼블리시 |

**확인 방법:**

```bash
# TF 트리 확인
ros2 run tf2_tools view_frames

# 토픽 확인
ros2 topic list
ros2 topic echo /robot_description
```

### 4.2 네비게이션 모드

EKF 센서 퓨전 + Nav2 전체 스택을 실행합니다.

```bash
# 방법 1: Docker (simulation 서비스 필요)
docker compose up simulation navigation

# 방법 2: 수동 실행
ros2 launch amr_fleet_system navigation.launch.py

# 방법 3: CLI
python3 main.py navigation
```

**실행되는 노드:**

| 노드 | 역할 | 파라미터 파일 |
|------|------|-------------|
| `ekf_node` | EKF 센서 퓨전 | `config/ekf_params.yaml` |
| `planner_server` | 글로벌 경로 계획 (SmacPlanner2D) | `config/nav2_params.yaml` |
| `controller_server` | 로컬 경로 추종 (DWB) | `config/nav2_params.yaml` |
| `behavior_server` | 복구 행동 (Spin, BackUp, Wait) | `config/nav2_params.yaml` |
| `lifecycle_manager` | 노드 수명 주기 관리 | - |

**네비게이션 목표 전송:**

```bash
# 명령줄에서 목표 전송
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 3.0}, orientation: {w: 1.0}}}}"
```

### 4.3 Fleet 모드

5대의 AMR을 동시에 관리하는 다중 로봇 시스템을 실행합니다.

```bash
# 방법 1: Docker
docker compose up amr_fleet

# 방법 2: 수동 실행
ros2 launch amr_fleet_system fleet.launch.py

# 방법 3: CLI
python3 main.py fleet
```

**실행되는 노드:**

| 네임스페이스 | 노드 | 수 |
|-------------|------|---|
| `/amr_01` ~ `/amr_05` | `ekf_node` | 5 |
| `/amr_01` ~ `/amr_05` | `planner_server` | 5 |
| `/amr_01` ~ `/amr_05` | `controller_server` | 5 |
| `/amr_01` ~ `/amr_05` | `behavior_server` | 5 |
| `/amr_01` ~ `/amr_05` | `lifecycle_manager` | 5 |
| `/` | `fleet_manager` | 1 |

**로봇 상태 확인:**

```bash
# 특정 로봇의 토픽 확인
ros2 topic list | grep amr_01

# 전체 Fleet 상태 확인
ros2 topic echo /fleet/status
```

### 4.4 테스트 모드

pytest 기반 단위 테스트를 실행합니다.

```bash
# CLI로 실행
python3 main.py test

# 또는 직접 pytest 실행
python3 -m pytest tests/ -v --tb=short
```

---

## 5. 데모 모드 (ROS2 불필요)

ROS2 없이 핵심 알고리즘의 동작을 확인할 수 있는 가장 간단한 실행 방법입니다.

### 5.1 요구사항

```bash
# Python 패키지만 있으면 됩니다
pip3 install numpy scipy
```

### 5.2 실행

```bash
cd 31_AMR_Fleet_System
python3 main.py demo
```

### 5.3 데모 내용

데모 모드는 6개의 핵심 알고리즘을 순차적으로 시연합니다:

```
╔══════════════════════════════════════════════════════════╗
║             AMR Fleet System - 데모 모드                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  [1] 차동 구동 기구학 및 오도메트리                          ║
║      • 직진 100스텝 → 위치/방향 확인                        ║
║      • 원호 주행 100스텝 → 곡선 경로 확인                    ║
║                                                          ║
║  [2] A* 경로 계획                                         ║
║      • 20×20 맵, 장애물 3줄                                ║
║      • (0,0) → (19,19) 최단 경로 탐색                     ║
║      • Gradient Descent 경로 스무딩 적용                   ║
║                                                          ║
║  [3] DWA 동적 경로 계획                                    ║
║      • 목표: (8.0, 3.0), 장애물 3개                       ║
║      • 실시간 속도 샘플링 + 장애물 회피                      ║
║      • 목표 도달 여부 및 스텝 수 출력                        ║
║                                                          ║
║  [4] EKF 센서 퓨전                                        ║
║      • 오도메트리 + IMU 센서 퓨전                           ║
║      • 100스텝 직진 시뮬레이션                              ║
║      • 평균 위치 오차 및 공분산 추적                         ║
║                                                          ║
║  [5] Fleet 작업 할당 (Hungarian)                          ║
║      • 로봇 5대, 작업 5개                                  ║
║      • 최적 1:1 매칭 결과 출력                              ║
║      • 비용(유클리드 거리) 기반 할당                         ║
║                                                          ║
║  [6] 물류센터 시뮬레이션 환경                                ║
║      • 60m × 40m 창고 환경 생성                            ║
║      • 선반, 충전소 배치 확인                                ║
║      • 동적 장애물 (작업자 3, 지게차 2) 이동                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### 5.4 예상 출력

```
============================================================
  AMR Fleet System - 핵심 알고리즘 데모
============================================================

[1] 차동 구동 기구학 및 오도메트리
----------------------------------------
  직진 100스텝: x=0.0767m, y=0.0000m, theta=0.0000rad
  원호 주행 100스텝: x=0.0734m, y=0.0098m, theta=0.2560rad

[2] A* 경로 계획
----------------------------------------
  맵 크기: 20x20, 장애물 3줄
  경로 길이: XX 노드
  시작: (0, 0), 끝: (19, 19)

[3] DWA 동적 경로 계획
----------------------------------------
  목표: (8.0, 3.0), 장애물: 3개
  목표 도달! XX스텝, 최종 거리: X.XXXm

[4] EKF 센서 퓨전
----------------------------------------
  100스텝 직진 퓨전
  평균 위치 오차: X.XXXXm
  최종 공분산 trace: X.XXXXXX

[5] Fleet 작업 할당 (Hungarian)
----------------------------------------
  로봇 5대, 작업 5개
    AMR-01 -> T1 (비용: X.XX)
    AMR-02 -> T2 (비용: X.XX)
    ...
  총 할당: 5건

[6] 물류센터 시뮬레이션 환경
----------------------------------------
  환경 크기: 60m x 40m
  그리드: (400, 600)
  선반 수: XX
  충전 스테이션: 5개
  동적 장애물: 5개 (작업자 3, 지게차 2)
  5초 시뮬레이션 완료

============================================================
  데모 완료
============================================================
```

---

## 6. 설정 파일 안내

### 설정 파일 목록

| 파일 | 용도 | 주요 설정 항목 |
|------|------|--------------|
| `config/robot_params.yaml` | 로봇 물리 파라미터 | 치수, 속도 제한, 배터리, 적재 동역학 |
| `config/fleet_config.yaml` | Fleet 관리 설정 | 로봇 구성, 충전소, 작업 할당, 교통 관리 |
| `config/nav2_params.yaml` | Nav2 파라미터 | Costmap, Planner, Controller, Behavior |
| `config/ekf_params.yaml` | EKF 센서 퓨전 | 센서 소스, 공분산 행렬, 퓨전 주기 |

### 주요 파라미터 조정 가이드

#### 로봇 속도 제한 변경

```yaml
# config/robot_params.yaml
robot:
  max_velocity: 1.0          # m/s → 물류센터: 0.5~1.0, 외부: 1.0~2.0
  max_acceleration: 0.5      # m/s² → 적재 시 낮춤
  max_angular_velocity: 1.5  # rad/s
```

#### 작업 할당 알고리즘 변경

```yaml
# config/fleet_config.yaml
fleet:
  task_allocation:
    algorithm: hungarian     # hungarian | greedy | auction
    cost_metric: distance    # distance | time | battery_aware
```

#### 안전 거리 변경

```yaml
# config/fleet_config.yaml
fleet:
  traffic:
    collision_avoidance:
      safety_distance: 1.0   # m → 좁은 통로: 0.5, 넓은 공간: 1.5
```

#### 배터리 충전 임계값 변경

```yaml
# config/robot_params.yaml
robot:
  battery:
    low_threshold: 20.0      # % → 충전 필요 알림
    critical_threshold: 10.0  # % → 긴급 충전 (즉시 이동)
```

---

## 7. 문제 해결

### 7.1 Docker 관련

#### 문제: `permission denied` 오류

```bash
# 원인: docker 그룹에 사용자가 없음
sudo usermod -aG docker $USER
# 로그아웃 후 재로그인 필요
```

#### 문제: GPU를 인식하지 못함

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# nvidia-container-toolkit 설치 확인
nvidia-ctk --version

# Docker 런타임 설정 확인
docker info | grep -i runtime

# GPU 없이 실행 (docker-compose.yml에서 deploy 섹션 제거)
# services > amr_fleet > deploy 블록을 주석 처리
```

#### 문제: GUI가 표시되지 않음

```bash
# X11 포워딩 허용
xhost +local:docker

# DISPLAY 변수 확인
echo $DISPLAY

# SSH 원격 접속 시: X11 포워딩 활성화
ssh -X user@host
```

#### 문제: `docker compose build` 실패

```bash
# Docker 캐시 정리 후 재빌드
docker compose build --no-cache

# 디스크 공간 확인
df -h

# 미사용 Docker 리소스 정리
docker system prune -a
```

### 7.2 ROS2 관련

#### 문제: 노드 간 통신이 되지 않음

```bash
# ROS_DOMAIN_ID 확인 (모든 터미널에서 동일해야 함)
echo $ROS_DOMAIN_ID
export ROS_DOMAIN_ID=42

# DDS 미들웨어 확인
echo $RMW_IMPLEMENTATION

# FastDDS 사용 시 멀티캐스트 설정 확인
ros2 doctor
```

#### 문제: `package not found` 오류

```bash
# 워크스페이스 소스 설정 확인
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 패키지 목록 확인
ros2 pkg list | grep amr

# 재빌드
cd ~/ros2_ws
colcon build --symlink-install
```

#### 문제: Nav2 노드가 시작되지 않음

```bash
# lifecycle 노드 상태 확인
ros2 lifecycle list /planner_server
ros2 lifecycle get /planner_server

# lifecycle 수동 전환
ros2 lifecycle set /planner_server configure
ros2 lifecycle set /planner_server activate

# 로그 확인
ros2 topic echo /rosout --qos-reliability reliable
```

#### 문제: TF 트리가 불완전함

```bash
# TF 트리 시각화
ros2 run tf2_tools view_frames

# 특정 TF 변환 확인
ros2 run tf2_ros tf2_echo map base_link

# URDF 파싱 오류 확인
ros2 run xacro xacro robot_description/amr_robot.urdf.xacro
```

### 7.3 Python / 알고리즘 관련

#### 문제: `ModuleNotFoundError: No module named 'core'`

```bash
# 프로젝트 루트에서 실행해야 합니다
cd 31_AMR_Fleet_System
python3 main.py demo

# 또는 PYTHONPATH 설정
export PYTHONPATH=/path/to/31_AMR_Fleet_System:$PYTHONPATH
```

#### 문제: `ModuleNotFoundError: No module named 'scipy'`

```bash
pip3 install scipy>=1.10
```

#### 문제: NumPy 버전 충돌

```bash
# 가상환경 사용 권장
python3 -m venv venv
source venv/bin/activate
pip install numpy>=1.24 scipy>=1.10
```

### 7.4 시뮬레이션 관련

#### 문제: Gazebo가 검은 화면으로 시작됨

```bash
# GPU 드라이버 확인
glxinfo | grep "OpenGL renderer"

# 소프트웨어 렌더링 사용
export LIBGL_ALWAYS_SOFTWARE=1
```

#### 문제: 시뮬레이션이 너무 느림

```bash
# Gazebo 물리 엔진 주기 확인 (기본 1ms)
# 실시간 비율 확인: Gazebo 좌하단 표시

# GUI 비활성화로 속도 향상
# simulation.launch.py에서 gui:=false 옵션 추가
```

---

## 8. 개발 환경 구성

### 8.1 VS Code 확장 프로그램 (권장)

| 확장 | 용도 |
|------|------|
| Python (ms-python.python) | Python 개발 |
| ROS (ms-iot.vscode-ros) | ROS2 통합 |
| YAML (redhat.vscode-yaml) | YAML 설정 편집 |
| Docker (ms-azuretools.vscode-docker) | Docker 통합 |
| Remote - Containers | Docker 내부 개발 |

### 8.2 프로젝트 구조

```
31_AMR_Fleet_System/
├── main.py                      # CLI 진입점
├── Dockerfile                   # Docker 이미지 정의
├── docker-compose.yml           # 다중 서비스 구성
│
├── core/                        # 핵심 알고리즘
│   ├── astar.py                 # A* 경로 계획
│   ├── dwa.py                   # DWA 로컬 경로 계획
│   ├── ekf.py                   # EKF 센서 퓨전
│   ├── costmap.py               # 2D 코스트맵
│   ├── kinematics.py            # 차동 구동 기구학
│   ├── pid_controller.py        # PID 제어기
│   ├── pure_pursuit.py          # Pure Pursuit 경로 추종
│   └── kalman_tracker.py        # 물체 추적
│
├── perception/                  # 인식 모듈
│   ├── lidar_processor.py       # LiDAR 처리 파이프라인
│   ├── yolo_detector.py         # YOLOv8 객체 감지
│   └── camera_projection.py     # 카메라 투영 변환
│
├── fleet/                       # Fleet 관리
│   ├── task_allocator.py        # 작업 할당 (Hungarian)
│   ├── traffic_manager.py       # 교통 관리/교착 탐지
│   └── monitor.py               # Fleet 모니터링/KPI
│
├── task/                        # 작업 실행
│   ├── behavior_tree.py         # 행동 트리 프레임워크
│   └── docking.py               # 도킹 제어기
│
├── simulation/                  # 시뮬레이션
│   ├── warehouse_world.py       # 물류센터 환경
│   └── sensor_noise.py          # 센서 노이즈 모델
│
├── config/                      # 설정 파일
│   ├── robot_params.yaml        # 로봇 물리 파라미터
│   ├── fleet_config.yaml        # Fleet 관리 설정
│   ├── nav2_params.yaml         # Nav2 파라미터
│   └── ekf_params.yaml          # EKF 설정
│
├── launch/                      # ROS2 Launch 파일
│   ├── simulation.launch.py     # 시뮬레이션 런치
│   ├── navigation.launch.py     # 네비게이션 런치
│   └── fleet.launch.py          # Fleet 런치
│
├── robot_description/           # 로봇 모델
│   └── amr_robot.urdf.xacro     # URDF 로봇 기술
│
├── tests/                       # 단위 테스트
│
└── docs/                        # 문서
    ├── ARCHITECTURE.md           # 시스템 아키텍처
    ├── API_REFERENCE.md          # API 레퍼런스
    ├── ALGORITHM_COMPARISON.md   # 알고리즘 비교 분석
    └── SETUP_GUIDE.md            # 설치 가이드 (본 문서)
```

### 8.3 테스트 실행

```bash
# 전체 테스트
python3 -m pytest tests/ -v

# 특정 모듈 테스트
python3 -m pytest tests/test_astar.py -v

# 커버리지 측정
pip3 install pytest-cov
python3 -m pytest tests/ --cov=core --cov=fleet --cov=perception --cov-report=html
```

### 8.4 빠른 시작 요약

```
+----------------------------------------------------------+
|                   빠른 시작 가이드                          |
+----------------------------------------------------------+
|                                                          |
|  1. ROS2 없이 바로 테스트하고 싶다면:                       |
|     $ pip install numpy scipy                            |
|     $ python3 main.py demo                               |
|                                                          |
|  2. Docker로 전체 시스템을 실행하고 싶다면:                  |
|     $ docker compose build                               |
|     $ docker compose up                                  |
|                                                          |
|  3. 수동 설치 후 시뮬레이션을 실행하고 싶다면:                |
|     $ ros2 launch amr_fleet_system simulation.launch.py  |
|     $ ros2 launch amr_fleet_system navigation.launch.py  |
|                                                          |
|  4. 5대 로봇 Fleet을 실행하고 싶다면:                       |
|     $ ros2 launch amr_fleet_system fleet.launch.py       |
|                                                          |
+----------------------------------------------------------+
```

---

> **문의**: 추가 질문이나 이슈는 프로젝트 리포지토리의 Issues 탭에 등록해 주세요.
