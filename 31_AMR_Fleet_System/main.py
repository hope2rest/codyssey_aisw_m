"""
AMR Fleet System CLI 진입점.

모드: simulation, navigation, fleet, test, demo
demo 모드는 ROS2 없이 핵심 알고리즘을 시연한다.
"""

import sys
import argparse
import numpy as np


def run_demo():
    """ROS2 없이 핵심 알고리즘 데모를 실행한다."""
    print("=" * 60)
    print("  AMR Fleet System - 핵심 알고리즘 데모")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 기구학 및 오도메트리
    # ------------------------------------------------------------------
    print("\n[1] 차동 구동 기구학 및 오도메트리")
    print("-" * 40)
    from core.kinematics import DifferentialDriveRobot

    robot = DifferentialDriveRobot()
    for _ in range(100):
        robot.update(left_ticks=10, right_ticks=10)
    x, y, theta = robot.get_pose()
    print(f"  직진 100스텝: x={x:.4f}m, y={y:.4f}m, theta={theta:.4f}rad")

    robot.reset()
    for _ in range(100):
        robot.update(left_ticks=8, right_ticks=10)
    x, y, theta = robot.get_pose()
    print(f"  원호 주행 100스텝: x={x:.4f}m, y={y:.4f}m, theta={theta:.4f}rad")

    # ------------------------------------------------------------------
    # 2. A* 경로 계획
    # ------------------------------------------------------------------
    print("\n[2] A* 경로 계획")
    print("-" * 40)
    from core.astar import plan_path

    grid = np.zeros((20, 20))
    grid[5, 3:15] = 1
    grid[10, 5:18] = 1
    grid[15, 0:12] = 1

    path = plan_path(grid, (0, 0), (19, 19), smooth=True)
    print(f"  맵 크기: 20x20, 장애물 3줄")
    print(f"  경로 길이: {len(path)} 노드")
    print(f"  시작: {path[0]}, 끝: {path[-1]}")

    # ------------------------------------------------------------------
    # 3. DWA 로컬 경로 계획
    # ------------------------------------------------------------------
    print("\n[3] DWA 동적 경로 계획")
    print("-" * 40)
    from core.dwa import dwa_planning, DWAConfig

    config = DWAConfig(max_v=2.0, predict_time=1.5, dt=0.1)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([8.0, 3.0])
    obstacles = np.array([[3.0, 0.5], [5.0, 2.0], [6.0, 1.0]])

    print(f"  목표: ({goal[0]}, {goal[1]}), 장애물: {len(obstacles)}개")
    reached = False
    for step in range(200):
        v, w = dwa_planning(state, goal, obstacles, config)
        state[0] += v * np.cos(state[2]) * config.dt
        state[1] += v * np.sin(state[2]) * config.dt
        state[2] += w * config.dt
        state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
        state[3] = v
        state[4] = w

        dist = np.sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)
        if dist < 0.5:
            print(f"  목표 도달! {step}스텝, 최종 거리: {dist:.3f}m")
            reached = True
            break

    if not reached:
        print(f"  200스텝 후 거리: {dist:.3f}m")

    # ------------------------------------------------------------------
    # 4. EKF 센서 퓨전
    # ------------------------------------------------------------------
    print("\n[4] EKF 센서 퓨전")
    print("-" * 40)
    from core.ekf import ExtendedKalmanFilter, fuse_sensors

    np.random.seed(42)
    ekf = ExtendedKalmanFilter(initial_state=[0, 0, 0, 1, 0])
    dt = 0.1
    errors = []

    for step in range(100):
        true_x = 1.0 * (step + 1) * dt
        odom = np.array([true_x, 0.0, 0.0]) + np.random.randn(3) * 0.05
        imu = np.array([0.0, 0.0]) + np.random.randn(2) * 0.01

        state = fuse_sensors(ekf, dt, odom=odom, imu=imu)
        errors.append(abs(state[0] - true_x))

    print(f"  100스텝 직진 퓨전")
    print(f"  평균 위치 오차: {np.mean(errors):.4f}m")
    print(f"  최종 공분산 trace: {np.trace(ekf.get_covariance()):.6f}")

    # ------------------------------------------------------------------
    # 5. Fleet 작업 할당 (Hungarian)
    # ------------------------------------------------------------------
    print("\n[5] Fleet 작업 할당 (Hungarian)")
    print("-" * 40)
    from fleet.task_allocator import TaskAllocator, Task, Robot

    task_list = [
        Task("T1", (2.0, 3.0), priority=5),
        Task("T2", (8.0, 2.0), priority=3),
        Task("T3", (5.0, 5.0), priority=4),
        Task("T4", (1.0, 8.0), priority=2),
        Task("T5", (9.0, 9.0), priority=1),
    ]
    robot_list = [
        Robot("AMR-01", (1.0, 1.0)),
        Robot("AMR-02", (10.0, 1.0)),
        Robot("AMR-03", (5.0, 8.0)),
        Robot("AMR-04", (2.0, 6.0)),
        Robot("AMR-05", (9.0, 7.0)),
    ]

    allocator = TaskAllocator()
    assignments = allocator.allocate(task_list, robot_list)

    print(f"  로봇 5대, 작업 5개")
    for a in assignments:
        print(f"    {a.robot_id} -> {a.task_id} (비용: {a.cost:.2f})")
    print(f"  총 할당: {len(assignments)}건")

    # ------------------------------------------------------------------
    # 6. 물류센터 시뮬레이션
    # ------------------------------------------------------------------
    print("\n[6] 물류센터 시뮬레이션 환경")
    print("-" * 40)
    from simulation.warehouse_world import WarehouseWorld

    world = WarehouseWorld()
    grid = world.get_occupancy_grid()
    print(f"  환경 크기: {world.width}m x {world.height}m")
    print(f"  그리드: {grid.shape}")
    print(f"  선반 수: {len(world.shelf_positions)}")
    print(f"  충전 스테이션: {len(world.charging_stations)}개")

    obs = world.spawn_dynamic_obstacles(num_workers=3, num_forklifts=2)
    print(f"  동적 장애물: {len(obs)}개 (작업자 3, 지게차 2)")

    for _ in range(50):
        world.step(0.1)
    print(f"  5초 시뮬레이션 완료")

    print("\n" + "=" * 60)
    print("  데모 완료")
    print("=" * 60)


def run_test():
    """pytest로 단위 테스트를 실행한다."""
    import subprocess
    test_dir = sys.path[0] if sys.path[0] else '.'
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


def run_simulation():
    """시뮬레이션 모드 (ROS2 필요)."""
    print("시뮬레이션 모드: ros2 launch 31_AMR_Fleet_System simulation.launch.py")
    print("Docker 환경에서 실행하세요: docker-compose up simulation")


def run_navigation():
    """네비게이션 모드 (ROS2 필요)."""
    print("네비게이션 모드: ros2 launch 31_AMR_Fleet_System navigation.launch.py")
    print("Docker 환경에서 실행하세요: docker-compose up navigation")


def run_fleet():
    """Fleet 모드 (ROS2 필요)."""
    print("Fleet 모드: ros2 launch 31_AMR_Fleet_System fleet.launch.py")
    print("Docker 환경에서 실행하세요: docker-compose up amr_fleet")


def main():
    import os

    parser = argparse.ArgumentParser(
        description='AMR Fleet System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드 설명:
  demo         ROS2 없이 핵심 알고리즘 데모 실행
  test         pytest 기반 단위 테스트 실행
  simulation   Gazebo 시뮬레이션 실행 (ROS2 필요)
  navigation   네비게이션 스택 실행 (ROS2 필요)
  fleet        Fleet 관리 시스템 실행 (ROS2 필요)
        """,
    )
    parser.add_argument(
        'mode',
        choices=['simulation', 'navigation', 'fleet', 'test', 'demo'],
        help='실행 모드',
    )

    args = parser.parse_args()

    mode_map = {
        'demo': run_demo,
        'test': run_test,
        'simulation': run_simulation,
        'navigation': run_navigation,
        'fleet': run_fleet,
    }

    mode_map[args.mode]()


if __name__ == '__main__':
    main()
