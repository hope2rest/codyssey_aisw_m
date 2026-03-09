"""
중앙 집중식 작업 할당 모듈.

Hungarian Algorithm을 직접 구현하여 로봇-작업 간 최적 할당을 수행한다.
최소 거리 기반, 부하 균형, 작업 우선순위 및 마감 시간을 종합적으로 고려한다.

주요 기능:
    - Hungarian Algorithm (Kuhn-Munkres) 직접 구현
    - 비용 행렬 계산: 거리 + 부하 + 우선순위 + 마감시간 가중합
    - 부하 균형: 로봇별 현재 작업량을 비용에 반영
    - 작업 우선순위 및 마감 시간 고려

외부 의존성:
    - numpy (필수)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    작업 정보.

    Attributes:
        task_id: 작업 고유 식별자.
        position: 작업 위치 (x, y).
        priority: 우선순위 (1~5, 5가 최고).
        deadline: 마감 시간 (Unix timestamp). None이면 마감 없음.
        estimated_duration: 예상 소요 시간(초).
        task_type: 작업 유형 ('pickup', 'delivery', 'inspect').
    """
    task_id: str
    position: Tuple[float, float]
    priority: int = 3
    deadline: Optional[float] = None
    estimated_duration: float = 60.0
    task_type: str = "delivery"


@dataclass
class Robot:
    """
    로봇 정보.

    Attributes:
        robot_id: 로봇 고유 식별자.
        position: 현재 위치 (x, y).
        speed: 이동 속도 (m/s).
        current_load: 현재 할당된 작업 수.
        max_load: 최대 동시 작업 수.
        battery_level: 배터리 잔량 (0~100).
        available: 가용 여부.
    """
    robot_id: str
    position: Tuple[float, float]
    speed: float = 1.0
    current_load: int = 0
    max_load: int = 5
    battery_level: float = 100.0
    available: bool = True


@dataclass
class Assignment:
    """
    작업 할당 결과.

    Attributes:
        robot_id: 할당된 로봇 ID.
        task_id: 할당된 작업 ID.
        cost: 할당 비용.
        estimated_arrival: 예상 도착 시간(초).
    """
    robot_id: str
    task_id: str
    cost: float
    estimated_arrival: float


class TaskAllocator:
    """
    중앙 집중식 작업 할당기.

    Hungarian Algorithm을 사용하여 로봇-작업 간 최적 할당을 수행한다.
    비용 함수는 거리, 부하, 우선순위, 마감시간을 종합적으로 고려한다.

    사용 예시::

        allocator = TaskAllocator()
        tasks = [
            Task("T1", (10, 5), priority=5),
            Task("T2", (3, 8), priority=3),
        ]
        robots = [
            Robot("R1", (0, 0)),
            Robot("R2", (5, 5)),
        ]
        assignments = allocator.allocate(tasks, robots)
        for a in assignments:
            print(f"{a.robot_id} -> {a.task_id} (비용: {a.cost:.2f})")

    Args:
        distance_weight: 거리 비용 가중치.
        load_weight: 부하 균형 비용 가중치.
        priority_weight: 우선순위 비용 가중치.
        deadline_weight: 마감시간 비용 가중치.
    """

    def __init__(
        self,
        distance_weight: float = 1.0,
        load_weight: float = 2.0,
        priority_weight: float = 1.5,
        deadline_weight: float = 3.0,
    ):
        self.distance_weight = distance_weight
        self.load_weight = load_weight
        self.priority_weight = priority_weight
        self.deadline_weight = deadline_weight

    def compute_cost_matrix(
        self, tasks: List[Task], robots: List[Robot]
    ) -> np.ndarray:
        """
        비용 행렬을 계산한다.

        비용 = w_d * 거리비용 + w_l * 부하비용 + w_p * 우선순위비용 + w_t * 마감비용

        Args:
            tasks: 작업 리스트.
            robots: 로봇 리스트.

        Returns:
            (n_robots, n_tasks) 형태의 비용 행렬.
        """
        n_robots = len(robots)
        n_tasks = len(tasks)
        cost_matrix = np.zeros((n_robots, n_tasks), dtype=np.float64)
        current_time = time.time()

        for i, robot in enumerate(robots):
            for j, task in enumerate(tasks):
                # 1. 거리 비용
                dx = robot.position[0] - task.position[0]
                dy = robot.position[1] - task.position[1]
                distance = math.sqrt(dx * dx + dy * dy)
                distance_cost = distance / max(robot.speed, 0.1)

                # 2. 부하 균형 비용 (현재 부하가 높을수록 비용 증가)
                load_ratio = robot.current_load / max(robot.max_load, 1)
                load_cost = load_ratio * 10.0

                # 로봇 사용 불가 시 매우 높은 비용
                if not robot.available or robot.current_load >= robot.max_load:
                    load_cost = 1e6

                # 배터리 부족 시 비용 증가
                if robot.battery_level < 20:
                    load_cost += 1e4

                # 3. 우선순위 비용 (높은 우선순위 작업에 낮은 비용 -> 빨리 할당)
                # 우선순위 5(최고) -> 비용 낮음, 1(최저) -> 비용 높음
                priority_cost = (6 - task.priority) * 5.0

                # 4. 마감시간 비용
                deadline_cost = 0.0
                if task.deadline is not None:
                    time_remaining = task.deadline - current_time
                    travel_time = distance / max(robot.speed, 0.1)
                    slack = time_remaining - travel_time - task.estimated_duration

                    if slack < 0:
                        # 마감 불가능 -> 높은 비용(하지만 할당은 가능)
                        deadline_cost = abs(slack) * 10.0
                    else:
                        # 여유 시간이 적을수록 비용 증가
                        deadline_cost = max(0, 100.0 - slack)

                cost_matrix[i, j] = (
                    self.distance_weight * distance_cost
                    + self.load_weight * load_cost
                    + self.priority_weight * priority_cost
                    + self.deadline_weight * deadline_cost
                )

        return cost_matrix

    def hungarian_algorithm(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Hungarian Algorithm (Kuhn-Munkres) 직접 구현.

        n x m 비용 행렬에 대해 최적 할당을 계산한다.
        행(로봇)과 열(작업)의 수가 다를 수 있으며, 정방행렬로 패딩한다.

        알고리즘 단계:
            1. 행 감소 (각 행의 최솟값 차감)
            2. 열 감소 (각 열의 최솟값 차감)
            3. 최소 수의 선으로 모든 0을 커버
            4. 커버되지 않은 값의 최솟값으로 행렬 조정
            5. 최적 할당 추출

        Args:
            cost_matrix: (n, m) 비용 행렬.

        Returns:
            (행 인덱스, 열 인덱스) 쌍의 리스트.
        """
        # 정방행렬로 패딩
        n, m = cost_matrix.shape
        size = max(n, m)
        padded = np.full((size, size), 0.0, dtype=np.float64)
        padded[:n, :m] = cost_matrix.copy()

        # 더미 행/열에 큰 비용 할당
        if n < size:
            padded[n:, :m] = cost_matrix.max() * 10 if cost_matrix.size > 0 else 0
        if m < size:
            padded[:n, m:] = cost_matrix.max() * 10 if cost_matrix.size > 0 else 0

        C = padded.copy()

        # 1단계: 행 감소
        for i in range(size):
            row_min = C[i].min()
            C[i] -= row_min

        # 2단계: 열 감소
        for j in range(size):
            col_min = C[:, j].min()
            C[:, j] -= col_min

        # 반복적 할당 최적화
        max_iter = size * size + 100
        for iteration in range(max_iter):
            # 0 위치에서 최대 매칭 찾기
            row_assigned = np.full(size, -1, dtype=int)
            col_assigned = np.full(size, -1, dtype=int)

            for i in range(size):
                for j in range(size):
                    if C[i, j] == 0 and row_assigned[i] == -1 and col_assigned[j] == -1:
                        row_assigned[i] = j
                        col_assigned[j] = i

            num_assigned = np.sum(row_assigned >= 0)
            if num_assigned >= size:
                # 최적 할당 완료
                result = []
                for i in range(min(n, size)):
                    j = row_assigned[i]
                    if j < m:
                        result.append((i, j))
                return result

            # 최소 커버를 위한 마킹 프로세스
            # 할당되지 않은 행 마킹 -> 해당 행의 0이 있는 열 마킹 -> 해당 열에 할당된 행 마킹
            marked_rows = set()
            marked_cols = set()

            # 할당되지 않은 행 마킹
            unassigned_rows = {i for i in range(size) if row_assigned[i] == -1}
            new_marked = set(unassigned_rows)

            while new_marked:
                marked_rows |= new_marked
                # 마킹된 행의 0이 있는 열 마킹
                new_cols = set()
                for i in new_marked:
                    for j in range(size):
                        if C[i, j] == 0 and j not in marked_cols:
                            new_cols.add(j)
                marked_cols |= new_cols

                # 마킹된 열에 할당된 행 마킹
                new_marked = set()
                for j in new_cols:
                    if col_assigned[j] >= 0 and col_assigned[j] not in marked_rows:
                        new_marked.add(col_assigned[j])

            # 커버 선: 마킹되지 않은 행 + 마킹된 열
            covered_rows = set(range(size)) - marked_rows
            covered_cols = marked_cols

            # 커버되지 않은 영역에서 최솟값 찾기
            min_val = float("inf")
            for i in range(size):
                if i in covered_rows:
                    continue
                for j in range(size):
                    if j in covered_cols:
                        continue
                    min_val = min(min_val, C[i, j])

            if min_val == float("inf") or min_val == 0:
                # 더 이상 개선 불가
                break

            # 행렬 조정: 커버되지 않은 값에서 빼고, 교차점에 더하기
            for i in range(size):
                for j in range(size):
                    if i not in covered_rows and j not in covered_cols:
                        C[i, j] -= min_val
                    elif i in covered_rows and j in covered_cols:
                        C[i, j] += min_val

        # 최종 할당 추출
        row_assigned = np.full(size, -1, dtype=int)
        col_assigned = np.full(size, -1, dtype=int)

        for i in range(size):
            for j in range(size):
                if C[i, j] == 0 and row_assigned[i] == -1 and col_assigned[j] == -1:
                    row_assigned[i] = j
                    col_assigned[j] = i

        result = []
        for i in range(min(n, size)):
            j = row_assigned[i]
            if 0 <= j < m:
                result.append((i, j))

        return result

    def allocate(
        self, tasks: List[Task], robots: List[Robot]
    ) -> List[Assignment]:
        """
        작업을 로봇에 최적 할당한다.

        비용 행렬을 계산하고 Hungarian Algorithm으로 최적 매칭을 수행한다.

        Args:
            tasks: 할당할 작업 리스트.
            robots: 가용 로봇 리스트.

        Returns:
            Assignment 리스트 (우선순위 높은 작업 순서로 정렬).
        """
        if not tasks or not robots:
            logger.warning("작업 또는 로봇이 없어 할당을 수행할 수 없습니다.")
            return []

        # 가용 로봇 필터링
        available_robots = [r for r in robots if r.available and r.current_load < r.max_load]
        if not available_robots:
            logger.warning("가용 로봇이 없습니다.")
            return []

        # 작업을 우선순위 내림차순, 마감시간 오름차순으로 정렬
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (-t.priority, t.deadline or float("inf")),
        )

        # 비용 행렬 계산
        cost_matrix = self.compute_cost_matrix(sorted_tasks, available_robots)
        logger.debug(f"비용 행렬 크기: {cost_matrix.shape}")

        # Hungarian Algorithm으로 최적 할당
        matches = self.hungarian_algorithm(cost_matrix)

        # Assignment 결과 생성
        assignments: List[Assignment] = []
        for robot_idx, task_idx in matches:
            if robot_idx >= len(available_robots) or task_idx >= len(sorted_tasks):
                continue

            robot = available_robots[robot_idx]
            task = sorted_tasks[task_idx]
            cost = cost_matrix[robot_idx, task_idx]

            # 비정상적으로 높은 비용이면 스킵 (불가능한 할당)
            if cost >= 1e5:
                logger.info(
                    f"할당 스킵: {robot.robot_id} -> {task.task_id} (비용 {cost:.0f} 초과)"
                )
                continue

            # 예상 도착 시간 계산
            dx = robot.position[0] - task.position[0]
            dy = robot.position[1] - task.position[1]
            distance = math.sqrt(dx * dx + dy * dy)
            estimated_arrival = distance / max(robot.speed, 0.1)

            assignments.append(
                Assignment(
                    robot_id=robot.robot_id,
                    task_id=task.task_id,
                    cost=cost,
                    estimated_arrival=estimated_arrival,
                )
            )

        assignments.sort(key=lambda a: a.cost)

        logger.info(f"할당 완료: {len(assignments)}개 작업 -> 로봇 할당")
        return assignments


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tasks = [
        Task("T-001", (10.0, 5.0), priority=5, estimated_duration=120),
        Task("T-002", (3.0, 8.0), priority=3, estimated_duration=90),
        Task("T-003", (15.0, 2.0), priority=4, estimated_duration=60),
        Task("T-004", (7.0, 12.0), priority=2, estimated_duration=150),
    ]

    robots = [
        Robot("AMR-01", (0.0, 0.0), speed=1.5, current_load=1),
        Robot("AMR-02", (5.0, 5.0), speed=1.2, current_load=0),
        Robot("AMR-03", (12.0, 8.0), speed=1.0, current_load=2),
    ]

    allocator = TaskAllocator()
    assignments = allocator.allocate(tasks, robots)

    print("\n작업 할당 결과:")
    print("-" * 60)
    for a in assignments:
        print(
            f"  {a.robot_id} -> {a.task_id}  "
            f"| 비용: {a.cost:.2f}  "
            f"| 예상 도착: {a.estimated_arrival:.1f}초"
        )
