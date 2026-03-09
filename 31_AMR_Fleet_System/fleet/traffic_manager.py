"""
경로 충돌 및 교통 관리 모듈.

다수의 AMR 로봇이 동시에 운행할 때 발생하는 경로 충돌을 예측하고 해소한다.
교차로/병목 구간의 우선순위를 결정하고, 교착(Deadlock) 상태를
Wait-For Graph 기반으로 탐지하여 해소한다.

주요 기능:
    - 경로 충돌 예측: 시간-공간 기반 충돌 검사
    - 교차로/병목 구간 우선순위 결정
    - 교착(Deadlock) 탐지: Wait-For Graph + 사이클 감지(DFS)
    - 교착 해소: 우선순위 양보 + 대체 경로

외부 의존성:
    - numpy (필수)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """
    충돌 유형.

    HEAD_ON: 정면 충돌 (마주보고 접근).
    CROSSING: 교차 충돌 (교차로에서 교차).
    REAR_END: 추돌 (동일 방향, 속도 차이).
    MERGE: 합류 충돌 (동일 지점으로 합류).
    """
    HEAD_ON = auto()
    CROSSING = auto()
    REAR_END = auto()
    MERGE = auto()


@dataclass
class PathSegment:
    """
    경로의 한 구간(세그먼트).

    Attributes:
        position: 구간 위치 (x, y).
        time_enter: 구간 진입 예상 시간(초).
        time_exit: 구간 이탈 예상 시간(초).
    """
    position: Tuple[float, float]
    time_enter: float
    time_exit: float


@dataclass
class RobotPath:
    """
    로봇의 계획된 경로.

    Attributes:
        robot_id: 로봇 고유 식별자.
        segments: 경로 세그먼트 리스트.
        priority: 로봇 우선순위 (높을수록 우선).
        speed: 이동 속도 (m/s).
    """
    robot_id: str
    segments: List[PathSegment]
    priority: int = 1
    speed: float = 1.0


@dataclass
class Conflict:
    """
    탐지된 충돌 정보.

    Attributes:
        robot_a: 첫 번째 로봇 ID.
        robot_b: 두 번째 로봇 ID.
        conflict_type: 충돌 유형.
        position: 충돌 위치 (x, y).
        time: 충돌 예상 시간(초).
        resolved: 해소 여부.
        resolution: 해소 방법 설명.
    """
    robot_a: str
    robot_b: str
    conflict_type: ConflictType
    position: Tuple[float, float]
    time: float
    resolved: bool = False
    resolution: str = ""


@dataclass
class Zone:
    """
    교차로/병목 구간 정보.

    Attributes:
        zone_id: 구간 고유 식별자.
        position: 구간 중심 위치 (x, y).
        radius: 구간 반경 (m).
        capacity: 동시 통과 가능 로봇 수.
        current_occupants: 현재 점유 로봇 ID 리스트.
        queue: 대기 중인 로봇 ID 리스트.
    """
    zone_id: str
    position: Tuple[float, float]
    radius: float = 1.0
    capacity: int = 1
    current_occupants: List[str] = field(default_factory=list)
    queue: List[str] = field(default_factory=list)


class TrafficManager:
    """
    경로 교통 관리기.

    다수의 AMR 로봇 경로를 관리하며 충돌을 예측하고 해소한다.
    교착 상태를 탐지하고 해소하는 기능도 포함한다.

    사용 예시::

        manager = TrafficManager(safety_distance=0.5)
        manager.register_zone(Zone("교차로A", (5, 5), radius=1.5))

        path1 = RobotPath("R1", segments=[...], priority=2)
        path2 = RobotPath("R2", segments=[...], priority=1)

        conflicts = manager.predict_conflicts([path1, path2])
        manager.resolve_conflicts(conflicts, [path1, path2])

        deadlocks = manager.detect_deadlock()

    Args:
        safety_distance: 로봇 간 최소 안전 거리 (m).
        time_resolution: 시간 해상도 (초). 충돌 검사 간격.
    """

    def __init__(self, safety_distance: float = 0.5, time_resolution: float = 0.5):
        self.safety_distance = safety_distance
        self.time_resolution = time_resolution

        self._zones: Dict[str, Zone] = {}
        self._robot_paths: Dict[str, RobotPath] = {}
        self._wait_for_graph: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # 구간 관리
    # ------------------------------------------------------------------

    def register_zone(self, zone: Zone) -> None:
        """교차로/병목 구간을 등록한다."""
        self._zones[zone.zone_id] = zone
        logger.info(f"구간 등록: {zone.zone_id} at {zone.position}")

    def register_path(self, path: RobotPath) -> None:
        """로봇 경로를 등록한다."""
        self._robot_paths[path.robot_id] = path

    # ------------------------------------------------------------------
    # 충돌 예측
    # ------------------------------------------------------------------

    def predict_conflicts(self, paths: List[RobotPath]) -> List[Conflict]:
        """
        경로 충돌을 예측한다.

        모든 로봇 쌍에 대해 시간-공간적으로 안전 거리 이내에
        접근하는 구간을 탐색한다.

        Args:
            paths: 검사할 로봇 경로 리스트.

        Returns:
            탐지된 충돌 리스트.
        """
        conflicts: List[Conflict] = []

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                path_a = paths[i]
                path_b = paths[j]
                pair_conflicts = self._check_pair_conflict(path_a, path_b)
                conflicts.extend(pair_conflicts)

        logger.info(f"충돌 예측 완료: {len(conflicts)}건 탐지")
        return conflicts

    def _check_pair_conflict(
        self, path_a: RobotPath, path_b: RobotPath
    ) -> List[Conflict]:
        """
        두 로봇 경로 간 충돌을 검사한다.

        각 세그먼트의 시간 겹침 여부와 공간 거리를 확인한다.

        Args:
            path_a: 첫 번째 로봇 경로.
            path_b: 두 번째 로봇 경로.

        Returns:
            탐지된 충돌 리스트.
        """
        conflicts = []

        for seg_a in path_a.segments:
            for seg_b in path_b.segments:
                # 시간 겹침 확인
                time_overlap = (
                    seg_a.time_enter < seg_b.time_exit
                    and seg_b.time_enter < seg_a.time_exit
                )
                if not time_overlap:
                    continue

                # 공간 거리 확인
                dx = seg_a.position[0] - seg_b.position[0]
                dy = seg_a.position[1] - seg_b.position[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < self.safety_distance:
                    # 충돌 유형 판별
                    conflict_type = self._classify_conflict(
                        path_a, path_b, seg_a, seg_b
                    )

                    conflict_time = max(seg_a.time_enter, seg_b.time_enter)
                    conflict_pos = (
                        (seg_a.position[0] + seg_b.position[0]) / 2,
                        (seg_a.position[1] + seg_b.position[1]) / 2,
                    )

                    conflicts.append(
                        Conflict(
                            robot_a=path_a.robot_id,
                            robot_b=path_b.robot_id,
                            conflict_type=conflict_type,
                            position=conflict_pos,
                            time=conflict_time,
                        )
                    )

        return conflicts

    def _classify_conflict(
        self,
        path_a: RobotPath,
        path_b: RobotPath,
        seg_a: PathSegment,
        seg_b: PathSegment,
    ) -> ConflictType:
        """충돌 유형을 분류한다."""
        # 간단한 분류: 이동 방향 기반
        idx_a = path_a.segments.index(seg_a)
        idx_b = path_b.segments.index(seg_b)

        if idx_a > 0 and idx_b > 0:
            prev_a = path_a.segments[idx_a - 1]
            prev_b = path_b.segments[idx_b - 1]

            dir_a = (
                seg_a.position[0] - prev_a.position[0],
                seg_a.position[1] - prev_a.position[1],
            )
            dir_b = (
                seg_b.position[0] - prev_b.position[0],
                seg_b.position[1] - prev_b.position[1],
            )

            # 방향 벡터 내적
            dot = dir_a[0] * dir_b[0] + dir_a[1] * dir_b[1]
            mag_a = math.sqrt(dir_a[0] ** 2 + dir_a[1] ** 2)
            mag_b = math.sqrt(dir_b[0] ** 2 + dir_b[1] ** 2)

            if mag_a > 1e-6 and mag_b > 1e-6:
                cos_angle = dot / (mag_a * mag_b)
                cos_angle = max(-1.0, min(1.0, cos_angle))

                if cos_angle < -0.7:
                    return ConflictType.HEAD_ON
                elif cos_angle > 0.7:
                    return ConflictType.REAR_END
                else:
                    return ConflictType.CROSSING

        return ConflictType.MERGE

    # ------------------------------------------------------------------
    # 충돌 해소
    # ------------------------------------------------------------------

    def resolve_conflicts(
        self, conflicts: List[Conflict], paths: List[RobotPath]
    ) -> List[Conflict]:
        """
        탐지된 충돌을 해소한다.

        해소 전략:
            1. 우선순위가 낮은 로봇이 양보 (대기)
            2. 동일 우선순위면 경로가 짧은 로봇이 우선
            3. 대체 경로가 가능하면 경로 변경

        Args:
            conflicts: 충돌 리스트.
            paths: 로봇 경로 리스트.

        Returns:
            해소 처리된 충돌 리스트.
        """
        path_map = {p.robot_id: p for p in paths}

        for conflict in conflicts:
            if conflict.resolved:
                continue

            path_a = path_map.get(conflict.robot_a)
            path_b = path_map.get(conflict.robot_b)

            if path_a is None or path_b is None:
                continue

            # 우선순위 비교
            if path_a.priority > path_b.priority:
                yielding_robot = conflict.robot_b
                priority_robot = conflict.robot_a
            elif path_b.priority > path_a.priority:
                yielding_robot = conflict.robot_a
                priority_robot = conflict.robot_b
            else:
                # 동일 우선순위 -> 경로가 긴 로봇이 양보
                if len(path_a.segments) >= len(path_b.segments):
                    yielding_robot = conflict.robot_a
                    priority_robot = conflict.robot_b
                else:
                    yielding_robot = conflict.robot_b
                    priority_robot = conflict.robot_a

            # 양보 로봇의 해당 세그먼트 시간을 지연
            yielding_path = path_map[yielding_robot]
            delay = 2.0  # 2초 대기

            for seg in yielding_path.segments:
                if seg.time_enter >= conflict.time - self.time_resolution:
                    seg.time_enter += delay
                    seg.time_exit += delay

            # Wait-For Graph 업데이트
            self._wait_for_graph[yielding_robot].add(priority_robot)

            conflict.resolved = True
            conflict.resolution = (
                f"{yielding_robot}이(가) {priority_robot}에게 양보 ({delay}초 대기)"
            )
            logger.info(f"충돌 해소: {conflict.resolution}")

        return conflicts

    # ------------------------------------------------------------------
    # 교차로/병목 구간 관리
    # ------------------------------------------------------------------

    def request_zone_entry(self, zone_id: str, robot_id: str, priority: int = 1) -> bool:
        """
        교차로/병목 구간 진입을 요청한다.

        구간 용량이 여유 있으면 즉시 진입, 아니면 대기열에 추가.

        Args:
            zone_id: 구간 ID.
            robot_id: 로봇 ID.
            priority: 로봇 우선순위.

        Returns:
            진입 허용 여부.
        """
        zone = self._zones.get(zone_id)
        if zone is None:
            logger.warning(f"등록되지 않은 구간: {zone_id}")
            return True  # 등록 안 된 구간은 자유 통과

        if robot_id in zone.current_occupants:
            return True  # 이미 진입 중

        if len(zone.current_occupants) < zone.capacity:
            zone.current_occupants.append(robot_id)
            logger.info(f"[{zone_id}] {robot_id} 진입 허용")
            return True
        else:
            if robot_id not in zone.queue:
                # 우선순위 기반 대기열 삽입
                zone.queue.append(robot_id)
                zone.queue.sort(key=lambda r: -priority)

                # 대기 중인 로봇의 Wait-For Graph 갱신
                for occupant in zone.current_occupants:
                    self._wait_for_graph[robot_id].add(occupant)

            logger.info(f"[{zone_id}] {robot_id} 대기 (대기열: {len(zone.queue)})")
            return False

    def release_zone(self, zone_id: str, robot_id: str) -> None:
        """
        교차로/병목 구간을 해제한다.

        해제 후 대기열의 다음 로봇을 자동 진입시킨다.

        Args:
            zone_id: 구간 ID.
            robot_id: 로봇 ID.
        """
        zone = self._zones.get(zone_id)
        if zone is None:
            return

        if robot_id in zone.current_occupants:
            zone.current_occupants.remove(robot_id)
            logger.info(f"[{zone_id}] {robot_id} 이탈")

            # Wait-For Graph에서 관련 엣지 제거
            for waiting in list(self._wait_for_graph.keys()):
                self._wait_for_graph[waiting].discard(robot_id)

            # 대기열에서 다음 로봇 진입
            while zone.queue and len(zone.current_occupants) < zone.capacity:
                next_robot = zone.queue.pop(0)
                zone.current_occupants.append(next_robot)
                logger.info(f"[{zone_id}] {next_robot} 대기열에서 진입")

    # ------------------------------------------------------------------
    # 교착(Deadlock) 탐지 및 해소
    # ------------------------------------------------------------------

    def detect_deadlock(self) -> List[List[str]]:
        """
        Wait-For Graph에서 교착 상태를 탐지한다.

        DFS(깊이 우선 탐색)를 사용하여 사이클을 검출한다.
        사이클이 존재하면 교착 상태이다.

        Returns:
            탐지된 교착 사이클 리스트. 각 사이클은 로봇 ID 리스트.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            """DFS로 사이클을 탐지한다."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._wait_for_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # 사이클 발견
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    logger.warning(f"교착 탐지: {' -> '.join(cycle)}")

            path.pop()
            rec_stack.discard(node)

        all_nodes = set(self._wait_for_graph.keys())
        for node in all_nodes:
            if node not in visited:
                dfs(node, [])

        if cycles:
            logger.warning(f"총 {len(cycles)}건의 교착 상태 탐지됨")
        else:
            logger.info("교착 상태 없음")

        return cycles

    def resolve_deadlock(self, cycles: List[List[str]]) -> List[str]:
        """
        교착 상태를 해소한다.

        해소 전략:
            1. 사이클 내 가장 낮은 우선순위의 로봇을 선택
            2. 해당 로봇의 Wait-For 관계를 해제 (양보)
            3. 필요시 대체 경로 할당

        Args:
            cycles: 교착 사이클 리스트.

        Returns:
            양보한 로봇 ID 리스트.
        """
        yielded_robots: List[str] = []

        for cycle in cycles:
            if len(cycle) < 2:
                continue

            # 사이클 내 로봇의 우선순위 비교
            unique_robots = list(dict.fromkeys(cycle))  # 순서 유지 중복 제거
            priorities = {}
            for rid in unique_robots:
                path = self._robot_paths.get(rid)
                priorities[rid] = path.priority if path else 0

            # 최저 우선순위 로봇 선택
            victim = min(unique_robots, key=lambda r: priorities.get(r, 0))

            # Wait-For 관계 해제
            self._wait_for_graph[victim].clear()

            # 모든 구간 대기열에서 제거
            for zone in self._zones.values():
                if victim in zone.queue:
                    zone.queue.remove(victim)

            yielded_robots.append(victim)
            logger.info(
                f"교착 해소: {victim}이(가) 양보 (우선순위: {priorities.get(victim, 0)})"
            )

        return yielded_robots

    def get_zone_status(self) -> Dict[str, Dict]:
        """
        모든 구간의 현재 상태를 반환한다.

        Returns:
            구간별 상태 딕셔너리.
        """
        status = {}
        for zone_id, zone in self._zones.items():
            status[zone_id] = {
                "occupants": list(zone.current_occupants),
                "queue": list(zone.queue),
                "capacity": zone.capacity,
                "utilization": len(zone.current_occupants) / max(zone.capacity, 1),
            }
        return status

    def clear_wait_graph(self) -> None:
        """Wait-For Graph를 초기화한다."""
        self._wait_for_graph.clear()


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manager = TrafficManager(safety_distance=0.8)

    # 교차로 등록
    manager.register_zone(Zone("교차로A", (5.0, 5.0), radius=1.5, capacity=1))

    # 두 로봇의 교차 경로
    path1 = RobotPath(
        "AMR-01",
        segments=[
            PathSegment((0, 5), 0, 2),
            PathSegment((2, 5), 2, 4),
            PathSegment((5, 5), 4, 6),  # 교차로 통과
            PathSegment((8, 5), 6, 8),
        ],
        priority=2,
    )

    path2 = RobotPath(
        "AMR-02",
        segments=[
            PathSegment((5, 0), 0, 2),
            PathSegment((5, 2), 2, 4),
            PathSegment((5, 5), 4, 6),  # 교차로 통과 (충돌!)
            PathSegment((5, 8), 6, 8),
        ],
        priority=1,
    )

    manager.register_path(path1)
    manager.register_path(path2)

    # 충돌 예측
    conflicts = manager.predict_conflicts([path1, path2])
    print(f"\n탐지된 충돌: {len(conflicts)}건")
    for c in conflicts:
        print(f"  {c.robot_a} vs {c.robot_b} ({c.conflict_type.name}) "
              f"at {c.position}, t={c.time:.1f}s")

    # 충돌 해소
    manager.resolve_conflicts(conflicts, [path1, path2])

    # 교착 탐지 시뮬레이션
    manager._wait_for_graph["AMR-01"].add("AMR-02")
    manager._wait_for_graph["AMR-02"].add("AMR-03")
    manager._wait_for_graph["AMR-03"].add("AMR-01")

    cycles = manager.detect_deadlock()
    if cycles:
        yielded = manager.resolve_deadlock(cycles)
        print(f"\n교착 해소: {yielded}")
