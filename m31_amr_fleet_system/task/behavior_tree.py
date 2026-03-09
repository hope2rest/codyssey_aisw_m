"""
Behavior Tree 프레임워크 모듈.

BehaviorTree.CPP 개념을 참고하여 순수 Python으로 구현한 Behavior Tree
프레임워크이다. AMR 로봇의 '대기-이동-인식-작업-복귀' 미션 시나리오를
트리 구조로 구성하고 실행한다.

노드 타입:
    - ActionNode: 실제 동작을 수행하는 리프 노드
    - ConditionNode: 조건을 검사하는 리프 노드
    - SequenceNode: 자식 노드를 순서대로 실행, 하나라도 실패하면 중단
    - SelectorNode: 자식 노드를 순서대로 시도, 하나라도 성공하면 중단
    - ParallelNode: 자식 노드를 동시에 실행
    - DecoratorNode: 자식 노드의 결과를 변형

노드 상태:
    - SUCCESS: 성공적으로 완료됨
    - FAILURE: 실행 실패
    - RUNNING: 아직 실행 중 (비동기 작업)

외부 의존성:
    - 없음 (순수 Python 표준 라이브러리만 사용)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """
    Behavior Tree 노드 실행 상태.

    SUCCESS: 노드가 성공적으로 완료됨.
    FAILURE: 노드 실행이 실패함.
    RUNNING: 노드가 아직 실행 중(비동기 작업 진행 중).
    """
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class Blackboard:
    """
    Behavior Tree 전역 데이터 저장소(Blackboard).

    노드 간 데이터를 공유하기 위한 key-value 저장소이다.
    모든 노드가 동일한 Blackboard 인스턴스를 참조하여 상태를 공유한다.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """값을 가져온다."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """값을 설정한다."""
        self._data[key] = value

    def has(self, key: str) -> bool:
        """키 존재 여부를 확인한다."""
        return key in self._data

    def clear(self) -> None:
        """모든 데이터를 삭제한다."""
        self._data.clear()


class TreeNode(ABC):
    """
    Behavior Tree 노드의 추상 기반 클래스.

    모든 노드 타입은 이 클래스를 상속받아 tick() 메서드를 구현해야 한다.

    Args:
        name: 노드 이름(디버깅/로깅용).
        blackboard: 공유 데이터 저장소.
    """

    def __init__(self, name: str, blackboard: Optional[Blackboard] = None):
        self.name = name
        self.blackboard = blackboard or Blackboard()
        self._status = NodeStatus.FAILURE

    @abstractmethod
    def tick(self) -> NodeStatus:
        """
        노드를 한 번 실행(tick)한다.

        Returns:
            실행 결과 상태.
        """
        pass

    def reset(self) -> None:
        """노드 상태를 초기화한다."""
        self._status = NodeStatus.FAILURE

    @property
    def status(self) -> NodeStatus:
        """현재 노드 상태를 반환한다."""
        return self._status


class ActionNode(TreeNode):
    """
    동작(Action) 노드.

    실제 로봇 동작(이동, 인식, 집기 등)을 수행하는 리프 노드이다.
    외부에서 주입된 콜백 함수를 실행하여 결과를 반환한다.

    Args:
        name: 노드 이름.
        action_fn: 실행할 함수. Blackboard를 인자로 받고 NodeStatus를 반환.
        blackboard: 공유 데이터 저장소.
    """

    def __init__(
        self,
        name: str,
        action_fn: Callable[[Blackboard], NodeStatus],
        blackboard: Optional[Blackboard] = None,
    ):
        super().__init__(name, blackboard)
        self._action_fn = action_fn

    def tick(self) -> NodeStatus:
        """동작을 실행하고 결과를 반환한다."""
        logger.debug(f"[Action] '{self.name}' 실행")
        try:
            self._status = self._action_fn(self.blackboard)
        except Exception as e:
            logger.error(f"[Action] '{self.name}' 예외 발생: {e}")
            self._status = NodeStatus.FAILURE
        return self._status


class ConditionNode(TreeNode):
    """
    조건(Condition) 노드.

    특정 조건을 검사하여 SUCCESS 또는 FAILURE를 반환하는 리프 노드이다.
    RUNNING 상태는 반환하지 않는다.

    Args:
        name: 노드 이름.
        condition_fn: 조건 검사 함수. Blackboard를 인자로 받고 bool을 반환.
        blackboard: 공유 데이터 저장소.
    """

    def __init__(
        self,
        name: str,
        condition_fn: Callable[[Blackboard], bool],
        blackboard: Optional[Blackboard] = None,
    ):
        super().__init__(name, blackboard)
        self._condition_fn = condition_fn

    def tick(self) -> NodeStatus:
        """조건을 검사하고 결과를 반환한다."""
        logger.debug(f"[Condition] '{self.name}' 검사")
        try:
            result = self._condition_fn(self.blackboard)
            self._status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
        except Exception as e:
            logger.error(f"[Condition] '{self.name}' 예외 발생: {e}")
            self._status = NodeStatus.FAILURE
        return self._status


class SequenceNode(TreeNode):
    """
    시퀀스(Sequence) 노드.

    자식 노드를 왼쪽부터 순서대로 실행한다.
    - 자식이 SUCCESS를 반환하면 다음 자식으로 진행한다.
    - 자식이 FAILURE를 반환하면 즉시 FAILURE를 반환한다.
    - 자식이 RUNNING을 반환하면 RUNNING을 반환하고, 다음 tick에서 해당 자식부터 재개한다.

    Args:
        name: 노드 이름.
        children: 자식 노드 리스트.
        blackboard: 공유 데이터 저장소.
    """

    def __init__(
        self,
        name: str,
        children: Optional[List[TreeNode]] = None,
        blackboard: Optional[Blackboard] = None,
    ):
        super().__init__(name, blackboard)
        self.children: List[TreeNode] = children or []
        self._current_index = 0

    def add_child(self, child: TreeNode) -> "SequenceNode":
        """자식 노드를 추가한다."""
        child.blackboard = self.blackboard
        self.children.append(child)
        return self

    def tick(self) -> NodeStatus:
        """시퀀스를 실행한다."""
        logger.debug(f"[Sequence] '{self.name}' 실행 (인덱스: {self._current_index})")

        while self._current_index < len(self.children):
            child = self.children[self._current_index]
            status = child.tick()

            if status == NodeStatus.FAILURE:
                self._current_index = 0
                self._status = NodeStatus.FAILURE
                return self._status
            elif status == NodeStatus.RUNNING:
                self._status = NodeStatus.RUNNING
                return self._status

            # SUCCESS → 다음 자식으로
            self._current_index += 1

        self._current_index = 0
        self._status = NodeStatus.SUCCESS
        return self._status

    def reset(self) -> None:
        """시퀀스 상태를 초기화한다."""
        super().reset()
        self._current_index = 0
        for child in self.children:
            child.reset()


class SelectorNode(TreeNode):
    """
    셀렉터(Selector / Fallback) 노드.

    자식 노드를 왼쪽부터 순서대로 시도한다.
    - 자식이 SUCCESS를 반환하면 즉시 SUCCESS를 반환한다.
    - 자식이 FAILURE를 반환하면 다음 자식을 시도한다.
    - 모든 자식이 FAILURE이면 FAILURE를 반환한다.

    Args:
        name: 노드 이름.
        children: 자식 노드 리스트.
        blackboard: 공유 데이터 저장소.
    """

    def __init__(
        self,
        name: str,
        children: Optional[List[TreeNode]] = None,
        blackboard: Optional[Blackboard] = None,
    ):
        super().__init__(name, blackboard)
        self.children: List[TreeNode] = children or []
        self._current_index = 0

    def add_child(self, child: TreeNode) -> "SelectorNode":
        """자식 노드를 추가한다."""
        child.blackboard = self.blackboard
        self.children.append(child)
        return self

    def tick(self) -> NodeStatus:
        """셀렉터를 실행한다."""
        logger.debug(f"[Selector] '{self.name}' 실행 (인덱스: {self._current_index})")

        while self._current_index < len(self.children):
            child = self.children[self._current_index]
            status = child.tick()

            if status == NodeStatus.SUCCESS:
                self._current_index = 0
                self._status = NodeStatus.SUCCESS
                return self._status
            elif status == NodeStatus.RUNNING:
                self._status = NodeStatus.RUNNING
                return self._status

            # FAILURE → 다음 자식 시도
            self._current_index += 1

        self._current_index = 0
        self._status = NodeStatus.FAILURE
        return self._status

    def reset(self) -> None:
        """셀렉터 상태를 초기화한다."""
        super().reset()
        self._current_index = 0
        for child in self.children:
            child.reset()


class ParallelNode(TreeNode):
    """
    병렬(Parallel) 노드.

    모든 자식 노드를 매 tick마다 동시에 실행한다.
    성공/실패 정책에 따라 결과를 결정한다.

    Args:
        name: 노드 이름.
        children: 자식 노드 리스트.
        success_threshold: 성공으로 판단하기 위한 최소 성공 자식 수.
        blackboard: 공유 데이터 저장소.
    """

    def __init__(
        self,
        name: str,
        children: Optional[List[TreeNode]] = None,
        success_threshold: int = -1,
        blackboard: Optional[Blackboard] = None,
    ):
        super().__init__(name, blackboard)
        self.children: List[TreeNode] = children or []
        self.success_threshold = success_threshold

    def add_child(self, child: TreeNode) -> "ParallelNode":
        """자식 노드를 추가한다."""
        child.blackboard = self.blackboard
        self.children.append(child)
        return self

    def tick(self) -> NodeStatus:
        """모든 자식을 병렬 실행한다."""
        logger.debug(f"[Parallel] '{self.name}' 실행")

        threshold = self.success_threshold
        if threshold < 0:
            threshold = len(self.children)

        success_count = 0
        failure_count = 0
        running_count = 0

        for child in self.children:
            status = child.tick()
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
            else:
                running_count += 1

        if success_count >= threshold:
            self._status = NodeStatus.SUCCESS
        elif failure_count > len(self.children) - threshold:
            # 성공 임계값을 더 이상 달성할 수 없음
            self._status = NodeStatus.FAILURE
        else:
            self._status = NodeStatus.RUNNING

        return self._status

    def reset(self) -> None:
        """병렬 노드 상태를 초기화한다."""
        super().reset()
        for child in self.children:
            child.reset()


class DecoratorNode(TreeNode):
    """
    데코레이터(Decorator) 노드.

    단일 자식 노드의 실행 결과를 변형한다.

    지원되는 데코레이터 타입:
        - 'inverter': SUCCESS ↔ FAILURE 반전
        - 'repeat': N회 반복 실행
        - 'retry': 실패 시 최대 N회 재시도
        - 'force_success': 항상 SUCCESS 반환
        - 'force_failure': 항상 FAILURE 반환

    Args:
        name: 노드 이름.
        child: 자식 노드.
        decorator_type: 데코레이터 타입 문자열.
        max_count: repeat/retry 시 최대 반복 횟수.
        blackboard: 공유 데이터 저장소.
    """

    def __init__(
        self,
        name: str,
        child: TreeNode,
        decorator_type: str = "inverter",
        max_count: int = 3,
        blackboard: Optional[Blackboard] = None,
    ):
        super().__init__(name, blackboard)
        self.child = child
        self.decorator_type = decorator_type
        self.max_count = max_count
        self._current_count = 0

    def tick(self) -> NodeStatus:
        """자식 노드를 실행하고 결과를 변형한다."""
        logger.debug(f"[Decorator:{self.decorator_type}] '{self.name}' 실행")

        if self.decorator_type == "inverter":
            return self._tick_inverter()
        elif self.decorator_type == "repeat":
            return self._tick_repeat()
        elif self.decorator_type == "retry":
            return self._tick_retry()
        elif self.decorator_type == "force_success":
            self.child.tick()
            self._status = NodeStatus.SUCCESS
            return self._status
        elif self.decorator_type == "force_failure":
            self.child.tick()
            self._status = NodeStatus.FAILURE
            return self._status
        else:
            logger.warning(f"알 수 없는 데코레이터 타입: {self.decorator_type}")
            return self.child.tick()

    def _tick_inverter(self) -> NodeStatus:
        """SUCCESS와 FAILURE를 반전시킨다."""
        status = self.child.tick()
        if status == NodeStatus.SUCCESS:
            self._status = NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            self._status = NodeStatus.SUCCESS
        else:
            self._status = NodeStatus.RUNNING
        return self._status

    def _tick_repeat(self) -> NodeStatus:
        """자식 노드를 N회 반복 실행한다."""
        if self._current_count >= self.max_count:
            self._current_count = 0
            self._status = NodeStatus.SUCCESS
            return self._status

        status = self.child.tick()
        if status == NodeStatus.SUCCESS:
            self._current_count += 1
            if self._current_count >= self.max_count:
                self._current_count = 0
                self._status = NodeStatus.SUCCESS
            else:
                self._status = NodeStatus.RUNNING
        elif status == NodeStatus.RUNNING:
            self._status = NodeStatus.RUNNING
        else:
            self._current_count = 0
            self._status = NodeStatus.FAILURE
        return self._status

    def _tick_retry(self) -> NodeStatus:
        """실패 시 최대 N회 재시도한다."""
        status = self.child.tick()
        if status == NodeStatus.SUCCESS:
            self._current_count = 0
            self._status = NodeStatus.SUCCESS
        elif status == NodeStatus.RUNNING:
            self._status = NodeStatus.RUNNING
        else:
            self._current_count += 1
            if self._current_count < self.max_count:
                logger.info(f"[Retry] '{self.name}' 재시도 {self._current_count}/{self.max_count}")
                self._status = NodeStatus.RUNNING
            else:
                self._current_count = 0
                self._status = NodeStatus.FAILURE
        return self._status

    def reset(self) -> None:
        """데코레이터 상태를 초기화한다."""
        super().reset()
        self._current_count = 0
        self.child.reset()


# ==========================================================================
# AMR 미션 트리 구성: '대기-이동-인식-작업-복귀' 시나리오
# 최소 15개 노드 사용, 에러 복구(Recovery) 로직 포함
# ==========================================================================

def build_amr_mission_tree(blackboard: Optional[Blackboard] = None) -> TreeNode:
    """
    AMR 미션 Behavior Tree를 구성한다.

    시나리오 흐름:
        1. 대기: 작업 명령 수신 대기
        2. 이동: 목표 위치로 이동 (경로 계획 → 내비게이션)
        3. 인식: 목표 객체 인식 (카메라 + LiDAR)
        4. 작업: 화물 적재/하역
        5. 복귀: 홈 위치로 복귀

    에러 복구:
        - 이동 실패 시 경로 재계획 후 재시도(최대 3회)
        - 인식 실패 시 위치 보정 후 재시도
        - 작업 실패 시 재정렬 후 재시도

    Returns:
        미션 트리의 루트 노드.

    노드 구성 (15개 이상):
        Root (Sequence)
        ├── 1. [Condition] 배터리 충분 확인
        ├── 2. [Action] 작업 명령 수신 대기
        ├── 3. [Sequence] 이동 시퀀스
        │   ├── 4. [Action] 경로 계획
        │   ├── 5. [Decorator:retry] 내비게이션 (재시도 3회)
        │   │   └── 6. [Selector] 내비게이션 셀렉터
        │   │       ├── 7. [Action] 주 경로 이동
        │   │       └── 8. [Action] 대체 경로 이동
        ├── 9. [Parallel] 인식 (카메라 + LiDAR 병렬)
        │   ├── 10. [Action] 카메라 인식
        │   └── 11. [Action] LiDAR 스캔
        ├── 12. [Selector] 작업 셀렉터 (작업 or 복구)
        │   ├── 13. [Sequence] 정상 작업
        │   │   ├── 14. [Action] 도킹
        │   │   └── 15. [Action] 화물 적재/하역
        │   └── 16. [Sequence] 작업 복구
        │       ├── 17. [Action] 위치 재정렬
        │       └── 18. [Action] 재시도 작업
        └── 19. [Decorator:retry] 복귀 (재시도 2회)
            └── 20. [Action] 홈 복귀
    """
    bb = blackboard or Blackboard()

    # --- 액션/조건 함수 정의 ---

    def check_battery(board: Blackboard) -> bool:
        """배터리 잔량이 충분한지 확인한다."""
        battery = board.get("battery_level", 100)
        result = battery > 20
        logger.info(f"배터리 확인: {battery}% → {'충분' if result else '부족'}")
        return result

    def wait_for_task(board: Blackboard) -> NodeStatus:
        """작업 명령 수신을 대기한다."""
        if board.has("task_assigned"):
            logger.info("작업 명령 수신 완료")
            return NodeStatus.SUCCESS
        logger.info("작업 명령 대기 중...")
        board.set("task_assigned", True)
        return NodeStatus.SUCCESS

    def plan_path(board: Blackboard) -> NodeStatus:
        """경로를 계획한다."""
        target = board.get("target_position", (10.0, 5.0))
        logger.info(f"경로 계획: 목표 위치 {target}")
        board.set("path_planned", True)
        return NodeStatus.SUCCESS

    def navigate_primary(board: Blackboard) -> NodeStatus:
        """주 경로로 이동한다."""
        if board.get("primary_path_blocked", False):
            logger.warning("주 경로 차단됨")
            return NodeStatus.FAILURE
        progress = board.get("nav_progress", 0.0)
        progress += 0.5
        board.set("nav_progress", progress)
        if progress >= 1.0:
            logger.info("주 경로 이동 완료")
            board.set("nav_progress", 0.0)
            return NodeStatus.SUCCESS
        logger.info(f"이동 중... {progress * 100:.0f}%")
        return NodeStatus.RUNNING

    def navigate_alternative(board: Blackboard) -> NodeStatus:
        """대체 경로로 이동한다."""
        logger.info("대체 경로로 이동")
        board.set("primary_path_blocked", False)
        return NodeStatus.SUCCESS

    def camera_detect(board: Blackboard) -> NodeStatus:
        """카메라로 객체를 인식한다."""
        logger.info("카메라 객체 인식 수행")
        board.set("camera_detected", True)
        return NodeStatus.SUCCESS

    def lidar_scan(board: Blackboard) -> NodeStatus:
        """LiDAR로 주변을 스캔한다."""
        logger.info("LiDAR 스캔 수행")
        board.set("lidar_scanned", True)
        return NodeStatus.SUCCESS

    def dock_to_station(board: Blackboard) -> NodeStatus:
        """스테이션에 도킹한다."""
        if not board.get("docking_aligned", True):
            logger.warning("도킹 정렬 실패")
            return NodeStatus.FAILURE
        logger.info("도킹 완료")
        board.set("docked", True)
        return NodeStatus.SUCCESS

    def load_unload_cargo(board: Blackboard) -> NodeStatus:
        """화물을 적재/하역한다."""
        logger.info("화물 적재/하역 수행")
        board.set("cargo_handled", True)
        return NodeStatus.SUCCESS

    def realign_position(board: Blackboard) -> NodeStatus:
        """위치를 재정렬한다(복구 동작)."""
        logger.info("위치 재정렬 수행 (복구)")
        board.set("docking_aligned", True)
        return NodeStatus.SUCCESS

    def retry_task(board: Blackboard) -> NodeStatus:
        """작업을 재시도한다(복구 동작)."""
        logger.info("작업 재시도 (복구)")
        return NodeStatus.SUCCESS

    def return_home(board: Blackboard) -> NodeStatus:
        """홈 위치로 복귀한다."""
        logger.info("홈 위치 복귀 완료")
        board.set("returned_home", True)
        return NodeStatus.SUCCESS

    # --- 트리 구성 (20개 노드) ---

    # 노드 1: 배터리 확인 (Condition)
    node_battery = ConditionNode("배터리_확인", check_battery, bb)

    # 노드 2: 작업 대기 (Action)
    node_wait = ActionNode("작업_대기", wait_for_task, bb)

    # 노드 7, 8: 내비게이션 셀렉터의 자식들
    node_nav_primary = ActionNode("주경로_이동", navigate_primary, bb)
    node_nav_alt = ActionNode("대체경로_이동", navigate_alternative, bb)

    # 노드 6: 내비게이션 셀렉터
    node_nav_selector = SelectorNode("내비게이션_셀렉터", blackboard=bb)
    node_nav_selector.add_child(node_nav_primary)
    node_nav_selector.add_child(node_nav_alt)

    # 노드 5: 내비게이션 재시도 데코레이터
    node_nav_retry = DecoratorNode(
        "내비게이션_재시도", node_nav_selector, decorator_type="retry", max_count=3, blackboard=bb
    )

    # 노드 4: 경로 계획
    node_plan = ActionNode("경로_계획", plan_path, bb)

    # 노드 3: 이동 시퀀스
    node_move_seq = SequenceNode("이동_시퀀스", blackboard=bb)
    node_move_seq.add_child(node_plan)
    node_move_seq.add_child(node_nav_retry)

    # 노드 10, 11: 인식 병렬 자식들
    node_camera = ActionNode("카메라_인식", camera_detect, bb)
    node_lidar = ActionNode("LiDAR_스캔", lidar_scan, bb)

    # 노드 9: 인식 병렬 노드
    node_perception = ParallelNode("인식_병렬", success_threshold=1, blackboard=bb)
    node_perception.add_child(node_camera)
    node_perception.add_child(node_lidar)

    # 노드 14, 15: 정상 작업 시퀀스 자식들
    node_dock = ActionNode("도킹", dock_to_station, bb)
    node_cargo = ActionNode("화물_처리", load_unload_cargo, bb)

    # 노드 13: 정상 작업 시퀀스
    node_work_seq = SequenceNode("정상_작업", blackboard=bb)
    node_work_seq.add_child(node_dock)
    node_work_seq.add_child(node_cargo)

    # 노드 17, 18: 복구 시퀀스 자식들
    node_realign = ActionNode("위치_재정렬", realign_position, bb)
    node_retry = ActionNode("재시도_작업", retry_task, bb)

    # 노드 16: 복구 시퀀스
    node_recovery_seq = SequenceNode("작업_복구", blackboard=bb)
    node_recovery_seq.add_child(node_realign)
    node_recovery_seq.add_child(node_retry)

    # 노드 12: 작업 셀렉터 (정상 or 복구)
    node_work_selector = SelectorNode("작업_셀렉터", blackboard=bb)
    node_work_selector.add_child(node_work_seq)
    node_work_selector.add_child(node_recovery_seq)

    # 노드 20: 홈 복귀
    node_home = ActionNode("홈_복귀", return_home, bb)

    # 노드 19: 복귀 재시도 데코레이터
    node_return_retry = DecoratorNode(
        "복귀_재시도", node_home, decorator_type="retry", max_count=2, blackboard=bb
    )

    # 루트: 전체 미션 시퀀스
    root = SequenceNode("AMR_미션_루트", blackboard=bb)
    root.add_child(node_battery)       # 1
    root.add_child(node_wait)          # 2
    root.add_child(node_move_seq)      # 3 (→ 4, 5 → 6 → 7, 8)
    root.add_child(node_perception)    # 9 (→ 10, 11)
    root.add_child(node_work_selector) # 12 (→ 13 → 14,15 / 16 → 17,18)
    root.add_child(node_return_retry)  # 19 (→ 20)

    return root


def count_nodes(node: TreeNode) -> int:
    """트리의 총 노드 수를 재귀적으로 세어 반환한다."""
    count = 1
    if isinstance(node, (SequenceNode, SelectorNode, ParallelNode)):
        for child in node.children:
            count += count_nodes(child)
    elif isinstance(node, DecoratorNode):
        count += count_nodes(node.child)
    return count


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bb = Blackboard()
    bb.set("battery_level", 80)
    bb.set("target_position", (10.0, 5.0))

    root = build_amr_mission_tree(bb)

    print(f"트리 노드 수: {count_nodes(root)}")
    print("=" * 60)
    print("AMR 미션 실행 시작")
    print("=" * 60)

    max_ticks = 10
    for tick_num in range(max_ticks):
        print(f"\n--- Tick {tick_num + 1} ---")
        status = root.tick()
        print(f"→ 루트 상태: {status.name}")

        if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
            break

    print("\n" + "=" * 60)
    print(f"미션 최종 결과: {root.status.name}")
