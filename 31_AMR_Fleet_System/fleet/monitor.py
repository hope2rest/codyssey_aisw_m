"""
Fleet 실시간 모니터링 모듈.

다수의 AMR 로봇 상태를 실시간으로 모니터링하고, KPI(핵심 성과 지표)를
계산하며, 이상 상황 발생 시 알림을 생성한다. 텍스트 기반 대시보드를
콘솔에 출력한다.

주요 기능:
    - 실시간 로봇 상태 모니터링 (위치, 배터리, 작업 상태)
    - KPI 계산: 작업 처리량, 평균 작업 시간, 로봇 가동률
    - 이상 상황 알림: 긴급 정지, 작업 실패, 교착, 배터리 부족
    - 텍스트 기반 대시보드 출력

외부 의존성:
    - numpy (필수)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RobotStatus(Enum):
    """
    로봇 운행 상태.

    IDLE: 대기 중.
    MOVING: 이동 중.
    WORKING: 작업 수행 중.
    CHARGING: 충전 중.
    ERROR: 오류 발생.
    EMERGENCY_STOP: 긴급 정지.
    """
    IDLE = auto()
    MOVING = auto()
    WORKING = auto()
    CHARGING = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()


class AlertLevel(Enum):
    """
    알림 등급.

    INFO: 정보성 알림.
    WARNING: 경고 (주의 필요).
    CRITICAL: 위험 (즉시 조치 필요).
    """
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class RobotState:
    """
    로봇의 현재 상태 정보.

    Attributes:
        robot_id: 로봇 고유 식별자.
        position: 현재 위치 (x, y).
        status: 운행 상태.
        battery_level: 배터리 잔량 (0~100).
        current_task: 현재 수행 중인 작업 ID. None이면 미할당.
        speed: 현재 속도 (m/s).
        total_tasks_completed: 누적 완료 작업 수.
        total_distance: 누적 이동 거리 (m).
        uptime: 가동 시간(초).
        last_update: 마지막 상태 업데이트 시간 (Unix timestamp).
        error_count: 누적 오류 횟수.
    """
    robot_id: str
    position: Tuple[float, float] = (0.0, 0.0)
    status: RobotStatus = RobotStatus.IDLE
    battery_level: float = 100.0
    current_task: Optional[str] = None
    speed: float = 0.0
    total_tasks_completed: int = 0
    total_distance: float = 0.0
    uptime: float = 0.0
    last_update: float = field(default_factory=time.time)
    error_count: int = 0


@dataclass
class Alert:
    """
    이상 상황 알림.

    Attributes:
        alert_id: 알림 고유 식별자.
        level: 알림 등급.
        robot_id: 관련 로봇 ID.
        message: 알림 메시지.
        timestamp: 발생 시간.
        acknowledged: 확인 여부.
    """
    alert_id: int
    level: AlertLevel
    robot_id: str
    message: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


@dataclass
class TaskRecord:
    """
    완료된 작업 기록.

    Attributes:
        task_id: 작업 ID.
        robot_id: 수행 로봇 ID.
        start_time: 시작 시간.
        end_time: 종료 시간.
        success: 성공 여부.
    """
    task_id: str
    robot_id: str
    start_time: float
    end_time: float
    success: bool


@dataclass
class KPI:
    """
    핵심 성과 지표(KPI).

    Attributes:
        throughput: 시간당 작업 처리량 (tasks/hour).
        avg_task_time: 평균 작업 소요 시간 (초).
        robot_utilization: 로봇 평균 가동률 (0.0 ~ 1.0).
        total_completed: 총 완료 작업 수.
        total_failed: 총 실패 작업 수.
        success_rate: 작업 성공률 (0.0 ~ 1.0).
        avg_battery: 평균 배터리 잔량 (%).
        fleet_availability: Fleet 가용률 (가용 로봇 / 전체 로봇).
    """
    throughput: float = 0.0
    avg_task_time: float = 0.0
    robot_utilization: float = 0.0
    total_completed: int = 0
    total_failed: int = 0
    success_rate: float = 0.0
    avg_battery: float = 0.0
    fleet_availability: float = 0.0


class FleetMonitor:
    """
    Fleet 실시간 모니터링 시스템.

    다수의 AMR 로봇 상태를 추적하고, KPI를 계산하며,
    이상 상황 발생 시 알림을 생성한다.

    사용 예시::

        monitor = FleetMonitor()
        monitor.register_robot("AMR-01")
        monitor.register_robot("AMR-02")

        monitor.update_state("AMR-01", position=(3, 5), status=RobotStatus.MOVING)
        monitor.record_task("T-001", "AMR-01", start, end, success=True)

        kpi = monitor.compute_kpi()
        monitor.print_dashboard()

    Args:
        battery_warning_threshold: 배터리 경고 임계값 (%).
        battery_critical_threshold: 배터리 위험 임계값 (%).
        max_alerts: 최대 알림 보관 수.
        task_history_size: 작업 기록 보관 수.
    """

    def __init__(
        self,
        battery_warning_threshold: float = 30.0,
        battery_critical_threshold: float = 10.0,
        max_alerts: int = 100,
        task_history_size: int = 500,
    ):
        self.battery_warning_threshold = battery_warning_threshold
        self.battery_critical_threshold = battery_critical_threshold

        self._robots: Dict[str, RobotState] = {}
        self._alerts: Deque[Alert] = deque(maxlen=max_alerts)
        self._task_history: Deque[TaskRecord] = deque(maxlen=task_history_size)
        self._alert_counter = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # 로봇 등록 및 상태 업데이트
    # ------------------------------------------------------------------

    def register_robot(self, robot_id: str, position: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        로봇을 모니터링 시스템에 등록한다.

        Args:
            robot_id: 로봇 ID.
            position: 초기 위치.
        """
        self._robots[robot_id] = RobotState(robot_id=robot_id, position=position)
        logger.info(f"로봇 등록: {robot_id}")

    def update_state(
        self,
        robot_id: str,
        position: Optional[Tuple[float, float]] = None,
        status: Optional[RobotStatus] = None,
        battery_level: Optional[float] = None,
        current_task: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None:
        """
        로봇 상태를 업데이트한다.

        상태 변경 시 이상 상황을 자동 감지하여 알림을 생성한다.

        Args:
            robot_id: 로봇 ID.
            position: 새 위치.
            status: 새 상태.
            battery_level: 새 배터리 잔량.
            current_task: 새 작업 ID.
            speed: 새 속도.
        """
        state = self._robots.get(robot_id)
        if state is None:
            logger.warning(f"등록되지 않은 로봇: {robot_id}")
            return

        prev_status = state.status

        # 위치 업데이트 및 이동 거리 누적
        if position is not None:
            dx = position[0] - state.position[0]
            dy = position[1] - state.position[1]
            state.total_distance += (dx * dx + dy * dy) ** 0.5
            state.position = position

        if status is not None:
            state.status = status
        if battery_level is not None:
            state.battery_level = battery_level
        if current_task is not None:
            state.current_task = current_task
        if speed is not None:
            state.speed = speed

        state.last_update = time.time()
        state.uptime = state.last_update - self._start_time

        # 이상 상황 감지
        self._check_anomalies(state, prev_status)

    def _check_anomalies(self, state: RobotState, prev_status: RobotStatus) -> None:
        """
        이상 상황을 감지하고 알림을 생성한다.

        Args:
            state: 현재 로봇 상태.
            prev_status: 이전 상태.
        """
        # 긴급 정지 감지
        if state.status == RobotStatus.EMERGENCY_STOP and prev_status != RobotStatus.EMERGENCY_STOP:
            self._create_alert(
                AlertLevel.CRITICAL,
                state.robot_id,
                f"긴급 정지 발생! 위치: ({state.position[0]:.1f}, {state.position[1]:.1f})",
            )

        # 오류 상태 감지
        if state.status == RobotStatus.ERROR and prev_status != RobotStatus.ERROR:
            state.error_count += 1
            self._create_alert(
                AlertLevel.WARNING,
                state.robot_id,
                f"오류 발생 (누적: {state.error_count}회)",
            )

        # 배터리 부족 경고
        if state.battery_level <= self.battery_critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL,
                state.robot_id,
                f"배터리 위험! {state.battery_level:.1f}%",
            )
        elif state.battery_level <= self.battery_warning_threshold:
            self._create_alert(
                AlertLevel.WARNING,
                state.robot_id,
                f"배터리 부족 경고: {state.battery_level:.1f}%",
            )

    def _create_alert(self, level: AlertLevel, robot_id: str, message: str) -> Alert:
        """알림을 생성하고 저장한다."""
        self._alert_counter += 1
        alert = Alert(
            alert_id=self._alert_counter,
            level=level,
            robot_id=robot_id,
            message=message,
        )
        self._alerts.append(alert)
        logger.warning(f"[{level.name}] {robot_id}: {message}")
        return alert

    def report_deadlock(self, robot_ids: List[str]) -> None:
        """
        교착 상태를 보고한다.

        Args:
            robot_ids: 교착에 관련된 로봇 ID 리스트.
        """
        robots_str = ", ".join(robot_ids)
        for rid in robot_ids:
            self._create_alert(
                AlertLevel.CRITICAL,
                rid,
                f"교착 상태 탐지! 관련 로봇: {robots_str}",
            )

    # ------------------------------------------------------------------
    # 작업 기록
    # ------------------------------------------------------------------

    def record_task(
        self,
        task_id: str,
        robot_id: str,
        start_time: float,
        end_time: float,
        success: bool = True,
    ) -> None:
        """
        완료된 작업을 기록한다.

        Args:
            task_id: 작업 ID.
            robot_id: 수행 로봇 ID.
            start_time: 시작 시간.
            end_time: 종료 시간.
            success: 성공 여부.
        """
        record = TaskRecord(
            task_id=task_id,
            robot_id=robot_id,
            start_time=start_time,
            end_time=end_time,
            success=success,
        )
        self._task_history.append(record)

        # 로봇 통계 업데이트
        state = self._robots.get(robot_id)
        if state is not None:
            state.total_tasks_completed += 1
            state.current_task = None

        if not success:
            self._create_alert(
                AlertLevel.WARNING,
                robot_id,
                f"작업 실패: {task_id}",
            )

    # ------------------------------------------------------------------
    # KPI 계산
    # ------------------------------------------------------------------

    def compute_kpi(self, time_window: float = 3600.0) -> KPI:
        """
        핵심 성과 지표(KPI)를 계산한다.

        Args:
            time_window: KPI 계산 시간 윈도우(초). 기본 1시간.

        Returns:
            KPI 데이터.
        """
        current_time = time.time()
        window_start = current_time - time_window

        # 시간 윈도우 내 작업 필터링
        recent_tasks = [
            t for t in self._task_history if t.end_time >= window_start
        ]

        completed = [t for t in recent_tasks if t.success]
        failed = [t for t in recent_tasks if not t.success]

        # 작업 처리량 (tasks/hour)
        elapsed_hours = min(time_window, current_time - self._start_time) / 3600.0
        throughput = len(completed) / max(elapsed_hours, 1e-6)

        # 평균 작업 시간
        task_times = [t.end_time - t.start_time for t in completed]
        avg_task_time = float(np.mean(task_times)) if task_times else 0.0

        # 성공률
        total = len(completed) + len(failed)
        success_rate = len(completed) / max(total, 1)

        # 로봇 가동률 (IDLE, CHARGING이 아닌 상태의 비율)
        active_statuses = {RobotStatus.MOVING, RobotStatus.WORKING}
        available_statuses = {
            RobotStatus.IDLE, RobotStatus.MOVING, RobotStatus.WORKING
        }

        active_count = sum(
            1 for r in self._robots.values() if r.status in active_statuses
        )
        available_count = sum(
            1 for r in self._robots.values() if r.status in available_statuses
        )
        total_robots = len(self._robots)

        utilization = active_count / max(total_robots, 1)
        availability = available_count / max(total_robots, 1)

        # 평균 배터리
        batteries = [r.battery_level for r in self._robots.values()]
        avg_battery = float(np.mean(batteries)) if batteries else 0.0

        return KPI(
            throughput=throughput,
            avg_task_time=avg_task_time,
            robot_utilization=utilization,
            total_completed=len(completed),
            total_failed=len(failed),
            success_rate=success_rate,
            avg_battery=avg_battery,
            fleet_availability=availability,
        )

    # ------------------------------------------------------------------
    # 대시보드 출력
    # ------------------------------------------------------------------

    def print_dashboard(self) -> str:
        """
        텍스트 기반 대시보드를 생성하고 출력한다.

        콘솔에 현재 Fleet 상태, KPI, 최근 알림을 표시한다.

        Returns:
            대시보드 문자열.
        """
        kpi = self.compute_kpi()
        now = time.time()

        lines: List[str] = []
        width = 70

        # 헤더
        lines.append("=" * width)
        lines.append(" AMR Fleet 모니터링 대시보드 ".center(width, "="))
        lines.append("=" * width)

        # KPI 섹션
        lines.append("")
        lines.append("[KPI 요약]")
        lines.append(f"  작업 처리량      : {kpi.throughput:.1f} tasks/hour")
        lines.append(f"  평균 작업 시간   : {kpi.avg_task_time:.1f} 초")
        lines.append(f"  작업 성공률      : {kpi.success_rate * 100:.1f}%")
        lines.append(f"  로봇 가동률      : {kpi.robot_utilization * 100:.1f}%")
        lines.append(f"  Fleet 가용률     : {kpi.fleet_availability * 100:.1f}%")
        lines.append(f"  평균 배터리      : {kpi.avg_battery:.1f}%")
        lines.append(f"  완료/실패        : {kpi.total_completed} / {kpi.total_failed}")

        # 로봇 상태 섹션
        lines.append("")
        lines.append("[로봇 상태]")
        lines.append(
            f"  {'ID':<10} {'상태':<14} {'배터리':>6} {'위치':>16} "
            f"{'현재 작업':>10} {'완료':>4}"
        )
        lines.append("  " + "-" * (width - 4))

        for robot_id in sorted(self._robots.keys()):
            state = self._robots[robot_id]
            status_str = state.status.name
            pos_str = f"({state.position[0]:.1f}, {state.position[1]:.1f})"
            task_str = state.current_task or "-"
            bat_str = f"{state.battery_level:.0f}%"

            # 상태별 표시
            if state.status == RobotStatus.EMERGENCY_STOP:
                status_str = "!! E-STOP !!"
            elif state.status == RobotStatus.ERROR:
                status_str = "** ERROR **"

            # 배터리 표시
            if state.battery_level <= self.battery_critical_threshold:
                bat_str = f"{state.battery_level:.0f}% [!]"
            elif state.battery_level <= self.battery_warning_threshold:
                bat_str = f"{state.battery_level:.0f}% [W]"

            lines.append(
                f"  {robot_id:<10} {status_str:<14} {bat_str:>6} "
                f"{pos_str:>16} {task_str:>10} {state.total_tasks_completed:>4}"
            )

        # 최근 알림 섹션
        lines.append("")
        lines.append("[최근 알림]")
        recent_alerts = list(self._alerts)[-5:]  # 최근 5건
        if recent_alerts:
            for alert in reversed(recent_alerts):
                level_tag = f"[{alert.level.name}]"
                lines.append(
                    f"  {level_tag:<12} {alert.robot_id:<10} {alert.message}"
                )
        else:
            lines.append("  (알림 없음)")

        lines.append("")
        lines.append("=" * width)

        dashboard = "\n".join(lines)
        print(dashboard)
        return dashboard

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------

    def get_robot_state(self, robot_id: str) -> Optional[RobotState]:
        """특정 로봇의 상태를 반환한다."""
        return self._robots.get(robot_id)

    def get_all_states(self) -> Dict[str, RobotState]:
        """모든 로봇의 상태를 반환한다."""
        return dict(self._robots)

    def get_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """
        알림을 조회한다.

        Args:
            level: 필터링할 알림 등급. None이면 전체 반환.

        Returns:
            알림 리스트.
        """
        if level is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.level == level]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        알림을 확인 처리한다.

        Args:
            alert_id: 알림 ID.

        Returns:
            확인 성공 여부.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    @property
    def robot_count(self) -> int:
        """등록된 로봇 수를 반환한다."""
        return len(self._robots)


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    monitor = FleetMonitor()

    # 로봇 등록
    monitor.register_robot("AMR-01", (0, 0))
    monitor.register_robot("AMR-02", (5, 5))
    monitor.register_robot("AMR-03", (10, 0))
    monitor.register_robot("AMR-04", (0, 10))

    # 상태 업데이트
    monitor.update_state("AMR-01", position=(3, 2), status=RobotStatus.MOVING,
                         battery_level=75, current_task="T-001", speed=1.2)
    monitor.update_state("AMR-02", position=(6, 7), status=RobotStatus.WORKING,
                         battery_level=45, current_task="T-002", speed=0.0)
    monitor.update_state("AMR-03", position=(10, 0), status=RobotStatus.IDLE,
                         battery_level=8, speed=0.0)  # 배터리 위험!
    monitor.update_state("AMR-04", position=(2, 8), status=RobotStatus.EMERGENCY_STOP,
                         battery_level=60, speed=0.0)  # 긴급 정지!

    # 작업 기록
    now = time.time()
    monitor.record_task("T-010", "AMR-01", now - 300, now - 60, success=True)
    monitor.record_task("T-011", "AMR-02", now - 200, now - 50, success=True)
    monitor.record_task("T-012", "AMR-01", now - 500, now - 400, success=False)

    # 교착 보고
    monitor.report_deadlock(["AMR-01", "AMR-02"])

    # 대시보드 출력
    monitor.print_dashboard()
