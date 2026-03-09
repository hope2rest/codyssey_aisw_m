"""
정밀 도킹 제어 모듈.

선반/스테이션 정밀 도킹 알고리즘을 구현한다. ArUco 마커 인식 기반으로
정밀 접근하며, 위치 2cm / 각도 1도 이내의 정밀도를 목표로 한다.

주요 기능:
    - ArUco 마커 인식 시뮬레이션
    - 비례-미분(PD) 제어 기반 정밀 접근
    - 도킹 정밀도 검증 (위치 2cm, 각도 1도 이내)
    - 재시도 로직 (최대 3회)
    - 도킹 상태 머신

외부 의존성:
    - numpy (필수)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DockingState(Enum):
    """
    도킹 상태 머신의 상태.

    IDLE: 대기 상태.
    APPROACHING: 스테이션 접근 중 (거친 접근).
    ALIGNING: 정밀 정렬 중 (ArUco 마커 기반).
    FINAL_APPROACH: 최종 접근 (저속 전진).
    DOCKED: 도킹 완료.
    FAILED: 도킹 실패.
    """
    IDLE = auto()
    APPROACHING = auto()
    ALIGNING = auto()
    FINAL_APPROACH = auto()
    DOCKED = auto()
    FAILED = auto()


@dataclass
class DockingTarget:
    """
    도킹 대상 스테이션 정보.

    Attributes:
        station_id: 스테이션 고유 식별자.
        position: 스테이션 위치 (x, y) (m).
        orientation: 스테이션 방향 (라디안). 로봇이 진입해야 할 방향.
        marker_id: ArUco 마커 ID.
    """
    station_id: str
    position: Tuple[float, float]
    orientation: float
    marker_id: int = 0


@dataclass
class RobotPose:
    """
    로봇의 현재 자세(Pose).

    Attributes:
        x: x 좌표 (m).
        y: y 좌표 (m).
        theta: 방향각 (라디안).
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class MarkerDetection:
    """
    ArUco 마커 인식 결과 (시뮬레이션).

    Attributes:
        marker_id: 인식된 마커 ID.
        distance: 마커까지의 거리 (m).
        lateral_offset: 좌우 오프셋 (m). 양수=오른쪽.
        angular_offset: 각도 오프셋 (라디안). 양수=시계방향.
        detected: 인식 성공 여부.
    """
    marker_id: int = 0
    distance: float = 0.0
    lateral_offset: float = 0.0
    angular_offset: float = 0.0
    detected: bool = False


@dataclass
class DockingResult:
    """
    도킹 최종 결과.

    Attributes:
        success: 도킹 성공 여부.
        position_error: 최종 위치 오차 (m).
        angle_error: 최종 각도 오차 (도).
        attempts: 총 시도 횟수.
        final_state: 최종 상태.
    """
    success: bool
    position_error: float
    angle_error: float
    attempts: int
    final_state: DockingState


class DockingController:
    """
    정밀 도킹 제어기.

    ArUco 마커 인식 기반으로 선반/스테이션에 정밀 도킹한다.
    PD 제어를 사용하여 위치 2cm / 각도 1도 이내의 정밀도를 달성한다.

    사용 예시::

        controller = DockingController()
        target = DockingTarget(
            station_id="ST-01",
            position=(5.0, 3.0),
            orientation=math.pi / 2,
            marker_id=42,
        )
        result = controller.execute_docking(target, initial_pose)
        if result.success:
            print(f"도킹 성공! 위치 오차: {result.position_error*100:.1f}cm")

    Args:
        position_tolerance: 위치 허용 오차 (m). 기본 0.02 (2cm).
        angle_tolerance_deg: 각도 허용 오차 (도). 기본 1.0.
        max_attempts: 최대 재시도 횟수. 기본 3.
        approach_speed: 접근 속도 (m/s). 기본 0.3.
        final_speed: 최종 접근 속도 (m/s). 기본 0.05.
        kp_linear: 선속도 비례 게인.
        kd_linear: 선속도 미분 게인.
        kp_angular: 각속도 비례 게인.
        kd_angular: 각속도 미분 게인.
        dt: 제어 주기 (초).
    """

    def __init__(
        self,
        position_tolerance: float = 0.02,
        angle_tolerance_deg: float = 1.0,
        max_attempts: int = 3,
        approach_speed: float = 0.3,
        final_speed: float = 0.05,
        kp_linear: float = 1.0,
        kd_linear: float = 0.1,
        kp_angular: float = 2.0,
        kd_angular: float = 0.2,
        dt: float = 0.05,
    ):
        self.position_tolerance = position_tolerance
        self.angle_tolerance_rad = math.radians(angle_tolerance_deg)
        self.angle_tolerance_deg = angle_tolerance_deg
        self.max_attempts = max_attempts
        self.approach_speed = approach_speed
        self.final_speed = final_speed

        # PD 제어 게인
        self.kp_linear = kp_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.dt = dt

        # 내부 상태
        self._state = DockingState.IDLE
        self._prev_linear_error = 0.0
        self._prev_angular_error = 0.0
        self._rng = np.random.default_rng(seed=42)

    @property
    def state(self) -> DockingState:
        """현재 도킹 상태를 반환한다."""
        return self._state

    def detect_aruco_marker(
        self, robot_pose: RobotPose, target: DockingTarget
    ) -> MarkerDetection:
        """
        ArUco 마커 인식을 시뮬레이션한다.

        실제 환경에서는 카메라 이미지에서 ArUco 마커를 인식하여
        상대 위치/방향을 추정한다. 여기서는 로봇과 타겟의 실제 위치로부터
        시뮬레이션된 인식 결과를 생성한다.

        Args:
            robot_pose: 로봇의 현재 자세.
            target: 도킹 대상 스테이션.

        Returns:
            마커 인식 결과.
        """
        dx = target.position[0] - robot_pose.x
        dy = target.position[1] - robot_pose.y
        distance = math.sqrt(dx * dx + dy * dy)

        # 마커가 너무 멀면 인식 실패 (시야 밖)
        if distance > 5.0:
            return MarkerDetection(marker_id=target.marker_id, detected=False)

        # 타겟 방향과 로봇 방향 차이
        target_angle = math.atan2(dy, dx)
        angular_offset = self._normalize_angle(target_angle - robot_pose.theta)

        # 로봇 로컬 좌표계에서의 좌우 오프셋
        lateral_offset = distance * math.sin(angular_offset)

        # 센서 노이즈 추가 (거리에 비례하는 노이즈)
        noise_scale = 0.002 * distance  # 거리 1m당 2mm 노이즈
        distance += self._rng.normal(0, noise_scale)
        lateral_offset += self._rng.normal(0, noise_scale * 0.5)
        angular_offset += self._rng.normal(0, math.radians(0.1) * distance)

        return MarkerDetection(
            marker_id=target.marker_id,
            distance=max(0, distance),
            lateral_offset=lateral_offset,
            angular_offset=angular_offset,
            detected=True,
        )

    def compute_control(
        self, marker: MarkerDetection, phase: DockingState
    ) -> Tuple[float, float]:
        """
        PD 제어기로 선속도(v)와 각속도(w)를 계산한다.

        Args:
            marker: 마커 인식 결과.
            phase: 현재 도킹 단계.

        Returns:
            (선속도, 각속도) 튜플. (m/s, rad/s).
        """
        linear_error = marker.distance
        angular_error = marker.angular_offset

        # 미분 항 계산
        d_linear = (linear_error - self._prev_linear_error) / self.dt
        d_angular = (angular_error - self._prev_angular_error) / self.dt

        self._prev_linear_error = linear_error
        self._prev_angular_error = angular_error

        # PD 제어
        v = self.kp_linear * linear_error + self.kd_linear * d_linear
        w = self.kp_angular * angular_error + self.kd_angular * d_angular

        # 속도 제한
        if phase == DockingState.FINAL_APPROACH:
            v = min(v, self.final_speed)
        else:
            v = min(v, self.approach_speed)

        # 최소 속도 보장
        if linear_error > self.position_tolerance:
            v = max(v, 0.01)

        return v, w

    def update_pose(
        self, pose: RobotPose, v: float, w: float
    ) -> RobotPose:
        """
        로봇 자세를 운동 모델에 따라 업데이트한다.

        차동 구동(differential drive) 운동학을 사용한다.

        Args:
            pose: 현재 자세.
            v: 선속도 (m/s).
            w: 각속도 (rad/s).

        Returns:
            업데이트된 자세.
        """
        new_theta = self._normalize_angle(pose.theta + w * self.dt)
        new_x = pose.x + v * math.cos(new_theta) * self.dt
        new_y = pose.y + v * math.sin(new_theta) * self.dt

        return RobotPose(x=new_x, y=new_y, theta=new_theta)

    def check_docking_precision(
        self, robot_pose: RobotPose, target: DockingTarget
    ) -> Tuple[float, float]:
        """
        도킹 정밀도를 확인한다.

        Args:
            robot_pose: 로봇의 현재 자세.
            target: 도킹 대상 스테이션.

        Returns:
            (위치 오차(m), 각도 오차(도)) 튜플.
        """
        dx = target.position[0] - robot_pose.x
        dy = target.position[1] - robot_pose.y
        position_error = math.sqrt(dx * dx + dy * dy)

        angle_error = abs(
            math.degrees(self._normalize_angle(target.orientation - robot_pose.theta))
        )

        return position_error, angle_error

    def execute_docking(
        self, target: DockingTarget, initial_pose: RobotPose
    ) -> DockingResult:
        """
        도킹 전체 시퀀스를 실행한다.

        상태 머신 흐름:
            IDLE → APPROACHING → ALIGNING → FINAL_APPROACH → DOCKED
            실패 시 최대 max_attempts회 재시도 후 FAILED.

        Args:
            target: 도킹 대상 스테이션.
            initial_pose: 로봇 초기 자세.

        Returns:
            도킹 결과.
        """
        logger.info(f"도킹 시작: 스테이션 '{target.station_id}'")

        best_result = None

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"--- 도킹 시도 {attempt}/{self.max_attempts} ---")

            # 상태 초기화
            self._state = DockingState.APPROACHING
            self._prev_linear_error = 0.0
            self._prev_angular_error = 0.0
            pose = RobotPose(
                x=initial_pose.x + self._rng.normal(0, 0.01) * (attempt - 1),
                y=initial_pose.y + self._rng.normal(0, 0.01) * (attempt - 1),
                theta=initial_pose.theta,
            )

            max_steps = 2000
            step = 0

            while step < max_steps:
                step += 1

                # ArUco 마커 인식
                marker = self.detect_aruco_marker(pose, target)
                if not marker.detected:
                    logger.warning("마커 인식 실패, 접근 계속")
                    # 타겟 방향으로 맹목적 전진
                    pose = self.update_pose(pose, self.approach_speed * 0.5, 0.0)
                    continue

                # 상태 전환 로직
                if self._state == DockingState.APPROACHING and marker.distance < 1.0:
                    self._state = DockingState.ALIGNING
                    logger.info(f"정렬 단계 진입 (거리: {marker.distance:.3f}m)")
                elif self._state == DockingState.ALIGNING:
                    if (
                        abs(marker.angular_offset) < self.angle_tolerance_rad * 2
                        and abs(marker.lateral_offset) < self.position_tolerance * 5
                    ):
                        self._state = DockingState.FINAL_APPROACH
                        logger.info("최종 접근 단계 진입")
                elif self._state == DockingState.FINAL_APPROACH:
                    pos_err, ang_err = self.check_docking_precision(pose, target)
                    if (
                        pos_err <= self.position_tolerance
                        and ang_err <= self.angle_tolerance_deg
                    ):
                        self._state = DockingState.DOCKED
                        result = DockingResult(
                            success=True,
                            position_error=pos_err,
                            angle_error=ang_err,
                            attempts=attempt,
                            final_state=DockingState.DOCKED,
                        )
                        logger.info(
                            f"도킹 성공! 위치 오차: {pos_err*100:.2f}cm, "
                            f"각도 오차: {ang_err:.3f}도"
                        )
                        return result

                # 제어 입력 계산 및 자세 업데이트
                v, w = self.compute_control(marker, self._state)
                pose = self.update_pose(pose, v, w)

            # 최대 스텝 초과 → 이 시도 실패
            pos_err, ang_err = self.check_docking_precision(pose, target)
            logger.warning(
                f"시도 {attempt} 시간 초과. "
                f"위치 오차: {pos_err*100:.2f}cm, 각도 오차: {ang_err:.3f}도"
            )

            if best_result is None or pos_err < best_result.position_error:
                best_result = DockingResult(
                    success=False,
                    position_error=pos_err,
                    angle_error=ang_err,
                    attempts=attempt,
                    final_state=DockingState.FAILED,
                )

        # 모든 재시도 소진
        self._state = DockingState.FAILED
        logger.error(f"도킹 실패: {self.max_attempts}회 시도 모두 실패")

        return best_result or DockingResult(
            success=False,
            position_error=float("inf"),
            angle_error=float("inf"),
            attempts=self.max_attempts,
            final_state=DockingState.FAILED,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """각도를 [-pi, pi) 범위로 정규화한다."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    target = DockingTarget(
        station_id="ST-01",
        position=(5.0, 3.0),
        orientation=math.pi / 2,
        marker_id=42,
    )

    initial_pose = RobotPose(x=3.5, y=1.5, theta=math.pi / 3)

    controller = DockingController(
        position_tolerance=0.02,
        angle_tolerance_deg=1.0,
        max_attempts=3,
    )

    result = controller.execute_docking(target, initial_pose)

    print("\n" + "=" * 50)
    print(f"도킹 결과: {'성공' if result.success else '실패'}")
    print(f"위치 오차: {result.position_error * 100:.2f} cm")
    print(f"각도 오차: {result.angle_error:.3f} 도")
    print(f"시도 횟수: {result.attempts}")
