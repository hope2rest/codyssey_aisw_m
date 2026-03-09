"""
PID 속도 제어기 및 사다리꼴 속도 프로파일 생성 모듈.

Anti-windup 기능을 포함한 PID 제어기와,
가감속을 부드럽게 처리하는 사다리꼴(Trapezoidal) 속도 프로파일 생성기를 제공한다.
"""

import numpy as np


class PIDController:
    """
    Anti-windup 기능을 포함한 PID 제어기.

    매개변수
    --------
    kp : float
        비례 이득.
    ki : float
        적분 이득.
    kd : float
        미분 이득.
    output_min : float
        출력 하한값.
    output_max : float
        출력 상한값.
    anti_windup_limit : float
        적분 누적 상한값 (Anti-windup). None이면 output_max 사용.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_min: float = -float("inf"),
        output_max: float = float("inf"),
        anti_windup_limit: float | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.anti_windup_limit = (
            anti_windup_limit if anti_windup_limit is not None else abs(output_max)
        )

        # 내부 상태
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def reset(self):
        """내부 상태(적분, 이전 오차)를 초기화한다."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def compute(self, setpoint: float, measured: float, dt: float) -> float:
        """
        PID 제어 출력을 계산한다.

        매개변수
        --------
        setpoint : float
            목표값.
        measured : float
            현재 측정값.
        dt : float
            시간 간격 (초). 0 이하이면 오류를 방지하기 위해 무시한다.

        반환
        ----
        float
            제어 출력값 (output_min ~ output_max 범위로 클리핑).
        """
        if dt <= 0.0:
            return 0.0

        error = setpoint - measured

        # 비례 항
        p_term = self.kp * error

        # 적분 항 (Anti-windup: 클리핑 방식)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral, -self.anti_windup_limit, self.anti_windup_limit
        )
        i_term = self.ki * self._integral

        # 미분 항
        if self._first_call:
            d_term = 0.0
            self._first_call = False
        else:
            d_term = self.kd * (error - self._prev_error) / dt

        self._prev_error = error

        # 출력 합산 및 클리핑
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)

        return float(output)

    def set_gains(self, kp: float, ki: float, kd: float):
        """PID 이득을 변경한다."""
        self.kp = kp
        self.ki = ki
        self.kd = kd


class TrapezoidalProfile:
    """
    사다리꼴 속도 프로파일 생성기.

    가속 → 등속 → 감속의 3단계로 목표 위치까지의 속도 프로파일을 생성한다.
    거리가 짧아 최대 속도에 도달하지 못하는 경우 삼각형 프로파일로 자동 전환된다.

    매개변수
    --------
    max_velocity : float
        최대 속도 (m/s 또는 rad/s).
    max_acceleration : float
        최대 가속도 (m/s² 또는 rad/s²).
    max_deceleration : float, optional
        최대 감속도. None이면 max_acceleration과 동일하게 설정.
    """

    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 1.0,
        max_deceleration: float | None = None,
    ):
        self.max_velocity = abs(max_velocity)
        self.max_acceleration = abs(max_acceleration)
        self.max_deceleration = (
            abs(max_deceleration) if max_deceleration is not None else self.max_acceleration
        )

    def generate(
        self, distance: float, dt: float = 0.01
    ) -> dict[str, np.ndarray]:
        """
        사다리꼴 속도 프로파일을 생성한다.

        매개변수
        --------
        distance : float
            이동 거리 (양수).
        dt : float
            시간 간격 (초). 기본값 0.01.

        반환
        ----
        dict
            - 'time': 시간 배열 (초)
            - 'velocity': 속도 배열
            - 'position': 위치 배열
            - 'acceleration': 가속도 배열
            - 'phase': 단계 배열 ('accel', 'cruise', 'decel')
        """
        distance = abs(distance)
        if distance < 1e-9:
            return {
                "time": np.array([0.0]),
                "velocity": np.array([0.0]),
                "position": np.array([0.0]),
                "acceleration": np.array([0.0]),
                "phase": np.array(["stop"]),
            }

        v_max = self.max_velocity
        a = self.max_acceleration
        d = self.max_deceleration

        # 가속/감속에 필요한 거리
        t_acc = v_max / a
        t_dec = v_max / d
        d_acc = 0.5 * a * t_acc ** 2
        d_dec = 0.5 * d * t_dec ** 2

        if d_acc + d_dec > distance:
            # 삼각형 프로파일: 최대 속도 도달 불가
            # v_peak = sqrt(2 * a * d * distance / (a + d))
            v_peak = np.sqrt(2.0 * a * d * distance / (a + d))
            t_acc = v_peak / a
            t_dec = v_peak / d
            t_cruise = 0.0
        else:
            # 사다리꼴 프로파일
            v_peak = v_max
            d_cruise = distance - d_acc - d_dec
            t_cruise = d_cruise / v_peak

        total_time = t_acc + t_cruise + t_dec
        n_steps = max(int(np.ceil(total_time / dt)) + 1, 2)
        times = np.linspace(0.0, total_time, n_steps)

        velocities = np.zeros(n_steps)
        positions = np.zeros(n_steps)
        accelerations = np.zeros(n_steps)
        phases = np.empty(n_steps, dtype=object)

        for i, t in enumerate(times):
            if t <= t_acc:
                # 가속 구간
                velocities[i] = a * t
                positions[i] = 0.5 * a * t ** 2
                accelerations[i] = a
                phases[i] = "accel"
            elif t <= t_acc + t_cruise:
                # 등속 구간
                dt_cruise = t - t_acc
                velocities[i] = v_peak
                positions[i] = d_acc + v_peak * dt_cruise
                accelerations[i] = 0.0
                phases[i] = "cruise"
            else:
                # 감속 구간
                dt_dec = t - t_acc - t_cruise
                velocities[i] = v_peak - d * dt_dec
                velocities[i] = max(velocities[i], 0.0)
                positions[i] = (
                    d_acc
                    + v_peak * t_cruise
                    + v_peak * dt_dec
                    - 0.5 * d * dt_dec ** 2
                )
                accelerations[i] = -d
                phases[i] = "decel"

        # 위치가 distance를 초과하지 않도록 클리핑
        positions = np.clip(positions, 0.0, distance)

        return {
            "time": times,
            "velocity": velocities,
            "position": positions,
            "acceleration": accelerations,
            "phase": phases,
        }

    def get_velocity_at_time(
        self, distance: float, t: float
    ) -> float:
        """
        특정 시각에서의 목표 속도를 계산한다.

        매개변수
        --------
        distance : float
            총 이동 거리.
        t : float
            경과 시간 (초).

        반환
        ----
        float
            해당 시각의 목표 속도.
        """
        distance = abs(distance)
        a = self.max_acceleration
        d = self.max_deceleration
        v_max = self.max_velocity

        t_acc = v_max / a
        t_dec = v_max / d
        d_acc = 0.5 * a * t_acc ** 2
        d_dec = 0.5 * d * t_dec ** 2

        if d_acc + d_dec > distance:
            v_peak = np.sqrt(2.0 * a * d * distance / (a + d))
            t_acc = v_peak / a
            t_dec = v_peak / d
            t_cruise = 0.0
        else:
            v_peak = v_max
            d_cruise = distance - d_acc - d_dec
            t_cruise = d_cruise / v_peak

        total_time = t_acc + t_cruise + t_dec

        if t < 0.0:
            return 0.0
        elif t <= t_acc:
            return float(a * t)
        elif t <= t_acc + t_cruise:
            return float(v_peak)
        elif t <= total_time:
            dt_dec = t - t_acc - t_cruise
            return float(max(v_peak - d * dt_dec, 0.0))
        else:
            return 0.0


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== PID 제어기 테스트 ===")

    pid = PIDController(kp=2.0, ki=0.5, kd=0.1, output_min=-5.0, output_max=5.0)

    # 1차 시스템 시뮬레이션: dx/dt = u (적분기)
    dt = 0.01
    measured = 0.0
    setpoint = 1.0

    for step in range(200):
        u = pid.compute(setpoint, measured, dt)
        measured += u * dt  # 단순 적분 모델

        if step % 50 == 0:
            print(f"  스텝 {step:3d}: 목표={setpoint:.2f}, 측정={measured:.4f}, 출력={u:.4f}")

    print(f"  최종 측정값: {measured:.4f} (목표: {setpoint})")

    print("\n=== 사다리꼴 속도 프로파일 테스트 ===")

    profile_gen = TrapezoidalProfile(max_velocity=2.0, max_acceleration=1.0)

    # 정상 사다리꼴
    profile = profile_gen.generate(distance=10.0, dt=0.1)
    print(f"거리 10m: 총 시간={profile['time'][-1]:.2f}s, "
          f"최대속도={max(profile['velocity']):.2f}m/s")

    # 삼각형 프로파일 (짧은 거리)
    profile_short = profile_gen.generate(distance=1.0, dt=0.1)
    print(f"거리  1m: 총 시간={profile_short['time'][-1]:.2f}s, "
          f"최대속도={max(profile_short['velocity']):.2f}m/s")
