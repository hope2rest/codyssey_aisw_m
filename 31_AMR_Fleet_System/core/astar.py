"""
A* 경로 계획 알고리즘 모듈.

2D Occupancy Grid 기반으로 A* 알고리즘을 직접 구현한다.
8방향 이동을 지원하며, 대각선 이동 비용은 sqrt(2)이다.
탐색된 경로에 대한 평활화(Path Smoothing) 기능도 제공한다.
"""

import heapq
import numpy as np


# 8방향 이동: (dy, dx, 비용)
_DIRECTIONS = [
    (-1, 0, 1.0),    # 상
    (1, 0, 1.0),     # 하
    (0, -1, 1.0),    # 좌
    (0, 1, 1.0),     # 우
    (-1, -1, np.sqrt(2)),  # 좌상 대각선
    (-1, 1, np.sqrt(2)),   # 우상 대각선
    (1, -1, np.sqrt(2)),   # 좌하 대각선
    (1, 1, np.sqrt(2)),    # 우하 대각선
]


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    휴리스틱 함수: 8방향 이동에 적합한 옥타일(Octile) 거리를 반환한다.

    매개변수
    --------
    a, b : tuple[int, int]
        (row, col) 좌표.

    반환
    ----
    float
        옥타일 거리.
    """
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return max(dy, dx) + (np.sqrt(2) - 1) * min(dy, dx)


def plan_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacle_threshold: float = 0.5,
    smooth: bool = True,
    smooth_weight_data: float = 0.5,
    smooth_weight_smooth: float = 0.3,
    smooth_tolerance: float = 1e-4,
) -> list[tuple[int, int]]:
    """
    A* 알고리즘으로 최단 경로를 탐색한다.

    매개변수
    --------
    grid : np.ndarray, shape (H, W)
        2D Occupancy Grid. 0=자유 공간, 1=장애물.
        [0, 1] 사이 값을 가질 수 있으며 obstacle_threshold 이상이면 장애물로 간주한다.
    start : tuple[int, int]
        시작 좌표 (row, col).
    goal : tuple[int, int]
        목표 좌표 (row, col).
    obstacle_threshold : float
        장애물 판단 기준값. 기본값 0.5.
    smooth : bool
        경로 평활화 적용 여부. 기본값 True.
    smooth_weight_data : float
        평활화 시 원본 경로 가중치. 기본값 0.5.
    smooth_weight_smooth : float
        평활화 시 매끄러움 가중치. 기본값 0.3.
    smooth_tolerance : float
        평활화 수렴 허용 오차. 기본값 1e-4.

    반환
    ----
    list[tuple[int, int]]
        경로 좌표 리스트 [(row, col), ...].
        경로를 찾을 수 없으면 빈 리스트를 반환한다.

    예외
    ----
    ValueError
        시작/목표 좌표가 맵 범위 밖이거나 장애물 위에 있을 때.
    """
    rows, cols = grid.shape

    # 입력 검증
    for label, pt in [("시작", start), ("목표", goal)]:
        if not (0 <= pt[0] < rows and 0 <= pt[1] < cols):
            raise ValueError(f"{label} 좌표 {pt}가 맵 범위({rows}x{cols}) 밖입니다.")
        if grid[pt[0], pt[1]] >= obstacle_threshold:
            raise ValueError(f"{label} 좌표 {pt}가 장애물 위에 있습니다.")

    # A* 탐색
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0  # tie-breaker
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            # 경로 복원
            path = _reconstruct_path(came_from, goal)
            if smooth and len(path) > 2:
                path = smooth_path(
                    path, grid, obstacle_threshold,
                    smooth_weight_data, smooth_weight_smooth, smooth_tolerance,
                )
            return path

        for dy, dx, move_cost in _DIRECTIONS:
            ny, nx = current[0] + dy, current[1] + dx

            # 범위 검사
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue

            # 장애물 검사
            if grid[ny, nx] >= obstacle_threshold:
                continue

            # 대각선 이동 시 인접 셀 통과 가능 여부 검사 (코너 절단 방지)
            if dy != 0 and dx != 0:
                if (
                    grid[current[0] + dy, current[1]] >= obstacle_threshold
                    or grid[current[0], current[1] + dx] >= obstacle_threshold
                ):
                    continue

            neighbor = (ny, nx)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                f_score = tentative_g + _heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))
                came_from[neighbor] = current

    # 경로를 찾지 못함
    return []


def _reconstruct_path(
    came_from: dict, goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """came_from 딕셔너리로부터 경로를 역추적하여 복원한다."""
    path = []
    current: tuple[int, int] | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def smooth_path(
    path: list[tuple[int, int]],
    grid: np.ndarray,
    obstacle_threshold: float = 0.5,
    weight_data: float = 0.5,
    weight_smooth: float = 0.3,
    tolerance: float = 1e-4,
) -> list[tuple[int, int]]:
    """
    경로 평활화 (Gradient Descent 기반).

    원본 경로와의 근접성(weight_data)과 매끄러움(weight_smooth) 사이의
    균형을 맞추어 경로를 부드럽게 만든다.

    매개변수
    --------
    path : list[tuple[int, int]]
        원본 경로 좌표 리스트.
    grid : np.ndarray
        Occupancy Grid.
    obstacle_threshold : float
        장애물 판단 기준값.
    weight_data : float
        원본 경로 유지 가중치.
    weight_smooth : float
        매끄러움 가중치.
    tolerance : float
        수렴 허용 오차.

    반환
    ----
    list[tuple[int, int]]
        평활화된 경로 좌표 리스트.
    """
    # float 배열로 변환 (시작/끝은 고정)
    smooth = np.array(path, dtype=np.float64)
    original = smooth.copy()

    rows, cols = grid.shape
    change = tolerance + 1.0

    while change >= tolerance:
        change = 0.0
        for i in range(1, len(smooth) - 1):
            for dim in range(2):
                old_val = smooth[i, dim]
                smooth[i, dim] += weight_data * (original[i, dim] - smooth[i, dim])
                smooth[i, dim] += weight_smooth * (
                    smooth[i - 1, dim] + smooth[i + 1, dim] - 2.0 * smooth[i, dim]
                )
                change += abs(smooth[i, dim] - old_val)

    # 정수 좌표로 반올림 후 장애물 위의 점은 원본으로 복구
    result: list[tuple[int, int]] = []
    for i in range(len(smooth)):
        r = int(round(smooth[i, 0]))
        c = int(round(smooth[i, 1]))
        r = np.clip(r, 0, rows - 1)
        c = np.clip(c, 0, cols - 1)
        if grid[r, c] >= obstacle_threshold:
            r, c = path[i]
        result.append((r, c))

    return result


# ======================================================================
# 독립 실행 테스트
# ======================================================================
if __name__ == "__main__":
    print("=== A* 경로 계획 테스트 ===")

    # 20x20 그리드 생성 (0=자유, 1=장애물)
    grid = np.zeros((20, 20))
    # 장애물 배치
    grid[5, 3:15] = 1
    grid[10, 5:18] = 1
    grid[15, 0:12] = 1

    start = (0, 0)
    goal = (19, 19)

    path = plan_path(grid, start, goal, smooth=True)
    print(f"경로 길이: {len(path)}")
    print(f"시작: {path[0]}, 끝: {path[-1]}")

    # 간단한 시각화
    display = grid.copy()
    for r, c in path:
        display[r, c] = 0.5
    display[start[0], start[1]] = 0.2
    display[goal[0], goal[1]] = 0.8

    print("\n맵 (0=자유, 1=장애물, *=경로, S=시작, G=목표):")
    symbols = {0.0: ".", 1.0: "#", 0.5: "*", 0.2: "S", 0.8: "G"}
    for row in display:
        print(" ".join(symbols.get(v, "?") for v in row))
