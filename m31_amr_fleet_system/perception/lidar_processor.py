"""
LiDAR 포인트 클라우드 처리 모듈.

LiDAR 센서로부터 수집된 포인트 클라우드 데이터를 필터링, 다운샘플링,
클러스터링하여 주변 환경의 장애물 및 객체를 탐지한다.

주요 기능:
    - 거리 필터: 최소/최대 거리 범위 내 포인트만 유지
    - 각도 필터: 관심 영역(FOV) 내 포인트만 유지
    - 아웃라이어 제거: 통계적 기법(SOR) 기반 이상치 제거
    - Voxel Grid 다운샘플링: 균일한 밀도의 포인트 클라우드 생성
    - DBSCAN 클러스터링: 밀도 기반 클러스터링으로 개별 객체 분리

외부 의존성:
    - numpy (필수)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PointCloud:
    """
    포인트 클라우드 데이터 컨테이너.

    Attributes:
        points: (N, 3) 형태의 numpy 배열. 각 행은 (x, y, z) 좌표.
        intensities: (N,) 형태의 반사 강도 배열. 선택적.
    """
    points: np.ndarray
    intensities: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        """포인트 수를 반환한다."""
        return self.points.shape[0] if self.points.size > 0 else 0


@dataclass
class Cluster:
    """
    클러스터링 결과 하나의 클러스터를 나타낸다.

    Attributes:
        cluster_id: 클러스터 고유 식별자.
        points: 클러스터에 속하는 포인트 배열 (M, 3).
        centroid: 클러스터 중심점 (x, y, z).
        bbox_min: 바운딩 박스 최소 좌표.
        bbox_max: 바운딩 박스 최대 좌표.
    """
    cluster_id: int
    points: np.ndarray
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        """클러스터 통계를 자동 계산한다."""
        if self.points.size > 0:
            self.centroid = np.mean(self.points, axis=0)
            self.bbox_min = np.min(self.points, axis=0)
            self.bbox_max = np.max(self.points, axis=0)


class LidarProcessor:
    """
    LiDAR 포인트 클라우드 처리 파이프라인.

    2D/3D LiDAR 센서에서 수집된 원시 포인트 클라우드를 처리하여
    주변 장애물 및 객체를 탐지한다.

    사용 예시::

        processor = LidarProcessor(min_range=0.3, max_range=30.0)
        cloud = PointCloud(points=raw_points)
        filtered = processor.filter_by_range(cloud)
        filtered = processor.filter_by_angle(filtered, -90, 90)
        filtered = processor.remove_outliers(filtered)
        downsampled = processor.voxel_grid_downsample(filtered, voxel_size=0.1)
        clusters = processor.dbscan_cluster(downsampled, eps=0.5, min_samples=5)

    Args:
        min_range: 최소 거리 임계값(m). 이보다 가까운 포인트는 제거.
        max_range: 최대 거리 임계값(m). 이보다 먼 포인트는 제거.
        min_height: 최소 높이(m). 지면 아래 포인트를 제거.
        max_height: 최대 높이(m). 천장 위 포인트를 제거.
    """

    def __init__(
        self,
        min_range: float = 0.2,
        max_range: float = 50.0,
        min_height: float = -0.5,
        max_height: float = 3.0,
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.min_height = min_height
        self.max_height = max_height

    # ------------------------------------------------------------------
    # 필터링
    # ------------------------------------------------------------------

    def filter_by_range(self, cloud: PointCloud) -> PointCloud:
        """
        거리 기반 필터링.

        원점(센서)으로부터의 유클리드 거리를 계산하여
        [min_range, max_range] 범위 밖의 포인트를 제거한다.

        Args:
            cloud: 입력 포인트 클라우드.

        Returns:
            필터링된 포인트 클라우드.
        """
        if cloud.size == 0:
            return cloud

        distances = np.linalg.norm(cloud.points[:, :2], axis=1)
        mask = (distances >= self.min_range) & (distances <= self.max_range)

        # 높이 필터도 동시에 적용
        if cloud.points.shape[1] >= 3:
            mask &= (cloud.points[:, 2] >= self.min_height) & (
                cloud.points[:, 2] <= self.max_height
            )

        return PointCloud(
            points=cloud.points[mask],
            intensities=cloud.intensities[mask] if cloud.intensities is not None else None,
        )

    def filter_by_angle(
        self, cloud: PointCloud, min_angle_deg: float = -180.0, max_angle_deg: float = 180.0
    ) -> PointCloud:
        """
        각도(방위각) 기반 필터링.

        XY 평면에서 x축 기준 방위각을 계산하여 관심 영역(FOV) 밖의
        포인트를 제거한다.

        Args:
            cloud: 입력 포인트 클라우드.
            min_angle_deg: 최소 방위각(도).
            max_angle_deg: 최대 방위각(도).

        Returns:
            필터링된 포인트 클라우드.
        """
        if cloud.size == 0:
            return cloud

        angles = np.degrees(np.arctan2(cloud.points[:, 1], cloud.points[:, 0]))
        mask = (angles >= min_angle_deg) & (angles <= max_angle_deg)

        return PointCloud(
            points=cloud.points[mask],
            intensities=cloud.intensities[mask] if cloud.intensities is not None else None,
        )

    def remove_outliers(
        self, cloud: PointCloud, k_neighbors: int = 20, std_ratio: float = 2.0
    ) -> PointCloud:
        """
        통계적 아웃라이어 제거(Statistical Outlier Removal, SOR).

        각 포인트에 대해 k개의 최근접 이웃까지의 평균 거리를 계산하고,
        전체 평균 + std_ratio * 표준편차보다 큰 포인트를 제거한다.

        Args:
            cloud: 입력 포인트 클라우드.
            k_neighbors: 이웃 탐색 수.
            std_ratio: 표준편차 배수 임계값.

        Returns:
            아웃라이어가 제거된 포인트 클라우드.
        """
        if cloud.size <= k_neighbors:
            return cloud

        points = cloud.points
        n = points.shape[0]

        # 모든 포인트 쌍 간 거리 행렬 계산 (메모리 효율을 위해 분할 가능)
        # 포인트 수가 많으면 배치 처리
        batch_size = 2000
        mean_distances = np.zeros(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = points[start:end]

            # 배치와 전체 포인트 간 거리 계산
            diff = batch[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)

            # 자기 자신 제외, k개 최근접 이웃의 평균 거리
            for i in range(end - start):
                sorted_dists = np.sort(dists[i])
                # 인덱스 0은 자기 자신(거리=0)이므로 1부터 시작
                mean_distances[start + i] = np.mean(sorted_dists[1 : k_neighbors + 1])

        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std

        mask = mean_distances <= threshold

        return PointCloud(
            points=cloud.points[mask],
            intensities=cloud.intensities[mask] if cloud.intensities is not None else None,
        )

    # ------------------------------------------------------------------
    # 다운샘플링
    # ------------------------------------------------------------------

    def voxel_grid_downsample(self, cloud: PointCloud, voxel_size: float = 0.1) -> PointCloud:
        """
        Voxel Grid 다운샘플링.

        3D 공간을 voxel_size 크기의 격자로 분할한 후,
        각 격자 내 포인트들의 중심점(centroid)만 남긴다.
        균일한 밀도의 포인트 클라우드를 생성하여 처리 속도를 향상시킨다.

        Args:
            cloud: 입력 포인트 클라우드.
            voxel_size: 복셀 크기(m).

        Returns:
            다운샘플링된 포인트 클라우드.
        """
        if cloud.size == 0 or voxel_size <= 0:
            return cloud

        points = cloud.points

        # 각 포인트의 복셀 인덱스 계산
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)

        # 복셀 인덱스를 고유 키로 변환
        # 해시 충돌 방지를 위해 큰 소수 사용
        primes = np.array([73856093, 19349663, 83492791], dtype=np.int64)
        keys = np.sum(voxel_indices * primes, axis=1)

        # 동일 복셀 내 포인트들의 중심점 계산
        unique_keys = np.unique(keys)
        centroids = []
        new_intensities = [] if cloud.intensities is not None else None

        for key in unique_keys:
            mask = keys == key
            centroid = np.mean(points[mask], axis=0)
            centroids.append(centroid)
            if cloud.intensities is not None:
                new_intensities.append(np.mean(cloud.intensities[mask]))

        result_points = np.array(centroids) if centroids else np.empty((0, 3))
        result_intensities = np.array(new_intensities) if new_intensities else None

        return PointCloud(points=result_points, intensities=result_intensities)

    # ------------------------------------------------------------------
    # 클러스터링 (DBSCAN 직접 구현)
    # ------------------------------------------------------------------

    def dbscan_cluster(
        self, cloud: PointCloud, eps: float = 0.5, min_samples: int = 5
    ) -> List[Cluster]:
        """
        DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
        클러스터링 알고리즘 직접 구현.

        밀도 기반으로 포인트들을 클러스터링하여 개별 객체를 분리한다.
        노이즈 포인트는 어떤 클러스터에도 포함되지 않는다.

        알고리즘 흐름:
            1. 각 포인트에 대해 eps 반경 내 이웃 포인트를 탐색
            2. min_samples 이상의 이웃을 가진 포인트를 핵심 포인트로 지정
            3. 핵심 포인트로부터 밀도 연결된 포인트들을 동일 클러스터로 그룹화

        Args:
            cloud: 입력 포인트 클라우드.
            eps: 이웃 탐색 반경(m).
            min_samples: 핵심 포인트 판별을 위한 최소 이웃 수.

        Returns:
            Cluster 객체 리스트. 노이즈 포인트는 제외.
        """
        if cloud.size == 0:
            return []

        points = cloud.points
        n = points.shape[0]

        # --- 이웃 탐색 (거리 행렬 기반) ---
        # 포인트 수가 많으면 배치 처리로 메모리 절약
        neighbors: List[List[int]] = [[] for _ in range(n)]
        batch_size = 2000

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = points[start:end]
            diff = batch[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)

            for i in range(end - start):
                neighbor_indices = np.where(dists[i] <= eps)[0].tolist()
                neighbors[start + i] = neighbor_indices

        # --- DBSCAN 핵심 알고리즘 ---
        labels = np.full(n, -1, dtype=int)  # -1: 미방문/노이즈
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue  # 이미 클러스터에 할당됨

            if len(neighbors[i]) < min_samples:
                continue  # 노이즈 포인트

            # 새 클러스터 시작 (BFS 확장)
            labels[i] = cluster_id
            queue = list(neighbors[i])
            visited = {i}

            while queue:
                j = queue.pop(0)
                if j in visited:
                    continue
                visited.add(j)

                if labels[j] == -1 or labels[j] == -2:
                    labels[j] = cluster_id

                # j가 핵심 포인트이면 이웃을 큐에 추가
                if len(neighbors[j]) >= min_samples:
                    for k in neighbors[j]:
                        if k not in visited:
                            queue.append(k)

            cluster_id += 1

        # --- 클러스터 객체 생성 ---
        clusters = []
        for cid in range(cluster_id):
            mask = labels == cid
            cluster_points = points[mask]
            if cluster_points.shape[0] > 0:
                clusters.append(Cluster(cluster_id=cid, points=cluster_points))

        return clusters

    # ------------------------------------------------------------------
    # 통합 파이프라인
    # ------------------------------------------------------------------

    def process(
        self,
        cloud: PointCloud,
        fov_min_deg: float = -120.0,
        fov_max_deg: float = 120.0,
        voxel_size: float = 0.1,
        cluster_eps: float = 0.5,
        cluster_min_samples: int = 5,
    ) -> List[Cluster]:
        """
        전체 포인트 클라우드 처리 파이프라인을 실행한다.

        순서: 거리 필터 → 각도 필터 → 아웃라이어 제거 → 다운샘플링 → 클러스터링

        Args:
            cloud: 원시 포인트 클라우드.
            fov_min_deg: FOV 최소 각도(도).
            fov_max_deg: FOV 최대 각도(도).
            voxel_size: 복셀 크기(m).
            cluster_eps: DBSCAN eps 파라미터(m).
            cluster_min_samples: DBSCAN min_samples 파라미터.

        Returns:
            탐지된 클러스터 리스트.
        """
        filtered = self.filter_by_range(cloud)
        filtered = self.filter_by_angle(filtered, fov_min_deg, fov_max_deg)
        filtered = self.remove_outliers(filtered)
        downsampled = self.voxel_grid_downsample(filtered, voxel_size)
        clusters = self.dbscan_cluster(downsampled, cluster_eps, cluster_min_samples)
        return clusters


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 시뮬레이션: 두 개의 장애물 + 노이즈
    obstacle1 = np.random.randn(50, 3) * 0.2 + np.array([5.0, 2.0, 0.5])
    obstacle2 = np.random.randn(40, 3) * 0.3 + np.array([8.0, -3.0, 0.8])
    noise = np.random.uniform(-50, 50, size=(20, 3))
    all_points = np.vstack([obstacle1, obstacle2, noise])

    cloud = PointCloud(points=all_points)
    processor = LidarProcessor(min_range=0.5, max_range=30.0)

    clusters = processor.process(cloud, voxel_size=0.15, cluster_eps=1.0, cluster_min_samples=3)

    print(f"입력 포인트 수: {cloud.size}")
    print(f"탐지된 클러스터 수: {len(clusters)}")
    for c in clusters:
        print(f"  클러스터 {c.cluster_id}: {c.points.shape[0]}개 포인트, "
              f"중심=({c.centroid[0]:.2f}, {c.centroid[1]:.2f}, {c.centroid[2]:.2f})")
