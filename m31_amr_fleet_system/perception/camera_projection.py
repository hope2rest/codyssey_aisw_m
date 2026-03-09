"""
카메라 투영 변환 모듈.

Pinhole Camera Model을 직접 구현하여 2D 픽셀 좌표와 깊이(depth) 정보를
3D 카메라 좌표 및 Map(월드) 좌표로 변환한다.

주요 기능:
    - 2D 픽셀 좌표 + Depth → 3D 카메라 좌표 변환 (역투영)
    - 3D 카메라 좌표 → 2D 픽셀 좌표 변환 (정투영)
    - 카메라 좌표 → Map 좌표 변환 (외부 파라미터 적용)
    - Map 좌표 → 카메라 좌표 역변환
    - 배치 처리를 통한 다수 포인트 동시 변환

외부 의존성:
    - numpy (필수)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    카메라 내부 파라미터(Intrinsics).

    Pinhole Camera Model의 내부 파라미터를 정의한다.

    Attributes:
        fx: x축 초점 거리(픽셀).
        fy: y축 초점 거리(픽셀).
        cx: x축 주점(principal point) 좌표(픽셀).
        cy: y축 주점 좌표(픽셀).
        width: 이미지 너비(픽셀).
        height: 이미지 높이(픽셀).
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480

    @property
    def matrix(self) -> np.ndarray:
        """
        3x3 카메라 내부 파라미터 행렬(K)을 반환한다.

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def inv_matrix(self) -> np.ndarray:
        """3x3 카메라 내부 파라미터 역행렬(K^-1)을 반환한다."""
        return np.linalg.inv(self.matrix)


@dataclass
class CameraExtrinsics:
    """
    카메라 외부 파라미터(Extrinsics).

    카메라 좌표계에서 월드(Map) 좌표계로의 변환을 정의한다.

    Attributes:
        rotation: 3x3 회전 행렬 (카메라 → 월드).
        translation: 3x1 이동 벡터 (카메라 → 월드).
    """
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        """입력 형태를 검증하고 numpy 배열로 변환한다."""
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    @property
    def transform_matrix(self) -> np.ndarray:
        """
        4x4 동차 변환 행렬(카메라 → 월드)을 반환한다.

        T = [[R, t],
             [0, 1]]
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    @property
    def inv_transform_matrix(self) -> np.ndarray:
        """4x4 역변환 행렬(월드 → 카메라)을 반환한다."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    @classmethod
    def from_euler_angles(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
        tx: float,
        ty: float,
        tz: float,
    ) -> "CameraExtrinsics":
        """
        오일러 각도와 이동 벡터로부터 외부 파라미터를 생성한다.

        Args:
            roll: X축 회전각(라디안).
            pitch: Y축 회전각(라디안).
            yaw: Z축 회전각(라디안).
            tx, ty, tz: 이동 벡터 성분.

        Returns:
            CameraExtrinsics 인스턴스.
        """
        # Rz * Ry * Rx 순서
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        R = Rz @ Ry @ Rx
        return cls(rotation=R, translation=np.array([tx, ty, tz]))


class CameraProjector:
    """
    Pinhole Camera Model 기반 좌표 변환기.

    2D 픽셀 좌표와 깊이 정보를 이용하여 3D 카메라 좌표 및 Map 좌표로
    변환하는 기능을 제공한다.

    사용 예시::

        intrinsics = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
        extrinsics = CameraExtrinsics.from_euler_angles(0, 0, 0, 1.0, 0.0, 0.5)
        projector = CameraProjector(intrinsics, extrinsics)

        # 픽셀 → 3D 카메라 좌표
        cam_point = projector.pixel_to_3d(320, 240, depth=2.5)

        # 카메라 좌표 → Map 좌표
        map_point = projector.camera_to_map(cam_point)

    Args:
        intrinsics: 카메라 내부 파라미터.
        extrinsics: 카메라 외부 파라미터. None이면 항등 변환 사용.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: Optional[CameraExtrinsics] = None,
    ):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics or CameraExtrinsics(
            rotation=np.eye(3), translation=np.zeros(3)
        )

        # 사전 계산
        self._K = intrinsics.matrix
        self._K_inv = intrinsics.inv_matrix
        self._T_cam_to_world = self.extrinsics.transform_matrix
        self._T_world_to_cam = self.extrinsics.inv_transform_matrix

    def pixel_to_3d(
        self, u: Union[float, np.ndarray], v: Union[float, np.ndarray], depth: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        2D 픽셀 좌표와 깊이(depth)로부터 3D 카메라 좌표를 계산한다.

        Pinhole Camera Model 역투영 공식:
            X_c = (u - cx) * depth / fx
            Y_c = (v - cy) * depth / fy
            Z_c = depth

        Args:
            u: 픽셀 x좌표. 스칼라 또는 배열.
            v: 픽셀 y좌표. 스칼라 또는 배열.
            depth: 깊이 값(m). 스칼라 또는 배열.

        Returns:
            3D 카메라 좌표 (x, y, z). 단일 포인트는 (3,), 배치는 (N, 3).
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)

        x_c = (u - self.intrinsics.cx) * depth / self.intrinsics.fx
        y_c = (v - self.intrinsics.cy) * depth / self.intrinsics.fy
        z_c = depth

        if u.ndim == 0:
            return np.array([x_c.item(), y_c.item(), z_c.item()])
        else:
            return np.stack([x_c, y_c, z_c], axis=-1)

    def project_3d_to_pixel(self, point_3d: np.ndarray) -> np.ndarray:
        """
        3D 카메라 좌표를 2D 픽셀 좌표로 투영한다.

        정투영 공식:
            u = fx * X_c / Z_c + cx
            v = fy * Y_c / Z_c + cy

        Args:
            point_3d: 3D 카메라 좌표 (x, y, z). (3,) 또는 (N, 3).

        Returns:
            2D 픽셀 좌표 (u, v). (2,) 또는 (N, 2).
        """
        point_3d = np.asarray(point_3d, dtype=np.float64)
        single = point_3d.ndim == 1

        if single:
            point_3d = point_3d.reshape(1, 3)

        z = point_3d[:, 2]
        # 0 나눗셈 방지
        z_safe = np.where(np.abs(z) < 1e-10, 1e-10, z)

        u = self.intrinsics.fx * point_3d[:, 0] / z_safe + self.intrinsics.cx
        v = self.intrinsics.fy * point_3d[:, 1] / z_safe + self.intrinsics.cy

        result = np.stack([u, v], axis=-1)
        return result[0] if single else result

    def camera_to_map(self, point_cam: np.ndarray) -> np.ndarray:
        """
        카메라 좌표를 Map(월드) 좌표로 변환한다.

        외부 파라미터(회전 + 이동)를 적용한다.

        Args:
            point_cam: 카메라 좌표 (x, y, z). (3,) 또는 (N, 3).

        Returns:
            Map 좌표 (x, y, z). (3,) 또는 (N, 3).
        """
        point_cam = np.asarray(point_cam, dtype=np.float64)
        single = point_cam.ndim == 1

        if single:
            point_cam = point_cam.reshape(1, 3)

        R = self.extrinsics.rotation
        t = self.extrinsics.translation

        # P_world = R * P_cam + t
        result = (R @ point_cam.T).T + t

        return result[0] if single else result

    def map_to_camera(self, point_map: np.ndarray) -> np.ndarray:
        """
        Map(월드) 좌표를 카메라 좌표로 역변환한다.

        Args:
            point_map: Map 좌표 (x, y, z). (3,) 또는 (N, 3).

        Returns:
            카메라 좌표 (x, y, z). (3,) 또는 (N, 3).
        """
        point_map = np.asarray(point_map, dtype=np.float64)
        single = point_map.ndim == 1

        if single:
            point_map = point_map.reshape(1, 3)

        R_inv = self.extrinsics.rotation.T
        t = self.extrinsics.translation

        # P_cam = R^T * (P_world - t)
        result = (R_inv @ (point_map - t).T).T

        return result[0] if single else result

    def pixel_to_map(
        self, u: Union[float, np.ndarray], v: Union[float, np.ndarray], depth: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        2D 픽셀 좌표 + 깊이를 한 번에 Map 좌표로 변환한다.

        pixel → 카메라 좌표 → Map 좌표 순서로 변환.

        Args:
            u: 픽셀 x좌표.
            v: 픽셀 y좌표.
            depth: 깊이 값(m).

        Returns:
            Map 좌표 (x, y, z).
        """
        cam_point = self.pixel_to_3d(u, v, depth)
        return self.camera_to_map(cam_point)

    def is_in_image(self, u: float, v: float) -> bool:
        """
        픽셀 좌표가 이미지 범위 내에 있는지 확인한다.

        Args:
            u: 픽셀 x좌표.
            v: 픽셀 y좌표.

        Returns:
            이미지 범위 내이면 True.
        """
        return 0 <= u < self.intrinsics.width and 0 <= v < self.intrinsics.height


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 카메라 파라미터 설정
    intrinsics = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    extrinsics = CameraExtrinsics.from_euler_angles(
        roll=0.0, pitch=-0.3, yaw=0.0,
        tx=1.0, ty=0.0, tz=0.5,
    )

    projector = CameraProjector(intrinsics, extrinsics)

    # 이미지 중심 픽셀, 깊이 2m
    u, v, depth = 320.0, 240.0, 2.0
    cam_3d = projector.pixel_to_3d(u, v, depth)
    map_3d = projector.camera_to_map(cam_3d)

    print(f"픽셀 좌표: ({u}, {v}), 깊이: {depth}m")
    print(f"카메라 좌표: ({cam_3d[0]:.4f}, {cam_3d[1]:.4f}, {cam_3d[2]:.4f})")
    print(f"Map 좌표: ({map_3d[0]:.4f}, {map_3d[1]:.4f}, {map_3d[2]:.4f})")

    # 역투영 확인
    reprojected = projector.project_3d_to_pixel(cam_3d)
    print(f"역투영 픽셀 좌표: ({reprojected[0]:.2f}, {reprojected[1]:.2f})")

    # 배치 처리
    us = np.array([100, 200, 320, 400, 500])
    vs = np.array([100, 150, 240, 300, 350])
    depths = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    map_points = projector.pixel_to_map(us, vs, depths)
    print(f"\n배치 변환 결과 ({map_points.shape[0]}개 포인트):")
    for i, pt in enumerate(map_points):
        print(f"  ({us[i]}, {vs[i]}, d={depths[i]}) → Map ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
