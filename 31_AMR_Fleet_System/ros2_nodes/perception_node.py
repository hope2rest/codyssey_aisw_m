"""perception_node.py - AI 인지 시스템 ROS2 노드.

RGB 카메라 + Depth 카메라를 입력받아 YOLOv8 객체 인식을 수행하고
인식된 객체의 3D 위치를 Pinhole Camera Model로 변환하여 발행한다.

Subscriptions:
    /camera/image_raw (Image): RGB 이미지
    /camera/depth (Image): 깊이 이미지
    /camera/camera_info (CameraInfo): 카메라 내부 파라미터

Publications:
    /detections (String): JSON 형식 인식 결과
    /detection_markers (MarkerArray): RViz2 3D 마커
"""

import sys
import os
import json
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from perception.yolo_detector import YOLODetector
from perception.camera_projection import CameraProjection


class PerceptionNode:
    """AI 인지 시스템 노드.

    YOLOv8 모델로 객체를 인식하고, Depth Camera를 이용하여
    Pinhole Camera Model 기반 2D→3D 좌표 변환을 수행한다.
    인식된 객체는 RViz2 마커로 시각화된다.

    인식 가능 객체:
        - Box (화물): 빨간색 마커
        - Person (작업자): 초록색 마커
        - Forklift (지게차): 파란색 마커
    """

    # 객체별 마커 색상 (R, G, B, A)
    MARKER_COLORS = {
        'Box': (1.0, 0.2, 0.2, 0.8),
        'Person': (0.2, 1.0, 0.2, 0.8),
        'Forklift': (0.2, 0.2, 1.0, 0.8),
    }

    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[PerceptionNode] rclpy 미설치 - 독립 실행 모드")
            return

        self.node = rclpy.create_node('perception_node')

        # 파라미터
        self.node.declare_parameter('confidence_threshold', 0.5)
        self.node.declare_parameter('model_path', 'yolov8n.pt')
        self.node.declare_parameter('inference_device', 'cpu')
        self.node.declare_parameter('publish_rate', 10.0)

        self.conf_threshold = self.node.get_parameter(
            'confidence_threshold').value

        # 인지 모듈 초기화
        self.detector = YOLODetector(
            model_path=self.node.get_parameter('model_path').value,
            device=self.node.get_parameter('inference_device').value,
            confidence_threshold=self.conf_threshold
        )
        self.projector = CameraProjection()

        # 상태
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.marker_id = 0

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, depth=5)

        # 구독
        self.node.create_subscription(
            Image, '/camera/image_raw', self._rgb_callback, sensor_qos)
        self.node.create_subscription(
            Image, '/camera/depth', self._depth_callback, sensor_qos)
        self.node.create_subscription(
            CameraInfo, '/camera/camera_info',
            self._camera_info_callback, sensor_qos)

        # 발행
        self.detection_pub = self.node.create_publisher(
            String, '/detections', 10)
        self.marker_pub = self.node.create_publisher(
            MarkerArray, '/detection_markers', 10)

        # 인식 루프 (10Hz)
        rate = self.node.get_parameter('publish_rate').value
        self.node.create_timer(1.0 / rate, self._detect_loop)

        self.node.get_logger().info(
            f'인지 노드 초기화 완료 (신뢰도 임계값: {self.conf_threshold})')

    def _rgb_callback(self, msg):
        """RGB 이미지 저장."""
        h, w = msg.height, msg.width
        if msg.encoding == 'rgb8':
            self.latest_rgb = np.frombuffer(
                msg.data, dtype=np.uint8).reshape((h, w, 3))
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(
                msg.data, dtype=np.uint8).reshape((h, w, 3))
            self.latest_rgb = img[:, :, ::-1]

    def _depth_callback(self, msg):
        """Depth 이미지 저장."""
        h, w = msg.height, msg.width
        if msg.encoding == '32FC1':
            self.latest_depth = np.frombuffer(
                msg.data, dtype=np.float32).reshape((h, w))
        elif msg.encoding == '16UC1':
            raw = np.frombuffer(
                msg.data, dtype=np.uint16).reshape((h, w))
            self.latest_depth = raw.astype(np.float32) / 1000.0

    def _camera_info_callback(self, msg):
        """카메라 내부 파라미터 저장."""
        K = np.array(msg.k).reshape((3, 3))
        self.projector.set_intrinsics(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        self.camera_info = msg

    def _detect_loop(self):
        """주기적 객체 인식 및 결과 발행."""
        if self.latest_rgb is None or self.latest_depth is None:
            return

        # YOLO 추론
        detections = self.detector.detect(self.latest_rgb)

        results = []
        markers = MarkerArray()
        self.marker_id = 0

        for det in detections:
            class_name = det['class']
            confidence = det['confidence']
            bbox = det['bbox']  # [x1, y1, x2, y2]

            if confidence < self.conf_threshold:
                continue

            # 바운딩 박스 중심의 depth 추출
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            h, w = self.latest_depth.shape
            cx = np.clip(cx, 0, w - 1)
            cy = np.clip(cy, 0, h - 1)

            # 중심 주변 5x5 영역의 중앙값 깊이
            r = 2
            patch = self.latest_depth[
                max(0, cy-r):min(h, cy+r+1),
                max(0, cx-r):min(w, cx+r+1)]
            depth = float(np.median(patch[patch > 0])) if np.any(
                patch > 0) else 0.0

            if depth <= 0:
                continue

            # 2D → 3D 좌표 변환 (Pinhole Camera Model)
            point_3d = self.projector.pixel_to_3d(cx, cy, depth)

            result = {
                'class': class_name,
                'confidence': round(confidence, 3),
                'bbox': bbox,
                'position_3d': {
                    'x': round(point_3d[0], 3),
                    'y': round(point_3d[1], 3),
                    'z': round(point_3d[2], 3)
                },
                'distance': round(depth, 3)
            }
            results.append(result)

            # RViz2 마커 생성
            marker = self._create_marker(
                class_name, confidence, point_3d, depth)
            markers.markers.append(marker)

        # 결과 발행
        if results:
            det_msg = String()
            det_msg.data = json.dumps(results, ensure_ascii=False)
            self.detection_pub.publish(det_msg)
            self.marker_pub.publish(markers)

    def _create_marker(self, class_name, confidence, position, distance):
        """RViz2 시각화 마커 생성."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0

        # 객체 크기 (근사)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # 색상
        color = self.MARKER_COLORS.get(
            class_name, (0.8, 0.8, 0.8, 0.8))
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        # 텍스트 마커 (라벨)
        marker.text = f"{class_name}: {confidence:.2f}, {distance:.1f}m"
        marker.lifetime.sec = 1

        return marker

    def spin(self):
        if ROS2_AVAILABLE:
            rclpy.spin(self.node)


def main():
    if ROS2_AVAILABLE:
        rclpy.init()
        node = PerceptionNode()
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.shutdown()
    else:
        print("[PerceptionNode] ROS2 미설치 - demo 모드로 main.py를 사용하세요.")


if __name__ == '__main__':
    main()
