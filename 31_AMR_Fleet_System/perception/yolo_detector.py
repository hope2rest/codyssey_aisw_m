"""
YOLOv8 기반 객체 인식 모듈.

ultralytics YOLOv8 모델을 래핑하여 AMR 환경에서 필요한
3종 객체(화물 Box, 사람 Person, 표지판 Sign)를 탐지한다.

주요 기능:
    - CPU/GPU 자동 선택
    - 인식 결과를 Detection 데이터 클래스로 구조화
    - 신뢰도(confidence) 기반 필터링
    - 관심 클래스만 선별 탐지

외부 의존성:
    - numpy (필수)
    - ultralytics (YOLOv8, 선택 - 미설치 시 시뮬레이션 모드)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ultralytics가 설치되어 있지 않을 경우를 대비한 안전 임포트
try:
    from ultralytics import YOLO

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    logger.warning(
        "ultralytics 패키지가 설치되어 있지 않습니다. "
        "시뮬레이션 모드로 동작합니다. "
        "실제 인식을 위해 `pip install ultralytics`를 실행하세요."
    )


@dataclass
class Detection:
    """
    단일 객체 인식 결과를 나타내는 데이터 클래스.

    Attributes:
        class_name: 인식된 객체 클래스 이름 ('Box', 'Person', 'Sign').
        confidence: 인식 신뢰도 (0.0 ~ 1.0).
        bbox: 바운딩 박스 [x1, y1, x2, y2] (픽셀 좌표).
        center: 바운딩 박스 중심 좌표 (cx, cy) (픽셀 좌표).
    """
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]

    def __repr__(self) -> str:
        return (
            f"Detection(class='{self.class_name}', conf={self.confidence:.3f}, "
            f"bbox=({self.bbox[0]:.1f},{self.bbox[1]:.1f},{self.bbox[2]:.1f},{self.bbox[3]:.1f}), "
            f"center=({self.center[0]:.1f},{self.center[1]:.1f}))"
        )


# COCO 클래스 → AMR 관심 클래스 매핑
# COCO 데이터셋 기준으로 Box, Person, Sign에 해당하는 클래스들
_COCO_TO_AMR: Dict[str, str] = {
    "person": "Person",
    "suitcase": "Box",
    "backpack": "Box",
    "handbag": "Box",
    "box": "Box",
    "stop sign": "Sign",
    "parking meter": "Sign",
    "fire hydrant": "Sign",
}


class YoloDetector:
    """
    YOLOv8 기반 객체 탐지기.

    AMR 운영 환경에서 화물(Box), 사람(Person), 표지판(Sign) 3종을
    실시간으로 탐지한다. ultralytics가 설치되지 않은 환경에서는
    시뮬레이션 모드로 동작하여 가상의 인식 결과를 반환한다.

    사용 예시::

        detector = YoloDetector(model_path='yolov8n.pt', conf_threshold=0.5)
        detections = detector.detect(image)
        for det in detections:
            print(f"{det.class_name}: {det.confidence:.2f}")

    Args:
        model_path: YOLOv8 모델 가중치 파일 경로.
        conf_threshold: 최소 신뢰도 임계값. 이보다 낮은 인식 결과는 무시.
        device: 추론 장치. None이면 자동 선택(GPU 우선).
        class_mapping: COCO 클래스 → AMR 클래스 매핑 딕셔너리.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        device: Optional[str] = None,
        class_mapping: Optional[Dict[str, str]] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.class_mapping = class_mapping or _COCO_TO_AMR
        self._simulation_mode = not _ULTRALYTICS_AVAILABLE

        # 장치 자동 선택
        if device is None:
            self.device = self._auto_select_device()
        else:
            self.device = device

        # 모델 로딩
        self.model = None
        if not self._simulation_mode:
            try:
                self.model = YOLO(model_path)
                logger.info(f"YOLOv8 모델 로드 완료: {model_path}, 장치: {self.device}")
            except Exception as e:
                logger.error(f"모델 로딩 실패: {e}. 시뮬레이션 모드로 전환.")
                self._simulation_mode = True

    @staticmethod
    def _auto_select_device() -> str:
        """
        추론 장치를 자동 선택한다.

        CUDA GPU가 사용 가능하면 'cuda', 아니면 'cpu'를 반환한다.

        Returns:
            장치 문자열 ('cuda' 또는 'cpu').
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA GPU를 탐지하여 GPU 모드로 동작합니다.")
                return "cuda"
        except ImportError:
            pass
        logger.info("CPU 모드로 동작합니다.")
        return "cpu"

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        이미지에서 AMR 관심 객체를 탐지한다.

        Args:
            image: BGR 또는 RGB 형식의 이미지 배열 (H, W, 3).

        Returns:
            Detection 객체 리스트. 신뢰도 내림차순 정렬.
        """
        if self._simulation_mode:
            return self._simulate_detection(image)

        return self._real_detection(image)

    def _real_detection(self, image: np.ndarray) -> List[Detection]:
        """
        실제 YOLOv8 모델을 사용한 객체 탐지.

        Args:
            image: 입력 이미지 배열.

        Returns:
            Detection 리스트.
        """
        results = self.model(image, device=self.device, verbose=False)
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # 클래스 이름 추출
                cls_id = int(boxes.cls[i].item())
                coco_name = result.names.get(cls_id, "unknown")
                amr_class = self.class_mapping.get(coco_name.lower())

                if amr_class is None:
                    continue  # 관심 클래스가 아님

                conf = float(boxes.conf[i].item())
                if conf < self.conf_threshold:
                    continue

                # 바운딩 박스 좌표
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                detections.append(
                    Detection(
                        class_name=amr_class,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                )

        # 신뢰도 내림차순 정렬
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _simulate_detection(self, image: np.ndarray) -> List[Detection]:
        """
        시뮬레이션 모드 객체 탐지.

        ultralytics가 없을 때 가상의 탐지 결과를 생성한다.
        이미지 크기를 기반으로 무작위 바운딩 박스를 생성.

        Args:
            image: 입력 이미지 배열.

        Returns:
            시뮬레이션된 Detection 리스트.
        """
        if image.ndim < 2:
            return []

        h, w = image.shape[:2]
        rng = np.random.default_rng()
        detections: List[Detection] = []

        # 시뮬레이션용 탐지 결과 생성
        sim_objects = [
            ("Box", 0.92, (w * 0.3, h * 0.4, w * 0.5, h * 0.7)),
            ("Person", 0.87, (w * 0.6, h * 0.2, w * 0.75, h * 0.9)),
            ("Sign", 0.78, (w * 0.1, h * 0.1, w * 0.2, h * 0.25)),
        ]

        for class_name, base_conf, (x1, y1, x2, y2) in sim_objects:
            # 약간의 노이즈 추가
            conf = base_conf + rng.uniform(-0.05, 0.05)
            conf = max(0.0, min(1.0, conf))

            if conf < self.conf_threshold:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info(f"[시뮬레이션] {len(detections)}개 객체 탐지됨.")
        return detections

    def detect_specific_class(self, image: np.ndarray, target_class: str) -> List[Detection]:
        """
        특정 클래스의 객체만 탐지한다.

        Args:
            image: 입력 이미지 배열.
            target_class: 탐지할 클래스 이름 ('Box', 'Person', 'Sign').

        Returns:
            해당 클래스의 Detection 리스트.
        """
        all_detections = self.detect(image)
        return [d for d in all_detections if d.class_name == target_class]

    @property
    def is_simulation(self) -> bool:
        """시뮬레이션 모드 여부를 반환한다."""
        return self._simulation_mode


# ---------------------------------------------------------------------------
# 모듈 단독 실행 시 데모
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 가상 이미지 생성 (640x480 검정 이미지)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = YoloDetector(conf_threshold=0.5)
    print(f"시뮬레이션 모드: {detector.is_simulation}")
    print(f"추론 장치: {detector.device}")

    results = detector.detect(dummy_image)
    print(f"\n탐지 결과 ({len(results)}개):")
    for det in results:
        print(f"  {det}")
