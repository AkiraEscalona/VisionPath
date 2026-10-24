"""Local object detection with OpenCV DNN (MobileNet-SSD, Caffe format).

Model: MobileNet-SSD (Caffe) from the chuanqi305/MobileNet-SSD repository,
trained on PASCAL VOC (20 object classes + background). It runs fully offline
through cv2.dnn; no Caffe, TensorFlow or PyTorch installation is needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

VOC_CLASSES: Tuple[str, ...] = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

MIN_WEIGHTS_BYTES = 5_000_000  # the real caffemodel is ~23 MB; smaller means a bad download

DOWNLOAD_HINT = "Run:  python scripts/download_model.py   (or see models/README.md for manual steps)"


class ModelNotFoundError(Exception):
    """Model files are missing."""


class ModelLoadError(Exception):
    """Model files exist but OpenCV could not load them."""


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]            # x1, y1, x2, y2 in pixels
    source: str = "dnn"                       # "dnn" or "edge"
    position: str = "center"                  # left / center / right
    proximity: Optional[str] = None           # near / medium / far / unknown
    distance_m: Optional[float] = None        # approximate, only when reliable
    distance_method: str = ""

    @property
    def box_height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def box_width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0


def position_for_box(box: Tuple[int, int, int, int], frame_width: int,
                     left_boundary: float = 0.33, right_boundary: float = 0.67) -> str:
    """Return 'left', 'center' or 'right' from the horizontal centre of the box."""
    if frame_width <= 0:
        return "center"
    cx = (box[0] + box[2]) / 2.0 / frame_width
    if cx < left_boundary:
        return "left"
    if cx > right_boundary:
        return "right"
    return "center"


def parse_detections(output: np.ndarray, frame_w: int, frame_h: int,
                     conf_threshold: float,
                     labels: Sequence[str] = VOC_CLASSES) -> List[Detection]:
    """Convert the raw SSD output (1 x 1 x N x 7) into Detection objects."""
    rows = np.asarray(output).reshape(-1, 7)
    detections: List[Detection] = []
    for row in rows:
        conf = float(row[2])
        class_id = int(row[1])
        if conf < conf_threshold or class_id <= 0 or class_id >= len(labels):
            continue
        x1 = int(max(0, min(frame_w - 1, row[3] * frame_w)))
        y1 = int(max(0, min(frame_h - 1, row[4] * frame_h)))
        x2 = int(max(0, min(frame_w - 1, row[5] * frame_w)))
        y2 = int(max(0, min(frame_h - 1, row[6] * frame_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(labels[class_id], conf, (x1, y1, x2, y2)))
    return detections


class ObjectDetector:
    """MobileNet-SSD detector using cv2.dnn on the CPU."""

    def __init__(self, prototxt_path: Path, weights_path: Path, conf_threshold: float = 0.5):
        self.prototxt_path = Path(prototxt_path)
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold

        missing = [str(p) for p in (self.prototxt_path, self.weights_path) if not p.is_file()]
        if missing:
            raise ModelNotFoundError(
                "Model file(s) not found:\n  " + "\n  ".join(missing) + "\n" + DOWNLOAD_HINT
            )
        size = self.weights_path.stat().st_size
        if size < MIN_WEIGHTS_BYTES:
            raise ModelLoadError(
                f"{self.weights_path.name} is only {size} bytes, so the download is incomplete "
                f"(it may be an HTML error page). Delete it and download again.\n" + DOWNLOAD_HINT
            )
        try:
            self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.weights_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise ModelLoadError(
                f"OpenCV could not load the model: {exc}\n"
                f"Make sure the .prototxt and .caffemodel belong together and are not corrupted.\n"
                + DOWNLOAD_HINT
            ) from exc

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        output = self.net.forward()
        return parse_detections(output, w, h, self.conf_threshold)

    def verify(self) -> str:
        """Run one dummy inference to prove the model works. Returns a status string."""
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(blank, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        output = self.net.forward()
        if output.ndim != 4 or output.shape[-1] != 7:
            raise ModelLoadError(f"Unexpected model output shape {output.shape}; expected (1, 1, N, 7).")
        return f"Model OK: output shape {tuple(output.shape)}, {len(VOC_CLASSES) - 1} object classes."
