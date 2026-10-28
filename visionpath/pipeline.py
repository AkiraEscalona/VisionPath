"""One-frame vision pipeline shared by the GUI session and the benchmark."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .config import Config
from .detector import Detection, ObjectDetector, position_for_box
from .distance import DistanceEstimator
from .edge_obstacles import EdgeObstacleDetector

PROXIMITY_COLORS = {          # BGR
    "near": (0, 0, 255),
    "medium": (0, 165, 255),
    "far": (0, 200, 0),
    "unknown": (200, 200, 200),
}


class VisionPipeline:
    def __init__(self, config: Config, detector=None,
                 estimator: Optional[DistanceEstimator] = None,
                 edge_detector: Optional[EdgeObstacleDetector] = None):
        self.config = config
        if detector is None:
            detector = ObjectDetector(
                config.resolve_path(config.model_prototxt),
                config.resolve_path(config.model_weights),
                config.confidence_threshold,
            )
        self.detector = detector
        self.estimator = estimator or DistanceEstimator(
            vertical_fov_deg=config.vertical_fov_deg,
            focal_length_norm=config.focal_length_norm,
            known_heights_m=config.known_heights_m,
            near_m=config.near_distance_m, medium_m=config.medium_distance_m,
            near_fraction=config.near_height_fraction, medium_fraction=config.medium_height_fraction,
        )
        self.edge_detector = edge_detector or EdgeObstacleDetector(
            config.edge_canny_low, config.edge_canny_high,
            config.edge_density_threshold, config.edge_min_contour_area_fraction,
            left_boundary=config.left_boundary, right_boundary=config.right_boundary,
        )
        self.edge_enabled = config.edge_obstacle_enabled

    def set_confidence(self, value: float) -> None:
        self.detector.conf_threshold = value

    def process(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        detections = list(self.detector.detect(frame))
        for det in detections:
            det.position = position_for_box(det.box, w, self.config.left_boundary, self.config.right_boundary)
            self.estimator.apply(det, h)
        if self.edge_enabled:
            detections.extend(self.edge_detector.detect(frame))
        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Return a copy of the frame with zone lines, boxes and labels drawn."""
        out = frame.copy()
        h, w = out.shape[:2]
        for frac in (self.config.left_boundary, self.config.right_boundary):
            x = int(w * frac)
            cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        for name, frac in (("LEFT", 0.165), ("CENTER", 0.5), ("RIGHT", 0.835)):
            cv2.putText(out, name, (int(w * frac) - 30, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for d in detections:
            color = PROXIMITY_COLORS.get(d.proximity or "unknown", (200, 200, 200))
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{d.label} {d.confidence:.0%} | {d.position}"
            if d.proximity and d.proximity != "unknown":
                text += f" | {d.proximity}"
            if d.distance_m is not None:
                text += f" ~{d.distance_m:.1f}m"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(out, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, "Distances are estimates only", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out
