"""Approximate distance estimation from a single camera.

METHOD (pinhole model, one assumption per class):
    distance = real_height * focal_length / box_height_in_pixels
Using the focal length in units of "frame heights" (focal_length_norm) keeps it
independent of the capture resolution:
    distance = real_height * focal_length_norm / (box_height / frame_height)

ASSUMPTIONS / LIMITATIONS
* The object's real height equals the configured average (people differ in height).
* The box covers the whole object; if it touches the top/bottom image edge the
  object is cut off, so only a coarse proximity category is given.
* The camera is roughly level and not tilted; lens distortion is ignored.
* The default focal length comes from an assumed field of view. Calibrate with a
  known distance for better results.
A single camera cannot measure distance directly: treat every value as an estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .detector import Detection


@dataclass
class DistanceEstimate:
    distance_m: Optional[float]
    proximity: str      # near / medium / far
    method: str         # "pinhole-height" or "box-height-fraction"


def default_focal_length_norm(vertical_fov_deg: float) -> float:
    """Focal length in frame-height units from an assumed vertical field of view."""
    return 0.5 / math.tan(math.radians(vertical_fov_deg) / 2.0)


def focal_length_from_reference(reference_distance_m: float, box_height_px: float,
                                frame_height_px: float, real_height_m: float) -> float:
    """Calibrate: object of known height at a known distance -> focal_length_norm."""
    if min(reference_distance_m, box_height_px, frame_height_px, real_height_m) <= 0:
        raise ValueError("Calibration inputs must all be positive.")
    return reference_distance_m * (box_height_px / frame_height_px) / real_height_m


def mean_absolute_error(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    """MAE = (1/N) * sum(|reference - estimate|) for (reference, estimate) pairs."""
    errors = [abs(ref - est) for ref, est in pairs]
    if not errors:
        return None
    return sum(errors) / len(errors)


class DistanceEstimator:
    def __init__(self, vertical_fov_deg: float = 50.0,
                 focal_length_norm: Optional[float] = None,
                 known_heights_m: Optional[Dict[str, float]] = None,
                 near_m: float = 1.5, medium_m: float = 3.0,
                 near_fraction: float = 0.6, medium_fraction: float = 0.3,
                 edge_margin_px: int = 3):
        self.focal_length_norm = focal_length_norm or default_focal_length_norm(vertical_fov_deg)
        self.calibrated = focal_length_norm is not None
        self.known_heights_m = dict(known_heights_m or {})
        self.near_m = near_m
        self.medium_m = medium_m
        self.near_fraction = near_fraction
        self.medium_fraction = medium_fraction
        self.edge_margin_px = edge_margin_px

    def proximity_from_distance(self, d: float) -> str:
        if d < self.near_m:
            return "near"
        if d < self.medium_m:
            return "medium"
        return "far"

    def proximity_from_fraction(self, frac: float) -> str:
        if frac >= self.near_fraction:
            return "near"
        if frac >= self.medium_fraction:
            return "medium"
        return "far"

    def is_truncated(self, box: Tuple[int, int, int, int], frame_h: int) -> bool:
        return box[1] <= self.edge_margin_px or box[3] >= frame_h - 1 - self.edge_margin_px

    def estimate(self, det: Detection, frame_h: int) -> DistanceEstimate:
        box_h = det.box_height
        frac = box_h / float(frame_h) if frame_h > 0 else 0.0
        real_h = self.known_heights_m.get(det.label)
        if real_h and box_h > 0 and not self.is_truncated(det.box, frame_h):
            d = real_h * self.focal_length_norm / frac
            return DistanceEstimate(round(d, 2), self.proximity_from_distance(d), "pinhole-height")
        return DistanceEstimate(None, self.proximity_from_fraction(frac), "box-height-fraction")

    def apply(self, det: Detection, frame_h: int) -> Detection:
        est = self.estimate(det, frame_h)
        det.distance_m = est.distance_m
        det.proximity = est.proximity
        det.distance_method = est.method
        return det

    def calibrate(self, label: str, reference_distance_m: float,
                  box: Tuple[int, int, int, int], frame_h: int) -> float:
        real_h = self.known_heights_m.get(label)
        if not real_h:
            raise ValueError(f"No known real height for '{label}'. Add it to known_heights_m in config.json.")
        if self.is_truncated(box, frame_h):
            raise ValueError("The object is cut off by the image edge; step back so it is fully visible.")
        self.focal_length_norm = focal_length_from_reference(
            reference_distance_m, box[3] - box[1], frame_h, real_h)
        self.calibrated = True
        return self.focal_length_norm
