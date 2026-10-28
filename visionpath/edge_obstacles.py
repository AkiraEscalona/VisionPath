"""Experimental Canny + contour heuristic for 'possible obstacle' regions.

IMPORTANT LIMITATION: edges and contours cannot recognise what an object is.
This module only flags image regions in the lower half of the frame that are
unusually rich in edges and contain large contours. Textured floors, carpets,
shadows and tiles can trigger false alarms, and low-contrast obstacles (a
plain white wall, a glass door) can be missed. It is OFF by default.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .detector import Detection


class EdgeObstacleDetector:
    def __init__(self, canny_low: int = 50, canny_high: int = 150,
                 density_threshold: float = 0.12, min_area_fraction: float = 0.02,
                 roi_top_fraction: float = 0.5,
                 left_boundary: float = 0.33, right_boundary: float = 0.67):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.density_threshold = density_threshold
        self.min_area_fraction = min_area_fraction
        self.roi_top_fraction = roi_top_fraction
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self._kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        y0 = int(h * self.roi_top_fraction)
        roi = cv2.dilate(edges[y0:, :], self._kernel, iterations=1)
        if roi.size == 0:
            return []

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_fraction * roi.shape[0] * roi.shape[1]
        large = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                large.append(cv2.boundingRect(c))  # x, y, w, h

        x_edges = [0, int(w * self.left_boundary), int(w * self.right_boundary), w]
        zones = ("left", "center", "right")
        results: List[Detection] = []
        for i, name in enumerate(zones):
            xa, xb = x_edges[i], x_edges[i + 1]
            zone = roi[:, xa:xb]
            if zone.size == 0:
                continue
            density = float(np.count_nonzero(zone)) / zone.size
            in_zone = [r for r in large if xa <= r[0] + r[2] / 2.0 < xb]
            if density >= self.density_threshold and in_zone:
                bx1 = min(r[0] for r in in_zone)
                by1 = min(r[1] for r in in_zone) + y0
                bx2 = max(r[0] + r[2] for r in in_zone)
                by2 = max(r[1] + r[3] for r in in_zone) + y0
                box = (max(0, bx1), max(0, by1), min(w - 1, bx2), min(h - 1, by2))
                results.append(Detection(
                    label="obstacle", confidence=min(1.0, density), box=box,
                    source="edge", position=name, proximity="unknown",
                    distance_method="edge-heuristic",
                ))
        return results
