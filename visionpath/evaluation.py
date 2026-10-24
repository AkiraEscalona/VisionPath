"""Compare detections with manually annotated observations.

ANNOTATION FILE (CSV)  columns: frame,label,position,reference_distance_m
  frame                 0-based frame index in the test video (as counted by the benchmark)
  label                 object class (e.g. person, chair) or 'none' = "this frame has no
                        supported object" (so detections there count as false positives)
  position              optional: left / center / right (if given, it must match too)
  reference_distance_m  optional: tape-measured distance for the distance-error (MAE) metric

Annotate EVERY supported object visible in each annotated frame; otherwise correct detections
of un-annotated objects are counted as false positives.

METRICS
  detection rate (recall) % = TP / (TP + FN) * 100     <- the report formula; NOT overall accuracy
  precision %               = TP / (TP + FP) * 100
  FP = detections in annotated frames that match no annotation; FN = annotations never detected
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .distance import mean_absolute_error

DETECTION_LOG_COLUMNS = ["frame", "label", "confidence", "position", "proximity", "distance_m", "source"]


@dataclass
class Annotation:
    frame: int
    label: str
    position: str = ""
    reference_distance_m: Optional[float] = None


@dataclass
class LoggedDetection:
    frame: int
    label: str
    confidence: float
    position: str
    distance_m: Optional[float]


@dataclass
class EvalResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    detection_rate_pct: Optional[float] = None
    precision_pct: Optional[float] = None
    distance_samples: int = 0
    distance_mae_m: Optional[float] = None
    annotated_frames: int = 0


def _opt_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    return float(text) if text else None


def load_annotations(path: Path) -> List[Annotation]:
    out: List[Annotation] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"frame", "label"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{Path(path).name} needs at least the columns: frame,label")
        for line_no, row in enumerate(reader, start=2):
            if not (row.get("frame") or "").strip():
                continue  # blank line
            try:
                out.append(Annotation(
                    frame=int(row["frame"]),
                    label=(row["label"] or "").strip().lower(),
                    position=(row.get("position") or "").strip().lower(),
                    reference_distance_m=_opt_float(row.get("reference_distance_m", "")),
                ))
            except ValueError as exc:
                raise ValueError(f"{Path(path).name}, line {line_no}: {exc}") from exc
    return out


def load_detection_log(path: Path) -> List[LoggedDetection]:
    out: List[LoggedDetection] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(LoggedDetection(
                frame=int(row["frame"]), label=row["label"].lower(),
                confidence=float(row["confidence"] or 0), position=row["position"],
                distance_m=_opt_float(row.get("distance_m", "")),
            ))
    return out


class DetectionLogger:
    """Writes one CSV row per detection (used by the benchmark). Contains no images."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(DETECTION_LOG_COLUMNS)

    def log(self, frame_index: int, detections) -> None:
        for d in detections:
            self._writer.writerow([
                frame_index, d.label, f"{d.confidence:.3f}", d.position, d.proximity or "",
                "" if d.distance_m is None else f"{d.distance_m:.2f}", d.source,
            ])

    def close(self) -> None:
        self._fh.close()


def evaluate(annotations: List[Annotation], detections: List[LoggedDetection],
             frame_tolerance: int = 0) -> EvalResult:
    """Match detections to annotations frame by frame (see module docstring)."""
    by_frame_ann: Dict[int, List[Annotation]] = defaultdict(list)
    for a in annotations:
        by_frame_ann[a.frame].append(a)
    by_frame_det: Dict[int, List[LoggedDetection]] = defaultdict(list)
    for d in detections:
        by_frame_det[d.frame].append(d)

    result = EvalResult(annotated_frames=len(by_frame_ann))
    pairs = []
    for frame, anns in by_frame_ann.items():
        window: List[LoggedDetection] = []
        for k in range(frame - frame_tolerance, frame + frame_tolerance + 1):
            window.extend(by_frame_det.get(k, []))
        window.sort(key=lambda d: -d.confidence)
        used = set()
        for a in anns:
            if a.label == "none":
                continue
            match = None
            for i, d in enumerate(window):
                if i in used or d.label != a.label:
                    continue
                if a.position and d.position != a.position:
                    continue
                match = i
                break
            if match is None:
                result.fn += 1
            else:
                used.add(match)
                result.tp += 1
                det = window[match]
                if a.reference_distance_m is not None and det.distance_m is not None:
                    pairs.append((a.reference_distance_m, det.distance_m))
        result.fp += len(window) - len(used)

    if result.tp + result.fn > 0:
        result.detection_rate_pct = 100.0 * result.tp / (result.tp + result.fn)
    if result.tp + result.fp > 0:
        result.precision_pct = 100.0 * result.tp / (result.tp + result.fp)
    result.distance_samples = len(pairs)
    result.distance_mae_m = mean_absolute_error(pairs)
    return result


# ----------------------------------------------------------------------
# Manual distance checks recorded from the GUI
# ----------------------------------------------------------------------
DISTANCE_CHECK_COLUMNS = ["timestamp", "label", "reference_m", "estimated_m", "abs_error_m"]


def append_distance_check(path: Path, label: str, reference_m: float, estimated_m: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(DISTANCE_CHECK_COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), label,
                         f"{reference_m:.2f}", f"{estimated_m:.2f}",
                         f"{abs(reference_m - estimated_m):.2f}"])


def distance_check_summary(path: Path):
    """Return (number_of_checks, MAE in metres) from the distance-check CSV."""
    path = Path(path)
    if not path.exists():
        return 0, None
    pairs = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pairs.append((float(row["reference_m"]), float(row["estimated_m"])))
    return len(pairs), mean_absolute_error(pairs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a detection log against manual annotations.")
    p.add_argument("--log", required=True, help="detections CSV written by the benchmark")
    p.add_argument("--annotations", required=True, help="annotation CSV (frame,label,position,reference_distance_m)")
    p.add_argument("--tolerance", type=int, default=0, help="frames of tolerance around each annotated frame")
    args = p.parse_args(argv)
    res = evaluate(load_annotations(Path(args.annotations)), load_detection_log(Path(args.log)), args.tolerance)
    print(f"Annotated frames : {res.annotated_frames}")
    print(f"TP / FP / FN     : {res.tp} / {res.fp} / {res.fn}")
    print(f"Detection rate   : {res.detection_rate_pct if res.detection_rate_pct is None else round(res.detection_rate_pct, 1)} %  (recall; not overall accuracy)")
    print(f"Precision        : {res.precision_pct if res.precision_pct is None else round(res.precision_pct, 1)} %")
    print(f"Distance MAE     : {res.distance_mae_m if res.distance_mae_m is None else round(res.distance_mae_m, 3)} m over {res.distance_samples} sample(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
