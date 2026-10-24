"""Performance measurement and CSV result output.

Definitions (match the project design document):
  latency per frame  = time spent in the detection pipeline for one frame (ms)
  average latency    = T_total / N_frames
  average FPS        = N_frames / T_total          (T_total = summed processing time)
  wall-clock FPS     = N_frames / elapsed time     (includes waiting for the camera)
  peak RAM           = max of process memory (RSS) samples, in MB

Latency covers detection + distance estimation (+ optional edge heuristic).
It does NOT include camera/network delay, drawing, or speech output.
"""
from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, Optional

try:
    import psutil
except ImportError:  # psutil is in requirements.txt, but fail softly
    psutil = None

_process = None


def get_process_ram_mb() -> Optional[float]:
    """Resident memory of this Python process in MB (None if psutil is missing)."""
    global _process
    if psutil is None:
        return None
    if _process is None:
        _process = psutil.Process()
    return _process.memory_info().rss / (1024.0 * 1024.0)


class PerformanceMonitor:
    def __init__(self, warmup_frames: int = 0, window: int = 30,
                 time_fn: Callable[[], float] = time.perf_counter,
                 ram_fn: Callable[[], Optional[float]] = get_process_ram_mb):
        self.warmup_frames = warmup_frames
        self._time_fn = time_fn
        self._ram_fn = ram_fn
        self._recent: Deque[float] = deque(maxlen=max(2, window))
        self.frames_seen = 0
        self.frames_measured = 0
        self._latency_sum = 0.0
        self._latency_max = 0.0
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None
        self.current_ram_mb: Optional[float] = None
        self.peak_ram_mb: Optional[float] = None
        self.last_latency_ms = 0.0
        self.sample_ram()

    def sample_ram(self) -> None:
        ram = self._ram_fn()
        if ram is None:
            return
        self.current_ram_mb = ram
        if self.peak_ram_mb is None or ram > self.peak_ram_mb:
            self.peak_ram_mb = ram

    def record_frame(self, latency_s: float) -> None:
        """Call once per processed frame with that frame's processing time (seconds)."""
        now = self._time_fn()
        self.frames_seen += 1
        self.last_latency_ms = latency_s * 1000.0
        self.sample_ram()
        self._recent.append(now)
        if self.frames_seen <= self.warmup_frames:
            return  # warm-up frames (model initialisation) are excluded from statistics
        self.frames_measured += 1
        self._latency_sum += latency_s
        self._latency_max = max(self._latency_max, latency_s)
        if self._first_ts is None:
            self._first_ts = now - latency_s
        self._last_ts = now

    @contextmanager
    def frame(self):
        start = self._time_fn()
        try:
            yield
        finally:
            self.record_frame(self._time_fn() - start)

    @property
    def live_fps(self) -> Optional[float]:
        if len(self._recent) < 2:
            return None
        span = self._recent[-1] - self._recent[0]
        return (len(self._recent) - 1) / span if span > 0 else None

    def summary(self) -> Dict[str, Optional[float]]:
        n = self.frames_measured
        avg_latency_ms = (self._latency_sum / n * 1000.0) if n else None
        avg_fps = (n / self._latency_sum) if n and self._latency_sum > 0 else None
        wall_fps = None
        if n and self._first_ts is not None and self._last_ts is not None:
            elapsed = self._last_ts - self._first_ts
            wall_fps = n / elapsed if elapsed > 0 else None
        return {
            "frames_processed": n,
            "warmup_frames_excluded": min(self.frames_seen, self.warmup_frames),
            "avg_latency_ms": avg_latency_ms,
            "max_latency_ms": self._latency_max * 1000.0 if n else None,
            "avg_fps": avg_fps,
            "wall_fps": wall_fps,
            "peak_ram_mb": self.peak_ram_mb,
        }


# ----------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------
RESULT_COLUMNS = [
    "timestamp", "config_name", "source", "frames_processed", "warmup_frames_excluded",
    "avg_latency_ms", "max_latency_ms", "avg_fps", "wall_fps", "peak_ram_mb",
    "detections_total", "alerts_generated",
    "tp", "fp", "fn", "detection_rate_pct", "precision_pct",
    "distance_samples", "distance_mae_m", "notes",
]


class ResultsFileError(Exception):
    """The results CSV cannot be written (locked, or has different columns)."""


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else value


def append_result_row(path: Path, row: Dict[str, object]) -> None:
    """Append one measurement row (creates the file and header if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    try:
        if exists:
            with open(path, "r", newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
            if header != RESULT_COLUMNS:
                raise ResultsFileError(
                    f"{path.name} has different columns than this version of VisionPath. "
                    f"Rename or move the old file and run again.")
        full = {col: "" for col in RESULT_COLUMNS}
        full["timestamp"] = datetime.now().isoformat(timespec="seconds")
        full.update({k: v for k, v in row.items() if k in full})
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow({k: _fmt(v) for k, v in full.items()})
    except PermissionError as exc:
        raise ResultsFileError(
            f"Cannot write {path.name}. Close it if it is open in Excel and try again.") from exc
    except OSError as exc:
        raise ResultsFileError(f"Cannot write {path}: {exc}") from exc
