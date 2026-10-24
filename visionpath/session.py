"""A running detection session: camera -> pipeline -> alerts -> latest result.

The processing loop runs in its own thread so the GUI stays responsive. The GUI
only calls get_update() / pop_messages() on a timer; it never blocks.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .audio import AlertManager, Speaker
from .camera import open_source
from .config import Config
from .detector import Detection
from .performance import PerformanceMonitor
from .pipeline import VisionPipeline

log = logging.getLogger(__name__)


@dataclass
class SessionUpdate:
    frame: np.ndarray
    detections: List[Detection]
    live_fps: Optional[float]
    latency_ms: float
    ram_mb: Optional[float]
    peak_ram_mb: Optional[float]
    frame_index: int


class DetectionSession:
    def __init__(self, config: Config, source_text: str,
                 pipeline: Optional[VisionPipeline] = None,
                 source_factory: Optional[Callable] = None,
                 use_speaker: bool = True,
                 pace_files: bool = True):
        self.config = config
        self.source_text = source_text
        self.pipeline = pipeline
        self._source_factory = source_factory or (
            lambda src, cfg: open_source(src, cfg.capture_width, cfg.capture_height,
                                         cfg.reconnect_attempts, cfg.reconnect_delay_s))
        self._use_speaker = use_speaker
        self._pace_files = pace_files

        self.speech_enabled = config.speech_enabled
        self.alerts = AlertManager(config.alert_cooldown_s, config.alert_global_interval_s,
                                   config.announce_far, config.ignore_labels)
        self.speaker: Optional[Speaker] = None
        self.monitor = PerformanceMonitor()
        self.detections_total = 0
        self.last_alert = ""

        self._source = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[SessionUpdate] = None
        self._messages: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._last_detections: List[Detection] = []
        self._last_frame_h = 0
        self.finished = threading.Event()

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open model, camera and speaker. Raises ModelNotFoundError / ModelLoadError / CameraError."""
        if self.running:
            return
        if self.pipeline is None:
            self.pipeline = VisionPipeline(self.config)   # fails early if the model is missing
        self._source = self._source_factory(self.source_text, self.config)
        self.monitor = PerformanceMonitor()
        if self._use_speaker:
            self.speaker = Speaker(self.config.speech_rate, self.config.speech_volume,
                                   on_status=lambda m: self._post("warning", m))
            self.speaker.start()
        self._stop.clear()
        self.finished.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="processing")
        self._thread.start()

    def stop(self) -> dict:
        """Stop everything cleanly and return the performance summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._source is not None:
            self._source.release()
            self._source = None
        if self.speaker is not None:
            self.speaker.stop()
            self.speaker = None
        summary = self.monitor.summary()
        summary["detections_total"] = self.detections_total
        summary["alerts_generated"] = self.alerts.alerts_generated
        return summary

    # ------------------------------------------------------------------
    def _post(self, level: str, text: str) -> None:
        self._messages.put((level, text))

    def pop_messages(self) -> List[Tuple[str, str]]:
        out = []
        while True:
            try:
                out.append(self._messages.get_nowait())
            except queue.Empty:
                return out

    def get_update(self) -> Optional[SessionUpdate]:
        with self._lock:
            update, self._latest = self._latest, None
        return update

    # ------------------------------------------------------------------
    def _run(self) -> None:
        n = max(1, self.config.process_every_n_frames)
        index = -1
        timeouts = 0
        try:
            while not self._stop.is_set():
                frame_start = time.perf_counter()
                frame = self._source.read(timeout=0.5)
                if frame is None:
                    if getattr(self._source, "ended", False):
                        self._post("info", "Video file finished.")
                        break
                    if getattr(self._source, "lost", False):
                        reason = getattr(self._source, "last_error", "")
                        self._post("error", "Camera or stream connection lost. "
                                            "Check the camera/Wi-Fi and press Start again." +
                                   (f" ({reason})" if reason else ""))
                        break
                    timeouts += 1
                    if timeouts in (4, 20):
                        self._post("warning", "Waiting for video frames...")
                    continue
                timeouts = 0
                index += 1
                if index % n != 0:
                    continue

                t0 = time.perf_counter()
                detections = self.pipeline.process(frame)
                self.monitor.record_frame(time.perf_counter() - t0)

                self.detections_total += len(detections)
                message = self.alerts.select(detections)
                if message:
                    self.last_alert = message
                    self._post("alert", message)
                    if self.speech_enabled and self.speaker is not None:
                        self.speaker.say(message)

                annotated = self.pipeline.draw(frame, detections)
                self._last_detections = detections
                self._last_frame_h = frame.shape[0]
                update = SessionUpdate(annotated, detections, self.monitor.live_fps,
                                       self.monitor.last_latency_ms, self.monitor.current_ram_mb,
                                       self.monitor.peak_ram_mb, index)
                with self._lock:
                    self._latest = update

                fps = getattr(self._source, "fps", 0.0)
                if self._pace_files and not getattr(self._source, "is_live", True) and fps > 0:
                    remaining = n / fps - (time.perf_counter() - frame_start)
                    if remaining > 0:
                        self._stop.wait(remaining)
        except Exception as exc:  # noqa: BLE001 - keep the GUI alive, report instead
            log.exception("Processing loop failed")
            self._post("error", f"Processing error: {exc}")
        finally:
            self.finished.set()

    # ------------------------------------------------------------------
    # Live settings (safe to change while running)
    def set_cooldown(self, seconds: float) -> None:
        self.alerts.cooldown_s = max(0.0, seconds)

    def set_confidence(self, value: float) -> None:
        if self.pipeline is not None:
            self.pipeline.set_confidence(value)

    def set_edge_enabled(self, enabled: bool) -> None:
        if self.pipeline is not None:
            self.pipeline.edge_enabled = enabled

    # ------------------------------------------------------------------
    # Calibration and manual distance checks (use the last processed frame)
    def _largest(self, label: Optional[str] = None) -> Optional[Detection]:
        pool = [d for d in self._last_detections if d.source == "dnn" and (label is None or d.label == label)]
        return max(pool, key=lambda d: d.box_height, default=None)

    def calibrate(self, label: str, reference_distance_m: float) -> float:
        det = self._largest(label)
        if det is None or self.pipeline is None:
            raise ValueError(f"No '{label}' is currently detected. Place it in view and try again.")
        return self.pipeline.estimator.calibrate(label, reference_distance_m, det.box, self._last_frame_h)

    def reference_check(self) -> Detection:
        """Return the currently largest detection that has a distance estimate."""
        pool = [d for d in self._last_detections if d.distance_m is not None]
        if not pool:
            raise ValueError("No detection with a distance estimate right now. "
                             "Make sure a supported object (e.g. a person) is fully visible.")
        return max(pool, key=lambda d: d.box_height)
