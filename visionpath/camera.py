"""Video input: webcam, smartphone IP/RTSP stream, or a video file.

Live sources are read in a background thread that keeps only the *latest*
frame. This avoids the growing delay that happens when a slow detector falls
behind a network stream. Video files are read frame-by-frame without dropping
frames (needed for repeatable benchmarks).
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Union

import cv2

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


class CameraError(Exception):
    """Raised when a video source cannot be opened."""


def parse_source(text: Union[str, int]) -> Union[int, str]:
    """'0' -> 0 (webcam index); anything else is kept as a URL / file path."""
    if isinstance(text, int):
        return text
    value = str(text).strip()
    return int(value) if value.isdigit() else value


def is_video_file(source: Union[int, str]) -> bool:
    if isinstance(source, int):
        return False
    if "://" in source:
        return False
    return os.path.isfile(source) or Path(source).suffix.lower() in VIDEO_EXTENSIONS


def _unopenable_message(source: Union[int, str]) -> str:
    if isinstance(source, int):
        return (f"Cannot open webcam {source}. Check that it is connected, enabled in "
                f"Windows privacy settings, and not in use by another app (Zoom, Teams, ...).")
    if "://" in source:
        return (f"Cannot open stream '{source}'. Check the URL, that the phone and PC are on the "
                f"same Wi-Fi network, and that the phone camera app is running.")
    return f"Cannot open video file '{source}'. Check that the path is correct and the format is supported."


class FileSource:
    """Sequential video-file reader (no frame dropping)."""

    is_live = False

    def __init__(self, path: str):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(_unopenable_message(self.path))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.ended = False
        self.lost = False

    def read(self, timeout: float = 1.0):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.ended = True
            return None
        return frame

    def release(self) -> None:
        self._cap.release()


class LiveCamera:
    """Webcam or network stream, read continuously in a background thread."""

    is_live = True
    fps = 0.0

    def __init__(self, source: Union[int, str], width: int = 0, height: int = 0,
                 reconnect_attempts: int = 5, reconnect_delay_s: float = 2.0):
        self.source = parse_source(source)
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame = None
        self._frame_id = 0
        self._last_read_id = 0
        self.lost = False
        self.ended = False
        self.last_error = ""

    # ------------------------------------------------------------------
    def _open_capture(self):
        try:
            if isinstance(self.source, int) and os.name == "nt":
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # faster start on Windows
            else:
                cap = cv2.VideoCapture(self.source)
        except Exception as exc:  # noqa: BLE001 - OpenCV can raise odd errors
            self.last_error = str(exc)
            return None
        if not cap.isOpened():
            cap.release()
            return None
        if isinstance(self.source, int):
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends; harmless
        return cap

    def open(self) -> None:
        cap = self._open_capture()
        if cap is None:
            raise CameraError(_unopenable_message(self.source))
        self._thread = threading.Thread(target=self._run, args=(cap,), daemon=True, name="camera-reader")
        self._thread.start()

    def _reconnect(self):
        for attempt in range(1, self.reconnect_attempts + 1):
            log.warning("Stream interrupted; reconnect attempt %d/%d", attempt, self.reconnect_attempts)
            if self._stop.wait(self.reconnect_delay_s):
                return None
            cap = self._open_capture()
            if cap is not None:
                return cap
        return None

    def _run(self, cap) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                ok, frame = cap.read()
            except Exception as exc:  # noqa: BLE001
                ok, frame = False, None
                self.last_error = str(exc)
            if ok and frame is not None:
                failures = 0
                with self._cond:
                    self._frame = frame
                    self._frame_id += 1
                    self._cond.notify_all()
                continue
            failures += 1
            if failures < 5:
                self._stop.wait(0.05)
                continue
            cap.release()
            cap = self._reconnect()
            failures = 0
            if cap is None:
                if not self._stop.is_set():
                    self.lost = True
                    self.last_error = self.last_error or "Connection to the camera was lost."
                with self._cond:
                    self._cond.notify_all()
                return
        cap.release()

    # ------------------------------------------------------------------
    def read(self, timeout: float = 1.0):
        """Return the newest unread frame, or None on timeout / lost connection."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame_id != self._last_read_id or self.lost or self._stop.is_set(),
                timeout,
            )
            if self._frame_id != self._last_read_id:
                self._last_read_id = self._frame_id
                return self._frame
        return None

    def release(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def open_source(source: Union[int, str], width: int = 0, height: int = 0,
                reconnect_attempts: int = 5, reconnect_delay_s: float = 2.0):
    """Open a webcam / stream / file and return an object with read() and release()."""
    parsed = parse_source(source)
    if is_video_file(parsed):
        return FileSource(str(parsed))
    cam = LiveCamera(parsed, width, height, reconnect_attempts, reconnect_delay_s)
    cam.open()
    return cam
