"""Spoken alerts: decide WHAT to say (AlertManager) and say it (Speaker).

Repetition control:
* the same (object, side) is not repeated within `cooldown_s`
* an object that moves closer (e.g. far -> near) is announced again immediately
* no two spoken alerts are closer than `global_interval_s`
* only one alert is spoken per decision; the most urgent (nearest) goes first
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .detector import Detection

log = logging.getLogger(__name__)

FRIENDLY_NAMES = {
    "diningtable": "table", "tvmonitor": "monitor", "pottedplant": "plant",
    "aeroplane": "airplane", "motorbike": "motorcycle", "obstacle": "possible obstacle",
}
POSITION_PHRASES = {"left": "on the left", "center": "ahead", "right": "on the right"}
PROXIMITY_PHRASES = {"near": "near", "medium": "medium distance", "far": "far", "unknown": ""}
PROXIMITY_RANK = {"near": 0, "medium": 1, "unknown": 1, "far": 2}


def build_message(det: Detection) -> str:
    name = FRIENDLY_NAMES.get(det.label, det.label)
    where = POSITION_PHRASES.get(det.position, "ahead")
    if det.distance_m is not None:
        # rounded to the nearest half metre to avoid implying false precision
        half = round(det.distance_m * 2) / 2
        how = f"about {half:g} meters"
    else:
        how = PROXIMITY_PHRASES.get(det.proximity or "", "")
    text = f"{name} {where}" + (f", {how}" if how else "")
    return text[0].upper() + text[1:]


class AlertManager:
    def __init__(self, cooldown_s: float = 4.0, global_interval_s: float = 1.0,
                 announce_far: bool = False, ignore_labels: Optional[Iterable[str]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cooldown_s = cooldown_s
        self.global_interval_s = global_interval_s
        self.announce_far = announce_far
        self.ignore_labels = {s.lower() for s in (ignore_labels or [])}
        self._clock = clock
        self._last: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self._last_any = -1e18
        self.alerts_generated = 0

    def reset(self) -> None:
        self._last.clear()
        self._last_any = -1e18

    @staticmethod
    def _rank(det: Detection) -> int:
        return PROXIMITY_RANK.get(det.proximity or "unknown", 1)

    def select(self, detections: List[Detection]) -> Optional[str]:
        """Return the message to speak now, or None."""
        now = self._clock()
        if now - self._last_any < self.global_interval_s:
            return None

        candidates = [
            d for d in detections
            if d.label.lower() not in self.ignore_labels
            and (self.announce_far or self._rank(d) < 2)
        ]
        candidates.sort(key=lambda d: (self._rank(d), -d.confidence))

        for det in candidates:
            key = (det.label, det.position)
            rank = self._rank(det)
            previous = self._last.get(key)
            if previous is not None:
                last_time, last_rank = previous
                if now - last_time < self.cooldown_s and rank >= last_rank:
                    continue  # within cooldown and not more urgent than before
            self._last[key] = (now, rank)
            self._last_any = now
            self.alerts_generated += 1
            self._prune(now)
            return build_message(det)
        return None

    def _prune(self, now: float) -> None:
        limit = max(self.cooldown_s * 10, 30.0)
        stale = [k for k, (t, _) in self._last.items() if now - t > limit]
        for k in stale:
            del self._last[k]


class Speaker:
    """Offline text-to-speech in its own thread so the video loop never blocks.

    Uses pyttsx3 (Windows SAPI5). Output goes to the Windows default audio
    device, so connecting Bluetooth headphones and selecting them as the
    default output device is enough. If speech is unavailable the application
    keeps running and alerts are only shown on screen.
    """

    def __init__(self, rate: int = 170, volume: float = 1.0,
                 on_status: Optional[Callable[[str], None]] = None):
        self.rate = rate
        self.volume = volume
        self.on_status = on_status
        self.available = False
        self.error = ""
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="speaker")
        self._thread.start()
        self._ready.wait(timeout=8.0)

    def _report(self, text: str) -> None:
        log.warning(text)
        if self.on_status:
            try:
                self.on_status(text)
            except Exception:  # noqa: BLE001
                pass

    def _run(self) -> None:
        try:
            import pyttsx3  # imported here: the engine must live in this thread
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as exc:  # noqa: BLE001 - missing package, no audio device, COM errors
            self.available = False
            self.error = f"Speech unavailable ({exc}). Alerts will only be shown on screen."
            self._report(self.error)
            self._ready.set()
            return
        self.available = True
        self._ready.set()

        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                self.available = False
                self.error = f"Audio output failed ({exc}). Alerts will only be shown on screen."
                self._report(self.error)
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass

    def say(self, text: str) -> bool:
        """Queue a message; a still-waiting older message is replaced (never piles up)."""
        if not self.available:
            return False
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                return False
        return True

    def stop(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
