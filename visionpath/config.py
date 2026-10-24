"""Configuration for VisionPath: defaults, loading, validation, saving."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.sample.json"


class ConfigError(Exception):
    """Raised when the configuration file cannot be read or is invalid."""


# Rough real-world heights (metres) used by the pinhole distance estimate.
# These are ASSUMPTIONS (average sizes). Real objects vary, so the resulting
# distance is only approximate. Classes not listed use proximity categories.
DEFAULT_KNOWN_HEIGHTS_M: Dict[str, float] = {
    "person": 1.70,
    "chair": 0.90,
    "bottle": 0.25,
    "bicycle": 1.00,
    "car": 1.50,
    "bus": 3.00,
    "motorbike": 1.10,
    "sofa": 0.85,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Config:
    # --- input ---
    camera_source: str = "0"          # "0" = webcam, URL = phone stream, or a video file path
    capture_width: int = 640          # requested webcam width (0 = camera default)
    capture_height: int = 480
    reconnect_attempts: int = 5       # for dropped streams
    reconnect_delay_s: float = 2.0

    # --- detection ---
    model_prototxt: str = "models/MobileNetSSD_deploy.prototxt"
    model_weights: str = "models/MobileNetSSD_deploy.caffemodel"
    confidence_threshold: float = 0.5
    process_every_n_frames: int = 1   # 1 = every frame; 2 = every second frame, ...

    # --- left / centre / right ---
    left_boundary: float = 0.33       # fraction of frame width
    right_boundary: float = 0.67

    # --- distance estimation ---
    near_distance_m: float = 1.5
    medium_distance_m: float = 3.0
    near_height_fraction: float = 0.6     # fallback: box height / frame height
    medium_height_fraction: float = 0.3
    vertical_fov_deg: float = 50.0        # assumed camera vertical field of view
    focal_length_norm: Optional[float] = None  # set by calibration (focal length / frame height)
    known_heights_m: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_KNOWN_HEIGHTS_M))

    # --- audio ---
    speech_enabled: bool = True
    speech_rate: int = 170
    speech_volume: float = 1.0
    alert_cooldown_s: float = 4.0         # same object+side is not repeated within this time
    alert_global_interval_s: float = 1.0  # minimum gap between any two spoken alerts
    announce_far: bool = False            # far objects are shown but not spoken by default
    ignore_labels: List[str] = field(default_factory=list)

    # --- Canny/contour "possible obstacle" heuristic (experimental, off by default) ---
    edge_obstacle_enabled: bool = False
    edge_canny_low: int = 50
    edge_canny_high: int = 150
    edge_density_threshold: float = 0.12
    edge_min_contour_area_fraction: float = 0.02

    # --- results ---
    results_csv: str = "results/results.csv"

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Clamp values into sensible ranges (logs a warning when it fixes something)."""
        self.camera_source = str(self.camera_source).strip() or "0"
        self.confidence_threshold = _clamp(float(self.confidence_threshold), 0.05, 0.99)
        self.process_every_n_frames = max(1, int(self.process_every_n_frames))
        self.capture_width = max(0, int(self.capture_width))
        self.capture_height = max(0, int(self.capture_height))
        self.reconnect_attempts = max(0, int(self.reconnect_attempts))
        self.reconnect_delay_s = max(0.1, float(self.reconnect_delay_s))

        if not (0.0 < self.left_boundary < self.right_boundary < 1.0):
            log.warning("Invalid left/right boundaries; using 0.33 / 0.67.")
            self.left_boundary, self.right_boundary = 0.33, 0.67
        if not (0.0 < self.near_distance_m < self.medium_distance_m):
            log.warning("Invalid near/medium distances; using 1.5 / 3.0 m.")
            self.near_distance_m, self.medium_distance_m = 1.5, 3.0
        if not (0.0 < self.medium_height_fraction < self.near_height_fraction <= 1.0):
            log.warning("Invalid height fractions; using 0.6 / 0.3.")
            self.near_height_fraction, self.medium_height_fraction = 0.6, 0.3

        self.vertical_fov_deg = _clamp(float(self.vertical_fov_deg), 10.0, 120.0)
        if self.focal_length_norm is not None:
            self.focal_length_norm = float(self.focal_length_norm)
            if self.focal_length_norm <= 0:
                self.focal_length_norm = None

        self.speech_rate = int(_clamp(int(self.speech_rate), 80, 300))
        self.speech_volume = _clamp(float(self.speech_volume), 0.0, 1.0)
        self.alert_cooldown_s = max(0.0, float(self.alert_cooldown_s))
        self.alert_global_interval_s = max(0.0, float(self.alert_global_interval_s))
        self.known_heights_m = {
            str(k): float(v) for k, v in dict(self.known_heights_m).items() if float(v) > 0
        }
        self.ignore_labels = [str(x).lower() for x in self.ignore_labels]
        self.edge_density_threshold = _clamp(float(self.edge_density_threshold), 0.01, 0.9)
        self.edge_min_contour_area_fraction = _clamp(float(self.edge_min_contour_area_fraction), 0.001, 0.5)

    # ------------------------------------------------------------------
    def resolve_path(self, relative_or_absolute: str) -> Path:
        p = Path(relative_or_absolute)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        valid = {f.name for f in fields(cls)}
        clean: Dict[str, Any] = {}
        for key, value in data.items():
            if key in valid:
                clean[key] = value
            elif not key.startswith("_"):
                log.warning("Ignoring unknown config key: %s", key)
        try:
            cfg = cls(**clean)
            cfg.validate()
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid configuration value: {exc}") from exc
        return cfg

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load config; a missing file simply gives the defaults."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path.name} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Cannot read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path.name} must contain a JSON object.")
        return cls.from_dict(data)

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
