"""Repeatable headless benchmark on a video file -> results CSV.

Example:
  python -m visionpath.benchmark --video test_videos/hallway.mp4 --name baseline
  python -m visionpath.benchmark --video test_videos/hallway.mp4 --name skip2 --set process_every_n_frames=2
  python -m visionpath.benchmark --video test_videos/hallway.mp4 --name baseline --annotations test_data/my_annotations.csv

Use the SAME video, computer and settings when comparing configurations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .audio import AlertManager, Speaker
from .camera import CameraError, FileSource
from .config import Config, ConfigError, DEFAULT_CONFIG_PATH, PROJECT_ROOT
from .detector import ModelLoadError, ModelNotFoundError
from .evaluation import DetectionLogger, evaluate, load_annotations, load_detection_log
from .performance import PerformanceMonitor, ResultsFileError, append_result_row
from .pipeline import VisionPipeline


def apply_overrides(config: Config, overrides: List[str]) -> Config:
    data = config.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        if key not in data:
            raise ValueError(f"Unknown config key '{key}'")
        try:
            data[key] = json.loads(raw)
        except json.JSONDecodeError:
            data[key] = raw
    return Config.from_dict(data)


def run_benchmark(config: Config, video: str, name: str, max_frames: int = 0, warmup: int = 5,
                  annotations_path: Optional[str] = None, tolerance: int = 0,
                  with_speech: bool = False, pipeline: Optional[VisionPipeline] = None,
                  results_csv: Optional[Path] = None, notes: str = "") -> Dict[str, object]:
    """Process a video as fast as possible and return (and save) the measurements."""
    pipeline = pipeline or VisionPipeline(config)
    source = FileSource(video)
    fps_video = source.fps if source.fps > 0 else 30.0
    n = max(1, config.process_every_n_frames)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "results" / f"detections_{name}_{stamp}.csv"
    logger = DetectionLogger(log_path)

    sim_time = [0.0]  # alert cooldowns follow VIDEO time, so results do not depend on CPU speed
    alerts = AlertManager(config.alert_cooldown_s, config.alert_global_interval_s,
                          config.announce_far, config.ignore_labels, clock=lambda: sim_time[0])
    speaker = None
    if with_speech:
        speaker = Speaker(config.speech_rate, config.speech_volume)
        speaker.start()

    monitor = PerformanceMonitor(warmup_frames=warmup)
    detections_total = 0
    index = -1
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            index += 1
            if max_frames and index >= max_frames:
                break
            if index % n != 0:
                continue
            sim_time[0] = index / fps_video
            t0 = time.perf_counter()
            detections = pipeline.process(frame)
            message = alerts.select(detections)      # decision logic is part of the pipeline cost
            monitor.record_frame(time.perf_counter() - t0)
            detections_total += len(detections)
            logger.log(index, detections)
            if message and speaker is not None:
                speaker.say(message)
    finally:
        source.release()
        logger.close()
        if speaker is not None:
            speaker.stop()

    summary = monitor.summary()
    row: Dict[str, object] = dict(summary)
    row.update({"config_name": name, "source": str(video), "detections_total": detections_total,
                "alerts_generated": alerts.alerts_generated, "notes": notes})

    if annotations_path:
        res = evaluate(load_annotations(Path(annotations_path)), load_detection_log(log_path), tolerance)
        row.update({"tp": res.tp, "fp": res.fp, "fn": res.fn,
                    "detection_rate_pct": res.detection_rate_pct, "precision_pct": res.precision_pct,
                    "distance_samples": res.distance_samples, "distance_mae_m": res.distance_mae_m})

    csv_path = Path(results_csv) if results_csv else config.resolve_path(config.results_csv)
    append_result_row(csv_path, row)
    row["_detections_log"] = str(log_path)
    row["_results_csv"] = str(csv_path)
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="VisionPath headless benchmark")
    p.add_argument("--video", required=True, help="test video file")
    p.add_argument("--name", required=True, help="label for this configuration, e.g. baseline")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. process_every_n_frames=2")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5, help="frames excluded from statistics")
    p.add_argument("--annotations", help="annotation CSV for detection rate / MAE")
    p.add_argument("--tolerance", type=int, default=0)
    p.add_argument("--with-speech", action="store_true", help="also speak alerts (slower; includes audio RAM)")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)

    try:
        config = apply_overrides(Config.load(Path(args.config)), args.set)
        row = run_benchmark(config, args.video, args.name, args.max_frames, args.warmup,
                            args.annotations, args.tolerance, args.with_speech, notes=args.notes)
    except (ConfigError, ValueError, ModelNotFoundError, ModelLoadError, CameraError, ResultsFileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    def show(label, key, fmt="{:.2f}"):
        v = row.get(key)
        print(f"{label:<22}: {'n/a' if v is None else (fmt.format(v) if isinstance(v, float) else v)}")

    print()
    show("Frames processed", "frames_processed")
    show("Avg latency (ms)", "avg_latency_ms")
    show("Avg FPS", "avg_fps")
    show("Peak RAM (MB)", "peak_ram_mb", "{:.1f}")
    show("Alerts generated", "alerts_generated")
    if args.annotations:
        show("Detection rate (%)", "detection_rate_pct", "{:.1f}")
        show("TP / FP / FN", "tp")
        print(f"{'':<22}  FP={row.get('fp')}  FN={row.get('fn')}")
        show("Distance MAE (m)", "distance_mae_m", "{:.3f}")
    print(f"\nSaved: {row['_results_csv']}\nDetections log: {row['_detections_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
