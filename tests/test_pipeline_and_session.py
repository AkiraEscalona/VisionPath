"""End-to-end tests with a stub detector (no model files needed)."""
import csv
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import context  # noqa: F401
from visionpath.benchmark import apply_overrides, run_benchmark
from visionpath.camera import CameraError, FileSource, is_video_file, open_source, parse_source
from visionpath.config import Config
from visionpath.detector import Detection
from visionpath.performance import RESULT_COLUMNS
from visionpath.pipeline import VisionPipeline
from visionpath.session import DetectionSession


class StubDetector:
    """Pretends a person is at the left of every frame."""
    conf_threshold = 0.5

    def detect(self, frame):
        h, w = frame.shape[:2]
        return [Detection("person", 0.9, (10, int(h * 0.1), int(w * 0.2), int(h * 0.9)))]


def make_video(path, frames=20, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV build cannot write MJPG video")
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 10) % 255, np.uint8))
    writer.release()


class PipelineTests(unittest.TestCase):
    def test_process_sets_position_and_distance(self):
        pipe = VisionPipeline(Config(), detector=StubDetector())
        frame = np.zeros((240, 320, 3), np.uint8)
        dets = pipe.process(frame)
        self.assertEqual(dets[0].position, "left")
        self.assertIsNotNone(dets[0].distance_m)
        self.assertIn(dets[0].proximity, ("near", "medium", "far"))
        out = pipe.draw(frame, dets)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)       # something was drawn
        self.assertEqual(int(frame.sum()), 0)       # original untouched


class CameraTests(unittest.TestCase):
    def test_parse_source(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source(" 2 "), 2)
        self.assertEqual(parse_source("http://1.2.3.4:8080/video"), "http://1.2.3.4:8080/video")

    def test_is_video_file(self):
        self.assertTrue(is_video_file("clip.mp4"))
        self.assertFalse(is_video_file("rtsp://1.2.3.4/live.mp4"))
        self.assertFalse(is_video_file(0))

    def test_missing_file_raises_camera_error(self):
        with self.assertRaises(CameraError) as ctx:
            open_source("no_such_video.mp4")
        self.assertIn("Cannot open video file", str(ctx.exception))

    def test_file_source_reads_all_frames(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.avi"
            make_video(p, frames=7)
            src = FileSource(str(p))
            count = 0
            while src.read() is not None:
                count += 1
            self.assertEqual(count, 7)
            self.assertTrue(src.ended)
            src.release()


class BenchmarkTests(unittest.TestCase):
    def test_apply_overrides(self):
        cfg = apply_overrides(Config(), ["process_every_n_frames=2", "edge_obstacle_enabled=true", "camera_source=1"])
        self.assertEqual(cfg.process_every_n_frames, 2)
        self.assertTrue(cfg.edge_obstacle_enabled)
        with self.assertRaises(ValueError):
            apply_overrides(Config(), ["bogus=1"])

    def test_benchmark_writes_csv_with_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "v.avi"
            make_video(video, frames=20)
            ann = Path(d) / "ann.csv"
            ann.write_text("frame,label,position,reference_distance_m\n"
                           "4,person,left,2.0\n8,person,left,2.0\n12,chair,,\n", encoding="utf-8")
            out_csv = Path(d) / "results.csv"
            cfg = Config()
            pipe = VisionPipeline(cfg, detector=StubDetector())
            row = run_benchmark(cfg, str(video), "unit", warmup=2, annotations_path=str(ann),
                                pipeline=pipe, results_csv=out_csv)
            self.assertEqual(row["frames_processed"], 18)          # 20 frames - 2 warm-up
            self.assertEqual((row["tp"], row["fn"]), (2, 1))        # chair never detected
            self.assertEqual(row["fp"], 1)                           # person detected at frame 12 not annotated
            self.assertAlmostEqual(row["detection_rate_pct"], 100 * 2 / 3)
            self.assertEqual(row["distance_samples"], 2)
            with open(out_csv, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(list(rows[0].keys()), RESULT_COLUMNS)
            self.assertEqual(rows[0]["config_name"], "unit")
            Path(row["_detections_log"]).unlink()

    def test_frame_skipping_reduces_processed_frames(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "v.avi"
            make_video(video, frames=20)
            cfg = Config(process_every_n_frames=2)
            row = run_benchmark(cfg, str(video), "skip2", warmup=0,
                                pipeline=VisionPipeline(cfg, detector=StubDetector()),
                                results_csv=Path(d) / "r.csv")
            self.assertEqual(row["frames_processed"], 10)
            Path(row["_detections_log"]).unlink()


class SessionTests(unittest.TestCase):
    def test_session_processes_video_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "v.avi"
            make_video(video, frames=15)
            cfg = Config()
            session = DetectionSession(cfg, str(video), pipeline=VisionPipeline(cfg, detector=StubDetector()),
                                       use_speaker=False, pace_files=False)
            session.start()
            self.assertTrue(session.finished.wait(timeout=10))
            messages = session.pop_messages()
            self.assertTrue(any(level == "alert" for level, _ in messages))      # alert produced
            self.assertTrue(any("Video file finished" in t for _, t in messages))
            summary = session.stop()
            self.assertEqual(summary["frames_processed"], 15)
            self.assertGreaterEqual(summary["alerts_generated"], 1)
            self.assertLess(summary["alerts_generated"], 15)        # cooldown prevents per-frame repeats
            self.assertFalse(session.running)

    def test_session_reports_missing_source(self):
        cfg = Config()
        session = DetectionSession(cfg, "missing.mp4", pipeline=VisionPipeline(cfg, detector=StubDetector()),
                                   use_speaker=False)
        with self.assertRaises(CameraError):
            session.start()

    def test_calibration_through_session(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "v.avi"
            make_video(video, frames=5)
            cfg = Config()
            session = DetectionSession(cfg, str(video), pipeline=VisionPipeline(cfg, detector=StubDetector()),
                                       use_speaker=False, pace_files=False)
            with self.assertRaises(ValueError):
                session.calibrate("person", 2.0)          # nothing detected yet
            session.start()
            session.finished.wait(timeout=10)
            f = session.calibrate("person", 2.0)
            self.assertGreater(f, 0)
            est = session.reference_check()
            self.assertIsNotNone(est.distance_m)
            session.stop()


if __name__ == "__main__":
    unittest.main()
