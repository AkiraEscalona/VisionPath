import csv
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from visionpath.performance import (RESULT_COLUMNS, PerformanceMonitor, ResultsFileError,
                                    append_result_row)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class PerformanceTests(unittest.TestCase):
    def test_latency_fps_and_peak_ram(self):
        clock = Clock()
        rams = iter([100.0, 120.0, 150.0, 130.0])    # initial sample + 3 frames
        mon = PerformanceMonitor(time_fn=clock, ram_fn=lambda: next(rams))
        for latency in (0.05, 0.10, 0.15):
            clock.t += latency
            mon.record_frame(latency)
        s = mon.summary()
        self.assertEqual(s["frames_processed"], 3)
        self.assertAlmostEqual(s["avg_latency_ms"], 100.0)          # T_total / N
        self.assertAlmostEqual(s["avg_fps"], 3 / 0.30)              # N / T_total
        self.assertAlmostEqual(s["max_latency_ms"], 150.0)
        self.assertEqual(s["peak_ram_mb"], 150.0)                   # max, not last

    def test_warmup_frames_excluded(self):
        clock = Clock()
        mon = PerformanceMonitor(warmup_frames=1, time_fn=clock, ram_fn=lambda: 10.0)
        for latency in (1.0, 0.1, 0.1):
            clock.t += latency
            mon.record_frame(latency)
        s = mon.summary()
        self.assertEqual(s["frames_processed"], 2)
        self.assertAlmostEqual(s["avg_latency_ms"], 100.0)
        self.assertEqual(s["warmup_frames_excluded"], 1)

    def test_empty_summary(self):
        s = PerformanceMonitor(ram_fn=lambda: None).summary()
        self.assertEqual(s["frames_processed"], 0)
        self.assertIsNone(s["avg_latency_ms"])
        self.assertIsNone(s["peak_ram_mb"])

    def test_csv_header_written_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "r.csv"
            append_result_row(p, {"config_name": "a", "avg_fps": 12.3456})
            append_result_row(p, {"config_name": "b"})
            with open(p, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0], RESULT_COLUMNS)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1][RESULT_COLUMNS.index("avg_fps")], "12.346")

    def test_csv_with_other_header_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ResultsFileError):
                append_result_row(p, {"config_name": "x"})


if __name__ == "__main__":
    unittest.main()
