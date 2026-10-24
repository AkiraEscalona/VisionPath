import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from visionpath.detector import Detection
from visionpath.evaluation import (Annotation, DetectionLogger, LoggedDetection, append_distance_check,
                                   distance_check_summary, evaluate, load_annotations, load_detection_log)


def L(frame, label, pos="center", dist=None, conf=0.9):
    return LoggedDetection(frame, label, conf, pos, dist)


class EvaluationTests(unittest.TestCase):
    def test_tp_fp_fn_counts(self):
        ann = [Annotation(10, "person"), Annotation(10, "chair"), Annotation(20, "none")]
        det = [L(10, "person"), L(10, "bottle"), L(20, "person")]
        r = evaluate(ann, det)
        self.assertEqual((r.tp, r.fp, r.fn), (1, 2, 1))          # chair missed; bottle + frame-20 person false
        self.assertAlmostEqual(r.detection_rate_pct, 50.0)
        self.assertAlmostEqual(r.precision_pct, 100 / 3)

    def test_position_must_match_when_annotated(self):
        r = evaluate([Annotation(5, "person", "left")], [L(5, "person", "right")])
        self.assertEqual((r.tp, r.fp, r.fn), (0, 1, 1))

    def test_frames_without_annotation_are_ignored(self):
        r = evaluate([Annotation(1, "person")], [L(1, "person"), L(2, "car")])
        self.assertEqual((r.tp, r.fp, r.fn), (1, 0, 0))
        self.assertEqual(r.detection_rate_pct, 100.0)

    def test_tolerance_window(self):
        ann = [Annotation(10, "person")]
        self.assertEqual(evaluate(ann, [L(11, "person")], 0).fn, 1)
        self.assertEqual(evaluate(ann, [L(11, "person")], 1).tp, 1)

    def test_distance_mae(self):
        ann = [Annotation(1, "person", reference_distance_m=2.0), Annotation(2, "person", reference_distance_m=3.0)]
        det = [L(1, "person", dist=2.5), L(2, "person", dist=2.0)]
        r = evaluate(ann, det)
        self.assertEqual(r.distance_samples, 2)
        self.assertAlmostEqual(r.distance_mae_m, 0.75)

    def test_empty_inputs(self):
        r = evaluate([], [])
        self.assertIsNone(r.detection_rate_pct)
        self.assertIsNone(r.distance_mae_m)

    def test_load_and_log_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            ann_path = Path(d) / "a.csv"
            ann_path.write_text("frame,label,position,reference_distance_m\n3,Person,left,2.0\n4,none,,\n",
                                encoding="utf-8")
            anns = load_annotations(ann_path)
            self.assertEqual(anns[0].label, "person")
            self.assertEqual(anns[0].reference_distance_m, 2.0)
            self.assertEqual(anns[1].label, "none")

            log_path = Path(d) / "log.csv"
            lg = DetectionLogger(log_path)
            lg.log(3, [Detection("person", 0.8, (0, 0, 5, 5), position="left", proximity="near", distance_m=2.2)])
            lg.close()
            loaded = load_detection_log(log_path)
            self.assertEqual((loaded[0].frame, loaded[0].label, loaded[0].distance_m), (3, "person", 2.2))

    def test_bad_annotation_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.csv"
            p.write_text("x,y\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_annotations(p)

    def test_distance_check_log(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checks.csv"
            append_distance_check(p, "person", 2.0, 2.4)
            append_distance_check(p, "person", 3.0, 2.8)
            n, mae = distance_check_summary(p)
            self.assertEqual(n, 2)
            self.assertAlmostEqual(mae, 0.3)


if __name__ == "__main__":
    unittest.main()
