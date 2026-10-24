import tempfile
import unittest
from pathlib import Path

import numpy as np

import context  # noqa: F401
from visionpath.detector import (ModelLoadError, ModelNotFoundError, ObjectDetector,
                                 VOC_CLASSES, parse_detections, position_for_box)


def row(class_id, conf, x1, y1, x2, y2):
    return [0, class_id, conf, x1, y1, x2, y2]


class DetectorTests(unittest.TestCase):
    def test_position_left_center_right(self):
        self.assertEqual(position_for_box((0, 0, 100, 50), 640), "left")
        self.assertEqual(position_for_box((270, 0, 370, 50), 640), "center")
        self.assertEqual(position_for_box((540, 0, 640, 50), 640), "right")

    def test_position_boundaries_configurable(self):
        self.assertEqual(position_for_box((300, 0, 340, 10), 640, 0.6, 0.9), "left")

    def test_parse_filters_by_confidence_and_background(self):
        out = np.array([[[
            row(15, 0.9, 0.1, 0.1, 0.5, 0.9),   # person, kept
            row(9, 0.3, 0.1, 0.1, 0.5, 0.9),    # chair, below threshold
            row(0, 0.99, 0.1, 0.1, 0.5, 0.9),   # background class, ignored
            row(99, 0.99, 0.1, 0.1, 0.5, 0.9),  # invalid class id, ignored
        ]]], dtype=np.float32)
        dets = parse_detections(out, 640, 480, 0.5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "person")
        self.assertEqual(dets[0].box, (64, 48, 320, 432))

    def test_parse_clamps_and_drops_degenerate_boxes(self):
        out = np.array([[[
            row(15, 0.9, -0.2, -0.1, 1.3, 1.2),   # clamped to the frame
            row(15, 0.9, 0.5, 0.5, 0.5, 0.9),     # zero width, dropped
        ]]], dtype=np.float32)
        dets = parse_detections(out, 100, 100, 0.5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].box, (0, 0, 99, 99))

    def test_class_list_has_20_objects(self):
        self.assertEqual(len(VOC_CLASSES) - 1, 20)
        self.assertIn("person", VOC_CLASSES)

    def test_missing_model_gives_helpful_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ModelNotFoundError) as ctx:
                ObjectDetector(Path(d) / "a.prototxt", Path(d) / "b.caffemodel")
            self.assertIn("download_model.py", str(ctx.exception))

    def test_truncated_weights_detected(self):
        with tempfile.TemporaryDirectory() as d:
            proto, weights = Path(d) / "a.prototxt", Path(d) / "b.caffemodel"
            proto.write_text("name: x", encoding="utf-8")
            weights.write_bytes(b"<html>not a model</html>")
            with self.assertRaises(ModelLoadError):
                ObjectDetector(proto, weights)


if __name__ == "__main__":
    unittest.main()
