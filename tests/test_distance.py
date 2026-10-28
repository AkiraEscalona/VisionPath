import math
import unittest

import context  # noqa: F401
from visionpath.detector import Detection
from visionpath.distance import (DistanceEstimator, default_focal_length_norm,
                                 focal_length_from_reference, mean_absolute_error)

HEIGHTS = {"person": 1.7}


def person(y1, y2):
    return Detection("person", 0.9, (100, y1, 200, y2))


class DistanceTests(unittest.TestCase):
    def test_default_focal_from_fov(self):
        self.assertAlmostEqual(default_focal_length_norm(90.0), 0.5, places=6)

    def test_pinhole_estimate(self):
        est = DistanceEstimator(focal_length_norm=1.0, known_heights_m=HEIGHTS)
        # 1.7 m person filling 0.5 of a 480 px frame at focal 1.0 -> 1.7 * 1.0 / 0.5 = 3.4 m
        r = est.estimate(person(100, 340), 480)
        self.assertAlmostEqual(r.distance_m, 3.4, places=2)
        self.assertEqual(r.method, "pinhole-height")
        self.assertEqual(r.proximity, "far")

    def test_calibration_roundtrip(self):
        f = focal_length_from_reference(2.0, 240, 480, 1.7)
        est = DistanceEstimator(focal_length_norm=f, known_heights_m=HEIGHTS)
        self.assertAlmostEqual(est.estimate(person(100, 340), 480).distance_m, 2.0, places=2)

    def test_estimator_calibrate_updates_focal(self):
        est = DistanceEstimator(known_heights_m=HEIGHTS)
        f = est.calibrate("person", 2.0, (100, 100, 200, 340), 480)
        self.assertAlmostEqual(est.focal_length_norm, f)
        self.assertTrue(est.calibrated)
        self.assertAlmostEqual(est.estimate(person(100, 340), 480).distance_m, 2.0, places=2)

    def test_calibrate_rejects_unknown_class_and_cut_off_box(self):
        est = DistanceEstimator(known_heights_m=HEIGHTS)
        with self.assertRaises(ValueError):
            est.calibrate("dog", 2.0, (0, 100, 50, 300), 480)
        with self.assertRaises(ValueError):
            est.calibrate("person", 2.0, (0, 0, 50, 300), 480)

    def test_truncated_box_falls_back_to_proximity(self):
        est = DistanceEstimator(known_heights_m=HEIGHTS)
        r = est.estimate(person(0, 470), 480)    # touches top and bottom edge
        self.assertIsNone(r.distance_m)
        self.assertEqual(r.proximity, "near")
        self.assertEqual(r.method, "box-height-fraction")

    def test_unknown_class_uses_fraction(self):
        est = DistanceEstimator(known_heights_m=HEIGHTS)
        d = Detection("dog", 0.9, (10, 100, 60, 250))     # 150/480 = 0.31 -> medium
        r = est.estimate(d, 480)
        self.assertIsNone(r.distance_m)
        self.assertEqual(r.proximity, "medium")

    def test_proximity_thresholds(self):
        est = DistanceEstimator(near_m=1.5, medium_m=3.0)
        self.assertEqual(est.proximity_from_distance(1.0), "near")
        self.assertEqual(est.proximity_from_distance(2.0), "medium")
        self.assertEqual(est.proximity_from_distance(5.0), "far")

    def test_apply_sets_fields(self):
        est = DistanceEstimator(focal_length_norm=1.0, known_heights_m=HEIGHTS)
        d = est.apply(person(100, 340), 480)
        self.assertIsNotNone(d.distance_m)
        self.assertEqual(d.proximity, "far")

    def test_mae(self):
        self.assertAlmostEqual(mean_absolute_error([(2.0, 2.5), (3.0, 2.0), (1.0, 1.0)]), 0.5)
        self.assertIsNone(mean_absolute_error([]))
        self.assertTrue(math.isclose(mean_absolute_error([(1, 3)]), 2.0))


if __name__ == "__main__":
    unittest.main()
