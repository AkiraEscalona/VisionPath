import unittest

import cv2
import numpy as np

import context  # noqa: F401
from visionpath.edge_obstacles import EdgeObstacleDetector


class EdgeTests(unittest.TestCase):
    def test_plain_scene_has_no_obstacle(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        self.assertEqual(EdgeObstacleDetector().detect(frame), [])

    def test_textured_block_in_lower_center_is_flagged(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        rng = np.random.RandomState(0)
        frame[260:470, 230:410] = rng.randint(0, 255, (210, 180, 3), dtype=np.uint8)
        found = EdgeObstacleDetector().detect(frame)
        self.assertEqual([d.position for d in found], ["center"])
        self.assertEqual(found[0].source, "edge")
        x1, y1, x2, y2 = found[0].box
        self.assertGreaterEqual(y1, 240)    # only the lower half is analysed

    def test_texture_in_upper_half_is_ignored(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        rng = np.random.RandomState(1)
        frame[20:200, 230:410] = rng.randint(0, 255, (180, 180, 3), dtype=np.uint8)
        self.assertEqual(EdgeObstacleDetector().detect(frame), [])

    def test_grayscale_input_supported(self):
        gray = np.zeros((100, 100), np.uint8)
        self.assertEqual(EdgeObstacleDetector().detect(gray), [])


if __name__ == "__main__":
    unittest.main()
