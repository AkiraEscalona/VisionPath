import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from visionpath.config import Config, ConfigError, SAMPLE_CONFIG_PATH


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = Config()
        cfg.validate()
        self.assertEqual(cfg.camera_source, "0")
        self.assertLess(cfg.left_boundary, cfg.right_boundary)

    def test_missing_file_gives_defaults(self):
        self.assertEqual(Config.load(Path("does_not_exist.json")).confidence_threshold, 0.5)

    def test_invalid_values_are_clamped(self):
        cfg = Config.from_dict({"confidence_threshold": 5, "process_every_n_frames": 0,
                                "left_boundary": 0.9, "right_boundary": 0.1})
        self.assertEqual(cfg.confidence_threshold, 0.99)
        self.assertEqual(cfg.process_every_n_frames, 1)
        self.assertEqual((cfg.left_boundary, cfg.right_boundary), (0.33, 0.67))

    def test_unknown_keys_ignored(self):
        cfg = Config.from_dict({"nonsense": 1, "_comment": "x"})
        self.assertFalse(hasattr(cfg, "nonsense"))

    def test_bad_json_raises_config_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config.load(p)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            cfg = Config(alert_cooldown_s=7.5, camera_source="http://1.2.3.4:8080/video")
            cfg.save(p)
            loaded = Config.load(p)
            self.assertEqual(loaded.alert_cooldown_s, 7.5)
            self.assertEqual(loaded.camera_source, "http://1.2.3.4:8080/video")

    def test_sample_config_matches_defaults(self):
        sample = json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sample, Config().to_dict())


if __name__ == "__main__":
    unittest.main()
