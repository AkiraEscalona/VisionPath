"""VisionPath entry point.

  python main.py                 start the desktop application
  python main.py --check-model   verify that the detection model loads, then exit
Benchmarking:  python -m visionpath.benchmark --help
"""
import argparse
import logging
import sys
from pathlib import Path

from visionpath.config import Config, ConfigError, DEFAULT_CONFIG_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionPath - experimental vision assistance prototype")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.json")
    parser.add_argument("--check-model", action="store_true", help="verify the model files and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    try:
        config = Config.load(Path(args.config))
    except ConfigError as exc:
        print(f"Config problem: {exc}\nUsing default settings instead.", file=sys.stderr)
        config = Config()

    if args.check_model:
        from visionpath.detector import ModelLoadError, ModelNotFoundError, ObjectDetector
        try:
            detector = ObjectDetector(config.resolve_path(config.model_prototxt),
                                      config.resolve_path(config.model_weights))
            print(detector.verify())
            return 0
        except (ModelNotFoundError, ModelLoadError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    try:
        from visionpath.gui import run_gui
    except ImportError as exc:
        print(f"Cannot start the interface ({exc}). On Windows, tkinter ships with the python.org "
              f"installer; re-run the installer and make sure 'tcl/tk' is ticked.", file=sys.stderr)
        return 1
    run_gui(config, Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
