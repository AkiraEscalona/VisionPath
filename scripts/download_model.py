"""Download the MobileNet-SSD (Caffe) files into models/ and verify that they load.

Source: https://github.com/chuanqi305/MobileNet-SSD  (Caffe MobileNet-SSD, trained on PASCAL VOC)
The script tries each candidate URL in turn and checks the result, so a wrong or
moved link produces a clear message instead of a broken model. If it fails, follow
the manual steps printed at the end (also in models/README.md).
"""
import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
FILES = {
    "MobileNetSSD_deploy.prototxt": [BASE + "deploy.prototxt", BASE + "MobileNetSSD_deploy.prototxt"],
    "MobileNetSSD_deploy.caffemodel": [BASE + "MobileNetSSD_deploy.caffemodel"],
}
MIN_BYTES = {"MobileNetSSD_deploy.prototxt": 2_000, "MobileNetSSD_deploy.caffemodel": 5_000_000}

MANUAL = """
MANUAL DOWNLOAD
  1. Open https://github.com/chuanqi305/MobileNet-SSD in your browser.
  2. Download the deploy prototxt (deploy.prototxt) and MobileNetSSD_deploy.caffemodel.
  3. Save them in the models/ folder as:
       models/MobileNetSSD_deploy.prototxt
       models/MobileNetSSD_deploy.caffemodel
  4. Check with:  python main.py --check-model
"""


def looks_valid(name: str, data: bytes) -> bool:
    if len(data) < MIN_BYTES[name] or data[:1] == b"<":   # '<' = HTML error page
        return False
    if name.endswith(".prototxt"):
        return b"layer" in data
    return True


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VisionPath-setup"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunks, got = [], 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    args = parser.parse_args()
    models = ROOT / "models"
    models.mkdir(exist_ok=True)

    ok = True
    for name, urls in FILES.items():
        target = models / name
        if target.exists() and not args.force:
            if looks_valid(name, target.read_bytes()):
                print(f"[skip] {name} already present")
                continue
            print(f"[warn] {name} exists but looks invalid; downloading again")
        done = False
        for url in urls:
            print(f"[get ] {name}\n    from {url}")
            try:
                data = download(url)
            except (urllib.error.URLError, OSError) as exc:
                print(f"    failed: {exc}")
                continue
            if not looks_valid(name, data):
                print("    downloaded file does not look valid; trying next source")
                continue
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)
            print(f"    saved {target} ({len(data) / 1e6:.1f} MB)")
            done = True
            break
        ok = ok and done

    if not ok:
        print("\nCould not download every file automatically." + MANUAL)
        return 1

    from visionpath.detector import ModelLoadError, ModelNotFoundError, ObjectDetector
    try:
        detector = ObjectDetector(models / "MobileNetSSD_deploy.prototxt",
                                  models / "MobileNetSSD_deploy.caffemodel")
        print(detector.verify())
    except (ModelNotFoundError, ModelLoadError) as exc:
        print(f"\nModel files downloaded but verification failed:\n{exc}" + MANUAL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
