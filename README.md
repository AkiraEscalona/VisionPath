# VisionPath

OpenCV-based real-time computer vision assistance prototype for blind and visually impaired users.
Python + OpenCV only: local MobileNet-SSD detection through `cv2.dnn`, spoken alerts, left/center/right
position, approximate distance, performance logging to CSV.

> **SAFETY NOTICE.** VisionPath is a student prototype and a *supplementary* aid. It is **not** a
> replacement for a white cane, guide dog, or established mobility techniques, it is **not**
> safety-certified, and it has **not** been validated for independent navigation. It can miss obstacles,
> raise false alarms, and its distances are rough estimates. No video is uploaded or saved; all processing is local.

## 1. Install (Windows, VS Code terminal)

Install Python 3.9+ from python.org (tick **Add python.exe to PATH**). Open the folder in VS Code, open a
terminal (Ctrl+`) and run:

```
setup.bat
```

or manually:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config.sample.json config.json
python scripts\download_model.py
python -m unittest discover -s tests
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first, or use `cmd`.
In VS Code choose the interpreter: Ctrl+Shift+P -> *Python: Select Interpreter* -> `.venv`.

## 2. Detection model

* **MobileNet-SSD (Caffe)**, https://github.com/chuanqi305/MobileNet-SSD, trained on PASCAL VOC.
* Files go in `models/`: `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` (~23 MB).
* Download: `python scripts\download_model.py` (checks the files, then loads the model). Manual steps: `models\README.md`.
* Verify: `python main.py --check-model` should print `Model OK: ...`.
* **Supported classes (20):** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable,
  dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor. It does **not** recognise stairs,
  doors, walls, poles, curbs, holes or glass.

## 3. Run

```
run.bat
```
or `python main.py`. Choose a camera source, press **Start**, press **Stop** to finish. On Stop the session's
FPS / latency / peak RAM are appended to `results/results.csv`. Settings are saved to `config.json` on exit.

Controls: camera source, Start/Stop, spoken alerts on/off, **alert cooldown slider** (alert frequency),
detection confidence, experimental Canny/contour "possible obstacle" mode, distance calibration, reference
distance check. Live FPS, latency and RAM are shown on the right.

## 4. Smartphone camera

PC and phone must be on the **same Wi-Fi network**. Start a camera-server app on the phone (examples:
*IP Webcam* on Android, *DroidCam*; apps differ, so use the exact URL the app displays), then in VisionPath choose
**Phone / IP camera URL** and paste it, e.g.

```
http://192.168.1.20:8080/video        (MJPEG/HTTP style)
rtsp://192.168.1.20:8554/live         (RTSP style)
```

Tips: test the URL in a browser or VLC first; lower the phone's resolution (640x480 to 1280x720) for lower
latency; disable VPNs; some routers isolate Wi-Fi clients ("AP isolation") and block this. If the stream drops, VisionPath
retries (`reconnect_attempts` in config) and then reports the loss.

## 5. Test detection and audio

1. `python main.py --check-model` (model loads).
2. Start with Webcam 0, sit in view: expect a box labelled `person` with side and proximity.
3. Hold a bottle or sit by a chair for other classes. Move left/right to see LEFT/CENTER/RIGHT change.
4. Audio: connect Bluetooth headphones in Windows and set them as the **default output device**; alerts such as
   "Person ahead, near" are spoken. Check the volume first. If speech is unavailable the app keeps running and shows alerts on screen.
5. Repeat-prevention: stay still; the same alert is not repeated until the cooldown passes. Moving closer re-announces immediately.

## 6. Distance estimate (approximate)

Pinhole model: `distance = real_height x focal_length / box_height_px`, using assumed average heights
(`known_heights_m` in config, e.g. person 1.7 m). Limitations: real objects differ in size; partly visible objects
(box touching the image edge) get only a *near/medium/far* category; classes without a known height use categories;
camera tilt and lens distortion are ignored. Default focal length comes from an assumed 50 degree vertical field of view.

* **Calibrate:** start the camera, stand a measured distance away (fully visible), click *Calibrate*, enter `person` and the distance.
* **Compare with measured distances:** stand at a tape-measured distance, click *Record reference distance check*,
  enter the real distance. Results go to `results/distance_checks.csv` (MAE is shown). Record several distances.

## 7. Measuring performance for the report

Use the **same video, computer and settings** for every configuration. Record a short test video (do not commit it; `*.mp4` is git-ignored) and run:

```
python -m visionpath.benchmark --video test_videos\hallway.mp4 --name baseline --annotations my_annotations.csv
python -m visionpath.benchmark --video test_videos\hallway.mp4 --name skip2 --set process_every_n_frames=2 --annotations my_annotations.csv
python -m visionpath.benchmark --video test_videos\hallway.mp4 --name edge_on --set edge_obstacle_enabled=true --annotations my_annotations.csv
```

Each run appends a row to `results/results.csv` and writes `results/detections_<name>_<time>.csv`.

| Column | Meaning |
|---|---|
| `avg_latency_ms` | T_total / N_frames (pipeline time per processed frame; excludes camera/network delay, drawing, speech) |
| `avg_fps` | N_frames / T_total (processing throughput, as in the design document) |
| `wall_fps` | frames / elapsed wall-clock time |
| `peak_ram_mb` | maximum resident memory of the Python process during the run (psutil) |
| `tp`, `fp`, `fn` | true positives, false positives, missed detections vs. your annotations |
| `detection_rate_pct` | TP / (TP + FN) x 100. This is **recall, not overall accuracy** (it ignores false positives) |
| `precision_pct` | TP / (TP + FP) x 100 |
| `distance_mae_m` | mean absolute error of distance estimates where `reference_distance_m` was annotated |

The first 5 frames are excluded as warm-up (`--warmup`). `--with-speech` also speaks the alerts (RAM then includes audio).

**Annotation file** (see `test_data\annotations_EXAMPLE.csv`; the example contains made-up values for format only): columns `frame,label,position,reference_distance_m`.
Pick frames, note the visible supported objects and measure distances with a tape. Use `none` for frames with no supported object.
Annotate *every* supported object visible in an annotated frame, otherwise correct detections of un-annotated objects count as false positives.
Evaluate an existing log again with `python -m visionpath.evaluation --log results\detections_X.csv --annotations my_annotations.csv`.

## 8. Project layout

```
main.py               entry point (GUI, --check-model)
setup.bat / run.bat   Windows helpers
config.sample.json    sample settings (copied to config.json)
visionpath/
  config.py           settings, validation, load/save
  camera.py           webcam / IP / RTSP / file input (threaded, reconnect)
  detector.py         MobileNet-SSD via cv2.dnn, left/center/right
  edge_obstacles.py   experimental Canny + contour heuristic
  distance.py         approximate distance, calibration, MAE
  audio.py            alert rules (cooldowns) + text-to-speech thread
  performance.py      latency / FPS / RAM monitor, results CSV
  evaluation.py       detection rate, FP/FN, distance MAE from annotations
  pipeline.py         one-frame pipeline + drawing
  session.py          processing thread used by the GUI
  benchmark.py        headless repeatable benchmark
  gui.py              Tkinter interface
scripts/download_model.py
tests/                automated tests (unittest)
models/  results/  test_data/
```

## 9. Tests

`python -m unittest discover -s tests` (no extra packages needed; `pytest` also works if installed).
The tests cover configuration, position logic, SSD output parsing, distance maths, alert cooldowns, speech failure
handling, the metric formulas, CSV output, video reading, and a full benchmark/session run using a stub detector.

## 10. Known limitations

* Detects only the 20 VOC classes; no stairs, curbs, doors, holes, walls. Poor lighting, motion blur and crowds reduce accuracy.
* Monocular distance is approximate; the near/medium/far categories are guides, not measurements.
* Canny/contour mode cannot recognise objects; it only flags edge-rich floor regions and gives false alarms on textured floors.
* Wi-Fi latency and camera quality affect warnings; the app measures its own processing time, not network delay.
* The GUI needs a display (tkinter); speech needs the Windows speech engine (pyttsx3/SAPI5).
