"""Tkinter desktop interface (Start/Stop, camera selection, live preview, settings)."""
from __future__ import annotations

import base64
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from pathlib import Path
from typing import Optional

import cv2

from .camera import CameraError
from .config import Config, DEFAULT_CONFIG_PATH
from .detector import ModelLoadError, ModelNotFoundError
from .evaluation import append_distance_check, distance_check_summary
from .performance import ResultsFileError, append_result_row
from .session import DetectionSession

log = logging.getLogger(__name__)

SOURCE_CHOICES = [
    "Webcam 0",
    "Webcam 1",
    "Webcam 2",
    "Phone / IP camera URL",
    "Video file",
]
MAX_VIDEO_W, MAX_VIDEO_H = 800, 600

DISCLAIMER = (
    "EXPERIMENTAL PROTOTYPE. A supplementary aid only - not a replacement for a white "
    "cane, guide dog or established mobility techniques, and not validated for "
    "independent navigation. Distances are rough estimates."
)


class VisionPathApp:
    def __init__(
        self,
        root: tk.Tk,
        config: Config,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ):
        self.root = root
        self.config = config
        self.config_path = Path(config_path)
        self.session: Optional[DetectionSession] = None
        self._source_text = ""

        root.title("VisionPath - experimental vision assistance prototype")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_widgets()
        self._load_source_from_config()
        self._poll()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        main = ttk.Frame(root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.video_label = tk.Label(
            main,
            bg="black",
            width=80,
            height=30,
            fg="white",
            text="Press Start to begin",
        )
        self.video_label.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 8),
        )

        panel = ttk.Frame(main)
        panel.grid(row=0, column=1, sticky="ns")

        # Camera source
        src = ttk.LabelFrame(panel, text="Camera source", padding=6)
        src.pack(fill="x", pady=(0, 6))

        self.source_var = tk.StringVar(value=SOURCE_CHOICES[0])
        self.source_combo = ttk.Combobox(
            src,
            textvariable=self.source_var,
            values=SOURCE_CHOICES,
            state="readonly",
            width=26,
        )
        self.source_combo.pack(fill="x")
        self.source_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self._on_source_changed(),
        )

        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(
            src,
            textvariable=self.entry_var,
            width=30,
        )
        self.entry.pack(fill="x", pady=(4, 0))

        self.browse_btn = ttk.Button(
            src,
            text="Browse...",
            command=self._browse,
        )
        self.browse_btn.pack(fill="x", pady=(4, 0))

        self.hint_var = tk.StringVar()
        ttk.Label(
            src,
            textvariable=self.hint_var,
            wraplength=230,
            foreground="#555",
        ).pack(fill="x", pady=(4, 0))

        # Start and stop buttons
        btns = ttk.Frame(panel)
        btns.pack(fill="x", pady=(0, 6))

        self.start_btn = ttk.Button(
            btns,
            text="Start",
            command=self.start,
        )
        self.start_btn.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 3),
        )

        self.stop_btn = ttk.Button(
            btns,
            text="Stop",
            command=self.stop,
            state="disabled",
        )
        self.stop_btn.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(3, 0),
        )

        # Settings
        sett = ttk.LabelFrame(panel, text="Settings", padding=6)
        sett.pack(fill="x", pady=(0, 6))

        self.speech_var = tk.BooleanVar(
            value=self.config.speech_enabled
        )
        ttk.Checkbutton(
            sett,
            text="Spoken alerts",
            variable=self.speech_var,
            command=self._on_speech_toggle,
        ).pack(anchor="w")

        self.edge_var = tk.BooleanVar(
            value=self.config.edge_obstacle_enabled
        )
        ttk.Checkbutton(
            sett,
            text="Canny/contour 'possible obstacle' (experimental)",
            variable=self.edge_var,
            command=self._on_edge_toggle,
        ).pack(anchor="w")

        ttk.Label(
            sett,
            text="Alert cooldown (seconds; higher = fewer repeats)",
        ).pack(anchor="w", pady=(6, 0))

        self.cooldown_var = tk.DoubleVar(
            value=self.config.alert_cooldown_s
        )
        tk.Scale(
            sett,
            from_=1,
            to=15,
            resolution=0.5,
            orient="horizontal",
            variable=self.cooldown_var,
            command=lambda value: (
                self.session
                and self.session.set_cooldown(float(value))
            ),
        ).pack(fill="x")

        ttk.Label(
            sett,
            text="Detection confidence",
        ).pack(anchor="w")

        self.conf_var = tk.DoubleVar(
            value=self.config.confidence_threshold
        )
        tk.Scale(
            sett,
            from_=0.2,
            to=0.9,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_var,
            command=lambda value: (
                self.session
                and self.session.set_confidence(float(value))
            ),
        ).pack(fill="x")

        # Distance tools
        dist = ttk.LabelFrame(
            panel,
            text="Distance estimate tools",
            padding=6,
        )
        dist.pack(fill="x", pady=(0, 6))

        ttk.Button(
            dist,
            text="Calibrate with known distance...",
            command=self.calibrate,
        ).pack(fill="x")

        ttk.Button(
            dist,
            text="Record reference distance check...",
            command=self.reference_check,
        ).pack(fill="x", pady=(4, 0))

        # Live statistics
        stats = ttk.LabelFrame(
            panel,
            text="Live status",
            padding=6,
        )
        stats.pack(fill="x", pady=(0, 6))

        self.fps_var = tk.StringVar(value="FPS: -")
        self.lat_var = tk.StringVar(value="Latency: -")
        self.ram_var = tk.StringVar(value="RAM: -")
        self.alert_var = tk.StringVar(value="Last alert: -")

        for var in (
            self.fps_var,
            self.lat_var,
            self.ram_var,
        ):
            ttk.Label(
                stats,
                textvariable=var,
            ).pack(anchor="w")

        ttk.Label(
            stats,
            textvariable=self.alert_var,
            wraplength=230,
        ).pack(anchor="w", pady=(4, 0))

        # Status bar and safety disclaimer
        self.status_var = tk.StringVar(value="Ready.")

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
        ).grid(row=1, column=0, sticky="ew")

        ttk.Label(
            self.root,
            text=DISCLAIMER,
            wraplength=1000,
            foreground="#a00",
            padding=4,
        ).grid(row=2, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Camera source handling
    # ------------------------------------------------------------------

    def _load_source_from_config(self) -> None:
        src = self.config.camera_source

        if src.isdigit() and int(src) in (0, 1, 2):
            self.source_var.set(f"Webcam {src}")

        elif "://" in src:
            self.source_var.set(SOURCE_CHOICES[3])
            self.entry_var.set(src)

        elif src and src != "0":
            self.source_var.set(SOURCE_CHOICES[4])
            self.entry_var.set(src)

        self._on_source_changed()

    def _on_source_changed(self) -> None:
        choice = self.source_var.get()
        is_url = choice == SOURCE_CHOICES[3]
        is_file = choice == SOURCE_CHOICES[4]

        self.entry.configure(
            state="normal" if (is_url or is_file) else "disabled"
        )
        self.browse_btn.configure(
            state="normal" if is_file else "disabled"
        )

        if is_url:
            self.hint_var.set(
                "Enter your phone's base address, e.g. "
                "http://192.168.1.2:8080. "
                "VisionPath automatically adds /video for IP Webcam."
            )
        elif is_file:
            self.hint_var.set(
                "Choose a recorded video file (useful for repeatable tests)."
            )
        else:
            self.hint_var.set("Built-in or USB webcam.")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[
                (
                    "Video",
                    "*.mp4 *.avi *.mov *.mkv *.wmv",
                ),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.entry_var.set(path)

    def _current_source(self) -> Optional[str]:
        choice = self.source_var.get()

        if choice.startswith("Webcam"):
            return choice.split()[-1]

        text = self.entry_var.get().strip()

        if not text:
            messagebox.showwarning(
                "No source",
                "Please enter a stream URL or choose a video file.",
            )
            return None

        if choice == SOURCE_CHOICES[3]:
            # Add the IP Webcam video endpoint automatically.
            text = text.rstrip("/")

            if not text.lower().endswith("/video"):
                text += "/video"

        return text

    # ------------------------------------------------------------------
    # Start and stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.session is not None and self.session.running:
            return

        source = self._current_source()

        if source is None:
            return

        self._source_text = source
        self.config.camera_source = source
        self.config.alert_cooldown_s = float(
            self.cooldown_var.get()
        )
        self.config.confidence_threshold = float(
            self.conf_var.get()
        )
        self.config.speech_enabled = bool(
            self.speech_var.get()
        )
        self.config.edge_obstacle_enabled = bool(
            self.edge_var.get()
        )

        self.status_var.set(
            "Starting... (loading model and opening camera)"
        )
        self.root.update_idletasks()

        session = DetectionSession(self.config, source)

        try:
            session.start()

        except (ModelNotFoundError, ModelLoadError) as exc:
            self.status_var.set("Model problem.")
            messagebox.showerror(
                "Detection model problem",
                str(exc),
            )
            return

        except CameraError as exc:
            self.status_var.set("Camera problem.")
            messagebox.showerror(
                "Camera problem",
                str(exc),
            )
            return

        except Exception as exc:
            log.exception("Unexpected start failure")
            self.status_var.set("Unexpected error.")
            messagebox.showerror(
                "Unexpected error",
                str(exc),
            )
            return

        session.speech_enabled = bool(self.speech_var.get())
        self.session = session

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.source_combo.configure(state="disabled")

        self.status_var.set(f"Running on source: {source}")

    def stop(self) -> None:
        session, self.session = self.session, None

        if session is None:
            return

        self.status_var.set("Stopping...")
        self.root.update_idletasks()

        summary = session.stop()

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.source_combo.configure(state="readonly")

        self.video_label.configure(
            image="",
            text="Stopped. Press Start to begin",
        )
        self.video_label.image = None

        if summary.get("frames_processed"):
            row = dict(summary)
            row.update({
                "config_name": "gui-session",
                "source": self._source_text,
                "notes": "live GUI session (no ground-truth evaluation)",
            })

            try:
                append_result_row(
                    self.config.resolve_path(self.config.results_csv),
                    row,
                )
                self.status_var.set(
                    f"Stopped. Session saved to {self.config.results_csv}"
                )

            except ResultsFileError as exc:
                self.status_var.set(
                    "Stopped. Results were not saved."
                )
                messagebox.showwarning(
                    "Results not saved",
                    str(exc),
                )
        else:
            self.status_var.set("Stopped.")

    # ------------------------------------------------------------------
    # Live updates and preview
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        session = self.session

        if session is not None:
            update = session.get_update()

            if update is not None:
                self._show_frame(update.frame)

                self.fps_var.set(
                    "FPS: "
                    + (
                        "-"
                        if update.live_fps is None
                        else f"{update.live_fps:.1f}"
                    )
                )

                self.lat_var.set(
                    f"Latency: {update.latency_ms:.1f} ms"
                )

                ram = (
                    "-"
                    if update.ram_mb is None
                    else f"{update.ram_mb:.0f} MB"
                )
                peak = (
                    "-"
                    if update.peak_ram_mb is None
                    else f"{update.peak_ram_mb:.0f} MB"
                )

                self.ram_var.set(
                    f"RAM: {ram} (peak {peak})"
                )

            fatal = False

            for level, text in session.pop_messages():
                if level == "alert":
                    self.alert_var.set(f"Last alert: {text}")

                elif level == "error":
                    fatal = True
                    self.status_var.set(text)
                    messagebox.showerror("VisionPath", text)

                else:
                    self.status_var.set(text)

            if fatal or (
                session.finished.is_set()
                and not session.running
            ):
                self.stop()

        self.root.after(30, self._poll)

    def _show_frame(self, frame) -> None:
        h, w = frame.shape[:2]
        scale = min(
            MAX_VIDEO_W / w,
            MAX_VIDEO_H / h,
        )

        if abs(scale - 1.0) > 0.01:
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
            )

        ok, buf = cv2.imencode(
            ".png",
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )

        if not ok:
            return

        photo = tk.PhotoImage(
            data=base64.b64encode(buf.tobytes())
        )

        self.video_label.configure(
            image=photo,
            text="",
            width=photo.width(),
            height=photo.height(),
        )
        self.video_label.image = photo

    # ------------------------------------------------------------------
    # Settings and distance tools
    # ------------------------------------------------------------------

    def _on_speech_toggle(self) -> None:
        if self.session is not None:
            self.session.speech_enabled = bool(
                self.speech_var.get()
            )

    def _on_edge_toggle(self) -> None:
        if self.session is not None:
            self.session.set_edge_enabled(
                bool(self.edge_var.get())
            )

    def calibrate(self) -> None:
        if self.session is None or not self.session.running:
            messagebox.showinfo(
                "Calibrate",
                "Start the camera first, then stand (or place an object) "
                "at a measured distance, fully visible in the frame.",
            )
            return

        label = simpledialog.askstring(
            "Calibrate",
            "Object class to calibrate with (e.g. person):",
            initialvalue="person",
            parent=self.root,
        )

        if not label:
            return

        dist = simpledialog.askfloat(
            "Calibrate",
            "Measured distance from camera to the object (meters):",
            minvalue=0.3,
            maxvalue=20.0,
            parent=self.root,
        )

        if dist is None:
            return

        try:
            value = self.session.calibrate(
                label.strip().lower(),
                dist,
            )

        except ValueError as exc:
            messagebox.showwarning(
                "Calibration failed",
                str(exc),
            )
            return

        self.config.focal_length_norm = value

        messagebox.showinfo(
            "Calibrated",
            f"Calibration stored (focal length = {value:.3f} frame heights). "
            "It is saved to config.json when you close the app.",
        )

    def reference_check(self) -> None:
        if self.session is None or not self.session.running:
            messagebox.showinfo(
                "Reference check",
                "Start the camera first.",
            )
            return

        try:
            det = self.session.reference_check()

        except ValueError as exc:
            messagebox.showwarning(
                "Reference check",
                str(exc),
            )
            return

        ref = simpledialog.askfloat(
            "Reference check",
            f"Estimate for '{det.label}': {det.distance_m:.2f} m.\n"
            "Enter the tape-measured distance (meters):",
            minvalue=0.1,
            maxvalue=50.0,
            parent=self.root,
        )

        if ref is None:
            return

        path = self.config.resolve_path(
            "results/distance_checks.csv"
        )

        try:
            append_distance_check(
                path,
                det.label,
                ref,
                det.distance_m,
            )
            n, mae = distance_check_summary(path)

        except OSError as exc:
            messagebox.showwarning(
                "Not saved",
                f"Could not write {path.name}: {exc}",
            )
            return

        messagebox.showinfo(
            "Saved",
            f"{n} check(s) recorded. "
            f"Mean absolute error so far: {mae:.2f} m\n({path})",
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def on_close(self) -> None:
        if self.session is not None:
            self.stop()

        try:
            self.config.alert_cooldown_s = float(
                self.cooldown_var.get()
            )
            self.config.confidence_threshold = float(
                self.conf_var.get()
            )
            self.config.speech_enabled = bool(
                self.speech_var.get()
            )
            self.config.edge_obstacle_enabled = bool(
                self.edge_var.get()
            )
            self.config.save(self.config_path)

        except OSError as exc:
            log.warning(
                "Could not save settings: %s",
                exc,
            )

        self.root.destroy()


def run_gui(
    config: Config,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    root = tk.Tk()
    VisionPathApp(root, config, config_path)
    root.mainloop()