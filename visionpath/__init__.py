"""VisionPath: OpenCV-based real-time vision assistance prototype (experimental)."""
import os

# Must be set before OpenCV opens any RTSP stream. TCP is more reliable than UDP
# on Wi-Fi (fewer corrupted frames). Override by setting the variable yourself.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

__version__ = "0.1.0"
