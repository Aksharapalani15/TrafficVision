"""TrafficVision - Member 1: YOLO vehicle detection module.

Typical use from another module (run from the TrafficVision root folder):

    from vehicle_detection import detect_frame
    record = detect_frame(frame)          # frame = OpenCV BGR image
    for det in record["detections"]:
        print(det["class"], det["confidence"], det["bbox"])

Importing this package is cheap: ultralytics/PyTorch are imported only when
the model is first loaded.
"""

from .config import VEHICLE_CLASSES
from .detector import InvalidFrameError, ModelLoadError, detect_frame, load_model
from .exporter import export_csv, export_json, load_json, summarize_detections
from .video_processor import (
    VideoError,
    VideoNotFoundError,
    VideoOpenError,
    draw_detections,
    iter_frames,
    process_video,
)

__all__ = [
    "VEHICLE_CLASSES",
    "load_model",
    "detect_frame",
    "process_video",
    "iter_frames",
    "draw_detections",
    "export_json",
    "export_csv",
    "load_json",
    "summarize_detections",
    "ModelLoadError",
    "InvalidFrameError",
    "VideoError",
    "VideoNotFoundError",
    "VideoOpenError",
]
