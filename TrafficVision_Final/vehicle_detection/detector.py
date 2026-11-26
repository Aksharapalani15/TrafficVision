"""YOLO vehicle detection (inference only).

This file never opens windows, draws, or writes files. It turns one OpenCV
frame into a structured detection record that other modules can consume.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)

# Models are cached so the weights load once, never once per frame.
_MODEL_CACHE: Dict[tuple, Any] = {}


class ModelLoadError(RuntimeError):
    """The YOLO model or the ultralytics package could not be loaded."""


class InvalidFrameError(ValueError):
    """The object passed to detect_frame() is not a usable image frame."""


@lru_cache(maxsize=None)
def _resolve_device(device: Any) -> Any:
    """'auto' -> GPU 0 if CUDA is available, else CPU."""
    if device not in (None, "auto"):
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing/broken -> CPU fallback
        return "cpu"


def _verify_class_names(model: Any) -> None:
    """Ensure the configured class IDs really mean car/motorcycle/bus/truck."""
    names = getattr(model, "names", None) or {}
    wrong = {
        cid: (expected, names.get(cid))
        for cid, expected in config.VEHICLE_CLASSES.items()
        if names.get(cid) != expected
    }
    if wrong:
        details = ", ".join(f"id {c}: expected '{e}', model has '{g}'" for c, (e, g) in wrong.items())
        raise ModelLoadError(
            f"Model class names do not match config.VEHICLE_CLASSES ({details}). "
            "This module expects a COCO-pretrained model such as yolo11n.pt. "
            "If you use a custom model, update VEHICLE_CLASSES in config.py."
        )


def load_model(model_path: Optional[str] = None, force_reload: bool = False) -> Any:
    """Load (and cache) the YOLO model.

    Args:
        model_path: weights file or Ultralytics model name. Defaults to
            ``config.MODEL_PATH`` (``yolo11n.pt``).
        force_reload: ignore the cache and load again.

    Raises:
        ModelLoadError: with a human-readable message and a suggested fix.
    """
    path = str(model_path or config.MODEL_PATH)
    key = (path, str(_resolve_device(config.DEVICE)))
    if not force_reload and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. "
            "Run: pip install -r vehicle_detection/requirements.txt"
        ) from exc

    try:
        model = YOLO(path)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load YOLO model '{path}': {exc}\n"
            "On the first run Ultralytics downloads yolo11n.pt from GitHub, so an "
            "internet connection is needed once. If you are offline, download the "
            "file manually and point to it with --model <path> or $env:VD_MODEL_PATH."
        ) from exc

    _verify_class_names(model)
    _MODEL_CACHE[key] = model
    logger.info("Loaded model %s on device %s", path, key[1])
    return model


def _prepare_frame(frame: Any) -> np.ndarray:
    """Validate a frame and normalise it to a 3-channel BGR uint8 array."""
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError(f"frame must be a NumPy array (OpenCV image), got {type(frame).__name__}")
    if frame.size == 0 or frame.ndim not in (2, 3):
        raise InvalidFrameError(f"frame is empty or has unsupported shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise InvalidFrameError(f"frame must have dtype uint8, got {frame.dtype}")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    channels = frame.shape[2]
    if channels == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return frame
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise InvalidFrameError(f"frame has unsupported channel count {channels}")


def _parse_results(results: Any, width: int, height: int) -> List[dict]:
    """Convert Ultralytics results into the project's detection dictionaries."""
    detections: List[dict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), score, cid in zip(xyxy, confs, class_ids):
            cid = int(cid)
            if cid not in config.VEHICLE_CLASSES:  # defensive; predict() already filters
                continue
            # Clamp to the frame so coordinates are always valid pixels.
            x1 = int(round(min(max(float(x1), 0), width)))
            x2 = int(round(min(max(float(x2), 0), width)))
            y1 = int(round(min(max(float(y1), 0), height)))
            y2 = int(round(min(max(float(y2), 0), height)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "class": config.VEHICLE_CLASSES[cid],
                    "class_id": cid,
                    "confidence": round(float(score), 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def detect_frame(
    frame: np.ndarray,
    frame_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    model: Any = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    imgsz: Optional[int] = None,
) -> dict:
    """Detect cars, motorcycles, buses and trucks in one OpenCV frame.

    Args:
        frame: BGR image as returned by ``cv2.VideoCapture.read()``.
        frame_id / timestamp: optional values copied into the record. Callers
            that own the video loop (process_video, the tracking module) supply
            them; left as ``None`` otherwise. Nothing is invented here.
        model: an already-loaded model (default: the cached shared model).
        conf / iou / imgsz: override the thresholds from ``config.py``.

    Returns:
        ``{"frame_id": ..., "timestamp": ..., "detections": [ {class, class_id,
        confidence, bbox:{x1,y1,x2,y2}}, ... ]}``. ``detections`` is an empty
        list when no vehicle is found.

    Raises:
        InvalidFrameError: the frame is not a valid image array.
        ModelLoadError: the model could not be loaded.
    """
    image = _prepare_frame(frame)
    conf = config.CONF_THRESHOLD if conf is None else conf
    iou = config.IOU_THRESHOLD if iou is None else iou
    imgsz = config.IMAGE_SIZE if imgsz is None else imgsz
    if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
        raise ValueError("conf and iou must be between 0 and 1")

    if model is None:
        model = load_model()

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=list(config.VEHICLE_CLASSES.keys()),  # ignore people, animals, etc.
        device=_resolve_device(config.DEVICE),
        verbose=False,
    )
    height, width = image.shape[:2]
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "detections": _parse_results(results, width, height),
    }
