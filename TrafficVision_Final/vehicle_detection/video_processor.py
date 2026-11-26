"""Video input, annotation, preview and output for the detector."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .detector import detect_frame, load_model
from .exporter import export_csv, export_json, summarize_detections

logger = logging.getLogger(__name__)


class VideoError(Exception):
    """Base class for video problems."""


class VideoNotFoundError(VideoError, FileNotFoundError):
    """The given path does not exist or is not a file."""


class VideoOpenError(VideoError):
    """The file exists but OpenCV cannot open it as a video."""


# --------------------------------------------------------------------------- #
# Video input
# --------------------------------------------------------------------------- #
def open_video(video_path) -> Tuple[cv2.VideoCapture, dict]:
    """Validate *video_path* and open it. Returns (capture, info dict)."""
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        raise VideoNotFoundError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoOpenError(
            f"OpenCV could not open '{path}'. The file may be corrupt, empty, or use an "
            "unsupported codec. Try converting it to H.264 .mp4."
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_assumed = not (fps and math.isfinite(fps) and fps > 0)
    if fps_assumed:
        logger.warning("Video reports no valid FPS; assuming %s fps for timestamps.", config.DEFAULT_FPS)
        fps = config.DEFAULT_FPS
    info = {
        "path": str(path),
        "fps": float(fps),
        "fps_assumed": fps_assumed,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "reported_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, info


def iter_frames(video_path) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield ``(frame_id, timestamp_seconds, frame)`` for each frame.

    ``frame_id`` starts at 1. ``timestamp = (frame_id - 1) / fps`` is the time
    at which the frame starts, so frame 1 is at 0.0 s.
    """
    cap, info = open_video(video_path)
    try:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_id += 1
            yield frame_id, round((frame_id - 1) / info["fps"], 4), frame
        expected = info["reported_frame_count"]
        if frame_id == 0:
            raise VideoOpenError(f"'{video_path}' opened but no frames could be read.")
        if expected > 0 and frame_id < expected * 0.98:
            logger.warning("Stream ended after %d frames but the file reports %d (read error?).", frame_id, expected)
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_detections(frame: np.ndarray, detections: List[dict], frame_id=None, timestamp=None) -> np.ndarray:
    """Return a copy of *frame* with boxes, class names and confidences drawn."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 900)
    thick = max(1, int(round(scale * 2)))
    for det in detections:
        b = det["bbox"]
        color = config.CLASS_COLORS.get(det["class"], (255, 255, 255))
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thick)
        label = f'{det["class"]} {det["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        top = max(b["y1"] - th - base - 2, 0)
        cv2.rectangle(out, (b["x1"], top), (b["x1"] + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(out, label, (b["x1"] + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)
    if frame_id is not None:
        hud = f"frame {frame_id}" + (f"  t={timestamp:.2f}s" if timestamp is not None else "")
        cv2.putText(out, hud, (10, int(28 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, hud, (10, int(28 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def _open_writer(base_path: Path, fps: float, size: Tuple[int, int]):
    """Try .mp4 (mp4v) first, then fall back to .avi (MJPG)."""
    for suffix, codec in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = base_path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
    return None, None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def process_video(
    video_path,
    output_dir=None,
    save_video: bool = True,
    save_json: bool = True,
    save_csv: bool = True,
    show: bool = False,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    model_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Run vehicle detection over every frame of a video.

    Returns a dict with:
        ``records``  - list of per-frame records (the standard output format)
        ``summary``  - frame counts and per-class detection counts
        ``video_info`` - fps, resolution, etc.
        ``outputs``  - paths of the files written (video/json/csv)

    Raises VideoNotFoundError / VideoOpenError for bad input, ModelLoadError if
    the model cannot be loaded.
    """
    # Validate the video and load the model BEFORE doing any work.
    cap, info = open_video(video_path)
    cap.release()
    model = load_model(model_path)

    out_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    stem = Path(video_path).stem
    records: List[dict] = []
    outputs = {"video": None, "json": None, "csv": None}
    writer = None
    show_enabled = show
    started = time.time()
    interrupted = False

    try:
        for frame_id, timestamp, frame in iter_frames(video_path):
            record = detect_frame(frame, frame_id=frame_id, timestamp=timestamp,
                                  model=model, conf=conf, iou=iou)
            records.append(record)

            if save_video or show_enabled:
                annotated = draw_detections(frame, record["detections"], frame_id, timestamp)
                if save_video:
                    if writer is None:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        h, w = annotated.shape[:2]
                        writer, vpath = _open_writer(out_dir / f"{stem}_annotated", info["fps"], (w, h))
                        if writer is None:
                            logger.error("Could not create an output video writer; video will not be saved.")
                            save_video = False
                        else:
                            outputs["video"] = str(vpath)
                    if writer is not None:
                        writer.write(annotated)
                if show_enabled:
                    try:
                        cv2.imshow("TrafficVision - vehicle detection (press q to stop)", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            interrupted = True
                    except cv2.error:
                        logger.warning("Preview window unavailable (headless OpenCV?). Continuing without it.")
                        show_enabled = False

            if verbose and frame_id % 25 == 0:
                total = info["reported_frame_count"]
                print(f"  processed {frame_id}" + (f"/{total}" if total > 0 else "") + " frames...", flush=True)
            if interrupted or (max_frames is not None and frame_id >= max_frames):
                break
    finally:
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.time() - started
    summary = summarize_detections(records)
    summary["processing_seconds"] = round(elapsed, 2)
    summary["processing_fps"] = round(len(records) / elapsed, 2) if elapsed > 0 else None

    metadata = {
        "source_video": Path(video_path).name,
        "fps": info["fps"],
        "fps_assumed": info["fps_assumed"],
        "width": info["width"],
        "height": info["height"],
        "model": str(model_path or config.MODEL_PATH),
        "conf_threshold": config.CONF_THRESHOLD if conf is None else conf,
        "iou_threshold": config.IOU_THRESHOLD if iou is None else iou,
        "classes": config.VEHICLE_CLASSES,
        "timestamp_convention": "(frame_id - 1) / fps; frame_id starts at 1",
        "summary": summary,
    }
    if save_json:
        outputs["json"] = str(export_json(records, out_dir / f"{stem}_detections.json", metadata))
    if save_csv:
        outputs["csv"] = str(export_csv(records, out_dir / f"{stem}_detections.csv"))

    result = {"records": records, "summary": summary, "video_info": info, "outputs": outputs}
    if verbose:
        print_summary(result)
    return result


def print_summary(result: dict) -> None:
    """Print a readable terminal summary of a process_video() result."""
    s, info, out = result["summary"], result["video_info"], result["outputs"]
    line = "=" * 58
    print(f"\n{line}\n VEHICLE DETECTION SUMMARY\n{line}")
    print(f" Video              : {Path(info['path']).name}  ({info['width']}x{info['height']}, "
          f"{info['fps']:.2f} fps{' - assumed' if info['fps_assumed'] else ''})")
    print(f" Frames processed   : {s['frames_processed']}")
    print(f" Frames w/ vehicles : {s['frames_with_detections']}  (without: {s['frames_without_detections']})")
    print(f" Total detections   : {s['total_detections']}")
    print(" Detections by class (summed over frames, NOT unique vehicles):")
    for name, n in s["counts_by_class"].items():
        print(f"   {name:<11}: {n}")
    if s.get("processing_fps"):
        print(f" Speed              : {s['processing_fps']} frames/s ({s['processing_seconds']} s total)")
    for label, key in (("Annotated video", "video"), ("JSON", "json"), ("CSV", "csv")):
        if out.get(key):
            print(f" {label:<19}: {out[key]}")
    print(line)
