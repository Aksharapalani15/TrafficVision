"""End-to-end TrafficVision pipeline connecting Modules 1, 2, 3, and 5."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from traffic_advisory import generate_advisory
from traffic_analysis import analyze_traffic
from vehicle_counting import VehicleTracker, draw_tracked_vehicles
from vehicle_detection import detect_frame, load_model
from vehicle_detection.video_processor import open_video

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _compute_output_size(width: int, height: int, max_width: int = 1920) -> tuple[int, int]:
    """Scale down oversized (e.g. 4K) output frames for fast browser playback while keeping even dimensions."""
    if width > max_width:
        scale = max_width / float(width)
        out_w = int(round(width * scale))
        out_h = int(round(height * scale))
    else:
        out_w, out_h = int(width), int(height)
    # Ensure even dimensions for H.264 / yuv420p compatibility.
    out_w = max(2, out_w - (out_w % 2))
    out_h = max(2, out_h - (out_h % 2))
    return out_w, out_h


def _open_video_writer(base_path: Path, fps: float, size: tuple[int, int]):
    """Open an OpenCV VideoWriter, trying mp4v first then WebM VP80."""
    for suffix, codec in ((".mp4", "mp4v"), (".webm", "VP80")):
        candidate = base_path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, candidate
        writer.release()
    return None, None


def _transcode_to_browser_mp4(raw_path: Path, final_mp4_path: Path) -> Optional[Path]:
    """Transcode OpenCV output to browser-playable H.264 MP4 using imageio-ffmpeg if available."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        str(final_mp4_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        logger.warning("ffmpeg transcode failed to execute: %s", exc)
        return None

    if proc.returncode == 0 and final_mp4_path.exists() and final_mp4_path.stat().st_size > 0:
        return final_mp4_path

    logger.warning("ffmpeg transcode exited with %s: %s", proc.returncode, proc.stderr[-300:])
    return None


def process_traffic_video(
    video_path,
    output_dir=None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    max_frames: Optional[int] = None,
    use_llm: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Run detection -> tracking/counting -> congestion analysis -> traffic advisory.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where the annotated video will be saved.
        conf: Optional YOLO confidence threshold override.
        iou: Optional YOLO IoU threshold override.
        max_frames: Optional cap on processed frames (None = full video).
        use_llm: Whether to call Gemini for advisory (falls back cleanly if unavailable).
        progress_callback: Optional callable ``(frames_done, total_reported_frames)``.

    Returns:
        Dict containing ``output_video_path``, ``video_info``, ``frames_processed``,
        ``tracking_summary``, ``analysis``, ``advisory``, and ``processing_seconds``.
    """
    cap, info = open_video(video_path)
    cap.release()
    model = load_model()

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    timestamp_tag = time.strftime("%Y%m%d_%H%M%S")
    raw_base = out_dir / f"{stem}_{timestamp_tag}_raw"
    final_mp4 = out_dir / f"{stem}_{timestamp_tag}_annotated.mp4"

    out_size = _compute_output_size(info["width"], info["height"])
    writer, raw_video_path = _open_video_writer(raw_base, info["fps"], out_size)
    if writer is None or raw_video_path is None:
        raise RuntimeError("Could not initialize an output video writer for the annotated video.")

    tracker = VehicleTracker(frame_width=info["width"], frame_height=info["height"])
    cap, info = open_video(video_path)

    started = time.time()
    frame_id = 0
    total_frame_detections = 0
    frame_timeline = []
    fps = info["fps"]
    reported_total = info["reported_frame_count"]

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_id += 1
            timestamp = round((frame_id - 1) / fps, 4)

            det_record = detect_frame(
                frame,
                frame_id=frame_id,
                timestamp=timestamp,
                model=model,
                conf=conf,
                iou=iou,
            )
            detections = det_record["detections"]
            total_frame_detections += len(detections)

            tracked_vehicles = tracker.update(detections, timestamp)
            per_class_frame = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
            for det in detections:
                cls_key = det["class"].lower()
                if cls_key in per_class_frame:
                    per_class_frame[cls_key] += 1
            frame_timeline.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "detections": len(detections),
                    "unique_cumulative": tracker.total_unique_vehicles,
                    **per_class_frame,
                }
            )
            annotated = draw_tracked_vehicles(
                frame,
                tracked_vehicles,
                unique_count=tracker.total_unique_vehicles,
                frame_id=frame_id,
                timestamp=timestamp,
            )

            if (annotated.shape[1], annotated.shape[0]) != out_size:
                annotated = cv2.resize(annotated, out_size, interpolation=cv2.INTER_AREA)
            writer.write(annotated)

            if progress_callback is not None:
                progress_callback(frame_id, reported_total)

            if max_frames is not None and max_frames > 0 and frame_id >= max_frames:
                break
    finally:
        cap.release()
        writer.release()

    if frame_id == 0:
        if raw_video_path.exists():
            raw_video_path.unlink(missing_ok=True)
        raise RuntimeError("No frames could be read from the uploaded video.")

    tracker.finalize(frames_processed=frame_id)

    # Convert to browser-playable H.264 MP4 when ffmpeg is available.
    transcoded = _transcode_to_browser_mp4(raw_video_path, final_mp4)
    if transcoded is not None:
        raw_video_path.unlink(missing_ok=True)
        output_video_path = transcoded
    else:
        fallback_path = out_dir / f"{stem}_{timestamp_tag}_annotated{raw_video_path.suffix}"
        if raw_video_path != fallback_path:
            raw_video_path.replace(fallback_path)
        output_video_path = fallback_path

    if not output_video_path.exists() or output_video_path.stat().st_size == 0:
        raise RuntimeError("Annotated output video file was not generated properly.")

    tracking_summary = tracker.get_summary()
    tracking_summary["frames_processed"] = frame_id
    tracking_summary["total_frame_detections"] = total_frame_detections
    tracking_summary["video_info"] = info

    analysis = analyze_traffic(tracking_summary, frame_timeline=frame_timeline)

    advisory_input = {
        "cars": analysis["category_counts"]["cars"],
        "buses": analysis["category_counts"]["buses"],
        "trucks": analysis["category_counts"]["trucks"],
        "two_wheelers": analysis["category_counts"]["two_wheelers"],
        "congestion": analysis["advisory_congestion"],
    }
    advisory_report = generate_advisory(advisory_input, use_llm=use_llm)

    return {
        "output_video_path": str(output_video_path),
        "video_info": info,
        "frames_processed": frame_id,
        "tracking_summary": tracking_summary,
        "analysis": analysis,
        "advisory": advisory_report,
        "processing_seconds": round(time.time() - started, 2),
    }
