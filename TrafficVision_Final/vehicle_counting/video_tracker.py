"""Member 2 video tracking & annotation helpers (safe to import or run via CLI)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vehicle_detection import config as vd_config
from vehicle_detection import detect_frame, load_model
from vehicle_detection.video_processor import open_video
from vehicle_counting.tracker import VehicleTracker


def draw_tracked_vehicles(
    frame,
    tracked_vehicles,
    unique_count: int,
    frame_id: Optional[int] = None,
    timestamp: Optional[float] = None,
):
    """Return an annotated copy of *frame* with tracked vehicle boxes, IDs, and count HUD."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    scale = max(0.45, min(w, h) / 900.0)
    thick = max(1, int(round(scale * 2)))

    for vehicle in tracked_vehicles:
        bbox = vehicle["bbox"]
        vehicle_id = vehicle["id"]
        vehicle_class = vehicle["class"]
        color = vd_config.CLASS_COLORS.get(vehicle_class.lower(), (0, 220, 0))

        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

        label = f"{vehicle_class} #{vehicle_id}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

    hud_parts = [f"Unique Vehicles: {unique_count}"]
    if frame_id is not None:
        hud_parts.insert(0, f"Frame {frame_id}")
    if timestamp is not None:
        hud_parts.append(f"t={timestamp:.1f}s")
    hud_text = " | ".join(hud_parts)

    hud_scale = max(0.55, scale * 1.1)
    hud_thick = max(2, thick)
    (hw, hh), hbase = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, hud_thick)
    cv2.rectangle(annotated, (10, 10), (24 + hw, 24 + hh + hbase), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        hud_text,
        (17, 17 + hh),
        cv2.FONT_HERSHEY_SIMPLEX,
        hud_scale,
        (0, 255, 255),
        hud_thick,
        cv2.LINE_AA,
    )
    return annotated


def process_video_tracking(
    video_path,
    model=None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    max_frames: Optional[int] = None,
    on_frame: Optional[Callable[[int, int, object, dict], None]] = None,
) -> dict:
    """Run Member 1 detection + Member 2 tracking frame-by-frame over a video."""
    cap, info = open_video(video_path)
    if model is None:
        model = load_model()

    tracker = VehicleTracker(frame_width=info["width"], frame_height=info["height"])
    fps = info["fps"]
    reported_total = info["reported_frame_count"]

    frame_id = 0
    total_frame_detections = 0

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
            annotated = draw_tracked_vehicles(
                frame,
                tracked_vehicles,
                unique_count=tracker.total_unique_vehicles,
                frame_id=frame_id,
                timestamp=timestamp,
            )

            if on_frame is not None:
                on_frame(frame_id, reported_total, annotated, det_record)

            if max_frames is not None and frame_id >= max_frames:
                break
    finally:
        cap.release()

    tracker.finalize(frames_processed=frame_id)
    summary = tracker.get_summary()
    summary["frames_processed"] = frame_id
    summary["total_frame_detections"] = total_frame_detections
    summary["video_info"] = info
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="TrafficVision - Vehicle Tracking & Counting")
    parser.add_argument("video", help="Path to traffic video")
    parser.add_argument("--json-out", default="vehicle_counting/vehicle_counts.json")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    summary = process_video_tracking(args.video, max_frames=args.max_frames)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Total unique vehicles: {summary['total_unique_vehicles']}")
    print(f"Category counts: {summary['category_counts']}")
    print(f"Saved summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
