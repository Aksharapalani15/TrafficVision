r"""How the tracking/counting module (Member 2) consumes this module.

Run from the TrafficVision root folder:
    python vehicle_detection\examples\member2_integration_example.py path\to\traffic.mp4

Member 2 owns the frame loop and passes each frame to detect_frame().
Tracking, ID assignment and counting are Member 2's job - this file only
shows how to read the detections.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # not needed once run from the repo root as a package

from vehicle_detection import detect_frame, load_model  # noqa: E402


def my_tracker_update(detections, frame_id):
    """Placeholder for Member 2's tracker (ByteTrack, SORT, ...)."""
    for d in detections:
        b = d["bbox"]
        cx, cy = (b["x1"] + b["x2"]) // 2, (b["y1"] + b["y2"]) // 2  # box centre
        print(f"  frame {frame_id}: {d['class']:<10} conf={d['confidence']:.2f} centre=({cx},{cy})")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: member2_integration_example.py <video path>")
        return 1
    model = load_model()  # load ONCE, reuse for every frame
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        print("Could not open video")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_id = 0
    while frame_id < 10:  # first 10 frames only, for the demo
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        record = detect_frame(frame, frame_id=frame_id, timestamp=(frame_id - 1) / fps, model=model)
        my_tracker_update(record["detections"], record["frame_id"])
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
