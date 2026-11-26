r"""Command-line runner: video in -> annotated video + JSON + CSV out.

Run from the TrafficVision root folder:
    python vehicle_detection\examples\run_detection.py path\to\traffic.mp4
"""

import argparse
import sys
from pathlib import Path

# Make `import vehicle_detection` work when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vehicle_detection import ModelLoadError, VideoError, config, process_video  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="TrafficVision vehicle detection (YOLO11n)")
    p.add_argument("video", help="path to a local video file")
    p.add_argument("--output-dir", default=None, help=f"default: {config.OUTPUT_DIR}")
    p.add_argument("--model", default=None, help=f"model path/name (default: {config.MODEL_PATH})")
    p.add_argument("--conf", type=float, default=None, help=f"confidence threshold (default {config.CONF_THRESHOLD})")
    p.add_argument("--iou", type=float, default=None, help=f"NMS IoU threshold (default {config.IOU_THRESHOLD})")
    p.add_argument("--show", action="store_true", help="live preview window (press q to stop)")
    p.add_argument("--no-video", action="store_true", help="do not save the annotated video")
    p.add_argument("--no-json", action="store_true", help="do not save JSON")
    p.add_argument("--no-csv", action="store_true", help="do not save CSV")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames (quick tests)")
    args = p.parse_args()

    try:
        process_video(
            args.video,
            output_dir=args.output_dir,
            save_video=not args.no_video,
            save_json=not args.no_json,
            save_csv=not args.no_csv,
            show=args.show,
            conf=args.conf,
            iou=args.iou,
            model_path=args.model,
            max_frames=args.max_frames,
        )
    except VideoError as exc:
        print(f"\nVIDEO ERROR: {exc}")
        return 2
    except ModelLoadError as exc:
        print(f"\nMODEL ERROR: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
