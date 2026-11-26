r"""Quick check on a single image (useful when you have no video yet).

    python vehicle_detection\examples\detect_image.py path\to\photo.jpg
Writes <name>_annotated.jpg into vehicle_detection\outputs\ and prints detections.
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vehicle_detection import config, detect_frame, draw_detections  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: detect_image.py <image path>")
        return 1
    path = Path(sys.argv[1])
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read image: {path}")
        return 2
    record = detect_frame(frame, frame_id=1, timestamp=0.0)
    print(json.dumps(record, indent=2))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"{path.stem}_annotated.jpg"
    cv2.imwrite(str(out), draw_detections(frame, record["detections"]))
    print(f"\n{len(record['detections'])} vehicle(s) detected. Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
