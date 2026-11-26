"""Saving detection records to JSON / CSV and summarising them."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import config

CSV_COLUMNS = ["frame_id", "timestamp", "class", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def summarize_detections(records: List[dict]) -> dict:
    """Count what was actually detected.

    ``counts_by_class`` sums detections over all frames. The same physical
    vehicle appears in many frames, so these are NOT unique-vehicle counts
    (unique counting needs tracking, which belongs to the tracking module).
    """
    counts: Dict[str, int] = {name: 0 for name in config.VEHICLE_CLASSES.values()}
    frames_with = 0
    for rec in records:
        dets = rec.get("detections", [])
        if dets:
            frames_with += 1
        for det in dets:
            counts[det["class"]] = counts.get(det["class"], 0) + 1
    return {
        "frames_processed": len(records),
        "frames_with_detections": frames_with,
        "frames_without_detections": len(records) - frames_with,
        "total_detections": sum(counts.values()),
        "counts_by_class": counts,
    }


def export_json(records: List[dict], json_path, metadata: Optional[dict] = None) -> Path:
    """Write ``{"metadata": {...}, "frames": [<frame records>]}`` to *json_path*."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    meta.update(metadata or {})
    payload = {"metadata": meta, "frames": records}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def export_csv(records: List[dict], csv_path) -> Path:
    """Write one row per detection. Frames with no detections produce no rows."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for rec in records:
            for det in rec.get("detections", []):
                box = det["bbox"]
                writer.writerow(
                    [
                        rec["frame_id"],
                        rec["timestamp"],
                        det["class"],
                        det["class_id"],
                        det["confidence"],
                        box["x1"],
                        box["y1"],
                        box["x2"],
                        box["y2"],
                    ]
                )
    return path


def load_json(json_path) -> dict:
    """Read a file written by export_json() (handy for offline consumers)."""
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
