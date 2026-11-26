"""End-to-end integration tests for TrafficVision_Final."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_components import (
    build_bar_chart,
    build_csv_report,
    build_donut_chart,
    build_forecast_chart,
    build_interval_heatmap,
    build_json_report,
    build_timeline_chart,
    render_junction_heatmap_svg,
)
from pipeline import process_traffic_video
from traffic_analysis import (
    analyze_traffic,
    build_interval_timeline,
    classify_traffic_condition,
    forecast_next_interval,
    get_congestion_zone,
)
from vehicle_counting import VehicleTracker


def test_congestion_thresholds():
    assert classify_traffic_condition(0.0) == ("LOW", "Free")
    assert classify_traffic_condition(4.9) == ("LOW", "Free")
    assert classify_traffic_condition(5.0) == ("MODERATE", "Moderate")
    assert classify_traffic_condition(11.9) == ("MODERATE", "Moderate")
    assert classify_traffic_condition(12.0) == ("HIGH", "Heavy")


def test_congestion_zones():
    assert get_congestion_zone(2.0)["zone"] == "LOW"
    assert get_congestion_zone(7.5)["zone"] == "MODERATE"
    assert get_congestion_zone(15.0)["zone"] == "HIGH"
    assert get_congestion_zone(22.0)["zone"] == "VERY HIGH"


def test_tracker_counts_unique_vehicles_without_line_crossing():
    tracker = VehicleTracker(frame_width=1280, frame_height=720, min_hits=3)
    for i in range(4):
        dets = [
            {"class": "car", "bbox": {"x1": 200, "y1": 100 + i * 10, "x2": 300, "y2": 180 + i * 10}},
            {"class": "bus", "bbox": {"x1": 900, "y1": 100 + i * 10, "x2": 1050, "y2": 220 + i * 10}},
        ]
        tracker.update(dets, timestamp=i * 0.04)

    summary = tracker.get_summary()
    assert summary["total_unique_vehicles"] == 2
    assert summary["class_counts"]["car"] == 1
    assert summary["class_counts"]["bus"] == 1
    assert summary["category_counts"]["cars"] == 1
    assert summary["category_counts"]["buses"] == 1


def test_interval_timeline_and_moving_average_forecast():
    timeline = [
        {"frame_id": i + 1, "timestamp": i * 1.0, "detections": 4 + (i % 3), "unique_cumulative": i + 1,
         "car": 3, "bus": 1, "truck": 0, "motorcycle": 0}
        for i in range(8)
    ]
    intervals = build_interval_timeline(timeline, interval_seconds=2.0)
    assert len(intervals) == 4
    forecast = forecast_next_interval(intervals, window=3)
    assert forecast["available"] is True
    assert forecast["estimated_avg_vehicles"] is not None
    assert forecast["estimated_condition"] in ("LOW", "MODERATE", "HIGH")

    short_forecast = forecast_next_interval(intervals[:2], window=3, min_intervals=3)
    assert short_forecast["available"] is False
    assert "Insufficient" in short_forecast["reason"]


def test_end_to_end_pipeline_on_sample_video_slice(tmp_path):
    """Run real YOLO + Tracker + Analyzer + Advisory + Charts/Reports on 24 frames of traffic.mp4."""
    sample_video = ROOT / "vehicle_detection" / "sample_data" / "traffic.mp4"
    if not sample_video.exists():
        pytest.skip("Sample traffic.mp4 not present")

    result = process_traffic_video(
        video_path=sample_video,
        output_dir=tmp_path,
        max_frames=24,
        use_llm=False,
    )

    out_video = Path(result["output_video_path"])
    assert out_video.exists()
    assert out_video.stat().st_size > 0

    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None

    assert result["frames_processed"] == 24
    analysis = result["analysis"]
    assert analysis["total_unique_vehicles"] > 0
    assert analysis["traffic_condition"] in ("LOW", "MODERATE", "HIGH")
    assert analysis["most_frequent_category"]["category"] != "None"
    assert len(analysis["intervals"]) >= 1

    # Verify charts, heatmap SVG, CSV and JSON reports build cleanly from real results
    assert build_donut_chart(analysis["class_counts"], ["car", "bus", "truck", "motorcycle"]) is not None
    assert build_bar_chart(analysis["class_counts"], ["car", "bus", "truck", "motorcycle"]) is not None
    assert build_timeline_chart(analysis["intervals"], ["car", "bus", "truck", "motorcycle"]) is not None
    assert build_interval_heatmap(analysis["intervals"]) is not None
    if analysis["forecast"]["available"]:
        assert build_forecast_chart(analysis["intervals"], analysis["forecast"]) is not None

    svg_html = render_junction_heatmap_svg(analysis)
    assert "Illustrative Junction Congestion Map" in svg_html

    csv_text = build_csv_report(result)
    assert "Total Unique Vehicles" in csv_text
    json_payload = json.loads(build_json_report(result))
    assert json_payload["frames_processed"] == 24

    advisory = result["advisory"]
    assert advisory["status"] == "success"
    assert advisory["source"] == "fallback"
    assert advisory["advisory"]
    assert len(advisory["recommended_actions"]) >= 1


def test_end_to_end_pipeline_on_empty_synthetic_video(tmp_path):
    """Ensure empty road video produces LOW condition and valid fallback advisory without crashing."""
    empty_vid = tmp_path / "empty.mp4"
    writer = cv2.VideoWriter(str(empty_vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(5):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    result = process_traffic_video(video_path=empty_vid, output_dir=tmp_path, use_llm=False)

    assert result["analysis"]["total_unique_vehicles"] == 0
    assert result["analysis"]["traffic_condition"] == "LOW"
    assert result["analysis"]["most_frequent_category"]["category"] == "None"
    assert result["advisory"]["status"] == "success"
    assert result["advisory"]["source"] == "fallback"
    assert Path(result["output_video_path"]).exists()
