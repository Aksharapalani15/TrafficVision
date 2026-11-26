"""Tests for the vehicle detection module.

Run from the TrafficVision root:   python -m pytest vehicle_detection/tests -v

Test categories (see the markers in each test's docstring):
  [no model]     pure logic, runs anywhere
  [model]        needs yolo11n.pt (skipped, not passed, if it cannot be loaded)
  [real photo]   uses Ultralytics' bundled bus.jpg sample photo (real image)
  [synthetic]    generated frames/videos - test mechanics only, NOT traffic realism
  [real video]   needs a real traffic video: set env var VD_TEST_VIDEO=<path>
"""

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vehicle_detection import (  # noqa: E402
    InvalidFrameError,
    ModelLoadError,
    VideoNotFoundError,
    VideoOpenError,
    config,
    detect_frame,
    draw_detections,
    export_csv,
    export_json,
    load_json,
    load_model,
    process_video,
    summarize_detections,
)

EXPECTED_KEYS = {"frame_id", "timestamp", "detections"}
DET_KEYS = {"class", "class_id", "confidence", "bbox"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def model():
    try:
        return load_model()
    except ModelLoadError as exc:
        pytest.skip(f"model unavailable: {exc}")


@pytest.fixture(scope="session")
def bus_image():
    """Real photograph (a bus + pedestrians) shipped with the ultralytics package."""
    try:
        from ultralytics.utils import ASSETS
    except Exception as exc:
        pytest.skip(f"ultralytics not available: {exc}")
    img = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    if img is None:
        pytest.skip("bus.jpg sample not found in ultralytics package")
    return img


def assert_valid_record(rec, width, height):
    assert EXPECTED_KEYS <= set(rec)
    assert isinstance(rec["detections"], list)
    for d in rec["detections"]:
        assert DET_KEYS <= set(d)
        assert d["class_id"] in config.VEHICLE_CLASSES
        assert d["class"] == config.VEHICLE_CLASSES[d["class_id"]]
        assert 0.0 <= d["confidence"] <= 1.0
        b = d["bbox"]
        assert set(b) == {"x1", "y1", "x2", "y2"}
        assert all(isinstance(v, int) for v in b.values())
        assert 0 <= b["x1"] < b["x2"] <= width
        assert 0 <= b["y1"] < b["y2"] <= height


# --------------------------------------------------------------------------- #
# [no model] configuration / input validation / export
# --------------------------------------------------------------------------- #
def test_class_map_matches_requirements():
    """[no model] Exactly car/motorcycle/bus/truck with COCO ids; no bicycle."""
    assert config.VEHICLE_CLASSES == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert 1 not in config.VEHICLE_CLASSES


def test_invalid_frame_inputs_rejected():
    """[no model] None, wrong type, empty and wrong dtype frames raise InvalidFrameError."""
    for bad in (None, "frame.jpg", np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10, 3), np.float32)):
        with pytest.raises(InvalidFrameError):
            detect_frame(bad)


def test_invalid_threshold_rejected():
    """[no model] Out-of-range thresholds raise ValueError before any model work."""
    with pytest.raises(ValueError):
        detect_frame(np.zeros((32, 32, 3), np.uint8), conf=1.5, model=object())


def test_invalid_video_path():
    """[no model] A path that does not exist raises VideoNotFoundError."""
    with pytest.raises(VideoNotFoundError):
        process_video("this/path/does/not/exist.mp4", save_video=False, verbose=False)


def test_unopenable_video(tmp_path):
    """[no model] An empty file and a text file named .mp4 raise VideoOpenError."""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    fake = tmp_path / "fake.mp4"
    fake.write_text("this is not a video")
    for p in (empty, fake):
        with pytest.raises(VideoOpenError):
            process_video(p, output_dir=tmp_path, verbose=False)


def test_export_json_csv_roundtrip(tmp_path):
    """[no model] JSON/CSV written from hand-made records (export logic only)."""
    records = [
        {"frame_id": 1, "timestamp": 0.0, "detections": [
            {"class": "car", "class_id": 2, "confidence": 0.9, "bbox": {"x1": 1, "y1": 2, "x2": 30, "y2": 40}},
            {"class": "bus", "class_id": 5, "confidence": 0.8, "bbox": {"x1": 5, "y1": 6, "x2": 70, "y2": 80}},
        ]},
        {"frame_id": 2, "timestamp": 0.04, "detections": []},
    ]
    jp = export_json(records, tmp_path / "out.json", {"note": "test"})
    cp = export_csv(records, tmp_path / "out.csv")
    data = load_json(jp)
    assert data["frames"] == records and data["metadata"]["note"] == "test"
    rows = list(csv.DictReader(open(cp, encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["class"] == "car" and rows[1]["x2"] == "70"
    s = summarize_detections(records)
    assert s["frames_processed"] == 2 and s["frames_with_detections"] == 1
    assert s["counts_by_class"] == {"car": 1, "motorcycle": 0, "bus": 1, "truck": 0}


def test_draw_detections_does_not_modify_input():
    """[no model] Drawing returns a copy and changes pixels only when boxes exist."""
    frame = np.zeros((120, 160, 3), np.uint8)
    det = [{"class": "car", "class_id": 2, "confidence": 0.9, "bbox": {"x1": 10, "y1": 20, "x2": 80, "y2": 90}}]
    out = draw_detections(frame, det)
    assert frame.sum() == 0 and out.sum() > 0
    assert draw_detections(frame, []).sum() == 0


# --------------------------------------------------------------------------- #
# [model] loading and behaviour on synthetic frames
# --------------------------------------------------------------------------- #
def test_model_loading_and_cache(model):
    """[model] Model loads, class names verified, second call returns the cached object."""
    assert load_model() is model
    assert all(model.names[i] == n for i, n in config.VEHICLE_CLASSES.items())


def test_bad_model_path_gives_clear_error():
    """[model-load failure] A nonexistent weights file raises ModelLoadError."""
    with pytest.raises(ModelLoadError):
        load_model("definitely_not_a_real_model_file.pt")


def test_empty_frame_has_no_detections(model):
    """[synthetic] A plain black frame returns an empty list (no fabricated detections)."""
    rec = detect_frame(np.zeros((480, 640, 3), np.uint8), frame_id=1, timestamp=0.0)
    assert rec["detections"] == [] and rec["frame_id"] == 1 and rec["timestamp"] == 0.0


@pytest.mark.parametrize("w,h", [(160, 120), (320, 240), (640, 480), (1280, 720), (1920, 1080)])
def test_different_resolutions_synthetic(model, w, h):
    """[synthetic] Blank frames of several sizes run without error and give a valid record."""
    rec = detect_frame(np.full((h, w, 3), 127, np.uint8))
    assert_valid_record(rec, w, h)


def test_grayscale_and_bgra_frames_accepted(model):
    """[synthetic] 2-D grayscale and 4-channel frames are normalised, not rejected."""
    assert detect_frame(np.zeros((100, 100), np.uint8))["detections"] == []
    assert detect_frame(np.zeros((100, 100, 4), np.uint8))["detections"] == []


# --------------------------------------------------------------------------- #
# [real photo] real detections
# --------------------------------------------------------------------------- #
def test_real_photo_detects_bus_and_ignores_people(model, bus_image):
    """[real photo] bus.jpg contains a bus and pedestrians: expect a bus, no person class."""
    h, w = bus_image.shape[:2]
    rec = detect_frame(bus_image)
    assert_valid_record(rec, w, h)
    classes = [d["class"] for d in rec["detections"]]
    assert "bus" in classes
    assert set(classes) <= set(config.VEHICLE_CLASSES.values())  # persons are filtered out


def test_multiple_vehicles_in_one_frame(model, bus_image):
    """[real photo, composite] Two copies of the real bus photo side by side -> >= 2 buses.

    This is a composite of a real image, used to exercise multi-detection output.
    """
    double = np.hstack([bus_image, bus_image])
    h, w = double.shape[:2]
    rec = detect_frame(double)
    assert_valid_record(rec, w, h)
    assert sum(d["class"] == "bus" for d in rec["detections"]) >= 2


def test_different_resolutions_real_photo(model, bus_image):
    """[real photo] The same real photo at several sizes gives valid, in-bounds boxes."""
    for scale in (0.5, 1.0, 1.5):
        img = cv2.resize(bus_image, None, fx=scale, fy=scale)
        assert_valid_record(detect_frame(img), img.shape[1], img.shape[0])


def test_high_threshold_filters_detections(model, bus_image):
    """[real photo] conf=0.99 returns fewer-or-equal detections than the default."""
    low = detect_frame(bus_image, conf=0.25)["detections"]
    high = detect_frame(bus_image, conf=0.99)["detections"]
    assert len(high) <= len(low)


# --------------------------------------------------------------------------- #
# Video pipeline
# --------------------------------------------------------------------------- #
def _write_video_from_photo(path, photo, n_frames=6, fps=10.0):
    """[synthetic video] Slightly shifted copies of a real photo. Pipeline test only."""
    h, w = photo.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        M = np.float32([[1, 0, i * 3], [0, 1, 0]])
        writer.write(cv2.warpAffine(photo, M, (w, h)))
    writer.release()


def test_video_pipeline_on_synthetic_video(model, bus_image, tmp_path):
    """[synthetic video] End-to-end process_video: frame numbering, timestamps, outputs."""
    vid = tmp_path / "pipeline_test.mp4"
    _write_video_from_photo(vid, bus_image, n_frames=6, fps=10.0)
    res = process_video(vid, output_dir=tmp_path / "out", verbose=False)

    recs = res["records"]
    assert [r["frame_id"] for r in recs] == list(range(1, len(recs) + 1))
    assert len(recs) == 6
    for r in recs:  # timestamp = (frame_id-1)/fps
        assert r["timestamp"] == pytest.approx((r["frame_id"] - 1) / 10.0, abs=1e-3)
    for key in ("video", "json", "csv"):
        assert Path(res["outputs"][key]).exists()
    assert json.load(open(res["outputs"]["json"]))["frames"] == recs
    assert res["summary"]["frames_processed"] == 6
    assert res["summary"]["counts_by_class"]["bus"] > 0

    out = cv2.VideoCapture(res["outputs"]["video"])  # annotated video is readable
    assert out.isOpened() and int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    out.release()


def test_max_frames_limits_processing(model, bus_image, tmp_path):
    """[synthetic video] max_frames stops early."""
    vid = tmp_path / "v.mp4"
    _write_video_from_photo(vid, bus_image, n_frames=6)
    res = process_video(vid, output_dir=tmp_path, save_video=False, save_csv=False, max_frames=2, verbose=False)
    assert res["summary"]["frames_processed"] == 2


@pytest.mark.skipif(not os.environ.get("VD_TEST_VIDEO"), reason="set VD_TEST_VIDEO to a real traffic video to run")
def test_real_traffic_video(model, tmp_path):
    """[real video] Runs the full pipeline on a real video supplied by the user."""
    res = process_video(os.environ["VD_TEST_VIDEO"], output_dir=tmp_path, max_frames=100, verbose=False)
    assert res["summary"]["frames_processed"] > 0
    w, h = res["video_info"]["width"], res["video_info"]["height"]
    for r in res["records"]:
        assert_valid_record(r, w, h)
