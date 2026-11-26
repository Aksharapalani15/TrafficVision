# TrafficVision – Vehicle Detection Module (Member 1)

YOLO11n-based detector that finds **cars, motorcycles, buses and trucks** in video
frames and returns them in a fixed, JSON-friendly format for the tracking/counting
module (Member 2).

**This module does:** per-frame detection, bounding boxes, confidence scores,
annotated video, JSON/CSV export.
**This module does NOT do:** tracking IDs, vehicle counting, line crossing,
congestion classification, prediction or dashboards (other members' work).

---

## 1. Setup (Windows PowerShell)

Requires Python 3.9 – 3.12 (check with `python --version`). Run from the
**TrafficVision root folder**.

```powershell
cd path\to\TrafficVision

# create and activate a virtual environment (kept inside the module, git-ignored)
python -m venv vehicle_detection\.venv
vehicle_detection\.venv\Scripts\Activate.ps1
# If PowerShell blocks the script:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m pip install --upgrade pip
pip install -r vehicle_detection\requirements.txt
```

PyTorch (installed by ultralytics) is a large download. It runs on **CPU by
default**; a CUDA GPU is used automatically if available.

**Model weights:** `yolo11n.pt` (~5.4 MB) is downloaded automatically from GitHub the
first time the model loads and is saved to `vehicle_detection\models\`. Internet is
needed once. To work offline, copy the file there yourself, or use
`--model C:\path\yolo11n.pt` / `$env:VD_MODEL_PATH`.

## 2. Run

```powershell
# Full pipeline: annotated video + JSON + CSV + terminal summary
python vehicle_detection\examples\run_detection.py "C:\path\to\traffic.mp4"

# Options
python vehicle_detection\examples\run_detection.py traffic.mp4 --show              # live preview (q = stop)
python vehicle_detection\examples\run_detection.py traffic.mp4 --conf 0.35 --iou 0.5
python vehicle_detection\examples\run_detection.py traffic.mp4 --max-frames 100    # quick test
python vehicle_detection\examples\run_detection.py traffic.mp4 --no-csv --output-dir D:\results

# No video yet? Check a single photo instead
python vehicle_detection\examples\detect_image.py "C:\path\to\photo.jpg"
```

**Outputs** (default folder `vehicle_detection\outputs\`, git-ignored):

| File | Content |
|---|---|
| `<name>_annotated.mp4` | video with boxes, class names, confidence (falls back to `.avi` if the mp4 codec is unavailable) |
| `<name>_detections.json` | `{"metadata": {...}, "frames": [ <frame records> ]}` |
| `<name>_detections.csv` | one row per detection: `frame_id,timestamp,class,class_id,confidence,x1,y1,x2,y2` |

The terminal summary reports frames processed, frames with/without vehicles,
detections per class and processing speed. **Per-class numbers are detections summed
over frames, not unique vehicles** – one car visible for 100 frames counts 100 times.
Unique counting needs tracking (Member 2).

## 3. Output format

```json
{
  "frame_id": 1,
  "timestamp": 0.0,
  "detections": [
    {"class": "car", "class_id": 2, "confidence": 0.92,
     "bbox": {"x1": 120, "y1": 80, "x2": 250, "y2": 190}}
  ]
}
```

* `frame_id` starts at **1**. `timestamp = (frame_id - 1) / fps` seconds, i.e. the
  time the frame starts (frame 1 = 0.0 s). If a video reports no FPS, 25 is assumed
  and `fps_assumed: true` is written to the JSON metadata.
* `bbox` values are integer pixels in the original frame, clamped to the frame.
* No vehicles → `"detections": []`. Detections are sorted by confidence (highest first).
* Class IDs are COCO IDs, **verified against the model's own class names at load time**.

## 4. Using it from another module (Member 2)

Run Python from the TrafficVision root so `vehicle_detection` is importable.

```python
import cv2
from vehicle_detection import detect_frame, load_model

model = load_model()                      # load ONCE (cached)
cap = cv2.VideoCapture("traffic.mp4")
frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_id += 1
    record = detect_frame(frame, frame_id=frame_id, model=model)
    for det in record["detections"]:
        b = det["bbox"]                   # b["x1"], b["y1"], b["x2"], b["y2"]
        print(det["class"], det["confidence"], b)   # -> feed to the tracker
```

A runnable version is in `examples\member2_integration_example.py`. Offline use:
`vehicle_detection.load_json("..._detections.json")["frames"]`.

## 5. API summary

| Function | Purpose |
|---|---|
| `load_model(model_path=None)` | Loads/caches YOLO11n, checks class names, raises `ModelLoadError` with a fix hint |
| `detect_frame(frame, frame_id=None, timestamp=None, model=None, conf=None, iou=None)` | One BGR frame → record. No windows, no files. Raises `InvalidFrameError` for bad input |
| `process_video(video_path, output_dir=None, save_video=True, save_json=True, save_csv=True, show=False, conf=None, iou=None, max_frames=None)` | Whole video → `{"records", "summary", "video_info", "outputs"}`. Raises `VideoNotFoundError` / `VideoOpenError` |
| `iter_frames(video_path)` | Yields `(frame_id, timestamp, frame)` |
| `draw_detections(frame, detections)` | Returns an annotated copy |
| `export_json(records, path, metadata)` / `export_csv(records, path)` | File export |

Settings (`config.py`, or environment variables `VD_MODEL_PATH`, `VD_CONF`, `VD_IOU`,
`VD_IMGSZ`, `VD_DEVICE`, `VD_OUTPUT_DIR`): defaults are confidence 0.25, IoU 0.45,
image size 640.

**Files:** `detector.py` (inference only) · `video_processor.py` (video I/O, drawing,
preview) · `exporter.py` (JSON/CSV/summary) · `config.py` (all settings).
CSV uses Python's built-in `csv` module, so pandas is not required.

## 6. Tests

```powershell
python -m pytest vehicle_detection\tests -v -rs
# Include a real traffic video:
$env:VD_TEST_VIDEO = "C:\path\to\traffic.mp4"; python -m pytest vehicle_detection\tests -v -rs
```

Each test's docstring says what kind of data it uses: **[no model]**, **[model]**,
**[real photo]** (Ultralytics' bundled `bus.jpg`), **[synthetic]** (blank frames/videos
built from that photo – these test mechanics, not traffic realism) or **[real video]**
(skipped unless `VD_TEST_VIDEO` is set).

**Status when this module was built** (Linux sandbox, Python 3.12, CPU-only PyTorch,
ultralytics 8.4.216): **22 passed, 1 skipped.** The one skip is the real-traffic-video
test. Consequently:

* **Verified on a real photo:** bus detection, people ignored, multi-object output
  (two-bus composite), boxes in-bounds at several resolutions.
* **NOT verified on real footage:** cars, trucks and motorcycles, congested scenes,
  night/rain, and real video at different resolutions. Run the real-video test on your
  own clips and check the annotated output yourself before reporting those classes as tested.

## 7. Getting a test video

No video is bundled. Download one from a source with a clear licence and **read the
licence on each video's page**:

* **Pexels Videos** (pexels.com/videos) – Pexels License: free for personal and
  commercial use, attribution not required. Search "traffic", "highway", "city street".
* **Pixabay Videos** (pixabay.com/videos) – Pixabay Content License: free to use,
  attribution not required; some restrictions (e.g. no redistribution as-is).
* **Wikimedia Commons** (commons.wikimedia.org) – licence is per file (often CC BY or
  CC BY-SA, which require attribution); the file page shows the exact terms.
* Research datasets (e.g. UA-DETRAC, AI City Challenge) usually restrict use to
  non-commercial research and require registration – read their terms first.

On these sites use the **Download** button in your browser, then:

```powershell
# put it somewhere outside the repo (or in a git-ignored folder) and run
python vehicle_detection\examples\run_detection.py "$HOME\Videos\traffic.mp4"
```

Use `Invoke-WebRequest -Uri <direct-file-URL> -OutFile traffic.mp4` only if a site
provides a direct file link. Pick clips with different vehicle types, and keep a note
of the source and licence for your report. Never commit videos.

## 8. Git

This module ships its own `vehicle_detection\.gitignore` (outputs, `*.pt`, videos,
`.venv`, caches). Suggested lines for the shared root `.gitignore` (not modified by this
module):

```
*.pt
*.mp4
*.avi
*.mov
outputs/
.venv/
__pycache__/
```

## 9. Model, licence, limitations, responsible AI

* **Model:** YOLO11n, COCO-pretrained, used as-is (no training). Chosen because it is
  small (~5 MB), runs on a laptop CPU, and covers all four required classes. It
  processed ~8 frames/s at 810×1080 on a CPU-only sandbox; your speed will differ.
  For better accuracy on small or distant vehicles, use `yolo11s.pt` (slower).
* **Licence:** Ultralytics YOLO is released under **AGPL-3.0** (a commercial licence is
  sold separately). That is generally workable for an open-source coursework project,
  but check the terms with your faculty guide before distributing or deploying it as a
  service. Weights are not committed to Git.
* **Accuracy varies** with lighting, weather, camera angle, occlusion, distance, motion
  blur and video quality. Detection is per frame, so boxes may flicker or be missed –
  smoothing is Member 2's job.
* **COCO limits:** there is no class for auto-rickshaws, vans/pickups or bicycles with
  riders. These may be labelled as car/truck/motorcycle, or missed. Bicycles are
  deliberately *not* counted as motorcycles. Truck/bus/car confusion happens.
* **Privacy:** the module outputs only vehicle class, confidence and box coordinates.
  It does not read number plates or recognise faces or drivers, and stores none. The
  annotated video contains the original frames, so treat it like the source video.
* **Preview:** `--show` needs the full `opencv-python` build (installed by the
  requirements). With a headless build it prints a warning and continues without a window.
