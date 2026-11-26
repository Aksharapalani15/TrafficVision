"""Central settings for the vehicle detection module.

Every tunable value lives here so nothing is hard-coded elsewhere.
Environment variables (PowerShell: ``$env:VD_CONF = "0.4"``) override the
defaults without editing code.
"""

import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent

# --- Model ------------------------------------------------------------------
# Weights are stored inside the module (vehicle_detection/models/yolo11n.pt) so
# they are covered by the module's .gitignore. If the file is missing,
# Ultralytics downloads it from GitHub on first use (~5.4 MB, internet needed once).
# Set VD_MODEL_PATH to any other local weights file to run fully offline.
MODEL_PATH = os.environ.get("VD_MODEL_PATH", str(MODULE_DIR / "models" / "yolo11n.pt"))

# "auto" -> first CUDA GPU if available, otherwise CPU. Can also be "cpu" or 0.
DEVICE = os.environ.get("VD_DEVICE", "auto")

# --- Thresholds ---------------------------------------------------------------
CONF_THRESHOLD = float(os.environ.get("VD_CONF", "0.25"))  # min confidence
IOU_THRESHOLD = float(os.environ.get("VD_IOU", "0.45"))    # NMS overlap limit
IMAGE_SIZE = int(os.environ.get("VD_IMGSZ", "640"))        # YOLO inference size

# --- Classes ------------------------------------------------------------------
# COCO class IDs used by the pretrained model. load_model() verifies these
# against model.names at load time, so a wrong mapping fails loudly.
# Bicycles (COCO id 1) are deliberately NOT included.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# --- Video / output -----------------------------------------------------------
DEFAULT_FPS = 25.0  # used only if the video reports no valid frame rate
OUTPUT_DIR = Path(os.environ.get("VD_OUTPUT_DIR", MODULE_DIR / "outputs"))

# Box colours (BGR) per class, used only for drawing.
CLASS_COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 120, 0),
    "truck": (0, 0, 230),
}
