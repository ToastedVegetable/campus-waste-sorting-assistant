"""
config.py
=========
ALL the knobs for the application live here, so you can tune behaviour without
digging through the rest of the code. This is also the file you will edit
the most when you later swap in your own trained model.

Sections:
  1. Waste categories (the three bins)
  2. Detection / confidence thresholds and timing
  3. Camera + performance settings
  4. The label -> waste-category mapping (the application's decision rules)
  5. Colours used by the UI
"""

import os

# ---------------------------------------------------------------------------
# 1. WASTE CATEGORIES  (the three bins shown in the UI)
# ---------------------------------------------------------------------------
# We use short string "keys" internally (RECYCLING, COMPOST, ...) and a friendly
# display name for the UI. Keep these keys stable -- the rest of the code
# refers to them.

LANDFILL = "LANDFILL"
COMPOST = "COMPOST"
RECYCLING = "RECYCLING"

# Order here = left-to-right order of the bin cards in the UI.
CATEGORIES = [LANDFILL, COMPOST, RECYCLING]

# Friendly names + a one-line hint shown on each bin card.
CATEGORY_DISPLAY = {
    RECYCLING: {"name": "Recycling", "hint": "Clean bottles, cans, paper, glass"},
    COMPOST:   {"name": "Compost",   "hint": "Food scraps and compostable paper"},
    LANDFILL:  {"name": "Landfill",  "hint": "Wrappers, foam, dirty mixed trash"},
}

# What to do with a detected object whose label we have NOT mapped to a bin.
# Set to LANDFILL to dump unknown items in general trash, or set to None to
# treat unknown objects as "unsure" (the app will ask you to hold steady).
DEFAULT_CATEGORY_FOR_UNMAPPED = LANDFILL


# ---------------------------------------------------------------------------
# 2. DETECTION / CONFIDENCE THRESHOLDS AND TIMING
# ---------------------------------------------------------------------------

# YOLO will ignore any detection it is less sure about than this (0.0 - 1.0).
DETECTION_CONFIDENCE = 0.40

# After smoothing, we only commit to a category if the averaged confidence
# is at least this high. Below this we show "Unsure -- hold item steady".
CATEGORY_CONFIDENCE = 0.50

# How long (seconds) the SAME category must stay on top before we "lock it in"
# and flash the matching bin. This is what creates the "hold it for a moment,
# then it tells you where to put it" behaviour.
STABLE_SECONDS = 2.0

# Number of recent detection results kept for majority-vote smoothing.
# Bigger = steadier but slower to react. (~ last second of detections.)
SMOOTHING_WINDOW = 8

# Object labels the detector should completely IGNORE. Since YOU are holding
# items up to the webcam, the model will almost always also see "person" --
# we drop that so the app focuses on the item, not you. A few large pieces of
# background furniture are included too, as they're not things you'd hold up to
# sort. Add or remove labels here as you like (use raw COCO label names).
IGNORED_LABELS = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",            # often the monitor behind you; remove if you WANT e-waste TVs
    "potted plant",
}


# ---------------------------------------------------------------------------
# 3. CAMERA + PERFORMANCE SETTINGS
# ---------------------------------------------------------------------------

# Which webcam to use. 0 is usually the built-in camera; 1, 2, ... are external
# cameras. Override without editing this file by launching with CAMERA_INDEX=1.
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

# Capture resolution requested from the webcam (the camera may pick the
# nearest supported size).
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# Run the (relatively expensive) YOLO model only on every Nth captured frame.
# 1 = every frame (smoothest detection, heaviest CPU).
# 2-3 = lighter on the CPU, still feels responsive. Increase if your laptop
# struggles. The live video itself stays smooth regardless of this value.
PROCESS_EVERY_N_FRAMES = 2

# Model weights file. Auto-downloaded the first time you run the app, then
# cached locally. Bigger model = more accurate but heavier:
#   yolov8n.pt  (~6 MB)  nano   - fastest, least accurate
#   yolo11s.pt  (~19 MB) small  - good balance
#   yolo11m.pt  (~40 MB) medium - more accurate
#   yolo11l.pt  (~50 MB) large  - very accurate (default here)
#   yolo11x.pt  (~110 MB) xlarge - most accurate, heaviest
# yolo11* is Ultralytics' newer, more accurate family. On an Apple Silicon
# "Pro" chip with GPU acceleration (see DEVICE below) the large model runs
# comfortably.
#
# To use YOUR OWN model trained on the TACO waste dataset (see the training/
# folder), point this at the trained weights AND switch MODEL_PROFILE to
# "taco" below, e.g.:
# MODEL_WEIGHTS = "training/runs/taco/weights/best.pt"
# MODEL_PROFILE = "taco"
MODEL_WEIGHTS = "yolo11l.pt"
MODEL_PROFILE = "coco"


# Which label vocabulary the loaded model speaks. This selects the right
# label -> bin mapping (see section 4):
#   "coco" -> a stock yolo11*.pt model (80 generic objects)
#   "taco" -> a model you trained on the TACO waste dataset (waste classes
#             like Bottle, Can, Carton, Battery, ...)

# Which compute device to run inference on:
#   None   -> auto-detect: Apple GPU ("mps") > NVIDIA GPU ("cuda") > "cpu"
#   "mps"  -> force Apple Silicon GPU (Metal)   "cpu" -> force CPU
# Using the GPU is what makes the large model feel real-time on a MacBook.
DEVICE = None

# Resolution the model runs inference at (square). The webcam feed is resized
# to this internally. Larger = better at spotting small/distant objects but
# slower. 640 is the YOLO default; 960 noticeably improves small-object
# detection and is comfortable on an Apple Silicon Pro GPU.
INFERENCE_IMGSZ = 960

# Ignore detections whose box covers less than this fraction of the frame.
# A held-up item fills a good chunk of the view, so tiny boxes are usually
# background clutter or false positives. Raise to be stricter, lower to catch
# smaller items. (0.0 disables the filter.)
MIN_BOX_AREA_FRACTION = 0.015


# ---------------------------------------------------------------------------
# 4. LABEL -> WASTE-CATEGORY MAPPING  (the "brain" that picks a bin)
# ---------------------------------------------------------------------------
# A detection model only ever outputs a *label* (e.g. "bottle"). These maps
# translate that label into one of our three bins. There are two maps because
# the app can run two very different kinds of model -- MODEL_PROFILE (above)
# chooses which one is active.
#
# Anything detected but NOT listed in the active map falls back to
# DEFAULT_CATEGORY_FOR_UNMAPPED (see section 1).
#
# --- Profile "coco": stock yolo11*.pt --------------------------------------
# COCO knows 80 everyday objects and is trained on 100k+ well-lit photos, so
# it is robust at close range -- a good fit for items held up to a webcam.
# This map is tuned for COLLEGE-CAMPUS waste (coffee cups, food, bowls, papers,
# bottles). Recycling rules vary by campus/city, so adjust to your local bins.
#
# Note: COCO has no "paper", "cardboard", "battery" or "can" class. Flat sheets
# of paper detect unreliably; "book" is the closest stand-in for paper items.

COCO_LABEL_TO_CATEGORY = {
    # ---- Recycling ----
    # COCO's closest label to notebooks / stacks of paper / magazines.
    "book": RECYCLING,
    "bottle": RECYCLING,       # plastic & glass drink bottles
    "wine glass": RECYCLING,
    "can": RECYCLING,          # not a stock COCO class, harmless to leave

    # ---- Compost (food scraps) ----
    "banana": COMPOST,
    "apple": COMPOST,
    "sandwich": COMPOST,
    "orange": COMPOST,
    "broccoli": COMPOST,
    "carrot": COMPOST,
    "hot dog": COMPOST,
    "pizza": COMPOST,
    "donut": COMPOST,
    "cake": COMPOST,

    # ---- Landfill (campus default for soiled / lined / mixed items) ----
    # IMPORTANT campus choice: disposable CUPS (Starbucks/coffee/foam) are
    # lined or food-soiled and usually are NOT curbside-recyclable -> Landfill.
    # If your campus recycles clean plastic cold cups, move "cup" to RECYCLING.
    "cup": LANDFILL,
    # Food bowls/containers are typically food-soiled -> Landfill. Move to
    # RECYCLING if you mostly see clean rigid-plastic bowls.
    "bowl": LANDFILL,
    # Disposable utensils (usually plastic) -> Landfill.
    "fork": LANDFILL,
    "knife": LANDFILL,
    "spoon": LANDFILL,
    "toothbrush": LANDFILL,
    "scissors": LANDFILL,

    # ---- No e-waste bin in this three-bin setup ----
    "cell phone": LANDFILL,
    "laptop": LANDFILL,
    "mouse": LANDFILL,
    "keyboard": LANDFILL,
    "remote": LANDFILL,
    "microwave": LANDFILL,
    "toaster": LANDFILL,
    "hair drier": LANDFILL,
}

# --- Profile "taco": a model trained on the TACO waste dataset -------------
# These keys are TACO's 28 "supercategories" (the class names produced by
# training/prepare_taco.py with the default --granularity supercategory).
# Unlike COCO, this includes a real "Battery" class, but this UI only has
# Recycling / Compost / Landfill.
#
# Recycling rules differ by city, so treat this as a sensible default and
# tweak it to match YOUR local rules. (E.g. some places recycle cartons or
# certain plastics; others send them to landfill.)
TACO_LABEL_TO_CATEGORY = {
    # ---- No e-waste bin in this three-bin setup ----
    "Battery": LANDFILL,

    # ---- Recycling (clean glass / metal / rigid plastic containers) ----
    "Bottle": RECYCLING,
    "Bottle cap": RECYCLING,
    "Can": RECYCLING,
    "Glass jar": RECYCLING,
    "Lid": RECYCLING,
    "Plastic container": RECYCLING,
    "Pop tab": RECYCLING,
    "Scrap metal": RECYCLING,
    "Aluminium foil": RECYCLING,
    "Cup": RECYCLING,

    # ---- Paper / cardboard ----
    "Paper": RECYCLING,
    "Paper bag": RECYCLING,
    "Carton": RECYCLING,

    # ---- Landfill (films, foams, composites, food, misc litter) ----
    "Plastic bag & wrapper": LANDFILL,
    "Other plastic": LANDFILL,
    "Plastic film": LANDFILL,
    "Styrofoam piece": LANDFILL,
    "Plastic utensils": LANDFILL,
    "Plastic glooves": LANDFILL,
    "Squeezable tube": LANDFILL,
    "Straw": LANDFILL,
    "Rope & strings": LANDFILL,
    "Shoe": LANDFILL,
    "Blister pack": LANDFILL,
    "Broken glass": LANDFILL,
    "Food waste": COMPOST,
    "Cigarette": LANDFILL,
    "Unlabeled litter": LANDFILL,
}

# Select the active map based on MODEL_PROFILE (see section 3).
LABEL_TO_CATEGORY = TACO_LABEL_TO_CATEGORY if MODEL_PROFILE == "taco" \
    else COCO_LABEL_TO_CATEGORY


def category_for_label(label: str):
    """Translate a raw model label (e.g. 'bottle') into a waste category key.

    Uses whichever map MODEL_PROFILE selected. Returns one of the category
    keys (RECYCLING/COMPOST/LANDFILL), or None if the label is unknown
    AND DEFAULT_CATEGORY_FOR_UNMAPPED is None.
    """
    if label in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[label]
    return DEFAULT_CATEGORY_FOR_UNMAPPED


# ---------------------------------------------------------------------------
# 5. COLOURS (R, G, B) used by the UI / overlays
# ---------------------------------------------------------------------------
# One signature colour per bin so the bounding box and the highlighted card
# match visually.

CATEGORY_COLOR = {
    RECYCLING: (0, 96, 255),      # blue
    COMPOST:   (0, 168, 84),      # green
    LANDFILL:  (5, 5, 5),         # black
}

# Neutral colour used when we are unsure / nothing is detected.
NEUTRAL_COLOR = (180, 180, 180)

# Tkinter background tones (hex) for a clean dark UI.
UI_BG = "#1e1e2e"
UI_PANEL_BG = "#2a2a3c"
UI_TEXT = "#f5f5f5"
UI_SUBTLE_TEXT = "#b0b0c0"
