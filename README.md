# Campus Waste Sorting Assistant

Campus Waste Sorting Assistant is a working waste-classification program built
for live presentation use for Northwestern DTD 208 team Six of Crows. Point 
your laptop webcam at an object, hold it steady for a moment, and it tells 
you which of three waste bins it belongs in:

- **Recycling** — bottles, cans, cups, glass
- **Compost** — food scraps and compostable paper
- **Landfill** — wrappers, foam, dirty / non-recyclable trash

The default local classifier runs on your machine. **No paid APIs, no cloud
calls, and no Claude/OpenAI/LLM tokens are used** by that path — not at startup
and not per frame. An optional LLM-assisted mode is also included for stronger
capture-on-demand classification with Gemini or local Ollama.

---

## 1. Architecture (how it works)

The local classifier is a small pipeline split across four files so each piece
is easy to understand and replace:

```
   Webcam ──▶ Detector (YOLOv8n) ──▶ Smoother ──▶ Tkinter UI
  (OpenCV)     local model           majority      video + box +
               object → label        vote + timer  3 bin cards
```

1. **Capture** — OpenCV grabs frames from your webcam.
2. **Detect** — A local **YOLOv8 "nano"** model (`yolov8n.pt`, ~6 MB) finds
   objects in the frame and returns labels like `bottle`, `book`, `cell phone`.
   This is the only "model" in the project and it runs entirely offline.
3. **Map label → bin** — A simple dictionary in `config.py`
   (`LABEL_TO_CATEGORY`) translates each object label into one of the three
   waste categories. This is the part you'll customise most.
4. **Smooth** — Raw predictions flicker frame-to-frame, so a `TemporalSmoother`
   keeps the last several results and reports the **majority vote** plus the
   **average confidence**. A **stability timer** waits until the same category
   has held for ~2 seconds before "locking it in".
5. **Display** — A Tkinter window shows the live video with a bounding box and
   confidence label, three bin cards, and a status line. When a category locks
   in, its bin card **flashes**.

To keep the UI smooth, the camera and the (slower) model run on a **background
thread**; the UI thread only reads the latest result. The model also runs only
on every Nth frame (configurable) to stay light on the CPU.

### Confidence handling

- Detections below `DETECTION_CONFIDENCE` are ignored by the model.
- If the smoothed confidence is below `CATEGORY_CONFIDENCE`, the UI shows
  **"Unsure — hold item steady"** instead of guessing a bin.
- A category is only "locked in" (and its bin flashed) after it stays on top
  *and* confident for `STABLE_SECONDS`.

---

## 2. File structure

```
campus-waste-sorting-assistant/
├── run_local_sorter.py    ← run this:  python run_local_sorter.py
├── run_llm_sorter.py      ← optional LLM-assisted mode
├── requirements.txt
├── requirements-llm.txt
├── README.md
├── waste_sorter/          ← the live app
│   ├── __init__.py
│   ├── config.py          ← all settings + the label→bin mappings (edit me!)
│   ├── detector.py        ← local YOLO wrapper (GPU-aware)
│   ├── smoothing.py       ← temporal smoothing + stability timer
│   └── app.py             ← Tkinter UI + camera thread + main loop
└── training/              ← train your own real waste model (see its README)
    ├── README.md
    ├── download_taco.py   ← fetch the TACO dataset
    ├── prepare_taco.py    ← convert TACO → YOLO format
    └── train.py           ← fine-tune YOLO on the waste classes
```

### Two ways to run it

The local app works in two modes, selected by `MODEL_PROFILE` in `config.py`:

- **`"coco"` (default)** — runs a stock, pretrained YOLO model. Works out of
  the box with zero training, but only knows 80 generic objects, so waste
  categories are approximated (see Limitations).
- **`"taco"`** — runs a model **you train** on the TACO waste dataset, which
  detects real waste classes including **Battery**. This is the trained-program
  path; see [`training/README.md`](training/README.md).

For capture-on-demand classification with Gemini or Ollama, install
`requirements-llm.txt` and run `python run_llm_sorter.py`. See
[`llm_sorter/README.md`](llm_sorter/README.md) for setup, privacy notes, voice
trigger options, and text-to-speech behavior.

---

## 3. Installation

You need **Python 3.9–3.12**. From inside the project folder:

```bash
# (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**What gets downloaded / installed:**

- `ultralytics` — the YOLOv8 library. Installing it also pulls in **PyTorch**
  (a few hundred MB; this is the largest download, and it's a one-time setup).
- `opencv-python` — webcam capture and image drawing.
- `pillow` — lets OpenCV frames display inside the Tkinter window.
- `numpy` — numerical helper.
- **On the first run only**, Ultralytics auto-downloads the default model
  weights (**`yolo11l.pt`, ~50 MB**) into the project folder and caches it.
  After that it works offline.

> **GPU note (Apple Silicon):** the app auto-detects and uses your Mac's GPU
> (Metal/`mps`), which is what lets the large model run in real time. No setup
> needed — `DEVICE = None` in `config.py` handles it.

> **Tkinter note:** Tkinter ships with most Python installs. On some Linux
> systems you may need `sudo apt install python3-tk`. On macOS, the
> python.org installer includes it.

> **Camera permission:** On macOS you'll be asked to allow camera access for
> your terminal/Python the first time.

---

## 4. How to run

```bash
python run_local_sorter.py
```

A window opens with your live webcam feed. Hold one item up to the camera
(a plastic **bottle**, a **book**, your **phone**, a piece of **fruit**) and
keep it steady. Within a couple of seconds the matching bin card lights up and
flashes, and the status line reads e.g. **"➜ Place in Recycling!"**.

Close the window (or press `Ctrl-C` in the terminal) to quit.

### Quick tuning (all in `waste_sorter/config.py`)

| Setting | What it does |
|---|---|
| `CAMERA_INDEX` | Change if the wrong camera opens (try `1`, `2`, …). |
| `PROCESS_EVERY_N_FRAMES` | Increase (e.g. `3`) if your laptop is slow. |
| `DETECTION_CONFIDENCE` | Raise to ignore weak detections. |
| `CATEGORY_CONFIDENCE` | Raise to make the classifier more cautious ("unsure"). |
| `STABLE_SECONDS` | How long to hold an item before the bin locks in. |
| `MODEL_WEIGHTS` | Which model to load (stock or your trained `best.pt`). |
| `MODEL_PROFILE` | `"coco"` (stock) or `"taco"` (your trained waste model). |
| `COCO_/TACO_LABEL_TO_CATEGORY` | The label → bin mappings (edit to taste). |

---

## 5. Limitations & accuracy

**In `"coco"` mode (default, no training):**

- The stock model is trained on COCO's 80 everyday objects and has **no real
  `battery`/`cardboard`/`wrapper` class**, so waste is approximated. Electronics
  and hazardous items route to Landfill in this three-bin setup, and real
  batteries, soda cans, or food wrappers may be missed or mislabeled. This mode
  is best thought of as a quick, zero-setup preview.

**In `"taco"` mode (after training — recommended):**

- The model detects real waste classes including items like bottles, cartons,
  wrappers, food waste, and batteries.
- TACO is a modestly sized dataset (~1500 images), so rarer classes are
  detected less reliably and accuracy still depends on lighting, distance and
  clutter. More epochs, a larger base model, or adding your own labeled photos
  all help (see `training/README.md`).

**Both modes:** the UI tracks **one item at a time** (the most prominent
object) to keep the experience simple, and works best with good lighting and an
uncluttered background.

---

## 6. Train your own waste model

The `training/` folder contains a complete, local pipeline to fine-tune a YOLO
model on the **TACO** waste dataset:

```bash
python training/download_taco.py --out training/datasets/TACO
python training/prepare_taco.py  --taco-dir training/datasets/TACO \
                                 --out training/datasets/taco_yolo
python training/train.py --data training/datasets/taco_yolo/data.yaml \
                         --model yolo11s.pt --epochs 100
```

Then point the live app at your trained model in `config.py`:

```python
MODEL_WEIGHTS = "training/runs/taco/weights/best.pt"
MODEL_PROFILE = "taco"
```

Full step-by-step instructions, knobs, and tips are in
[`training/README.md`](training/README.md). The class → bin rules live in
`config.TACO_LABEL_TO_CATEGORY` and are easy to adjust to your local recycling
rules.

### Other easy extensions

- **Add your own photos** to the dataset (especially of batteries / items you
  care about) to boost accuracy on the things that matter to you.
- **Add more bins** — extend `CATEGORIES` + `CATEGORY_DISPLAY` +
  `CATEGORY_COLOR` in `config.py`.
- **Show multiple objects** at once by returning a list from
  `WasteDetector.detect()` and looping in the UI.
- **Tune box selection** in `detector.py` (it currently favors large, central,
  confident objects).
```
