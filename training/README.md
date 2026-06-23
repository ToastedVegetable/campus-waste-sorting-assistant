# Training a real waste model on TACO

This folder turns the local classifier from generic-object recognition into a
program that
detects **actual waste classes** — including **batteries** — by fine-tuning a
YOLO model on the [TACO dataset](http://tacodataset.org) (Trash Annotations in
Context: ~1500 in-the-wild litter photos with bounding boxes, grouped into 28
waste *supercategories*).

Everything runs locally on your machine. No paid APIs or LLM calls are used.

## The three steps

```
download_taco.py   →   prepare_taco.py   →   train.py
 (get images +          (COCO → YOLO          (fine-tune YOLO,
  annotations)           format + data.yaml)   produces best.pt)
```

### 1. Download TACO

```bash
python training/download_taco.py --out training/datasets/TACO
```

Downloads `annotations.json` (from the official TACO repo) and every image
(TACO's 640px versions) into `training/datasets/TACO/`, preserving the
`batch_N/` folders. It's a few GB and skips files you already have, so it's
safe to re-run if interrupted. Tip: add `--limit 100` for a quick trial run.

### 2. Convert to YOLO format

```bash
python training/prepare_taco.py \
    --taco-dir training/datasets/TACO \
    --out      training/datasets/taco_yolo \
    --val-split 0.2
```

Produces a YOLO dataset (`images/`, `labels/`, `data.yaml`). By default it
trains on the **28 supercategories** (Bottle, Can, Carton, Battery, …). Pass
`--granularity category` to use the finer 60-class version instead. The script
prints the exact class names it used.

### 3. Train

```bash
python training/train.py \
    --data  training/datasets/taco_yolo/data.yaml \
    --model yolo11s.pt \
    --epochs 100
```

This fine-tunes a COCO-pretrained YOLO model on the waste classes (transfer
learning — much faster than training from scratch). It auto-uses your Apple
GPU (`mps`). Watch the validation **mAP** printed each epoch; higher is better.
When it finishes you'll get `training/runs/taco/weights/best.pt`.

Knobs:
- `--model` — `yolo11n.pt` (fastest) · `yolo11s.pt` (default) · `yolo11m.pt` (more accurate)
- `--epochs` — try 100; raise for more accuracy if mAP is still improving
- `--batch` — lower (8 or 4) if you hit out-of-memory
- `--imgsz` — 640 standard; 512 is lighter

## 4. Use your trained model in the live app

Edit `waste_sorter/config.py`:

```python
MODEL_WEIGHTS = "training/runs/taco/weights/best.pt"
MODEL_PROFILE = "taco"     # switches to the TACO label → bin mapping
```

Then run the app as usual:

```bash
python run_local_sorter.py
```

The bounding-box caption will now show real waste labels (e.g. `Bottle`,
`Battery`, `Carton`), and the three bins react via `TACO_LABEL_TO_CATEGORY`
in `config.py`.

## Customising the bin mapping

`config.TACO_LABEL_TO_CATEGORY` maps each TACO supercategory to one of the
three bins. **Recycling rules vary by city**, so adjust it to your local rules
(e.g. whether cartons or certain plastics are recyclable where you live). Any
class you don't list falls back to `DEFAULT_CATEGORY_FOR_UNMAPPED`.

## Notes & tips for better accuracy

- TACO is relatively small (~1500 images), so some rare classes have few
  examples and will be detected less reliably. More epochs and a larger base
  model (`yolo11m.pt`) help.
- You can grow the dataset by adding your own labeled photos in YOLO format
  into `images/` and `labels/` — especially photos of the items you care most
  about (e.g. batteries against realistic backgrounds).
- Keep `--granularity supercategory` unless you specifically need the fine
  60-class distinctions; fewer classes train better on limited data.
