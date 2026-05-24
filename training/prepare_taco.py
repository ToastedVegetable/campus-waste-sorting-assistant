"""
prepare_taco.py
===============
Convert the TACO dataset (Trash Annotations in Context) from its COCO-style
annotation format into the YOLO detection format that Ultralytics expects,
then write a `data.yaml` ready for training.

TACO: http://tacodataset.org  /  https://github.com/pedropro/TACO
  ~1500 in-the-wild photos of litter, annotated with bounding boxes.
  60 fine categories grouped into 28 "supercategories" (Bottle, Can, Carton,
  Battery, ...). We train on the SUPERCATEGORIES by default: there are fewer
  of them so each has more examples, which trains better on a small dataset.

What this script produces
-------------------------
    <out>/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt     # one YOLO label file per image
      labels/val/*.txt
      data.yaml              # points YOLO at the above + lists class names

Each label line is:  <class_index> <cx> <cy> <w> <h>   (all normalised 0-1)

Usage
-----
    python training/prepare_taco.py \
        --taco-dir /path/to/TACO/data \
        --out      training/datasets/taco_yolo \
        --val-split 0.2

`--taco-dir` is the folder that contains `annotations.json` and the image
batch folders (batch_1/, batch_2/, ...). Use download_taco.py first to get it.
"""

import argparse
import json
import os
import random
import shutil


# ---------------------------------------------------------------------------
# Testable helper functions (no file I/O) -- these are unit-tested separately.
# ---------------------------------------------------------------------------
def build_class_index(categories, granularity="supercategory"):
    """Map each COCO category id -> (class_index, class_name).

    `categories` is the list from the COCO json. `granularity` is either
    "supercategory" (recommended) or "category" (the fine 60-class version).

    Returns (cat_id_to_class_idx, class_names) where class_names is ordered
    so its position == the class index used in the labels.
    """
    if granularity not in ("supercategory", "category"):
        raise ValueError("granularity must be 'supercategory' or 'category'")

    # Collect the set of class names at the requested granularity.
    name_for_cat = {}
    for c in categories:
        name = c["supercategory"] if granularity == "supercategory" else c["name"]
        name_for_cat[c["id"]] = name

    # Stable, sorted ordering so indices are reproducible across runs.
    class_names = sorted(set(name_for_cat.values()))
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    cat_id_to_class_idx = {cid: name_to_idx[name]
                           for cid, name in name_for_cat.items()}
    return cat_id_to_class_idx, class_names


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert a COCO bbox [x, y, w, h] (top-left) to YOLO normalised
    [cx, cy, w, h] in 0-1, clamped to the image bounds."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    def clamp(v):
        return max(0.0, min(1.0, v))

    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def flatten_name(file_name):
    """TACO file names look like 'batch_1/000006.jpg'. Flatten the path so we
    can store every image/label in a single train|val folder without clashes."""
    return file_name.replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------------
# Main conversion routine (does the file I/O).
# ---------------------------------------------------------------------------
def convert(taco_dir, out_dir, val_split=0.2, granularity="supercategory",
            seed=42, symlink=False):
    ann_path = os.path.join(taco_dir, "annotations.json")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(
            f"Could not find {ann_path}. Run download_taco.py first, or pass "
            f"--taco-dir pointing at the folder that holds annotations.json.")

    print(f"[prepare] Reading {ann_path} ...")
    with open(ann_path, "r") as f:
        coco = json.load(f)

    cat_id_to_idx, class_names = build_class_index(coco["categories"], granularity)
    print(f"[prepare] {len(class_names)} classes ({granularity}): {class_names}")

    # Group annotations by image id for quick lookup.
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]

    # Reproducible train/val split.
    random.Random(seed).shuffle(images)
    n_val = int(len(images) * val_split)
    val_ids = {img["id"] for img in images[:n_val]}

    # Create output folder structure.
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    n_written = {"train": 0, "val": 0}
    n_missing = 0

    for img in images:
        split = "val" if img["id"] in val_ids else "train"
        src_img = os.path.join(taco_dir, img["file_name"])
        if not os.path.isfile(src_img):
            n_missing += 1
            continue

        flat = flatten_name(img["file_name"])
        dst_img = os.path.join(out_dir, "images", split, flat)

        # Copy (or symlink) the image into the dataset.
        if symlink:
            if not os.path.islink(dst_img) and not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src_img), dst_img)
        else:
            shutil.copy(src_img, dst_img)

        # Write the YOLO label file (same base name, .txt extension).
        label_name = os.path.splitext(flat)[0] + ".txt"
        dst_label = os.path.join(out_dir, "labels", split, label_name)
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            cls_idx = cat_id_to_idx[ann["category_id"]]
            cx, cy, w, h = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(dst_label, "w") as f:
            f.write("\n".join(lines))

        n_written[split] += 1

    # Write data.yaml for Ultralytics.
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("# Auto-generated by prepare_taco.py\n")
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"  - {name}\n")

    print(f"[prepare] Wrote {n_written['train']} train + {n_written['val']} val "
          f"images. ({n_missing} images referenced in annotations were missing "
          f"on disk -- run download_taco.py to fetch them all.)")
    print(f"[prepare] Dataset ready. data.yaml -> {yaml_path}")
    print(f"[prepare] Class names (copy these into config.TACO_LABEL_TO_CATEGORY "
          f"if you customise the mapping):\n           {class_names}")
    return yaml_path, class_names


def main():
    p = argparse.ArgumentParser(description="Convert TACO -> YOLO detection format")
    p.add_argument("--taco-dir", required=True,
                   help="Folder containing annotations.json and image batches")
    p.add_argument("--out", default="training/datasets/taco_yolo",
                   help="Output dataset folder")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of images used for validation (default 0.2)")
    p.add_argument("--granularity", choices=["supercategory", "category"],
                   default="supercategory",
                   help="Train on 28 supercategories (default) or 60 fine categories")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink images instead of copying (saves disk space)")
    args = p.parse_args()

    convert(args.taco_dir, args.out, val_split=args.val_split,
            granularity=args.granularity, symlink=args.symlink)


if __name__ == "__main__":
    main()
