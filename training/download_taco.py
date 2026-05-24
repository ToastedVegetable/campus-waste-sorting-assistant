"""
download_taco.py
================
Download the TACO dataset (annotations + images) to a local folder so that
prepare_taco.py can convert it.

It does two things:
  1. Fetches `annotations.json` from the official TACO repo if you don't
     already have it.
  2. Reads that file and downloads every image (TACO hosts a 640px version of
     each photo) into the same folder, preserving the batch_N/ subfolders so
     the file paths match the annotations.

This script uses only the Python standard library (urllib), so it needs no
extra packages. It runs ON YOUR machine and only contacts the public TACO /
Flickr image hosts -- no Claude/LLM/cloud-AI calls are involved.

Usage
-----
    python training/download_taco.py --out training/datasets/TACO

Then convert:
    python training/prepare_taco.py --taco-dir training/datasets/TACO \
        --out training/datasets/taco_yolo

Notes
-----
* The full image set is a few GB and can take a while. The script skips files
  that already exist, so you can safely re-run it if it gets interrupted.
* A few images occasionally fail to download (dead Flickr links); that's fine,
  prepare_taco.py just skips any image missing on disk.
"""

import argparse
import json
import os
import time
import urllib.request

# Official annotations file in the TACO GitHub repo.
ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"

# Pretend to be a normal browser; some hosts reject the default urllib agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (TACO-downloader)"}


def _download(url, dst, timeout=30):
    """Download a single URL to dst. Returns True on success."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    return True


def ensure_annotations(out_dir):
    """Make sure annotations.json exists locally; download it if not."""
    ann_path = os.path.join(out_dir, "annotations.json")
    if os.path.isfile(ann_path):
        print(f"[download] Using existing {ann_path}")
        return ann_path
    os.makedirs(out_dir, exist_ok=True)
    print(f"[download] Fetching annotations.json ...")
    _download(ANNOTATIONS_URL, ann_path)
    print(f"[download] Saved {ann_path}")
    return ann_path


def download_images(out_dir, ann_path, limit=None):
    """Download every image referenced in the annotations file."""
    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    if limit:
        images = images[:limit]

    total = len(images)
    done = skipped = failed = 0

    for i, img in enumerate(images, 1):
        dst = os.path.join(out_dir, img["file_name"])
        if os.path.isfile(dst):
            skipped += 1
            continue

        # TACO stores a 640px URL (smaller, plenty for training) and a full one.
        url = img.get("flickr_640_url") or img.get("flickr_url")
        if not url:
            failed += 1
            continue

        try:
            _download(url, dst)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[download]  ! failed {img['file_name']}: {exc}")

        if i % 50 == 0 or i == total:
            print(f"[download] {i}/{total}  (new={done} skipped={skipped} failed={failed})")
        # Be polite to the image host.
        time.sleep(0.05)

    print(f"[download] Done. {done} downloaded, {skipped} already present, "
          f"{failed} failed.")


def main():
    p = argparse.ArgumentParser(description="Download the TACO dataset")
    p.add_argument("--out", default="training/datasets/TACO",
                   help="Destination folder for annotations.json + images")
    p.add_argument("--limit", type=int, default=None,
                   help="Only download the first N images (handy for a quick test)")
    args = p.parse_args()

    ann_path = ensure_annotations(args.out)
    download_images(args.out, ann_path, limit=args.limit)


if __name__ == "__main__":
    main()
