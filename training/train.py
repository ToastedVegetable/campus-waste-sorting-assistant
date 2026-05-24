"""
train.py
========
Train a YOLO object-detection model on the converted TACO dataset.

This is a thin, friendly wrapper around Ultralytics' training API with
defaults that work well on an Apple Silicon laptop (it uses the GPU via
"mps" automatically). Training is transfer learning: we start from a model
already pretrained on COCO and fine-tune it on the waste classes, which needs
far less data and time than training from scratch.

Usage (after download_taco.py + prepare_taco.py):
    python training/train.py \
        --data  training/datasets/taco_yolo/data.yaml \
        --model yolo11s.pt \
        --epochs 100

When it finishes it prints the path to `best.pt`. Point the live app at it:
    in waste_sorter/config.py set
        MODEL_WEIGHTS = "training/runs/taco/weights/best.pt"
        MODEL_PROFILE = "taco"

Tips
----
* Base model size trades accuracy vs training speed:
    yolo11n.pt (fastest)  yolo11s.pt (good default)  yolo11m.pt (more accurate)
* More epochs = better, up to a point. 100 is a reasonable starting point on
  ~1500 images; watch the validation mAP printed each epoch.
* If you hit out-of-memory, lower --batch (e.g. 8 or 4) or --imgsz (e.g. 512).
"""

import argparse


def auto_device():
    """Apple GPU > NVIDIA GPU > CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"   # first CUDA GPU
    except Exception:
        pass
    return "cpu"


def main():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on the TACO waste dataset")
    p.add_argument("--data", default="training/datasets/taco_yolo/data.yaml",
                   help="Path to the data.yaml produced by prepare_taco.py")
    p.add_argument("--model", default="yolo11s.pt",
                   help="Base (pretrained) model to fine-tune from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Training image size (640 is standard)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size. Lower this if you run out of memory.")
    p.add_argument("--device", default=None,
                   help="mps / cpu / 0 (CUDA). Default: auto-detect.")
    p.add_argument("--project", default="training/runs",
                   help="Where to save training runs")
    p.add_argument("--name", default="taco",
                   help="Run name (subfolder under --project)")
    args = p.parse_args()

    # Imported here so the rest of the toolkit can be read without ultralytics.
    from ultralytics import YOLO

    device = args.device or auto_device()
    print(f"[train] Fine-tuning {args.model} on {args.data}")
    print(f"[train] device={device}  epochs={args.epochs}  imgsz={args.imgsz}  "
          f"batch={args.batch}")

    model = YOLO(args.model)  # downloads pretrained weights on first use
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best = f"{args.project}/{args.name}/weights/best.pt"
    print("\n[train] Training complete.")
    print(f"[train] Best weights: {best}")
    print("[train] To use them in the live app, edit waste_sorter/config.py:")
    print(f'           MODEL_WEIGHTS = "{best}"')
    print('           MODEL_PROFILE = "taco"')


if __name__ == "__main__":
    main()
