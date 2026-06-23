from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PreparedDatasetSummary:
    """Small run summary returned by prepare_dataset and printed by the CLI."""

    yaml_path: Path
    class_names: list[str]
    train_images: int
    val_images: int
    missing_images: int


def build_class_index(
    categories: Iterable[dict],
    granularity: str = "supercategory",
) -> tuple[dict[int, int], list[str]]:
    """Map raw TACO category ids to contiguous model class indices.

    TACO provides both fine-grained category names and broader supercategories.
    Supercategories are the default because they reduce sparsity in a relatively
    small dataset.
    """
    if granularity not in {"supercategory", "category"}:
        raise ValueError("granularity must be either 'supercategory' or 'category'")

    category_names_by_id: dict[int, str] = {}
    for category in categories:
        category_id = int(category["id"])
        category_names_by_id[category_id] = str(category[granularity])

    class_names = sorted(set(category_names_by_id.values()))
    class_index_by_name = {name: idx for idx, name in enumerate(class_names)}
    category_id_to_class_index = {
        category_id: class_index_by_name[name]
        for category_id, name in category_names_by_id.items()
    }

    return category_id_to_class_index, class_names


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, width, height] boxes to YOLO normalized boxes.

    YOLO expects center-x, center-y, width, height values normalized to [0, 1].
    The coordinates are clipped to the image bounds so a slightly noisy source
    annotation cannot produce invalid training labels.
    """
    x, y, width, height = bbox

    x_min = _clip(x, 0.0, float(image_width))
    y_min = _clip(y, 0.0, float(image_height))
    x_max = _clip(x + width, 0.0, float(image_width))
    y_max = _clip(y + height, 0.0, float(image_height))

    clipped_width = max(0.0, x_max - x_min)
    clipped_height = max(0.0, y_max - y_min)
    center_x = x_min + clipped_width / 2.0
    center_y = y_min + clipped_height / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        clipped_width / image_width,
        clipped_height / image_height,
    )


def flatten_image_name(file_name: str) -> str:
    """Flatten nested TACO names like 'batch_1/000006.jpg' for output folders."""
    return file_name.replace("/", "_").replace("\\", "_")


def prepare_dataset(
    taco_dir: Path,
    output_dir: Path,
    val_split: float = 0.2,
    granularity: str = "supercategory",
    seed: int = 42,
    symlink: bool = False,
) -> PreparedDatasetSummary:
    """Convert a raw TACO directory into a YOLO-ready dataset directory."""
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0.0, 1.0)")

    annotations_path = taco_dir / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(
            f"Expected annotations at {annotations_path}. "
            "Download TACO first or pass the correct --taco-dir."
        )

    with annotations_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    category_id_to_class_index, class_names = build_class_index(
        coco["categories"],
        granularity=granularity,
    )
    annotations_by_image_id = _group_annotations_by_image(coco["annotations"])
    images = list(coco["images"])

    rng = random.Random(seed)
    rng.shuffle(images)
    val_count = int(len(images) * val_split)
    val_image_ids = {image["id"] for image in images[:val_count]}

    _create_output_folders(output_dir)

    train_images = 0
    val_images = 0
    missing_images = 0

    for image in images:
        split = "val" if image["id"] in val_image_ids else "train"
        source_image = taco_dir / image["file_name"]
        if not source_image.is_file():
            missing_images += 1
            continue

        output_name = flatten_image_name(image["file_name"])
        destination_image = output_dir / "images" / split / output_name
        _copy_or_link_image(source_image, destination_image, symlink=symlink)

        label_name = f"{Path(output_name).stem}.txt"
        label_path = output_dir / "labels" / split / label_name
        label_lines = _format_yolo_labels(
            annotations_by_image_id.get(image["id"], []),
            category_id_to_class_index,
            image_width=int(image["width"]),
            image_height=int(image["height"]),
        )
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if split == "val":
            val_images += 1
        else:
            train_images += 1

    yaml_path = _write_data_yaml(output_dir, class_names)
    return PreparedDatasetSummary(
        yaml_path=yaml_path,
        class_names=class_names,
        train_images=train_images,
        val_images=val_images,
        missing_images=missing_images,
    )


def _group_annotations_by_image(annotations: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        grouped.setdefault(image_id, []).append(annotation)
    return grouped


def _format_yolo_labels(
    annotations: Iterable[dict],
    category_id_to_class_index: dict[int, int],
    image_width: int,
    image_height: int,
) -> list[str]:
    labels = []
    for annotation in annotations:
        class_index = category_id_to_class_index[int(annotation["category_id"])]
        center_x, center_y, width, height = coco_bbox_to_yolo(
            annotation["bbox"],
            image_width,
            image_height,
        )
        if width == 0.0 or height == 0.0:
            continue
        labels.append(
            f"{class_index} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )
    return labels


def _create_output_folders(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_or_link_image(source_image: Path, destination_image: Path, symlink: bool) -> None:
    if symlink:
        if not destination_image.exists():
            destination_image.symlink_to(source_image.resolve())
        return

    shutil.copy2(source_image, destination_image)


def _write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / "data.yaml"
    lines = [
        "# Auto-generated by prepare_taco_yolo_dataset.py",
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  - {name}" for name in class_names)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TACO COCO-style annotations into YOLO format."
    )
    parser.add_argument(
        "--taco-dir",
        type=Path,
        required=True,
        help="Directory containing annotations.json and TACO image batches.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training/datasets/taco_yolo"),
        help="Output directory for the YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of images reserved for validation.",
    )
    parser.add_argument(
        "--granularity",
        choices=("supercategory", "category"),
        default="supercategory",
        help="Use TACO supercategories or fine-grained category names.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        taco_dir=args.taco_dir,
        output_dir=args.out,
        val_split=args.val_split,
        granularity=args.granularity,
        seed=args.seed,
        symlink=args.symlink,
    )

    print("[prepare] Dataset ready")
    print(f"[prepare] data.yaml: {summary.yaml_path}")
    print(
        f"[prepare] train={summary.train_images} val={summary.val_images} "
        f"missing_images={summary.missing_images}"
    )
    print(f"[prepare] classes ({len(summary.class_names)}): {summary.class_names}")


if __name__ == "__main__":
    main()
