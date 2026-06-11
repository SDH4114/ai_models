from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_PATH = PROJECT_DIR / "model" / "aminatron.pt"
DEFAULT_SOURCE_MODEL = "yolo11m.pt"

# COCO class names in the same zero-based order used by YOLO.
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def convert_box(box: list[float], width: int, height: int, bbox_format: str) -> tuple[float, float, float, float]:
    """Convert COCO box to normalized YOLO xywh."""

    a, b, c, d = [float(value) for value in box]
    if bbox_format == "xywh":
        x1, y1, box_width, box_height = a, b, c, d
        x2, y2 = x1 + box_width, y1 + box_height
    else:
        x1, y1, x2, y2 = a, b, c, d

    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)

    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2
    return x_center / width, y_center / height, box_width / width, box_height / height


def export_split(dataset, split_name: str, output_dir: Path, max_items: int | None, bbox_format: str) -> int:
    """Export one Hugging Face COCO split to YOLO files."""

    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for index, item in enumerate(dataset):
        if max_items is not None and exported >= max_items:
            break

        image = item["image"].convert("RGB")
        width = int(item.get("width") or image.width)
        height = int(item.get("height") or image.height)
        objects = item.get("objects") or {}
        categories = objects.get("category") or []
        boxes = objects.get("bbox") or []

        label_lines = []
        for category, box in zip(categories, boxes):
            class_id = int(category)
            if class_id < 0 or class_id >= len(COCO_NAMES):
                continue

            x_center, y_center, box_width, box_height = convert_box(box, width, height, bbox_format)
            if box_width <= 0 or box_height <= 0:
                continue

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        if not label_lines:
            continue

        stem = f"{split_name}_{index:08d}"
        image.save(images_dir / f"{stem}.jpg", quality=95)
        (labels_dir / f"{stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        exported += 1

    return exported


def write_data_yaml(output_dir: Path) -> Path:
    """Create YOLO data.yaml for the exported COCO dataset."""

    import yaml

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(COCO_NAMES)},
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_coco(output_dir: Path, max_train: int | None, max_val: int | None, overwrite: bool, bbox_format: str) -> Path:
    """Download COCO from Hugging Face and export it to YOLO format."""

    from datasets import load_dataset

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("detection-datasets/coco")
    train_count = export_split(dataset["train"], "train", output_dir, max_train, bbox_format)
    val_count = export_split(dataset["val"], "val", output_dir, max_val, bbox_format)
    yaml_path = write_data_yaml(output_dir)

    print(f"Exported train images: {train_count}")
    print(f"Exported val images: {val_count}")
    print(f"Data YAML: {yaml_path}")
    return yaml_path


def save_final_model(best_model_path: Path):
    """Copy the best training checkpoint to model/aminatron.pt."""

    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model was not found: {best_model_path}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, MODEL_OUTPUT_PATH)
    print(f"Aminatron model saved: {MODEL_OUTPUT_PATH}")


def normalize_limit(value: int) -> int | None:
    """Treat 0 or negative limits as unlimited export."""

    return None if value <= 0 else value


def parse_args():
    parser = argparse.ArgumentParser(description="Train/fine-tune Aminatron object detector on COCO.")
    parser.add_argument("--prepare-only", action="store_true", help="Only download/export COCO to YOLO format.")
    parser.add_argument("--save-as-aminatron", action="store_true", help="Save base model as model/aminatron.pt without training.")
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL, help="Source model for --save-as-aminatron.")
    parser.add_argument("--data-dir", default="coco_yolo", help="Local YOLO dataset folder.")
    parser.add_argument("--model", default=str(MODEL_OUTPUT_PATH), help="Model to fine-tune. Default: model/aminatron.pt.")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Lower it if memory is not enough.")
    parser.add_argument("--max-train", type=int, default=5000, help="Limit train images for local experiments.")
    parser.add_argument("--max-val", type=int, default=1000, help="Limit validation images for local experiments.")
    parser.add_argument("--bbox-format", choices=["xyxy", "xywh"], default="xyxy", help="Box format in HF dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild exported COCO dataset.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.save_as_aminatron:
        from ultralytics import YOLO

        model = YOLO(args.source_model)
        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_OUTPUT_PATH)
        print(f"Aminatron base model saved: {MODEL_OUTPUT_PATH}")
        return

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_yaml = data_dir / "data.yaml"

    if args.overwrite or not data_yaml.exists():
        data_yaml = prepare_coco(
            output_dir=data_dir,
            max_train=normalize_limit(args.max_train),
            max_val=normalize_limit(args.max_val),
            overwrite=args.overwrite,
            bbox_format=args.bbox_format,
        )

    if args.prepare_only:
        return

    # Fine-tuning from pretrained weights is intentional: it keeps existing COCO knowledge
    # and improves the model without the degradation risk of random training from scratch.
    from ultralytics import YOLO

    if not Path(args.model).expanduser().exists() and args.model == str(MODEL_OUTPUT_PATH):
        raise FileNotFoundError(
            f"Aminatron model not found: {MODEL_OUTPUT_PATH}. "
            "Create it first with: python train.py --save-as-aminatron"
        )

    model = YOLO(args.model)
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="aminatron",
        pretrained=True,
    )

    save_dir = getattr(train_result, "save_dir", None) or getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Could not find Ultralytics training save_dir.")

    best_model_path = Path(save_dir) / "weights" / "best.pt"
    save_final_model(best_model_path)


if __name__ == "__main__":
    main()
