from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from aminatron_model import PROJECT_DIR, detect_image, format_detections, save_result_image, summarize_detections


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(input_path: Path) -> list[Path]:
    """Return one image or all supported images inside a folder."""

    if input_path.is_file():
        return [input_path]

    return sorted(
        path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Aminatron object detection.")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(PROJECT_DIR / "photos"),
        help="Image file or folder. Default: photos/.",
    )
    parser.add_argument("--model", default=None, help="Model path/name. Default: model/aminatron.pt.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image.")
    parser.add_argument("--output", default="runs/predict", help="Folder for annotated images.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    image_paths = find_images(input_path)

    if not image_paths:
        print(f"No images found: {input_path}")
        return

    for image_path in image_paths:
        image = Image.open(image_path)
        result = detect_image(
            image,
            model_path=args.model,
            confidence=args.confidence,
            iou=args.iou,
            image_size=args.imgsz,
            max_detections=args.max_det,
        )
        output_path = save_result_image(result, output_dir / image_path.name)
        detections = format_detections(result)
        summary = summarize_detections(detections)

        print(f"\n{image_path.name} -> {output_path}")
        if not detections:
            print("  no objects found")
            continue

        print(f"  summary: {summary}")
        for item in detections:
            print(f"  {item['class']}: {item['confidence'] * 100:.2f}% {item['box']}")


if __name__ == "__main__":
    main()
