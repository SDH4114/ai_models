from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = PROJECT_DIR / "model" / "aminatron.pt"


def resolve_model_path(model_path: str | None = None) -> str:
    """Return the Aminatron model path or fail with a clear setup message."""

    if model_path:
        return str(Path(model_path).expanduser().resolve()) if Path(model_path).exists() else model_path

    if LOCAL_MODEL_PATH.exists():
        return str(LOCAL_MODEL_PATH)

    raise FileNotFoundError(
        f"Aminatron model not found: {LOCAL_MODEL_PATH}. "
        "Create it with: python train.py --save-as-aminatron"
    )


@lru_cache(maxsize=3)
def load_model(model_path: str | None = None) -> YOLO:
    """Load and cache the detector model."""

    from ultralytics import YOLO

    return YOLO(resolve_model_path(model_path))


def detect_image(
    image: Image.Image,
    model_path: str | None = None,
    confidence: float = 0.25,
    iou: float = 0.45,
    image_size: int = 640,
    max_detections: int = 100,
) -> Any:
    """Detect multiple objects on a single image."""

    model = load_model(model_path)
    rgb_image = image.convert("RGB")
    results = model.predict(
        rgb_image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        max_det=max_detections,
        verbose=False,
    )
    return results[0]


def format_detections(result: Any) -> list[dict[str, object]]:
    """Convert YOLO boxes to readable dictionaries."""

    detections = []
    names = result.names

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "class": names[class_id],
                "confidence": round(confidence, 4),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def summarize_detections(detections: list[dict[str, object]]) -> dict[str, int]:
    """Count detected objects by class."""

    return dict(Counter(str(item["class"]) for item in detections))


def render_result_image(result: Any) -> Image.Image:
    """Return a PIL image with bounding boxes drawn by YOLO."""

    # Ultralytics returns a BGR numpy image from plot(), PIL expects RGB.
    annotated_bgr = result.plot()
    return Image.fromarray(annotated_bgr[..., ::-1])


def save_result_image(result: Any, output_path: Path) -> Path:
    """Save an annotated prediction image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_result_image(result).save(output_path)
    return output_path
