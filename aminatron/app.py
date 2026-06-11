from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from io import BytesIO

import gradio as gr
from PIL import Image

from aminatron_model import detect_image, format_detections, render_result_image, summarize_detections


SPACE_MODEL = os.getenv("AMINATRON_MODEL") or None
EMAIL_TO = os.getenv("EMAIL_TO", "mammadov.amin2000@gmail.com")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image to PNG bytes for email attachment."""

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def send_review_email(original: Image.Image, annotated: Image.Image, detections: list[dict[str, object]], summary: dict[str, int]):
    """Send original and processed images to email if SMTP secrets are configured."""

    if not SMTP_USER or not SMTP_PASSWORD:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    message = EmailMessage()
    message["Subject"] = f"Aminatron prediction review - {timestamp}"
    message["From"] = SMTP_USER
    message["To"] = EMAIL_TO
    message.set_content(
        "A new Aminatron Space prediction was processed.\n\n"
        f"Time: {timestamp}\n"
        f"Summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
        f"Detections:\n{json.dumps(detections, ensure_ascii=False, indent=2)}\n"
    )
    message.add_attachment(
        image_to_png_bytes(original),
        maintype="image",
        subtype="png",
        filename="original.png",
    )
    message.add_attachment(
        image_to_png_bytes(annotated),
        maintype="image",
        subtype="png",
        filename="aminatron_result.png",
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(message)
    except Exception as error:
        print(f"Email notification failed: {error}")


def predict(image: Image.Image, confidence: float, iou: float, image_size: int, max_detections: int):
    """Detect objects for the Gradio interface."""

    if image is None:
        return None, {}, []

    result = detect_image(
        image,
        model_path=SPACE_MODEL,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        max_detections=max_detections,
    )
    detections = format_detections(result)
    summary = summarize_detections(detections)
    annotated_image = render_result_image(result)
    rows = [
        [item["class"], f"{item['confidence'] * 100:.2f}%", item["box"]]
        for item in detections
    ]

    send_review_email(image, annotated_image, detections, summary)
    return annotated_image, summary, rows


with gr.Blocks(title="Aminatron") as demo:
    gr.Markdown(
        """
        # Aminatron

        Aminatron is a multi-object detection model. Upload one image and it can find
        several COCO objects at once: people, cats, dogs, birds, cows, cars, chairs and more.
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload image")
        image_output = gr.Image(type="pil", label="Detected objects")

    with gr.Row():
        confidence = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
        iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU")
        image_size = gr.Dropdown([416, 512, 640, 768], value=640, label="Image size")
        max_detections = gr.Slider(1, 100, value=50, step=1, label="Max detections")

    detect_button = gr.Button("Detect")
    summary_output = gr.JSON(label="Summary")
    detections_output = gr.Dataframe(
        headers=["Class", "Confidence", "Box [x1, y1, x2, y2]"],
        label="Detections",
        interactive=False,
    )

    detect_button.click(
        fn=predict,
        inputs=[image_input, confidence, iou, image_size, max_detections],
        outputs=[image_output, summary_output, detections_output],
    )


if __name__ == "__main__":
    demo.launch()
