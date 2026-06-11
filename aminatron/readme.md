---
title: Aminatron
emoji: A
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.12
pinned: false
license: mit
---

# Aminatron

Aminatron is a multi-object detection model.

It detects several objects on one image and returns object class, confidence, bounding box and an annotated image.

## Folders

```text
aminatron/
  photos/              default input photos
  model/aminatron.pt   main Aminatron model
  runs/predict/        output images with boxes
  coco_yolo/           prepared COCO dataset
```

`model/aminatron.pt` is the main model everywhere. No `yolo11s.pt` or `yolo11m.pt` files are needed after `aminatron.pt` is created.

## Install

```bash
cd /Users/aminmammadov/aiwork/models/aminatron
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create Aminatron Model

Create `model/aminatron.pt` from YOLO11m once:

```bash
python train.py --save-as-aminatron
```

This uses `yolo11m.pt` as source and saves it as:

```text
model/aminatron.pt
```

## Run On Laptop

Put images into:

```text
aminatron/photos/
```

Run:

```bash
python predict.py
```

Run on one custom image:

```bash
python predict.py path/to/image.jpg
```

Results are saved to:

```text
runs/predict/
```

## Prepare COCO

Prepare a smaller COCO subset:

```bash
python train.py --prepare-only --max-train 5000 --max-val 1000 --overwrite
```

Prepare full COCO:

```bash
python train.py --prepare-only --max-train 0 --max-val 0 --overwrite
```

`0` means no limit.

## Fine-Tune Aminatron

Fine-tune the existing `model/aminatron.pt`:

```bash
python train.py --epochs 10
```

Continue training later:

```bash
python train.py --epochs 5
```

Both commands use `model/aminatron.pt` by default and save the best result back to `model/aminatron.pt`.

## Web Demo

```bash
python app.py
```

## Hugging Face Space Email Review

The Space can email the original uploaded image and the processed result image to:

```text
mammadov.amin2000@gmail.com
```

For this to work, add these Hugging Face Space Secrets:

```text
SMTP_USER=your_gmail_address@gmail.com
SMTP_PASSWORD=your_gmail_app_password
EMAIL_TO=mammadov.amin2000@gmail.com
```

For Gmail, `SMTP_PASSWORD` must be a Gmail App Password, not your normal Gmail password.

Optional secrets:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
```

If SMTP secrets are missing, the Space still works, but photos are not emailed.

## Most Needed Commands

```bash
cd /Users/aminmammadov/aiwork/models/aminatron
source .venv/bin/activate
python train.py --save-as-aminatron
python predict.py
python train.py --prepare-only --max-train 5000 --max-val 1000 --overwrite
python train.py --epochs 10
python train.py --epochs 5
python app.py
```
