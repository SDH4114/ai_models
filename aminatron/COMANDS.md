**Самые нужные команды:**

cd /Users/aminmammadov/aiwork/models/aminatron
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

**Сохранить базовую COCO-модель как Aminatron:**
python train.py --save-as-aminatron

**Более мощная версия с большим числом параметров:**
python train.py --save-as-aminatron --model yolo11m.pt

**Просто запустить на ноуте:**
python predict.py

**Фотки для этого клади сюда:**
aminatron/photos/

**Проверить конкретную фотку:**
python predict.py path/to/image.jpg

**Скачать и подготовить COCO subset:**
python train.py --prepare-only --max-train 5000 --max-val 1000 --overwrite

**Дообучить Aminatron:**
python train.py --epochs 10

**Продолжить обучение уже своей модели:**
python train.py --model model/aminatron.pt --epochs 5

**Запустить веб-демо:**
python app.py

**Результаты с рамками сохраняются сюда:**
aminatron/runs/predict/