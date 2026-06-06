# Cat VS Dog ai

# Инструкцыя
source .venv/bin/activate
python main.py --train-more --epochs 50 --extra-data extra-data


# Важные команды

## Проверка фоток

python main.py

- загружает готовую модель из models/cat_dog.keras
- берет картинки из папки photo
- пишет в терминал: кот или собака

## Проверить Другую Папку
python main.py --images photo

## Обучить Модель С Нуля
python main.py --force-train --epochs 12

обнуляет все обучение и начинает его заново

## Продолжить Обучение
python main.py --train-more --epochs 12

- берет уже обученную модель из models/cat_dog.keras
- обучает её ещё 12 эпох
- сохраняет обновленную модель обратно

## Обучить На Дополнительном Датасете
Сначала структура должна быть такая:

добавляем в extra-data еше датасеты

python main.py --train-more --epochs 12 --extra-data extra-data

## Обучить С Нуля На Двух Датасетах
python main.py --force-train --epochs 12 --extra-data extra-data



# Все Аргументы

## --epochs N
Сколько эпох обучать.

## --images ПАПКА
Откуда брать фотки для проверки.

## --force-train
Обучить модель заново с нуля.

## --train-more
Продолжить обучение уже сохраненной модели.

## --extra-data ПАПКА
Добавить свой датасет с папками cat и dog.


# Где Что Хранится
## Код проекта:
cat-dog-ai/main.py

## Зависимости:
cat-dog-ai/requirements.txt

## Готовая обученная модель:
cat-dog-ai/models/cat_dog.keras

## Фотки для проверки по умолчанию:
cat-dog-ai/photo/

## Основной скачанный датасет:
cat-dog-ai/.cache/tensorflow_datasets/

## Скачанные веса MobileNetV2:
cat-dog-ai/.cache/keras/

## Виртуальное окружение Python:
cat-dog-ai/.venv/

## Дополнительный датасет, если создашь:
cat-dog-ai/extra-data/
  cat/
  dog/