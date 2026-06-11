# Первая ИИ-модель на Python: кошки против собак

Цель: сделать первую работающую модель компьютерного зрения, которая получает фотографию и отвечает: `cat` или `dog`.

Лучший старт для новичка: не писать нейросеть с нуля, а взять готовую предобученную модель и дообучить ее на кошках и собаках. Это называется `transfer learning`.

---

## 1. Что ты вообще будешь делать

Ты соберешь маленький ML-проект:

```text
cat-dog-ai/
  data/
    train/
      cats/
      dogs/
    val/
      cats/
      dogs/
    test/
      cats/
      dogs/
  notebooks/
  src/
    train.py
    predict.py
  models/
  first_cat_dog_ai_model.md
```

Что происходит внутри:

1. Берем много картинок кошек и собак.
2. Делим их на `train`, `val`, `test`.
3. Приводим картинки к одному размеру, например `224x224`.
4. Загружаем готовую модель, например `MobileNetV2`.
5. Меняем последний слой под 2 класса: кошка и собака.
6. Обучаем модель.
7. Проверяем точность на картинках, которые модель раньше не видела.
8. Сохраняем модель.
9. Пишем скрипт `predict.py`, который проверяет одну новую фотку.

---

## 2. Виды обучения моделей

### 2.1. Supervised Learning — обучение с учителем

Это твой случай.

У модели есть примеры и правильные ответы:

```text
photo_001.jpg -> cat
photo_002.jpg -> dog
photo_003.jpg -> dog
```

Модель смотрит на картинку, делает предсказание, сравнивает с правильным ответом и постепенно исправляет свои ошибки.

Используется для:

- классификации: кошка или собака, спам или не спам
- регрессии: предсказать цену квартиры, температуру, доход
- распознавания объектов на изображениях

### 2.2. Unsupervised Learning — обучение без учителя

У данных нет готовых ответов.

Модель сама ищет структуру:

- группирует похожие объекты
- находит скрытые темы
- сжимает данные
- ищет аномалии

Пример: дать модели тысячи фото животных без подписей, а она сама разделит их на группы похожих изображений.

### 2.3. Reinforcement Learning — обучение с подкреплением

Модель действует в среде и получает награду или штраф.

Примеры:

- ИИ играет в игру
- робот учится ходить
- агент учится принимать решения

Логика:

```text
действие -> результат -> награда/штраф -> новая стратегия
```

### 2.4. Self-Supervised Learning — самообучение на данных

Модель сама создает себе обучающую задачу из данных.

Примеры:

- закрыть часть текста и предсказать пропущенные слова
- скрыть часть изображения и восстановить ее
- научиться понимать похожесть изображений

Так обучают многие большие модели.

### 2.5. Transfer Learning — перенос обучения

Это самый практичный вариант для первой модели.

Идея: кто-то уже обучил большую модель на миллионах изображений. Она уже умеет видеть формы, края, текстуры, глаза, шерсть, силуэты. Ты берешь эту модель и дообучаешь ее под свою задачу.

Плюсы:

- нужно меньше данных
- обучение быстрее
- качество выше
- проще для новичка

Для кошек и собак можно использовать:

- `MobileNetV2` — легкая и быстрая
- `EfficientNetB0` — часто дает хорошее качество
- `ResNet50` — классика компьютерного зрения

---

## 3. Какие инструменты использовать

### Минимальный стек

Для первой модели лучше использовать `TensorFlow/Keras`, потому что там меньше кода для старта.

Нужно:

- `Python 3.10+`
- `tensorflow`
- `matplotlib`
- `numpy`
- `pillow`
- `scikit-learn`

Установка:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow matplotlib numpy pillow scikit-learn
```

Если хочешь работать через ноутбуки:

```bash
pip install jupyter
jupyter notebook
```

### TensorFlow или PyTorch?

Для старта:

- `TensorFlow/Keras` проще, быстрее начать.
- `PyTorch` гибче и чаще используется в research/современных ML-проектах.

Мой совет: первую модель сделай на `TensorFlow/Keras`, вторую похожую задачу повтори на `PyTorch`.

---

## 4. Где скачать датасет кошек и собак

### Вариант 1: Microsoft Kaggle Cats and Dogs Dataset

Официальная страница Microsoft:

<https://www.microsoft.com/en-us/download/details.aspx?id=54765>

Прямая ссылка на архив:

<https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip>

Размер: примерно `786.7 MB`.

Это хороший вариант для первой модели: много готовых изображений кошек и собак.

### Вариант 2: Kaggle Dogs vs. Cats

Страница соревнования:

<https://www.kaggle.com/c/dogs-vs-cats/data>

Плюсы:

- классический датасет для новичков
- много туториалов
- удобно тренироваться

Минус:

- нужен аккаунт Kaggle
- иногда нужно принять правила соревнования

### Вариант 3: TensorFlow Datasets

Каталог TensorFlow Datasets:

<https://www.tensorflow.org/datasets/catalog/cats_vs_dogs>

Плюс: можно загружать датасет прямо из Python через `tensorflow_datasets`.

Минус: для новичка иногда проще руками видеть папки `cats/` и `dogs/`.

---

## 5. Как правильно подготовить данные

Модель не должна видеть одни и те же картинки во время обучения и проверки.

Деление:

```text
train — 70%
val   — 15%
test  — 15%
```

Что означает:

- `train` — на этих картинках модель учится
- `val` — по этим картинкам ты следишь, не переобучается ли модель
- `test` — финальная честная проверка

Правильная структура папок:

```text
data/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

Важно:

- не клади одну и ту же фотку в разные наборы
- удали битые изображения
- не проверяй модель только на красивых идеальных фото
- добавь разные условия: свет, фон, позы, размеры животных

---

## 6. Базовый тренировочный код на Keras

Создай файл:

```text
src/train.py
```

Код:

```python
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_dataset(split: str, shuffle: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=shuffle,
        seed=SEED,
    )


def main() -> None:
    train_ds = load_dataset("train", shuffle=True)
    val_ds = load_dataset("val", shuffle=False)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ]
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
    )

    model.save(MODEL_DIR / "cat_dog_model.keras")


if __name__ == "__main__":
    main()
```

Запуск:

```bash
python src/train.py
```

---

## 7. Код для проверки одной фотографии

Создай файл:

```text
src/predict.py
```

Код:

```python
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "cat_dog_model.keras"
IMG_SIZE = (224, 224)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/image.jpg")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"File not found: {image_path}")

    model = keras.models.load_model(MODEL_PATH)

    image = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    array = keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    probability = float(model.predict(array, verbose=0)[0][0])
    label = "dog" if probability >= 0.5 else "cat"
    confidence = probability if label == "dog" else 1 - probability

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
```

Запуск:

```bash
python src/predict.py path/to/photo.jpg
```

---

## 8. Как понять, что модель реально учится

Смотри на две метрики:

- `train accuracy` — точность на обучающих данных
- `val accuracy` — точность на проверочных данных

Нормальная ситуация:

```text
train accuracy растет
val accuracy тоже растет
loss падает
```

Плохая ситуация, переобучение:

```text
train accuracy высокая
val accuracy низкая или падает
```

Это значит: модель запомнила тренировочные картинки, но плохо понимает новые.

Что делать при переобучении:

- добавить больше данных
- использовать `data augmentation`
- уменьшить количество эпох
- добавить `Dropout`
- использовать `EarlyStopping`

---

## 9. Что такое основные ML-слова простым языком

`Dataset` — набор данных.

`Label` — правильный ответ, например `cat` или `dog`.

`Feature` — признак. В картинках это пиксели и паттерны, которые модель сама учится находить.

`Model` — функция, которая превращает вход в ответ.

`Training` — процесс обучения модели.

`Epoch` — один полный проход по обучающим данным.

`Batch` — маленькая пачка картинок, которую модель обрабатывает за один шаг.

`Loss` — число ошибки. Чем меньше, тем лучше.

`Accuracy` — доля правильных ответов.

`Validation` — проверка во время обучения.

`Test` — финальный экзамен модели.

`Overfitting` — модель запомнила тренировочные данные, но плохо работает на новых.

`Data augmentation` — искусственное разнообразие данных: повороты, отражения, приближение.

`Transfer learning` — дообучение готовой модели под свою задачу.

---

## 10. Самый правильный план обучения на 2 недели

### День 1-2: Python для ML

Повтори:

- функции
- списки и словари
- работа с файлами через `pathlib`
- виртуальные окружения
- установка пакетов через `pip`

### День 3-4: NumPy и картинки

Разберись:

- что такое массив
- как картинка превращается в числа
- форма массива: `height x width x channels`
- почему RGB имеет 3 канала

### День 5-6: базовая теория ML

Пойми:

- train/val/test
- loss
- accuracy
- overfitting
- classification

### День 7-9: первая модель cats vs dogs

Сделай:

- скачай датасет
- разложи папки
- запусти обучение
- сохрани модель
- проверь на своих фото

### День 10-11: улучшение качества

Добавь:

- `EarlyStopping`
- больше `data augmentation`
- тестовый набор
- графики `loss` и `accuracy`

### День 12-14: мини-проект

Сделай маленькое приложение:

- CLI-скрипт: `python src/predict.py image.jpg`
- или простой веб-интерфейс через `Gradio`
- или Telegram-бот, которому отправляешь фото, а он отвечает `cat/dog`

---

## 11. Что делать после первой модели

Следующие проекты по сложности:

1. Кошка/собака/лиса/волк — классификация на 4 класса.
2. Определение породы собаки.
3. Детектор объектов: найти животное на фото и обвести рамкой.
4. Свой датасет: собрать 500-1000 картинок самому.
5. Веб-приложение на FastAPI или Flask.
6. ML-приложение с интерфейсом на Gradio.
7. Повторить проект на PyTorch.

---

## 12. Главная ошибка новичков

Не начинай с идеи "я сейчас напишу настоящий ChatGPT".

Правильный путь:

```text
маленькая понятная задача -> рабочая модель -> оценка качества -> улучшение -> новый проект
```

Кошки против собак — отличный первый проект, потому что ты сразу изучишь фундамент:

- данные
- классы
- обучение
- проверку качества
- переобучение
- сохранение модели
- инференс на новых данных

---

## 13. Мини-задание для тебя

Сделай проект в таком порядке:

1. Скачай Microsoft Kaggle Cats and Dogs Dataset.
2. Создай структуру `data/train`, `data/val`, `data/test`.
3. Разложи картинки по папкам `cats` и `dogs`.
4. Обучи модель `MobileNetV2` 5 эпох.
5. Проверь 10 своих фото из интернета.
6. Запиши результаты в таблицу:

```text
image | real_label | prediction | confidence | correct
```

Если модель ошиблась, не злись на нее. Смотри на ошибку как инженер: почему она ошиблась? плохой свет, странный ракурс, два животных на фото, игрушка похожа на животное, фото слишком маленькое?

Это и есть настоящее мышление ML-инженера.
