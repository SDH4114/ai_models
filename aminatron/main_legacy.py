import argparse
import os
from pathlib import Path

import certifi


# абсолютный путь к папке
PROJECT_DIR = Path(__file__).resolve().parent

# весь кеш проекта храним тут чтоб TensorFlow не сохранял файлы по системе
CACHE_DIR = PROJECT_DIR / ".cache"

# Сюда сохраняется обученная модель
MODEL_PATH = PROJECT_DIR / "models" / "cat_dog.keras"

# папка для фоток
DEFAULT_IMAGES_DIR = PROJECT_DIR / "photo"

# размер фото для MobileNetV2
SIZE = 224

# сколько картинок модель обрабатывает за один шаг обучения
BATCH_SIZE = 16

# фиксируем seed чтобы разбиение дополнительного датасета было повторяемым
SEED = 42

# создаем нужные папки заранее чтобы кеш и веса сохранялись в понятные места
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "tensorflow_datasets").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "keras").mkdir(parents=True, exist_ok=True)

# переменные окружения должны быть заданы до импорта TensorFlow
# и направляют кеш датасетов кеш Keras и SSL-сертификаты в правильные места
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TFDS_DATA_DIR", str(CACHE_DIR / "tensorflow_datasets"))
os.environ.setdefault("KERAS_HOME", str(CACHE_DIR / "keras"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# уменьшает количество технических логов TensorFlow в терминале
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# TensorFlow нормально импортируется только после настройки
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def prepare_image(img, label):
    """делаем картинку 224 на 224 для MobileNetV2"""

    # TensorFlow работает с изображениями как с числами float32
    img = tf.cast(img, tf.float32)

    # делаем все картинки одного размера для удобства обучения
    img = tf.image.resize(img, (SIZE, SIZE))

    # preprocess_input приводит значения пикселей к формату MobileNetV2
    img = preprocess_input(img)

    # label возвращаем без изменений 0/1 для кот/собака
    label = tf.cast(label, tf.float32)
    return img, label


def load_extra_dataset(extra_data_dir, subset):
    """загружает дополнительный датасет из папок cat и dog"""

    # структура должна быть такой:
    # extra-data/cat/*.jpg
    # extra-data/dog/*.jpg
    # class_names фиксирует порядок: cat = 0, dog = 1
    return tf.keras.utils.image_dataset_from_directory(
        extra_data_dir,
        labels="inferred",
        label_mode="int",
        class_names=["cat", "dog"],
        color_mode="rgb",
        batch_size=None,
        image_size=(SIZE, SIZE),
        validation_split=0.2,
        subset=subset,
        seed=SEED,
    )


def load_datasets(extra_data_dir=None):
    """Загружает cats_vs_dogs и при необходимости добавляет второй датасет."""

    # tfds скачивает и кеширует датасет cats_vs_dogs
    # 80% идут на обучение 20% на проверку качества во время обучения и сброс
    train, val = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        data_dir=str(CACHE_DIR / "tensorflow_datasets"),
    )

    # map — обработка картинок shuffle — перемешивание batch — пачки, prefetch — ускорение
    train = train.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Validation pipeline почти такой же но без shuffle: проверочные данные не нужно мешать
    val = val.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)

    # если передали --extra-data, добавляем свои картинки к основному датасету
    if extra_data_dir:
        extra_train = load_extra_dataset(extra_data_dir, subset="training")
        extra_val = load_extra_dataset(extra_data_dir, subset="validation")
        extra_train = extra_train.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        extra_val = extra_val.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.concatenate(extra_train)
        val = val.concatenate(extra_val)

    train = train.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val = val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train, val


def build_model():
    """создаем модель"""

    # MobileNetV2 — уже обученная на ImageNet модель
    # include_top=False убирает ее старый классификатор потому что нам нужны только кот/собака
    base_layers = tf.keras.applications.MobileNetV2(
        input_shape=(SIZE, SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # замораживаем базовую модель обучаем только наш маленький классификатор сверху
    # это и быстрее и требует меньше данных
    base_layers.trainable = False

    # Sequential это каждый слой идет друг за другом
    model = tf.keras.Sequential(
        [
            # создание базовой модели
            base_layers,

            # превращает карту признаков в один компактный вектор
            GlobalAveragePooling2D(),

            # Dropout отключаеь рандомно некоторые нейроны
            Dropout(0.2),

            # 0 кот    1 собака
            Dense(1),
        ]
    )

    # from_logits=True нужен потому что Dense(1) возвращает сырой logit а не проценты
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model(epochs, extra_data_dir=None, start_from_saved=False):
    """Обучение модели и сохранение ее на диск"""

    train, val = load_datasets(extra_data_dir)

    # если нужно продолжить обучение, грузим уже сохраненную модель
    if start_from_saved and MODEL_PATH.exists():
        print(f"Продолжаю обучение модели: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = build_model()

    # показывает структуру модели в терминале
    model.summary()

    # запускает обучение на train и проверку на val
    model.fit(train, validation_data=val, epochs=epochs)

    # создает папку models
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # сохраняем модель, чтобы не обучать ее заново
    model.save(MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    return model


def load_or_train_model(epochs, force_train, train_more, extra_data_dir):
    """загружает готовую модель или обучает новую"""

    # --train-more продолжает обучение сохраненной модели еще на N эпох
    if train_more:
        if not MODEL_PATH.exists():
            print("Готовой модели нет, поэтому обучаю новую модель с нуля.")
        return train_model(epochs, extra_data_dir, start_from_saved=True)

    # --force-train удаляет смысл старой модели: обучение начинается с нуля
    if force_train:
        return train_model(epochs, extra_data_dir)

    # если модель сушествует и не просит переобучение просто грузим файл
    if MODEL_PATH.exists():
        print(f"Загружаю готовую модель: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

    # если модели нет запускаем обучение
    return train_model(epochs, extra_data_dir)


def find_images(images_dir):
    """находит JPG/PNG-картинки в указанной папке"""

    # expanduser поддерживает путь вида ~/photo, resolve делает путь абсолютным
    image_dir = Path(images_dir).expanduser().resolve()

    # поддерживаем форматы изображений
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []

    # собираем все подходящие файлы
    for extension in extensions:
        image_paths.extend(image_dir.glob(extension))

    return sorted(image_paths)


def predict_images(model, images_dir):
    """проверяем картинки"""

    image_paths = find_images(images_dir)

    # если нет картинок пишем это
    if not image_paths:
        print(f"Нет картинок для проверки в папке: {Path(images_dir).resolve()}")
        print("Положи туда JPG/PNG-файлы или укажи другую папку через --images.")
        return

    for image_path in image_paths:
        # загружаем картинку через Keras или Pillow
        img = load_img(image_path)

        # переводим картинку в numpy массив
        img_array = img_to_array(img)

        # используем ту же обработку, что и при обучении
        img_resized, _ = prepare_image(img_array, 0)

        # одна картинка в пачке
        img_expanded = np.expand_dims(img_resized, axis=0)

        # predict возвращает logit чем больше logit значт там должна быть собака собака
        logit = model.predict(img_expanded, verbose=0)[0][0]

        # sigmoid превращает logit в вероятность от 0 до 1
        dog_probability = float(tf.sigmoid(logit).numpy())

        # больше или равно 50 то сабака если нет кот
        pred_label = "Собака" if dog_probability >= 0.5 else "Кот"
        print(f"{image_path.name}: {pred_label} ({dog_probability:.2%} dog)")


def parse_args():
    """аргументы командной строки"""

    parser = argparse.ArgumentParser(description="Обучение модели cat/dog и проверка картинок")

    # python main.py --epochs 12
    parser.add_argument("--epochs", type=int, default=3, help="Сколько эпох обучать модель")

    # python main.py --images photo
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGES_DIR,
        help="Папка с JPG/PNG-картинками для проверки. По умолчанию: photo",
    )

    # python main.py --force-train --epochs 12
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Переобучить модель",
    )

    # python main.py --train-more --epochs 12
    parser.add_argument(
        "--train-more",
        action="store_true",
        help="Продолжить обучение сохраненной модели еще на указанное число эпох",
    )

    # python main.py --force-train --epochs 12 --extra-data extra-data
    parser.add_argument(
        "--extra-data",
        default=None,
        help="Дополнительный датасет с папками cat и dog",
    )
    return parser.parse_args()


def main():
    """запуск програмы"""

    # читаем параметры из терминала
    args = parse_args()

    # берем готовую модель обучаем новую или продолжаем обучение
    model = load_or_train_model(
        args.epochs,
        args.force_train,
        args.train_more,
        args.extra_data,
    )

    # проверяем картинки из папки
    predict_images(model, args.images)


# python main.py
if __name__ == "__main__":
    main()
