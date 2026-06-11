import argparse
import json
import os
import warnings
from pathlib import Path

import certifi


# Абсолютный путь к папке проекта
PROJECT_DIR = Path(__file__).resolve().parent

# Весь кеш проекта хранится тут
CACHE_DIR = PROJECT_DIR / ".cache"

# Модель
MODEL_PATH = PROJECT_DIR / "models" / "aminatron.keras"

LEGACY_MODEL_PATHS = [
    PROJECT_DIR / "models" / "cat_dog_human.keras",
]

# Рядом с моделью сохраняем список классов, чтобы при проверке знать порядок выходов.
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

# Папка с картинками, которые программа проверяет по умолчанию.
DEFAULT_IMAGES_DIR = PROJECT_DIR / "photo"

# Классы модели. Порядок важен: номер класса = позиция в этом списке.
CLASS_NAMES = ["cat", "dog", "human"]

# Красивые названия для вывода в терминале.
CLASS_LABELS = {
    "cat": "Кот",
    "dog": "Собака",
    "human": "Человек",
}

# MobileNetV2 работает с изображениями 224x224.
SIZE = 224

# Количество картинок в одном batch во время обучения.
BATCH_SIZE = 16

# Seed нужен, чтобы разбиение дополнительного датасета было повторяемым.
SEED = 42


# Создаем папки кеша заранее.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "tensorflow_datasets").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "keras").mkdir(parents=True, exist_ok=True)

# Эти переменные должны быть заданы до импорта TensorFlow.
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TFDS_DATA_DIR", str(CACHE_DIR / "tensorflow_datasets"))
os.environ.setdefault("KERAS_HOME", str(CACHE_DIR / "keras"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Pillow иногда предупреждает о прозрачности PNG. Мы принудительно грузим RGB, поэтому это безопасно скрыть.
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
)


# TensorFlow импортируется после настройки окружения выше.
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def prepare_image(img, label):
    """Приводит картинку и label к формату, который нужен модели."""

    # Картинку переводим в float32, потому что нейросети работают с числами.
    img = tf.cast(img, tf.float32)

    # Все изображения должны иметь одинаковый размер.
    img = tf.image.resize(img, (SIZE, SIZE))

    # MobileNetV2 ожидает специальную нормализацию пикселей.
    img = preprocess_input(img)

    # SparseCategoricalCrossentropy ожидает целые номера классов: 0, 1, 2.
    label = tf.cast(label, tf.int32)
    return img, label


def validate_extra_dataset(extra_data_dir):
    """Проверяет, что дополнительный датасет содержит папки cat, dog и human."""

    if not extra_data_dir:
        return None

    data_dir = Path(extra_data_dir).expanduser().resolve()
    missing_dirs = [class_name for class_name in CLASS_NAMES if not (data_dir / class_name).is_dir()]
    if missing_dirs:
        missing = ", ".join(missing_dirs)
        raise FileNotFoundError(
            f"В дополнительном датасете нет папок: {missing}. "
            f"Нужна структура: {data_dir}/cat, {data_dir}/dog, {data_dir}/human"
        )
    return data_dir


def load_extra_dataset(extra_data_dir, subset):
    """Загружает дополнительный датасет из папок cat, dog и human."""

    # Структура файлов
    # extra-data/cat/*.jpg
    # extra-data/dog/*.jpg
    # extra-data/human/*.jpg
    return tf.keras.utils.image_dataset_from_directory(
        extra_data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=None,
        image_size=(SIZE, SIZE),
        validation_split=0.2,
        subset=subset,
        seed=SEED,
    )


def load_datasets(extra_data_dir=None):
    """Загружает cats_vs_dogs и добавляет extra-data, если он указан."""

    # Встроенный датасет дает только cat/dog:
    # cat = 0, dog = 1. Эти номера совпадают с CLASS_NAMES.
    train, val = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        data_dir=str(CACHE_DIR / "tensorflow_datasets"),
    )

    # Приводим основной датасет к формату MobileNetV2.
    train = train.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Если есть extra-data, добавляем его к основному датасету.
    # Именно extra-data должен содержать примеры human, потому что cats_vs_dogs людей не содержит.
    if extra_data_dir:
        extra_train = load_extra_dataset(extra_data_dir, subset="training")
        extra_val = load_extra_dataset(extra_data_dir, subset="validation")
        extra_train = extra_train.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        extra_val = extra_val.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.concatenate(extra_train)
        val = val.concatenate(extra_val)

    # shuffle перемешивает обучение, batch собирает картинки в пачки, prefetch ускоряет загрузку.
    train = train.shuffle(1000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val = val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train, val


def build_model(fine_tune_layers):
    """Создает multi-class модель cat/dog/human на базе MobileNetV2."""

    # MobileNetV2 уже умеет видеть общие признаки: формы, текстуры, части объектов.
    # include_top=False убирает старую ImageNet-голову, потому что мы делаем свои 3 класса.
    base_layers = tf.keras.applications.MobileNetV2(
        input_shape=(SIZE, SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Для роста качества и числа обучаемых параметров размораживаем последние слои MobileNetV2.
    # Раньше обучался только Dense-слой, теперь модель может подстроить часть feature extractor.
    base_layers.trainable = True
    for layer in base_layers.layers[:-fine_tune_layers]:
        layer.trainable = False

    # BatchNormalization лучше держать замороженным при fine-tuning на небольших датасетах.
    for layer in base_layers.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model = tf.keras.Sequential(
        [
            # Базовая часть достает признаки из изображения.
            base_layers,

            # Сжимает карту признаков в один вектор.
            GlobalAveragePooling2D(),

            # Dropout снижает переобучение.
            Dropout(0.3),

            # Один выход на каждый класс: cat, dog, human.
            Dense(len(CLASS_NAMES)),
        ]
    )

    # Для нескольких классов нужен SparseCategoricalCrossentropy.
    # from_logits=True потому что Dense выдает сырые числа, а не проценты.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def save_class_names():
    """Сохраняет список классов рядом с моделью."""

    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")


def load_class_names():
    """Загружает список классов для предсказания."""

    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return CLASS_NAMES


def get_existing_model_path():
    """Возвращает путь к новой модели Aminatron или к старому совместимому файлу."""

    if MODEL_PATH.exists():
        return MODEL_PATH

    for legacy_path in LEGACY_MODEL_PATHS:
        if legacy_path.exists():
            return legacy_path

    return None


def build_callbacks():
    """Создает callbacks, чтобы сохранять лучшую модель, а не последнюю."""

    return [
        # Если val_loss перестал улучшаться, обучение остановится само.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),

        # Сохраняет только лучшую эпоху по val_loss.
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),

        # Если модель застряла, уменьшаем learning rate и даем ей тоньше настроиться.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]


def train_model(epochs, extra_data_dir=None, start_from_saved=False, fine_tune_layers=30):
    """Обучает новую модель или продолжает обучение сохраненной."""

    extra_data_dir = validate_extra_dataset(extra_data_dir)
    if not extra_data_dir:
        print("Внимание: human-класс не будет нормально обучен без --extra-data с папкой human.")

    train, val = load_datasets(extra_data_dir)

    # Продолжаем обучение, если есть Aminatron или старый совместимый файл cat_dog_human.keras.
    existing_model_path = get_existing_model_path()
    if start_from_saved and existing_model_path:
        print(f"Продолжаю обучение модели: {existing_model_path}")
        model = tf.keras.models.load_model(existing_model_path)
    else:
        model = build_model(fine_tune_layers)

    model.summary()

    # Callbacks защищают от ситуации, когда лучшая эпоха была раньше, а последняя хуже.
    model.fit(train, validation_data=val, epochs=epochs, callbacks=build_callbacks())

    # EarlyStopping восстановил лучшие веса в памяти, поэтому сохраняем их явно.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_class_names()
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Классы сохранены: {CLASS_NAMES_PATH}")
    return model


def load_or_train_model(epochs, force_train, train_more, extra_data_dir, fine_tune_layers):
    """Загружает готовую модель, продолжает обучение или обучает новую."""

    if train_more:
        if not get_existing_model_path():
            print("Готовой 3-классовой модели нет, поэтому обучаю новую модель с нуля.")
        return train_model(
            epochs,
            extra_data_dir,
            start_from_saved=True,
            fine_tune_layers=fine_tune_layers,
        )

    if force_train:
        return train_model(epochs, extra_data_dir, fine_tune_layers=fine_tune_layers)

    existing_model_path = get_existing_model_path()
    if existing_model_path:
        print(f"Загружаю готовую модель: {existing_model_path}")
        model = tf.keras.models.load_model(existing_model_path)

        # Если модель была сохранена под старым именем, один раз пересохраняем ее как Aminatron.
        if existing_model_path != MODEL_PATH:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_PATH)
            save_class_names()
            print(f"Модель переименована и сохранена: {MODEL_PATH}")

        return model

    return train_model(epochs, extra_data_dir, fine_tune_layers=fine_tune_layers)


def find_images(images_dir):
    """Находит JPG/PNG-картинки в указанной папке."""

    image_dir = Path(images_dir).expanduser().resolve()
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for extension in extensions:
        image_paths.extend(image_dir.glob(extension))
    return sorted(image_paths)


def predict_images(model, images_dir):
    """Проверяет картинки и печатает самый вероятный класс."""

    class_names = load_class_names()
    image_paths = find_images(images_dir)

    if not image_paths:
        print(f"Нет картинок для проверки в папке: {Path(images_dir).resolve()}")
        print("Положи туда JPG/PNG-файлы или укажи другую папку через --images.")
        return

    for image_path in image_paths:
        # RGB убирает проблемы с PNG-прозрачностью и гарантирует 3 канала цвета.
        img = load_img(image_path, color_mode="rgb")
        img_array = img_to_array(img)
        img_resized, _ = prepare_image(img_array, 0)

        # Модель ожидает batch, поэтому добавляем ось: одна картинка в пачке.
        img_expanded = np.expand_dims(img_resized, axis=0)

        # На выходе теперь не один logit, а список logits: cat, dog, human.
        logits = model.predict(img_expanded, verbose=0)[0]

        # Softmax превращает logits в вероятности по всем классам.
        probabilities = tf.nn.softmax(logits).numpy()
        best_index = int(np.argmax(probabilities))
        if best_index >= len(class_names):
            raise ValueError(
                "Количество выходов модели не совпадает со списком классов. "
                "Переобучи модель или проверь models/class_names.json."
            )

        best_class = class_names[best_index]
        best_label = CLASS_LABELS.get(best_class, best_class)
        best_probability = float(probabilities[best_index])

        print(f"{image_path.name}: {best_label} ({best_probability:.2%})")


def parse_args():
    """Описывает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Обучение модели Aminatron и проверка картинок")

    parser.add_argument("--epochs", type=int, default=3, help="Сколько эпох обучать модель")
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGES_DIR,
        help="Папка с JPG/PNG-картинками для проверки. По умолчанию: photo",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Обучить новую 3-классовую модель с нуля",
    )
    parser.add_argument(
        "--train-more",
        action="store_true",
        help="Продолжить обучение сохраненной 3-классовой модели",
    )
    parser.add_argument(
        "--extra-data",
        default=None,
        help="Дополнительный датасет с папками cat, dog и human",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=30,
        help="Сколько последних слоев MobileNetV2 разморозить для обучения",
    )
    return parser.parse_args()


def main():
    """Главная точка входа программы."""

    args = parse_args()
    model = load_or_train_model(
        args.epochs,
        args.force_train,
        args.train_more,
        args.extra_data,
        args.fine_tune_layers,
    )
    predict_images(model, args.images)


if __name__ == "__main__":
    main()
