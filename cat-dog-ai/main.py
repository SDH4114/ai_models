import argparse
import os
from pathlib import Path

import certifi


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache"
MODEL_PATH = PROJECT_DIR / "models" / "cat_dog.keras"
SIZE = 224
BATCH_SIZE = 16

CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "tensorflow_datasets").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "keras").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TFDS_DATA_DIR", str(CACHE_DIR / "tensorflow_datasets"))
os.environ.setdefault("KERAS_HOME", str(CACHE_DIR / "keras"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt


def prepare_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = preprocess_input(img)
    return img, label


def load_datasets():
    train, val = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        data_dir=str(CACHE_DIR / "tensorflow_datasets"),
    )

    train = (
        train.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val = (
        val.map(prepare_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train, val


def build_model(use_pretrained=True):
    base_layers = tf.keras.applications.MobileNetV2(
        input_shape=(SIZE, SIZE, 3),
        include_top=False,
        weights="imagenet" if use_pretrained else None,
    )
    base_layers.trainable = False

    model = tf.keras.Sequential(
        [
            base_layers,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model(epochs, use_pretrained=True):
    train, val = load_datasets()
    model = build_model(use_pretrained=use_pretrained)
    model.summary()
    model.fit(train, validation_data=val, epochs=epochs)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")
    return model


def load_or_train_model(epochs, force_train, use_pretrained):
    if MODEL_PATH.exists() and not force_train:
        print(f"Загружаю готовую модель: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)
    return train_model(epochs, use_pretrained=use_pretrained)


def find_images(images_dir):
    image_dir = Path(images_dir).expanduser().resolve()
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for extension in extensions:
        image_paths.extend(image_dir.glob(extension))
    return sorted(image_paths)


def predict_images(model, images_dir, show_plots):
    image_paths = find_images(images_dir)

    if not image_paths:
        print(f"Нет картинок для проверки в папке: {Path(images_dir).resolve()}")
        print("Положи туда JPG/PNG-файлы или укажи другую папку через --images.")
        return

    for image_path in image_paths:
        img = load_img(image_path)
        img_array = img_to_array(img)
        img_resized, _ = prepare_image(img_array, 0)
        img_expanded = np.expand_dims(img_resized, axis=0)

        logit = model.predict(img_expanded, verbose=0)[0][0]
        dog_probability = float(tf.sigmoid(logit).numpy())
        pred_label = "СОБАКА" if dog_probability >= 0.5 else "КОТ"
        print(f"{image_path.name}: {pred_label} ({dog_probability:.2%} dog)")

        if show_plots:
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{image_path.name}: {pred_label} ({dog_probability:.2%} dog)")

    if show_plots:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели cat/dog и проверка картинок.")
    parser.add_argument("--epochs", type=int, default=3, help="Сколько эпох обучать модель.")
    parser.add_argument(
        "--images",
        default=PROJECT_DIR,
        help="Папка с JPG/PNG-картинками для проверки. По умолчанию папка проекта.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Переобучить модель, даже если models/cat_dog.keras уже существует.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не открывать окна matplotlib, только напечатать результат.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Не скачивать ImageNet-веса MobileNetV2. Используй, если интернет/SSL блокирует загрузку.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_or_train_model(
        args.epochs,
        args.force_train,
        use_pretrained=not args.no_pretrained,
    )
    predict_images(model, args.images, show_plots=not args.no_show)


if __name__ == "__main__":
    main()
