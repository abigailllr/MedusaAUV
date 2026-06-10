import os
import random
from typing import Any
import yaml
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from wandb.integration.keras import WandbMetricsLogger

BASE = "/content/dataset"
SPLITS = ["train", "val", "test"]
CLASSES = ["jellyfish", "no_jellyfish"]

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

IMG_SIZE = CFG["img_size"]
BATCH_SIZE = CFG["batch_size"]
EPOCHS = CFG["epochs"]
THRESHOLD = CFG["threshold"]


def _fetch_fathomnet(concepts, max_samples):
    try:
        from fathomnet.api import images as fn_images
    except ImportError:
        raise ImportError("Run: pip install fathomnet")

    all_records = []
    for concept in concepts:
        results = fn_images.find_by_concept(concept)
        print(f"  FathomNet '{concept}': {len(results)} images")
        all_records.extend(results)

    seen, unique = set(), []
    for r in all_records:
        if r.uuid not in seen:
            seen.add(r.uuid)
            unique.append(r)

    random.shuffle(unique)
    return unique[:max_samples]


def download_and_split(concepts, class_name, max_samples, ratios=(0.7, 0.15, 0.15)):
    records = _fetch_fathomnet(concepts, max_samples)
    n = len(records)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    split_map = {
        "train": records[:train_end],
        "val":   records[train_end:val_end],
        "test":  records[val_end:]
    }
    for split, batch in split_map.items():
        print(f"  Downloading {len(batch)} → {split}/{class_name}")
        for record in batch:
            ext = record.url.split(".")[-1].split("?")[0]
            if ext.lower() not in ["jpg", "jpeg", "png"]:
                ext = "jpg"
            dest = f"{BASE}/{split}/{class_name}/{record.uuid}.{ext}"
            try:
                urllib.request.urlretrieve(record.url, dest)
            except Exception as e:
                print(f"    skip {record.uuid}: {e}")


def build_folder_structure():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(f"{BASE}/{split}/{cls}", exist_ok=True)


def white_balance(images):
    means = tf.reduce_mean(images, axis=[1, 2], keepdims=True)
    gray = tf.reduce_mean(means, axis=-1, keepdims=True)
    scaled = images * (gray / (means + 1e-6))
    return tf.clip_by_value(scaled, 0.0, 255.0)


def color_jitter(images):
    x = images / 255.0
    x = tf.image.random_hue(x, 0.05)
    x = tf.image.random_saturation(x, 0.7, 1.3)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0


def load_datasets():
    normalize = tf.keras.layers.Rescaling(1.0 / 255)

    augment = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    def prep_train(ds):
        return ds.map(lambda x, y: (normalize(color_jitter(augment(white_balance(x), training=True))), y))

    def prep(ds):
        return ds.map(lambda x, y: (normalize(white_balance(x)), y))

    train_ds = prep_train(tf.keras.utils.image_dataset_from_directory(
        f"{BASE}/train", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    ))
    val_ds = prep(tf.keras.utils.image_dataset_from_directory(
        f"{BASE}/val", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
    ))
    test_ds = prep(tf.keras.utils.image_dataset_from_directory(
        f"{BASE}/test", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False
    ))
    return train_ds, val_ds, test_ds


def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(CFG["dropout1"])(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(CFG["dropout2"])(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fine_tune(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    for layer in model.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def evaluate(model, test_ds):
    y_true, y_pred, y_scores = [], [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds.flatten())
        y_pred.extend((preds > THRESHOLD).astype(int).flatten())
        y_true.extend(labels.numpy())

    report: Any = classification_report(y_true, y_pred,
              target_names=["no_jellyfish", "jellyfish"], output_dict=True)
    print(classification_report(y_true, y_pred,
          target_names=["no_jellyfish", "jellyfish"]))

    wandb.log({
        "test_accuracy": report["accuracy"],
        "test_precision_jellyfish": report["jellyfish"]["precision"],
        "test_recall_jellyfish": report["jellyfish"]["recall"],
        "test_f1_jellyfish": report["jellyfish"]["f1-score"],
    })

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                xticklabels=["No Jellyfish", "Jellyfish"],
                yticklabels=["No Jellyfish", "Jellyfish"])
    ax.set_ylabel("Tatsächlich")
    ax.set_xlabel("Vorhergesagt")
    ax.set_title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.savefig("confusion_matrix.png")
    plt.show()


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("jellyfish_model.tflite", "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Modellgroesse: {size_kb:.1f} KB")
    wandb.log({"model_size_kb": size_kb})
    wandb.save("jellyfish_model.tflite")


if __name__ == "__main__":
    wandb.init(project="medusa-auv", config=CFG)

    build_folder_structure()

    print("\nDownloading jellyfish images from FathomNet...")
    download_and_split(CFG["fathomnet_positive_concepts"], "jellyfish", CFG["max_samples"])

    print("\nDownloading negative images from FathomNet...")
    download_and_split(CFG["fathomnet_negative_concepts"], "no_jellyfish", CFG["max_samples"])

    counts = {}
    for split in SPLITS:
        for cls in CLASSES:
            count = len(os.listdir(f"{BASE}/{split}/{cls}"))
            counts[f"{split}_{cls}"] = count
            print(f"{split}/{cls}: {count} images")
    wandb.log(counts)

    train_ds, val_ds, test_ds = load_datasets()
    model = build_model()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)
    wandb_logger = WandbMetricsLogger()

    print("\n--- Phase 1: Transfer Learning ---")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=[early_stop, checkpoint, wandb_logger])

    print("\n--- Phase 2: Fine-tuning ---")
    model = fine_tune(model)
    model.fit(train_ds, validation_data=val_ds, epochs=CFG["fine_tune_epochs"],
              callbacks=[early_stop, checkpoint, wandb_logger])

    evaluate(model, test_ds)
    export_tflite(model)

    wandb.finish()
