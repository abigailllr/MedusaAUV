import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import fiftyone.zoo as foz

BASE = "/content/dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
THRESHOLD = 0.7
SPLITS = ["train", "val", "test"]
CLASSES = ["jellyfish", "no_jellyfish"]


def download_dataset():
    jellyfish = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["classifications"],
        classes=["Jellyfish"],
        max_samples=500,
        dataset_name="jellyfish_pos"
    )
    negative = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["classifications"],
        classes=["Person", "Car", "Tree", "Building", "Dog", "Cat", "Chair", "Flower"],
        max_samples=500,
        dataset_name="jellyfish_neg"
    )
    return jellyfish, negative


def build_folder_structure():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(f"{BASE}/{split}/{cls}", exist_ok=True)


def copy_samples(dataset, class_name, ratios=(0.7, 0.15, 0.15)):
    paths = [s.filepath for s in dataset]
    random.shuffle(paths)
    n = len(paths)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    split_map = {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:]
    }
    for split, files in split_map.items():
        for f in files:
            shutil.copy(f, f"{BASE}/{split}/{class_name}/")


def load_datasets():
    normalize = tf.keras.layers.Rescaling(1.0 / 255)

    augment = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomBrightness(0.2),
    ])

    def prep_train(ds):
        return ds.map(lambda x, y: (normalize(augment(x, training=True)), y))

    def prep(ds):
        return ds.map(lambda x, y: (normalize(x), y))

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
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fine_tune(model):
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def evaluate(model, test_ds):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend((preds > THRESHOLD).astype(int).flatten())
        y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred,
          target_names=["no_jellyfish", "jellyfish"]))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["No Jellyfish", "Jellyfish"],
                yticklabels=["No Jellyfish", "Jellyfish"])
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


def plot_history(histories):
    plt.figure(figsize=(12, 4))
    acc = histories[0].history["accuracy"] + histories[1].history["accuracy"]
    val_acc = histories[0].history["val_accuracy"] + histories[1].history["val_accuracy"]
    loss = histories[0].history["loss"] + histories[1].history["loss"]
    val_loss = histories[0].history["val_loss"] + histories[1].history["val_loss"]

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train")
    plt.plot(val_acc, label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("jellyfish_model.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"Modellgroesse: {len(tflite_model) / 1024:.1f} KB")


if __name__ == "__main__":
    jellyfish_ds, negative_ds = download_dataset()
    build_folder_structure()
    copy_samples(jellyfish_ds, "jellyfish")
    copy_samples(negative_ds, "no_jellyfish")

    for split in SPLITS:
        for cls in CLASSES:
            count = len(os.listdir(f"{BASE}/{split}/{cls}"))
            print(f"{split}/{cls}: {count} Bilder")

    train_ds, val_ds, test_ds = load_datasets()
    model = build_model()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)

    print("\n--- Phase 1: Transfer Learning ---")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                   callbacks=[early_stop, checkpoint])

    print("\n--- Phase 2: Fine-tuning ---")
    model = fine_tune(model)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=20,
                   callbacks=[early_stop, checkpoint])

    evaluate(model, test_ds)
    plot_history([h1, h2])
    export_tflite(model)
