"""
data_preprocessing.py
----------------------
Handles:
  1. Converting a classic FER2013-style CSV (pixels + label columns) into
     a folder-per-class structure under data/train and data/val.
  2. Building tf.keras ImageDataGenerators (with augmentation) for training.

Emotion classes (fixed order used throughout the project):
    0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = (48, 48)  # grayscale FER-style default; change to (96,96)/(224,224) for transfer learning


def csv_to_folders(csv_path: str, out_dir: str):
    """Convert a classic fer2013.csv (columns: emotion, pixels, Usage) into
    data/train/<class>/*.png and data/val/<class>/*.png."""
    df = pd.read_csv(csv_path)

    for split_name, usage_values in [("train", ["Training"]), ("val", ["PublicTest", "PrivateTest"])]:
        split_df = df[df["Usage"].isin(usage_values)] if "Usage" in df.columns else df
        for label_idx, label_name in enumerate(EMOTION_LABELS):
            os.makedirs(os.path.join(out_dir, split_name, label_name), exist_ok=True)

        counters = {label: 0 for label in EMOTION_LABELS}
        for _, row in split_df.iterrows():
            label_name = EMOTION_LABELS[int(row["emotion"])]
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(pixels, mode="L")
            counters[label_name] += 1
            fname = f"{label_name}_{counters[label_name]:05d}.png"
            img.save(os.path.join(out_dir, split_name, label_name, fname))

        print(f"[{split_name}] wrote {sum(counters.values())} images across {len(EMOTION_LABELS)} classes")


def build_generators(train_dir: str, val_dir: str, img_size=IMG_SIZE, batch_size=64, color_mode="grayscale"):
    """Return (train_generator, val_generator) with augmentation applied only to training data."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        classes=EMOTION_LABELS,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        classes=EMOTION_LABELS,
        shuffle=False,
    )
    return train_gen, val_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing utilities for DeepFER")
    parser.add_argument("--csv_to_folders", type=str, help="Path to fer2013.csv to convert")
    parser.add_argument("--out_dir", type=str, default="data", help="Output data directory")
    args = parser.parse_args()

    if args.csv_to_folders:
        csv_to_folders(args.csv_to_folders, args.out_dir)
    else:
        print("Nothing to do. Pass --csv_to_folders path/to/fer2013.csv to convert a CSV dataset.")
