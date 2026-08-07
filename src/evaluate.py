"""
evaluate.py
-----------
Loads a trained model and reports Accuracy, Precision, Recall, F1-Score
(per class and weighted average) plus a confusion matrix heatmap.

Usage:
  python src/evaluate.py --model_path models/best_model.keras --val_dir data/val
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import EMOTION_LABELS
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepFER model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--img_size", type=int, nargs=2, default=[48, 48])
    parser.add_argument("--color_mode", type=str, default="grayscale", choices=["grayscale", "rgb"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = datagen.flow_from_directory(
        args.val_dir,
        target_size=tuple(args.img_size),
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=EMOTION_LABELS,
        shuffle=False,
    )

    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DeepFER Confusion Matrix")
    plt.tight_layout()
    out_path = f"{args.out_dir}/confusion_matrix.png"
    plt.savefig(out_path)
    print(f"\nSaved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
