"""
train.py
--------
Trains DeepFER using either:
  --model_type cnn        a from-scratch CNN on 48x48 grayscale images
  --model_type transfer   MobileNetV2 transfer learning on 96x96 RGB images

Usage:
  python src/train.py --model_type cnn --epochs 40 --batch_size 64
  python src/train.py --model_type transfer --epochs 25 --fine_tune
"""

import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from data_preprocessing import build_generators, EMOTION_LABELS
from model import build_cnn_model, build_transfer_model


def compute_generator_class_weights(train_gen):
    """Compute class weights to counter class imbalance (e.g. under-represented
    'disgust'/'fear' classes common in FER datasets)."""
    classes = train_gen.classes
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(classes), y=classes)
    return dict(enumerate(weights))


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DeepFER emotion recognition model")
    parser.add_argument("--model_type", choices=["cnn", "transfer"], default="cnn")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune", action="store_true", help="(transfer only) unfreeze backbone for fine-tuning")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if args.model_type == "cnn":
        img_size, color_mode = (48, 48), "grayscale"
        input_shape = (48, 48, 1)
    else:
        img_size, color_mode = (96, 96), "rgb"
        input_shape = (96, 96, 3)

    train_gen, val_gen = build_generators(
        args.train_dir, args.val_dir, img_size=img_size, batch_size=args.batch_size, color_mode=color_mode
    )

    if args.model_type == "cnn":
        model = build_cnn_model(input_shape=input_shape)
    else:
        model = build_transfer_model(input_shape=input_shape, fine_tune=args.fine_tune)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_generator_class_weights(train_gen)
    print("Class weights:", dict(zip(EMOTION_LABELS, [round(class_weights[i], 2) for i in range(len(EMOTION_LABELS))])))

    best_model_path = os.path.join(args.out_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    final_path = os.path.join(args.out_dir, f"final_{args.model_type}_model.keras")
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    print(f"Best model (by val_accuracy) saved to {best_model_path}")

    plot_history(history, os.path.join(args.out_dir, f"training_curves_{args.model_type}.png"))


if __name__ == "__main__":
    main()
