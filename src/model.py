"""
model.py
--------
Defines two model builders:
  1. build_cnn_model()      -> a CNN trained from scratch on 48x48 grayscale images
  2. build_transfer_model() -> MobileNetV2 backbone with a custom classification head
                                (expects 96x96 or 224x224 RGB images)

Also includes a helper to convert a saved Keras model to TensorFlow Lite
for faster/lighter deployment.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

NUM_CLASSES = 7


def build_cnn_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    """A compact CNN built from scratch, suitable for grayscale 48x48 face crops."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="DeepFER_CNN")

    return model


def build_transfer_model(input_shape=(96, 96, 3), num_classes=NUM_CLASSES, fine_tune=False, fine_tune_at=100):
    """MobileNetV2-based transfer learning model.

    If fine_tune is False, the backbone is entirely frozen (feature extraction only).
    If fine_tune is True, layers from `fine_tune_at` onward are unfrozen for fine-tuning
    (call this after a first round of training with fine_tune=False for best results).
    """
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = fine_tune

    if fine_tune:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="DeepFER_TransferLearning")
    return model


def convert_to_tflite(keras_model_path: str, tflite_out_path: str, quantize: bool = True):
    """Convert a saved .keras/.h5 model to a lightweight TFLite model for
    faster inference on edge/mobile devices."""
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {tflite_out_path}")


if __name__ == "__main__":
    m1 = build_cnn_model()
    m1.summary()

    m2 = build_transfer_model()
    m2.summary()
