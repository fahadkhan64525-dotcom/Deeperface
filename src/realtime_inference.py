"""
realtime_inference.py
----------------------
Real-time facial emotion recognition from a webcam feed.

Pipeline:
  1. Capture frame from webcam (OpenCV).
  2. Detect face(s) with a Haar Cascade classifier.
  3. Crop + preprocess each face to match the trained model's input.
  4. Predict emotion and overlay label + confidence on the video feed.

Usage:
  python src/realtime_inference.py --model_path models/best_model.keras
Press 'q' to quit.
"""

import argparse
import time

import cv2
import numpy as np
import tensorflow as tf

from data_preprocessing import EMOTION_LABELS

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def preprocess_face(face_img, img_size, color_mode):
    if color_mode == "grayscale":
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, img_size)
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=-1)  # add channel dim
    else:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, img_size)
        face_img = face_img.astype("float32") / 255.0

    return np.expand_dims(face_img, axis=0)  # add batch dim


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam emotion recognition")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, nargs=2, default=[48, 48])
    parser.add_argument("--color_mode", type=str, default="grayscale", choices=["grayscale", "rgb"])
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check --camera_index or camera permissions.")

    prev_time = time.time()

    print("Starting real-time inference. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            input_tensor = preprocess_face(face_roi, tuple(args.img_size), args.color_mode)

            preds = model.predict(input_tensor, verbose=0)[0]
            label_idx = int(np.argmax(preds))
            label = EMOTION_LABELS[label_idx]
            confidence = float(preds[label_idx])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS overlay
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("DeepFER - Real-Time Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
