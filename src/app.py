"""
app.py
------
A simple Streamlit web app: upload a photo, detect the face, and
display the predicted emotion with a confidence bar chart.

Usage:
  streamlit run src/app.py
"""

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from data_preprocessing import EMOTION_LABELS

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_face(face_img, img_size, color_mode):
    if color_mode == "grayscale":
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        face_img = cv2.resize(face_img, img_size)
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=-1)
    else:
        face_img = cv2.resize(face_img, img_size)
        face_img = face_img.astype("float32") / 255.0
    return np.expand_dims(face_img, axis=0)


def main():
    st.set_page_config(page_title="DeepFER", page_icon="🙂", layout="centered")
    st.title("DeepFER: Facial Emotion Recognition.")
    st.caption("Upload a photo to detect the facial emotion using a  Model CNN / Transfer Learning model.")

    model_path = st.sidebar.text_input("Model path", value="models/best_model.keras")
    img_h = st.sidebar.number_input("Model input height", value=48)
    img_w = st.sidebar.number_input("Model input width", value=48)
    color_mode = st.sidebar.selectbox("Color mode", ["grayscale", "rgb"])

    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Uploaded image", use_container_width=True)

        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            st.warning("No face detected. Try a clearer, front-facing photo.")
            return

        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Could not load model at '{model_path}'. Train a model first. Error: {e}")
            return

        x, y, w, h = faces[0]
        face_roi = img_array[y:y + h, x:x + w]
        input_tensor = preprocess_face(face_roi, (img_w, img_h), color_mode)

        preds = model.predict(input_tensor, verbose=0)[0]
        label_idx = int(np.argmax(preds))

        st.subheader(f"Predicted emotion: **{EMOTION_LABELS[label_idx].capitalize()}**")
        st.bar_chart({label: float(p) for label, p in zip(EMOTION_LABELS, preds)})


if __name__ == "__main__":
    main()
