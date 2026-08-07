# DeepFER: Facial Emotion Recognition Using Deep Learning

DeepFER is a deep learning project that recognizes human facial emotions
in real time using Convolutional Neural Networks (CNNs) and Transfer
Learning. It classifies faces into **7 emotion categories**: Angry, Sad,
Happy, Fear, Neutral, Disgust, Surprise.

---

## 1. Project Structure

```
DeepFER/
├── data/
│   ├── train/                # one subfolder per emotion class
│   │   ├── angry/
│   │   ├── sad/
│   │   ├── happy/
│   │   ├── fear/
│   │   ├── neutral/
│   │   ├── disgust/
│   │   └── surprise/
│   └── val/                  # same structure as train/
│       ├── angry/ ... etc.
├── models/                   # saved trained models (.h5 / .keras)
├── notebooks/                # exploratory notebooks (optional)
├── src/
│   ├── data_preprocessing.py # data loading + augmentation pipeline
│   ├── model.py              # CNN-from-scratch + Transfer Learning models
│   ├── train.py               # training script with callbacks
│   ├── evaluate.py            # accuracy / precision / recall / F1 + confusion matrix
│   ├── realtime_inference.py  # webcam real-time emotion detection (OpenCV)
│   └── app.py                 # simple Streamlit web app (image upload)
├── requirements.txt
└── README.md
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Dataset

Organize your dataset into `data/train/<emotion>/*.jpg` and
`data/val/<emotion>/*.jpg`. This project is compatible with any
FER-style dataset (e.g. FER2013 converted to folder format, CK+, or a
custom crowd-sourced dataset) as long as it follows the 7-class
folder structure above.

If you only have a single CSV (like classic FER2013 `fer2013.csv`),
run:

```bash
python src/data_preprocessing.py --csv_to_folders data/fer2013.csv --out_dir data
```

This converts the CSV into the `train/` and `val/` folder structure
automatically.

## 4. Training

Train the custom CNN from scratch:

```bash
python src/train.py --model_type cnn --epochs 40 --batch_size 64
```

Train using Transfer Learning (MobileNetV2 backbone, recommended for
best accuracy/speed trade-off):

```bash
python src/train.py --model_type transfer --epochs 25 --batch_size 32 --fine_tune
```

Trained models and training curves are saved to `models/`.

## 5. Evaluation

```bash
python src/evaluate.py --model_path models/best_model.keras
```

Outputs accuracy, precision, recall, F1-score (per class + weighted
average) and a confusion matrix plot.

## 6. Real-Time Webcam Inference

```bash
python src/realtime_inference.py --model_path models/best_model.keras
```

Opens your webcam, detects faces with OpenCV's Haar Cascade, and
overlays the predicted emotion + confidence in real time. Press `q` to
quit.

## 7. Web App (image upload demo)

```bash
streamlit run src/app.py
```

Lets a user upload a photo and see the predicted emotion with a
confidence bar chart — a simple front end for demoing the model
without needing a webcam.

## 8. Technologies Used

- Python 3.9+
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib, Scikit-learn
- Streamlit (demo app)

## 9. Applications

- Human-Computer Interaction
- Mental Health Monitoring
- Customer Service
- Security Systems
- Smart Assistants
- Educational Technology

## 10. Notes on Performance Optimization

- Use `--model_type transfer` with MobileNetV2 for the best
  accuracy-to-latency trade-off on CPU/edge devices.
- Convert the final model to TensorFlow Lite (see
  `src/model.py::convert_to_tflite`) to reduce inference latency
  further for deployment on mobile/embedded devices.
- Mixed precision training (`--mixed_precision`) is supported for
  GPUs to speed up training.

## 11. Known Limitations

- Accuracy depends heavily on dataset quality/balance; emotions like
  "Disgust" and "Fear" are typically under-represented in public
  datasets and may need class-weighting (already implemented in
  `train.py`).
- Real-time performance depends on webcam resolution and hardware;
  reduce frame resolution in `realtime_inference.py` if FPS is low.
"# Deeperface" 
