# 🧠 MedScan AI — Medical Image Disease Detection

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/aathix/medscan-ai)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An end-to-end AI system for medical image disease detection using **EfficientNetV2** with **Grad-CAM++** explainability. Upload a medical scan and get an instant prediction with a heatmap showing exactly where the model detected the abnormality.

> ⚠️ For research and educational use only. Not a substitute for clinical diagnosis.

---

## 🔴 Live Demo

**👉 [Try it live on HuggingFace Spaces](https://huggingface.co/spaces/aathix/medscan-ai)**

---

## ✨ Features

- **Multi-module detection** — Brain MRI, Chest X-Ray, Skin Lesion, Diabetic Retinopathy
- **Grad-CAM++ heatmaps** — Visual explanation of model decisions overlaid on the scan
- **99.85% AUC** on Brain MRI tumor classification across 4 classes
- **FastAPI backend** with batch inference support
- **React frontend** with drag & drop image upload
- **Dockerized** and deployed on HuggingFace Spaces

---

## 🧠 Disease Modules

| Brain MRI | Glioma · Meningioma · Pituitary · No Tumor | ✅ Live |
| Chest X-Ray | Pneumonia · COVID-19 · Tuberculosis · Normal | ✅ Live |
| Skin Lesion | Melanoma · Nevus · Carcinoma · 4 more | ✅ Live |
| Diabetic Retinopathy | 5-stage DR grading | ✅ Live |

---

## 📊 Model Performance (Brain MRI)

| Metric | Score |
|---|---|
| Test Accuracy | **96.21%** |
| Macro AUC | **99.63%** |
| Best Val AUC | **99.85%** |
| Macro F1 | **96.18%** |

| Chest X-Ray Accuracy | 93.55% |
| Chest X-Ray AUC | 99.08% |
| Skin Lesion AUC | 90.59% |
| Retinal AUC | 88.03% |

**Per-class results:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Tumor | 0.977 | 0.995 | 0.986 |
| Glioma | 0.954 | 0.981 | 0.967 |
| Meningioma | 0.960 | 0.905 | 0.932 |
| Pituitary | 0.958 | 0.967 | 0.962 |

---

## 🏗️ Architecture

```
Medical Image
      │
      ▼
Domain Preprocessing          CLAHE (X-ray) · Ben Graham (Retinal) · Skull Strip (MRI)
      │
      ▼
EfficientNetV2-S Backbone     Pretrained on ImageNet-21k · 20.8M parameters
      │
      ▼
Custom Classification Head    FC(1280→512) · GELU · BN · Dropout → FC(512→N)
      │
      ▼
Prediction + Grad-CAM++       Class probabilities · Confidence · Heatmap overlay
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | EfficientNetV2-S (timm) |
| Framework | PyTorch 2.2+ |
| Explainability | Grad-CAM++ |
| Loss | Focal Loss + Label Smoothing |
| Scheduler | Cosine Annealing LR |
| Tracking | MLflow |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- Node.js 20+

### 1. Clone & install
```bash
git clone https://github.com/its-aathira/MedScan-AI.git
cd MedScan-AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download model weights
Weights are stored on HuggingFace (too large for GitHub):
```bash
mkdir -p backend/models/weights
# Download brain_mri_best.pt from:
# https://huggingface.co/spaces/aathix/medscan-ai/tree/main/backend/models/weights
```

### 3. Start the API
```bash
uvicorn backend.api.main:app --reload --port 8000
```

### 4. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

---

## 🏋️ Train Your Own Model

### Download datasets
```bash
python -m training.datasets.download_guide
python -m training.datasets.download_guide --validate
```

### Train
```bash
python -m training.train --module brain_mri  --epochs 30 --batch_size 32
python -m training.train --module chest_xray --epochs 30
python -m training.train --module skin_lesion --epochs 30
python -m training.train --module retinal    --epochs 30
```

### Evaluate
```bash
python -m backend.inference.predictor brain_mri "path/to/image.jpg"
```

---

## 📁 Project Structure

```
MedScan-AI/
├── configs/                  # Dataset config, class definitions
├── training/
│   ├── datasets/             # PyTorch Dataset + DataLoader factory
│   ├── augmentation/         # Module-aware augmentation pipelines
│   ├── train.py              # Training entry point
│   ├── trainer.py            # Training loop + early stopping + MLflow
│   ├── losses.py             # Focal Loss + Label Smoothing
│   └── evaluate.py           # Metrics, confusion matrix, ROC curves
├── backend/
│   ├── models/               # EfficientNetV2 model definition
│   ├── inference/            # Grad-CAM++ + prediction pipeline
│   ├── utils/                # CLAHE, Ben Graham, preprocessing
│   └── api/                  # FastAPI routes + model registry
├── frontend/
│   └── src/
│       └── App.jsx           # React UI
├── Dockerfile
└── README.md
```

---

## 🔬 Development Phases

| Phase | Description |
|---|---|
| 1 | Data pipeline — dataset config, augmentation, stratified DataLoaders |
| 2 | Model training — EfficientNetV2, Focal Loss, Cosine LR, MLflow tracking |
| 3 | Explainability — Grad-CAM++ heatmaps, domain-specific preprocessing |
| 4 | FastAPI backend — REST API, batch inference, model registry |
| 5 | React frontend — drag & drop upload, confidence bars, live Grad-CAM |
| 6 | Docker + deployment — HuggingFace Spaces, single-container full-stack |

---

## 📄 License

MIT License

---

<div align="center">
  Built by <a href="https://github.com/its-aathira">Aathira Shibu</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/aathix/medscan-ai">🤗 Live Demo</a>
</div>