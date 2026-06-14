---
title: MedScan AI — Medical Image Disease Detection
emoji: 🧠
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# MedScan AI — Medical Image Disease Detection

AI-powered disease detection system using **EfficientNetV2** with **Grad-CAM++** explainability.

## Modules
| Module | Classes | AUC |
|---|---|---|
| Brain MRI | Glioma · Meningioma · Pituitary · No Tumor | 99.85% |
| Chest X-Ray | Pneumonia · COVID-19 · TB · Normal | coming soon |
| Skin Lesion | Melanoma · Nevus · Carcinoma · 4 more | coming soon |
| Diabetic Retinopathy | 5-stage DR grading | coming soon |

## Tech Stack
- **Model**: EfficientNetV2-S fine-tuned with PyTorch
- **Explainability**: Grad-CAM++ heatmaps
- **Backend**: FastAPI
- **Frontend**: React + Vite
- **Training**: Focal Loss · Cosine LR · Mixed Precision

## Usage
1. Select a disease module
2. Upload a medical image
3. Get prediction + confidence scores + Grad-CAM heatmap
## Model Weights
Model weights are not stored in this repo due to size.
Download from HuggingFace: https://huggingface.co/spaces/aathix/medscan-ai
Place in: backend/models/weights/brain_mri_best.pt

