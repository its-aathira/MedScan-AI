# ─────────────────────────────────────────────
#  configs/dataset_config.py
#  Central config for all disease modules
# ─────────────────────────────────────────────

IMAGE_SIZE = 224          # Input size for EfficientNetV2 / ViT
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY  = True

# ── Module definitions ────────────────────────
MODULES = {
    "chest_xray": {
        "name":        "Chest X-Ray",
        "num_classes": 4,
        "classes":     ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"],
        "dataset":     "NIH ChestX-ray14 / CheXpert",
        "image_size":  224,
        "mean":        [0.485, 0.456, 0.406],   # ImageNet stats (grayscale→RGB)
        "std":         [0.229, 0.224, 0.225],
    },
    "skin_lesion": {
        "name":        "Skin Lesion",
        "num_classes": 7,
        "classes":     [
            "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
            "Actinic Keratosis", "Benign Keratosis",
            "Dermatofibroma", "Vascular Lesion"
        ],
        "dataset":     "ISIC 2020",
        "image_size":  224,
        "mean":        [0.763, 0.546, 0.570],   # ISIC-specific stats
        "std":         [0.141, 0.152, 0.169],
    },
    "brain_mri": {
        "name":        "Brain MRI",
        "num_classes": 4,
        "classes":     ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
        "dataset":     "Kaggle Brain Tumor MRI Dataset",
        "image_size":  224,
        "mean":        [0.485, 0.456, 0.406],
        "std":         [0.229, 0.224, 0.225],
    },
    "retinal": {
        "name":        "Diabetic Retinopathy",
        "num_classes": 5,
        "classes":     [
            "No DR", "Mild DR", "Moderate DR",
            "Severe DR", "Proliferative DR"
        ],
        "dataset":     "APTOS 2019 Blindness Detection",
        "image_size":  224,
        "mean":        [0.485, 0.456, 0.406],
        "std":         [0.229, 0.224, 0.225],
    },
}

# ── Train / Val / Test split ratios ──────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ── Paths (override via env vars in production)
import os
BASE_DATA_DIR   = os.getenv("DATA_DIR",   "data/")
BASE_MODEL_DIR  = os.getenv("MODEL_DIR",  "backend/models/weights/")
LOGS_DIR        = os.getenv("LOGS_DIR",   "logs/")
