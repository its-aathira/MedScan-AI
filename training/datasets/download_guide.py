# ─────────────────────────────────────────────
#  training/datasets/download_guide.py
#  Dataset download + directory setup helper
# ─────────────────────────────────────────────

"""
Run this script first!

    python -m training.datasets.download_guide

It will:
  1. Print download instructions for each dataset
  2. Create the expected directory structure
  3. Validate your data after you've downloaded it
"""

import os
import sys
from pathlib import Path
from configs.dataset_config import MODULES, BASE_DATA_DIR

DOWNLOAD_INSTRUCTIONS = {
    "chest_xray": {
        "url":     "https://www.kaggle.com/datasets/nih-chest-xrays/data",
        "size":    "~45 GB",
        "alt_url": "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
        "alt_size": "~3 GB (smaller, good for quick start)",
        "steps": [
            "1. Install Kaggle CLI:  pip install kaggle",
            "2. Place kaggle.json in ~/.kaggle/",
            "3. Run: kaggle datasets download -d tawsifurrahman/covid19-radiography-database",
            "4. Unzip and rename class folders to: Normal, Pneumonia, COVID-19, Tuberculosis",
            "5. Place under:  data/chest_xray/<class_name>/",
        ],
    },
    "skin_lesion": {
        "url":     "https://www.isic-archive.com/",
        "size":    "~10 GB",
        "alt_url": "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
        "alt_size": "~3 GB",
        "steps": [
            "1. Run: kaggle datasets download -d kmader/skin-cancer-mnist-ham10000",
            "2. Unzip to data/skin_lesion/",
            "3. Run the organizer script below to create class sub-folders from CSV metadata",
        ],
    },
    "brain_mri": {
        "url":     "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "size":    "~180 MB",
        "alt_url": None,
        "steps": [
            "1. Run: kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset",
            "2. Unzip — folder structure already matches expected layout",
            "3. Rename: notumor → No Tumor | glioma → Glioma | etc.",
            "4. Place under: data/brain_mri/<class_name>/",
        ],
    },
    "retinal": {
        "url":     "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
        "size":    "~9 GB",
        "alt_url": "https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data",
        "alt_size": "~1.5 GB (pre-resized)",
        "steps": [
            "1. Run: kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-2019-data",
            "2. Unzip and organize by label (0-4) into class sub-folders",
            "3. Rename: 0→No DR | 1→Mild DR | 2→Moderate DR | 3→Severe DR | 4→Proliferative DR",
            "4. Place under: data/retinal/<class_name>/",
        ],
    },
}


def print_instructions():
    print("\n" + "═" * 60)
    print("  DATASET DOWNLOAD GUIDE")
    print("═" * 60)
    for key, info in DOWNLOAD_INSTRUCTIONS.items():
        cfg = MODULES[key]
        print(f"\n📂  {cfg['name']}  ({info['size']})")
        print(f"    URL      : {info['url']}")
        if info.get("alt_url"):
            print(f"    Alt URL  : {info['alt_url']}  [{info['alt_size']}]  ← recommended")
        print(f"    Steps:")
        for step in info["steps"]:
            print(f"      {step}")


def create_directory_structure():
    """Creates the expected data/ folder tree."""
    print("\n\n📁  Creating directory structure...")
    for module_key, cfg in MODULES.items():
        for class_name in cfg["classes"]:
            path = Path(BASE_DATA_DIR) / module_key / class_name
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓  {path}")
    print("\nDirectory structure ready.\n")


def validate_data():
    """Counts images per class for each module."""
    print("\n📊  Data Validation Report")
    print("─" * 50)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    total_images = 0

    for module_key, cfg in MODULES.items():
        print(f"\n  {cfg['name']}")
        module_total = 0
        for class_name in cfg["classes"]:
            path = Path(BASE_DATA_DIR) / module_key / class_name
            if not path.exists():
                print(f"    {'⚠':2}  {class_name:<30}  MISSING")
                continue
            count = sum(1 for f in path.iterdir() if f.suffix.lower() in exts)
            status = "✓" if count > 0 else "⚠  EMPTY"
            print(f"    {status}  {class_name:<30}  {count:>6} images")
            module_total += count
        print(f"    {'─'*44}")
        print(f"    {'Total':<34}  {module_total:>6} images")
        total_images += module_total

    print(f"\n  Grand total: {total_images} images across all modules\n")


if __name__ == "__main__":
    print_instructions()
    create_directory_structure()

    if "--validate" in sys.argv:
        validate_data()
    else:
        print("\nTip: After downloading, run with --validate flag:")
        print("     python -m training.datasets.download_guide --validate\n")
