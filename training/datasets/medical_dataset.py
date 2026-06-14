# ─────────────────────────────────────────────
#  training/datasets/medical_dataset.py
#  Universal Dataset class for all 4 modules
# ─────────────────────────────────────────────

import os
import json
import random
from pathlib import Path
from typing  import Optional, Tuple, List, Dict

import numpy  as np
import pandas as pd
from PIL      import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from configs.dataset_config  import MODULES, SPLIT_RATIOS, BASE_DATA_DIR
from training.augmentation.transforms import get_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate corrupted JPEG tails


# ─────────────────────────────────────────────
class MedicalImageDataset(Dataset):
    """
    Unified dataset loader for all medical imaging modules.

    Expected directory layout (one per module):
        data/
        └── <module_key>/
            ├── <class_0>/
            │   ├── img001.jpg
            │   └── ...
            ├── <class_1>/
            └── ...

    Alternatively, pass a pre-built DataFrame with columns:
        ['filepath', 'label']
    """

    def __init__(
        self,
        module_key : str,
        split      : str,
        data_dir   : Optional[str] = None,
        dataframe  : Optional[pd.DataFrame] = None,
        transform  = None,
        seed       : int = 42,
    ):
        assert module_key in MODULES, f"Unknown module: {module_key}"
        assert split      in ("train", "val", "test")

        self.module_key  = module_key
        self.split       = split
        self.cfg         = MODULES[module_key]
        self.classes     = self.cfg["classes"]
        self.num_classes = self.cfg["num_classes"]
        self.class2idx   = {c: i for i, c in enumerate(self.classes)}
        self.transform   = transform or get_transforms(module_key, split)
        self.seed        = seed

        # ── Build file list ───────────────────
        if dataframe is not None:
            self.data = self._from_dataframe(dataframe)
        else:
            root = Path(data_dir or BASE_DATA_DIR) / module_key
            self.data = self._scan_directory(root)

        # ── Split ─────────────────────────────
        self.data = self._split(self.data)

        print(
            f"[{module_key}][{split}] "
            f"{len(self.data)} samples | "
            f"{self.num_classes} classes"
        )

    # ── Directory scanner ─────────────────────
    def _scan_directory(self, root: Path) -> List[Dict]:
        records = []
        for class_name in self.classes:
            class_dir = root / class_name
            if not class_dir.exists():
                print(f"  ⚠  Missing: {class_dir}")
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
                for fp in class_dir.glob(ext):
                    records.append({
                        "filepath": str(fp),
                        "label":    self.class2idx[class_name],
                    })
        return records

    # ── DataFrame loader ──────────────────────
    def _from_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        assert "filepath" in df.columns and "label" in df.columns
        return df[["filepath", "label"]].to_dict("records")

    # ── Stratified split ──────────────────────
    def _split(self, data: List[Dict]) -> List[Dict]:
        random.seed(self.seed)

        # Group by label
        buckets: Dict[int, List] = {}
        for rec in data:
            buckets.setdefault(rec["label"], []).append(rec)

        train_d, val_d, test_d = [], [], []
        tr, va, te = (
            SPLIT_RATIOS["train"],
            SPLIT_RATIOS["val"],
            SPLIT_RATIOS["test"],
        )

        for recs in buckets.values():
            random.shuffle(recs)
            n      = len(recs)
            n_tr   = int(n * tr)
            n_va   = int(n * va)
            train_d += recs[:n_tr]
            val_d   += recs[n_tr : n_tr + n_va]
            test_d  += recs[n_tr + n_va :]

        return {"train": train_d, "val": val_d, "test": test_d}[self.split]

    # ── PyTorch Dataset interface ─────────────
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec   = self.data[idx]
        image = Image.open(rec["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, rec["label"]

    # ── Class weights for imbalanced datasets ─
    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = np.zeros(self.num_classes)
        for rec in self.data:
            counts[rec["label"]] += 1
        counts = np.where(counts == 0, 1, counts)   # avoid div-by-zero
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    # ── WeightedRandomSampler (for DataLoader) ─
    def get_sampler(self) -> WeightedRandomSampler:
        """Use instead of shuffle=True to handle class imbalance."""
        class_weights = self.get_class_weights().numpy()
        sample_weights = np.array([
            class_weights[rec["label"]] for rec in self.data
        ])
        return WeightedRandomSampler(
            weights     = torch.from_numpy(sample_weights).float(),
            num_samples = len(self.data),
            replacement = True,
        )

    # ── Label distribution summary ────────────
    def class_distribution(self) -> Dict[str, int]:
        dist = {c: 0 for c in self.classes}
        for rec in self.data:
            dist[self.classes[rec["label"]]] += 1
        return dist
