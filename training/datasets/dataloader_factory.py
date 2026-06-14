# ─────────────────────────────────────────────
#  training/datasets/dataloader_factory.py
#  Builds train/val/test DataLoaders cleanly
# ─────────────────────────────────────────────

from typing import Dict, Optional
import pandas as pd
from torch.utils.data import DataLoader

from configs.dataset_config              import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
from training.datasets.medical_dataset  import MedicalImageDataset


def build_dataloaders(
    module_key  : str,
    data_dir    : Optional[str]          = None,
    dataframe   : Optional[pd.DataFrame] = None,
    batch_size  : int                    = BATCH_SIZE,
    num_workers : int                    = NUM_WORKERS,
    use_sampler : bool                   = True,     # handles class imbalance
    seed        : int                    = 42,
) -> Dict[str, DataLoader]:
    """
    Returns a dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}

    Args:
        module_key  : 'chest_xray' | 'skin_lesion' | 'brain_mri' | 'retinal'
        data_dir    : root path containing the module sub-folder
        dataframe   : optional pre-built DataFrame with [filepath, label] columns
        batch_size  : images per batch
        num_workers : parallel workers for data loading
        use_sampler : if True, uses WeightedRandomSampler for training split
                      (recommended for all medical datasets — they're imbalanced)
        seed        : random seed for reproducible splits
    """
    loaders = {}

    for split in ("train", "val", "test"):
        dataset = MedicalImageDataset(
            module_key = module_key,
            split      = split,
            data_dir   = data_dir,
            dataframe  = dataframe,
            seed       = seed,
        )

        # Training: use sampler OR shuffle (not both)
        if split == "train" and use_sampler:
            sampler = dataset.get_sampler()
            loader  = DataLoader(
                dataset,
                batch_size  = batch_size,
                sampler     = sampler,           # replaces shuffle
                num_workers = num_workers,
                pin_memory  = PIN_MEMORY,
                drop_last   = True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = PIN_MEMORY,
                drop_last   = False,
            )

        loaders[split] = loader

    _print_summary(module_key, loaders)
    return loaders


def _print_summary(module_key: str, loaders: Dict[str, DataLoader]):
    print(f"\n{'─'*50}")
    print(f"  DataLoaders ready → Module: {module_key}")
    print(f"{'─'*50}")
    for split, loader in loaders.items():
        n_batches = len(loader)
        n_samples = len(loader.dataset)
        print(f"  {split:<6} | {n_samples:>6} samples | {n_batches:>4} batches")
    print(f"{'─'*50}\n")


# ── Quick smoke-test ──────────────────────────
if __name__ == "__main__":
    import torch

    # Replace with your actual data path
    loaders = build_dataloaders(
        module_key = "chest_xray",
        data_dir   = "data/",
    )

    # Peek at one batch
    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape : {images.shape}")   # [B, 3, 224, 224]
    print(f"Label shape : {labels.shape}")   # [B]
    print(f"Pixel range : [{images.min():.2f}, {images.max():.2f}]")
