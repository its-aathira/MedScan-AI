# ─────────────────────────────────────────────
#  training/augmentation/transforms.py
#  Augmentation pipelines per split & module
# ─────────────────────────────────────────────

import torchvision.transforms as T
from configs.dataset_config import MODULES


def get_transforms(module_key: str, split: str):
    """
    Returns the appropriate torchvision transform pipeline.

    Args:
        module_key : one of 'chest_xray' | 'skin_lesion' | 'brain_mri' | 'retinal'
        split      : 'train' | 'val' | 'test'

    Returns:
        torchvision.transforms.Compose
    """
    cfg  = MODULES[module_key]
    mean = cfg["mean"]
    std  = cfg["std"]
    size = cfg["image_size"]

    # ── Shared base (val / test) ──────────────
    base = [
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    if split != "train":
        return T.Compose(base)

    # ── Training augmentations (module-aware) ─
    if module_key == "chest_xray":
        # Chest X-rays are grayscale; subtle augmentations only
        aug = [
            T.Resize((size + 20, size + 20)),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.Grayscale(num_output_channels=3),   # keep 3-ch for pretrained models
        ]

    elif module_key == "skin_lesion":
        # Skin lesions: aggressive augmentation + colour jitter (important for melanoma)
        aug = [
            T.Resize((size + 32, size + 32)),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.05),
        ]

    elif module_key == "brain_mri":
        # MRI: flips + slight zoom; no colour jitter (grayscale scans)
        aug = [
            T.Resize((size + 16, size + 16)),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]

    elif module_key == "retinal":
        # Fundus: circular FOV — aggressive flips, careful colour
        aug = [
            T.Resize((size + 20, size + 20)),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=360),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]

    else:
        aug = [
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
        ]

    return T.Compose(aug + [T.ToTensor(), T.Normalize(mean=mean, std=std)])


# ── Quick sanity-check ────────────────────────
if __name__ == "__main__":
    for mod in ["chest_xray", "skin_lesion", "brain_mri", "retinal"]:
        for split in ["train", "val", "test"]:
            tf = get_transforms(mod, split)
            print(f"[{mod}][{split}] → {len(tf.transforms)} transforms")
