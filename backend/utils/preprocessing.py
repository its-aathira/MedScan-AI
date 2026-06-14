# ─────────────────────────────────────────────
#  backend/utils/preprocessing.py
#  Domain-specific image preprocessing helpers
# ─────────────────────────────────────────────

import cv2
import numpy as np
from PIL import Image
from typing import Union


# ─────────────────────────────────────────────
#  CHEST X-RAY: CLAHE contrast enhancement
# ─────────────────────────────────────────────
def apply_clahe(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Dramatically improves visibility of lung findings in X-rays.

    Args:
        image: PIL Image or NumPy array (grayscale or RGB)
    Returns:
        PIL Image (RGB, CLAHE applied to luminance channel)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to LAB → apply CLAHE on L-channel only
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lab   = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)

    lab   = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


# ─────────────────────────────────────────────
#  RETINAL: Ben Graham preprocessing
# ─────────────────────────────────────────────
def ben_graham_preprocess(
    image     : Union[np.ndarray, Image.Image],
    sigmaX    : int = 10,
    scale     : float = 4.0,
) -> Image.Image:
    """
    Ben Graham's retinal fundus preprocessing.
    Removes uneven illumination and enhances microvasculature detail.
    Top-performing technique on APTOS / Kaggle DR competitions.

    Args:
        image  : PIL Image or NumPy array (RGB)
        sigmaX : Gaussian blur sigma (controls local averaging radius)
        scale  : blending weight
    Returns:
        PIL Image (preprocessed fundus)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    image  = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    result  = cv2.addWeighted(image, scale, blurred, -scale, 128)
    result  = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def crop_retinal_circle(
    image   : Union[np.ndarray, Image.Image],
    tol     : int = 7,
) -> Image.Image:
    """
    Crops the black border around retinal fundus images.
    Removes ~15% dead pixels and focuses the model on the disc.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    mask   = image > tol
    if mask.ndim == 3:
        mask = mask.any(axis=2)

    rows   = np.any(mask, axis=1)
    cols   = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = image[rmin:rmax + 1, cmin:cmax + 1]
    return Image.fromarray(cropped)


# ─────────────────────────────────────────────
#  BRAIN MRI: skull stripping (simple threshold)
# ─────────────────────────────────────────────
def skull_strip(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Lightweight skull-stripping via Otsu thresholding.
    Removes non-brain tissue to reduce noise for tumour detection.
    NOTE: For production, use FSL BET or HD-BET.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    result  = image.copy()
    if image.ndim == 3:
        result[mask == 0] = 0
    else:
        result[mask == 0] = 0

    return Image.fromarray(result)


# ─────────────────────────────────────────────
#  MODULE DISPATCHER
# ─────────────────────────────────────────────
def preprocess_image(
    image      : Union[np.ndarray, Image.Image],
    module_key : str,
) -> Image.Image:
    """
    Applies the correct domain-specific preprocessing for each module.

    Args:
        image      : raw PIL Image
        module_key : 'chest_xray' | 'skin_lesion' | 'brain_mri' | 'retinal'
    Returns:
        Preprocessed PIL Image ready for augmentation pipeline
    """
    if module_key == "chest_xray":
        return apply_clahe(image)

    elif module_key == "skin_lesion":
        # Skin lesions: just ensure RGB, no domain-specific preprocessing needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")

    elif module_key == "brain_mri":
        return skull_strip(image)

    elif module_key == "retinal":
        image = crop_retinal_circle(image)
        return ben_graham_preprocess(image)

    else:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")
