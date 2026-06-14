# ─────────────────────────────────────────────
#  backend/inference/predictor.py
#  Single-image inference pipeline
# ─────────────────────────────────────────────

import io
import base64
from pathlib import Path
from typing  import Optional

import numpy as np
import torch
from PIL   import Image

from configs.dataset_config              import MODULES
from backend.models.model                import build_model
from backend.utils.preprocessing         import preprocess_image
from training.augmentation.transforms    import get_transforms
from backend.inference.gradcam           import predict_with_gradcam


class MedicalPredictor:
    """
    Loads a trained model and runs full inference + Grad-CAM on a PIL Image.
    Designed to be instantiated once and reused across requests (FastAPI lifespan).
    """

    def __init__(
        self,
        module_key      : str,
        checkpoint_path : str,
        device          : Optional[str] = None,
    ):
        self.module_key = module_key
        self.cfg        = MODULES[module_key]
        self.classes    = self.cfg["classes"]
        self.device     = device or self._auto_device()
        self.transform  = get_transforms(module_key, split="val")

        # Load model
        self.model = build_model(
            module_key      = module_key,
            pretrained      = False,
            checkpoint_path = checkpoint_path,
            device          = self.device,
        )
        self.model.eval()
        print(f"  ✓ Predictor ready [{module_key}] on {self.device}")

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():       return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Args:
            pil_image: raw PIL Image (any size, any mode)
        Returns:
            Full result dict from predict_with_gradcam + base64 overlay
        """
        # ── Preprocess (domain-specific) ──────
        preprocessed = preprocess_image(pil_image, self.module_key)

        # ── Transform (resize + normalise) ────
        tensor = self.transform(preprocessed).unsqueeze(0)  # [1, 3, 224, 224]

        # ── Inference + Grad-CAM ──────────────
        result = predict_with_gradcam(
            model        = self.model,
            image_tensor = tensor,
            original_pil = pil_image,
            module_key   = self.module_key,
            classes      = self.classes,
            device       = self.device,
        )

        # ── Add base64 overlay for API response
        result["overlay_b64"] = pil_to_base64(result["overlay_pil"])

        # Remove non-serialisable fields
        result.pop("heatmap",      None)
        result.pop("overlay",      None)
        result.pop("overlay_pil",  None)

        return result


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── CLI quick-test ────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m backend.inference.predictor <module_key> <image_path>")
        print("Example: python -m backend.inference.predictor brain_mri data/brain_mri/Glioma/Tr-gl_0010.jpg")
        sys.exit(1)

    module_key  = sys.argv[1]
    image_path  = sys.argv[2]
    ckpt_path   = f"backend/models/weights/{module_key}_best.pt"

    predictor = MedicalPredictor(module_key, ckpt_path)
    image     = Image.open(image_path).convert("RGB")
    result    = predictor.predict(image)

    print(f"\n{'─'*45}")
    print(f"  Image     : {image_path}")
    print(f"  Prediction: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"\n  All class probabilities:")
    for cls, prob in sorted(result["class_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob / 5)
        print(f"    {cls:<20} {prob:>6.2f}%  {bar}")
    print(f"{'─'*45}")

    # Save overlay
    out_path = Path("logs") / module_key / "gradcam_sample.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import base64
    img_bytes = base64.b64decode(result["overlay_b64"])
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    print(f"\n  Grad-CAM overlay saved → {out_path}")