# ─────────────────────────────────────────────
#  backend/inference/gradcam.py
#  Grad-CAM heatmap generation for all modules
# ─────────────────────────────────────────────

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Selvaraju et al. (2017) — https://arxiv.org/abs/1610.02391

    Produces a heatmap highlighting image regions the model
    used to make its prediction — critical for clinical trust.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap, overlay = gradcam(image_tensor, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(
        self,
        image_tensor : torch.Tensor,        # [1, 3, H, W]
        class_idx    : Optional[int] = None, # None → predicted class
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            heatmap : np.ndarray [H, W]  float32 in [0, 1]
            overlay : np.ndarray [H, W, 3] uint8  RGB with heatmap blended
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # ── Forward pass ──────────────────────
        logits = self.model(image_tensor)           # [1, C]

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # ── Backward pass for target class ────
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # ── Grad-CAM computation ──────────────
        # gradients: [1, C, h, w]  activations: [1, C, h, w]
        weights    = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam        = (weights * self.activations).sum(dim=1, keepdim=True)
        cam        = F.relu(cam)                    # keep positive influence only

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

    def __del__(self):
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — Chattopadhyay et al. (2018)
    Better localisation for multiple instances of the same class.
    Drop-in replacement for GradCAM.
    """

    def __call__(
        self,
        image_tensor : torch.Tensor,
        class_idx    : Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        grads  = self.gradients              # [1, C, h, w]
        acts   = self.activations            # [1, C, h, w]

        # Grad-CAM++ alpha weights
        grads_sq   = grads ** 2
        grads_cub  = grads ** 3
        denom      = 2 * grads_sq + (grads_cub * acts).sum(dim=[2, 3], keepdim=True) + 1e-8
        alpha      = grads_sq / denom
        weights    = (alpha * F.relu(grads)).sum(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


# ─────────────────────────────────────────────
#  Overlay utilities
# ─────────────────────────────────────────────

def apply_heatmap_overlay(
    original_image : np.ndarray,    # [H, W, 3] uint8 RGB
    cam            : np.ndarray,    # [h, w]    float32 [0,1]
    alpha          : float = 0.45,  # heatmap opacity
    colormap       : int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Resizes CAM to match image, applies colormap, blends with original.

    Returns:
        overlay: [H, W, 3] uint8 RGB
    """
    H, W = original_image.shape[:2]

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_uint8   = (cam_resized * 255).astype(np.uint8)

    # Apply colormap (JET: blue=low, red=high)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def tensor_to_numpy_image(
    tensor : torch.Tensor,         # [1, 3, H, W] or [3, H, W]
    mean   : List[float] = [0.485, 0.456, 0.406],
    std    : List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Denormalises a tensor back to uint8 RGB numpy array."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    t = tensor.cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m

    t = t.permute(1, 2, 0).numpy()
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return t


# ─────────────────────────────────────────────
#  Factory — auto-selects correct target layer
# ─────────────────────────────────────────────

def build_gradcam(
    model      : nn.Module,
    module_key : str,
    use_plusplus: bool = True,
) -> GradCAM:
    """
    Selects the correct convolutional layer for Grad-CAM based on backbone.
    For EfficientNetV2, the last conv block before global pooling is best.

    Args:
        model       : trained MedicalClassifier
        module_key  : disease module (used for logging only)
        use_plusplus: use Grad-CAM++ (better localisation, recommended)
    """
    # Navigate to the last convolutional block of EfficientNetV2
    # timm's EfficientNetV2: backbone.blocks[-1] is the last MBConv block
    try:
        target_layer = model.backbone.blocks[-1]
    except (AttributeError, IndexError):
        # Fallback: use the last child of backbone
        target_layer = list(model.backbone.children())[-1]

    cls = GradCAMPlusPlus if use_plusplus else GradCAM
    print(f"  Grad-CAM{'++' if use_plusplus else ''} target layer: {type(target_layer).__name__}")
    return cls(model, target_layer)


# ─────────────────────────────────────────────
#  Full inference + Grad-CAM pipeline
# ─────────────────────────────────────────────

def predict_with_gradcam(
    model        : nn.Module,
    image_tensor : torch.Tensor,    # [1, 3, 224, 224]
    original_pil : Image.Image,     # original PIL image (for overlay)
    module_key   : str,
    classes      : List[str],
    device       : str = "cpu",
) -> dict:
    """
    End-to-end: image → prediction + confidence + heatmap overlay.

    Returns dict with:
        predicted_class : str
        confidence      : float
        class_probs     : dict {class_name: prob}
        heatmap         : np.ndarray [H, W] float32
        overlay         : np.ndarray [H, W, 3] uint8 RGB
        overlay_pil     : PIL Image (ready to save/serve)
    """
    from configs.dataset_config import MODULES

    gradcam = build_gradcam(model, module_key)
    image_tensor = image_tensor.to(device)

    # ── Prediction ────────────────────────────
    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])

    # ── Grad-CAM ─────────────────────────────
    # Need gradients — re-run with grad enabled
    cam, _ = gradcam(image_tensor, class_idx=pred_idx)
    gradcam.remove_hooks()

    # ── Overlay ───────────────────────────────
    cfg     = MODULES[module_key]
    orig_np = np.array(original_pil.convert("RGB").resize((224, 224)))
    overlay = apply_heatmap_overlay(orig_np, cam)

    return {
        "predicted_class" : classes[pred_idx],
        "confidence"      : round(confidence * 100, 2),
        "class_probs"     : {cls: round(float(p) * 100, 2)
                             for cls, p in zip(classes, probs)},
        "heatmap"         : cam,
        "overlay"         : overlay,
        "overlay_pil"     : Image.fromarray(overlay),
    }
