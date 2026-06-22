# ─────────────────────────────────────────────
#  training/losses.py
#  Focal Loss + Label Smoothing for medical AI
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al. (2017)  https://arxiv.org/abs/1708.02002

    Downweights easy (well-classified) examples so the model focuses
    on hard, misclassified samples. Critical for medical datasets
    where rare disease classes are the hardest and most important.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma      : focusing parameter (0 = CrossEntropy, 2 = recommended)
        alpha      : per-class weight tensor (handles class imbalance)
        reduction  : 'mean' | 'sum' | 'none'
        label_smoothing: mixes hard labels with uniform distribution
    """

    def __init__(
        self,
        gamma           : float             = 2.0,
        alpha           : Optional[torch.Tensor] = None,
        reduction       : str               = "mean",
        label_smoothing : float             = 0.1,
    ):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits : torch.Tensor,   # [B, C]  raw logits
        targets: torch.Tensor,   # [B]     class indices
    ) -> torch.Tensor:

        num_classes = logits.size(1)

        # ── Label smoothing ───────────────────────
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    logits, self.label_smoothing / (num_classes - 1)
                )
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss   = -(smooth_targets * log_probs).sum(dim=1)
        else:
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss   = F.nll_loss(log_probs, targets, reduction="none")

        # ── Focal weighting ───────────────────────
        probs    = torch.exp(log_probs)
        p_t      = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w  = (1.0 - p_t) ** self.gamma

        # ── Class (alpha) weighting ───────────────
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_w = alpha_t * focal_w

        loss = focal_w * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Standard CE with label smoothing (lighter alternative to Focal Loss)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs   = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        return -(smooth * log_probs).sum(dim=1).mean()


def build_loss(
    module_key    : str,
    class_weights : Optional[torch.Tensor] = None,
    focal_gamma   : float = 2.0,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """
    Returns the best loss function for each module.

    brain_mri    → balanced dataset → FocalLoss (mild)
    chest_xray   → imbalanced       → FocalLoss + class weights
    skin_lesion  → very imbalanced  → FocalLoss + class weights (strong)
    retinal      → ordinal classes  → FocalLoss + label smoothing
    """
    gamma_map = {
    "brain_mri"   : 1.5,
    "chest_xray"  : 2.0,
    "skin_lesion" : 1.5,
    "retinal"     : 1.0,
    }
    gamma = gamma_map.get(module_key, focal_gamma)

    return FocalLoss(
        gamma           = gamma,
        alpha           = class_weights,
        label_smoothing = label_smoothing,
    )