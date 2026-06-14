# ─────────────────────────────────────────────
#  backend/models/model.py
#  EfficientNetV2 backbone + per-module heads
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import timm
from configs.dataset_config import MODULES


class MedicalClassifier(nn.Module):
    """
    EfficientNetV2-S backbone with a custom classification head.
    Supports all 4 disease modules via dynamic num_classes.

    Architecture:
        EfficientNetV2-S (pretrained on ImageNet-21k)
            └── Global Average Pooling  [B, 1280]
                └── Dropout(0.3)
                    └── FC(1280 → 512) + GELU + BN
                        └── Dropout(0.2)
                            └── FC(512 → num_classes)
    """

    def __init__(
        self,
        module_key      : str,
        pretrained      : bool  = True,
        dropout_rate    : float = 0.3,
        freeze_backbone : bool  = False,
    ):
        super().__init__()
        assert module_key in MODULES, f"Unknown module: {module_key}"

        self.module_key  = module_key
        self.num_classes = MODULES[module_key]["num_classes"]

        # ── Backbone ─────────────────────────────
        # Try preferred model name first, fall back to alternatives
        for model_name in ["tf_efficientnetv2_s", "efficientnetv2_s", "efficientnet_b3"]:
            try:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained  = pretrained,
                    num_classes = 0,
                    global_pool = "avg",
                )
                print(f"  ✓ Backbone: {model_name}")
                break
            except RuntimeError:
                continue

        in_features = self.backbone.num_features

        # ── Optionally freeze backbone ────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Classification head ───────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(512, self.num_classes),
        )

        # ── Weight initialisation ─────────────────
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)     # [B, 1280]
        logits   = self.head(features)  # [B, num_classes]
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns feature embeddings before the classification head (for Grad-CAM)."""
        return self.backbone(x)

    def unfreeze_backbone(self, from_layer: int = -3):
        """
        Gradually unfreeze backbone layers for fine-tuning.
        Call after initial head-only training converges.

        Args:
            from_layer: unfreeze from this block index onwards (-3 = last 3 blocks)
        """
        blocks = list(self.backbone.children())
        for block in blocks[from_layer:]:
            for param in block.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze from layer {from_layer} | Trainable params: {n_trainable:,}")


# ── Factory function ──────────────────────────
def build_model(
    module_key      : str,
    pretrained      : bool  = True,
    freeze_backbone : bool  = False,
    checkpoint_path : str   = None,
    device          : str   = "cpu",
) -> MedicalClassifier:
    """
    Builds and optionally loads a checkpoint.

    Args:
        module_key      : 'chest_xray' | 'skin_lesion' | 'brain_mri' | 'retinal'
        pretrained      : use ImageNet pretrained weights
        freeze_backbone : freeze backbone, train head only (good for small datasets)
        checkpoint_path : path to a saved .pt checkpoint to resume from
        device          : 'cuda' | 'mps' | 'cpu'
    """
    model = MedicalClassifier(
        module_key      = module_key,
        pretrained      = pretrained,
        freeze_backbone = freeze_backbone,
    ).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNetV2-S → {MODULES[module_key]['name']}")
    print(f"  Total params    : {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Classes         : {MODULES[module_key]['classes']}\n")

    return model


# ── Sanity check ──────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model("brain_mri", pretrained=False, device=device)

    dummy  = torch.randn(4, 3, 224, 224).to(device)
    out    = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")    # [4, 4]