# ─────────────────────────────────────────────
#  training/train.py
#  Entry point — run this to train any module
# ─────────────────────────────────────────────
#
#  Usage:
#    python -m training.train --module brain_mri
#    python -m training.train --module chest_xray --epochs 50 --batch_size 64
#    python -m training.train --module brain_mri  --freeze_backbone --epochs 10
#

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.dataset_config              import MODULES, BASE_MODEL_DIR
from training.datasets.dataloader_factory import build_dataloaders
from backend.models.model                import build_model
from training.losses                     import build_loss
from training.trainer                    import Trainer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Medical AI Detector")

    parser.add_argument("--module",    type=str, default="brain_mri",
                        choices=list(MODULES.keys()),
                        help="Disease module to train")
    parser.add_argument("--data_dir",  type=str, default="data/",
                        help="Root data directory")
    parser.add_argument("--save_dir",  type=str, default=BASE_MODEL_DIR)
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--patience",  type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone; train head only first")
    parser.add_argument("--no_pretrained",   action="store_true",
                        help="Train from scratch (not recommended)")
    parser.add_argument("--checkpoint",      type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--experiment", type=str, default="medical-ai-detector",
                        help="MLflow experiment name")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    print(f"\n{'═'*55}")
    print(f"  Medical AI Detector — Training")
    print(f"  Module : {MODULES[args.module]['name']}")
    print(f"  Device : {device}")
    print(f"{'═'*55}\n")

    # ── 1. DataLoaders ────────────────────────
    loaders = build_dataloaders(
        module_key  = args.module,
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        use_sampler = True,
    )

    # ── 2. Model ──────────────────────────────
    model = build_model(
        module_key      = args.module,
        pretrained      = not args.no_pretrained,
        freeze_backbone = args.freeze_backbone,
        checkpoint_path = args.checkpoint,
        device          = device,
    )

    # ── 3. Loss ───────────────────────────────
    # Get class weights from training dataset for imbalanced modules
    train_dataset  = loaders["train"].dataset
    class_weights  = train_dataset.get_class_weights().to(device)
    criterion      = build_loss(args.module, class_weights=class_weights)

    # ── 4. Optimizer ──────────────────────────
    # Lower LR for backbone, higher for head (differential LR)
    head_params     = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},   # 1e-5
        {"params": head_params,     "lr": args.lr},          # 1e-4
    ], weight_decay=1e-2)

    # ── 5. Scheduler ──────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = args.epochs,
        eta_min= args.lr * 0.01,
    )

    # ── 6. Train ──────────────────────────────
    trainer = Trainer(
        model           = model,
        module_key      = args.module,
        train_loader    = loaders["train"],
        val_loader      = loaders["val"],
        criterion       = criterion,
        optimizer       = optimizer,
        scheduler       = scheduler,
        device          = device,
        num_epochs      = args.epochs,
        patience        = args.patience,
        save_dir        = args.save_dir,
        experiment_name = args.experiment,
        use_amp         = (device == "cuda"),
    )

    history = trainer.fit()

    # ── 7. Final test evaluation ──────────────
    print("\n📊  Running test set evaluation...")
    from training.evaluate import evaluate_model
    evaluate_model(
        model      = model,
        loader     = loaders["test"],
        module_key = args.module,
        device     = device,
    )


if __name__ == "__main__":
    main()