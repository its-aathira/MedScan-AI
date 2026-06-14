# ─────────────────────────────────────────────
#  training/trainer.py
#  Full training loop — mixed precision + MLflow
# ─────────────────────────────────────────────

import os
import time
import copy
from pathlib import Path
from typing  import Dict, Optional, Tuple

import numpy  as np
import torch
import torch.nn as nn
from torch.cuda.amp        import GradScaler, autocast
from torch.utils.data      import DataLoader
from sklearn.metrics       import (
    accuracy_score, roc_auc_score,
    f1_score, classification_report,
)
import mlflow
import mlflow.pytorch

from configs.dataset_config import MODULES, BASE_MODEL_DIR


# ─────────────────────────────────────────────
class EarlyStopping:
    """Stops training when val metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best + self.min_delta if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─────────────────────────────────────────────
class Trainer:
    """
    Full training engine with:
      - Mixed precision (AMP)
      - Gradient clipping
      - Cosine LR scheduler with warmup
      - Early stopping
      - Best-model checkpointing (by val AUC)
      - MLflow experiment tracking
    """

    def __init__(
        self,
        model         : nn.Module,
        module_key    : str,
        train_loader  : DataLoader,
        val_loader    : DataLoader,
        criterion     : nn.Module,
        optimizer     : torch.optim.Optimizer,
        scheduler     = None,
        device        : str  = "cpu",
        num_epochs    : int  = 30,
        patience      : int  = 7,
        save_dir      : str  = BASE_MODEL_DIR,
        experiment_name: str = "medical-ai-detector",
        use_amp       : bool = True,
    ):
        self.model          = model
        self.module_key     = module_key
        self.cfg            = MODULES[module_key]
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.num_epochs     = num_epochs
        self.save_dir       = Path(save_dir)
        self.experiment_name= experiment_name
        self.use_amp        = use_amp and ("cuda" in device)

        self.scaler         = GradScaler(enabled=self.use_amp)
        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.best_auc       = 0.0
        self.best_weights   = None

        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── One epoch ─────────────────────────────
    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds .extend(preds)
            all_labels.extend(labels.cpu().numpy())

        n    = len(self.train_loader.dataset)
        acc  = accuracy_score(all_labels, all_preds)
        return {"loss": total_loss / n, "acc": acc}

    # ── Validation ────────────────────────────
    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs .extend(probs)
            all_preds .extend(preds)
            all_labels.extend(labels.cpu().numpy())

        n           = len(self.val_loader.dataset)
        all_labels  = np.array(all_labels)
        all_probs   = np.array(all_probs)

        acc  = accuracy_score(all_labels, all_preds)
        f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # AUC: one-vs-rest for multi-class
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0

        return {"loss": total_loss / n, "acc": acc, "f1": f1, "auc": auc}

    # ── Save checkpoint ───────────────────────
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        path = self.save_dir / f"{self.module_key}_best.pt"
        torch.save({
            "epoch"             : epoch,
            "model_state_dict"  : self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auc"           : metrics["auc"],
            "val_acc"           : metrics["acc"],
            "module_key"        : self.module_key,
            "classes"           : self.cfg["classes"],
        }, path)
        print(f"  💾  Saved best model → {path}")

    # ── Main training loop ────────────────────
    def fit(self) -> Dict:
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.module_key):
            mlflow.log_params({
                "module"     : self.module_key,
                "num_classes": self.cfg["num_classes"],
                "num_epochs" : self.num_epochs,
                "optimizer"  : type(self.optimizer).__name__,
                "device"     : self.device,
                "amp"        : self.use_amp,
            })

            print(f"\n{'═'*55}")
            print(f"  Training: {self.cfg['name']}")
            print(f"  Device  : {self.device}  |  AMP: {self.use_amp}")
            print(f"{'═'*55}")

            history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "val_auc":[]}

            for epoch in range(1, self.num_epochs + 1):
                t0 = time.time()

                train_m = self._train_epoch()
                val_m   = self._val_epoch()

                if self.scheduler:
                    self.scheduler.step()

                elapsed = time.time() - t0

                # ── Log to MLflow ─────────────────
                mlflow.log_metrics({
                    "train_loss": train_m["loss"],
                    "train_acc" : train_m["acc"],
                    "val_loss"  : val_m["loss"],
                    "val_acc"   : val_m["acc"],
                    "val_f1"    : val_m["f1"],
                    "val_auc"   : val_m["auc"],
                }, step=epoch)

                # ── History ───────────────────────
                for k in history:
                    src = train_m if k.startswith("train") else val_m
                    history[k].append(src[k.split("_")[1]])

                # ── Print ─────────────────────────
                print(
                    f"  Ep {epoch:03d}/{self.num_epochs} "
                    f"| train loss {train_m['loss']:.4f} acc {train_m['acc']:.4f} "
                    f"| val loss {val_m['loss']:.4f} acc {val_m['acc']:.4f} "
                    f"auc {val_m['auc']:.4f} f1 {val_m['f1']:.4f} "
                    f"| {elapsed:.1f}s"
                )

                # ── Save best ─────────────────────
                if val_m["auc"] > self.best_auc:
                    self.best_auc     = val_m["auc"]
                    self.best_weights = copy.deepcopy(self.model.state_dict())
                    self._save_checkpoint(epoch, val_m)

                # ── Early stopping ────────────────
                if self.early_stopping(val_m["auc"]):
                    print(f"\n  ⏹  Early stopping at epoch {epoch}")
                    break

            # ── Restore best weights ──────────────
            if self.best_weights:
                self.model.load_state_dict(self.best_weights)

            print(f"\n  ✅  Best val AUC: {self.best_auc:.4f}")
            mlflow.log_metric("best_val_auc", self.best_auc)
            mlflow.pytorch.log_model(self.model, "model")

        return history