# ─────────────────────────────────────────────
#  training/evaluate.py
#  Test-set evaluation + plots
# ─────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp   import autocast

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
)
from pathlib import Path
from configs.dataset_config import MODULES


@torch.no_grad()
def evaluate_model(
    model      : nn.Module,
    loader     : DataLoader,
    module_key : str,
    device     : str  = "cpu",
    save_dir   : str  = "logs/",
    use_amp    : bool = False,
) -> dict:
    """
    Full evaluation on a DataLoader split.
    Computes accuracy, macro AUC, F1, and saves:
      - Confusion matrix heatmap
      - ROC curves (one-vs-rest per class)
      - Classification report (text)
    """
    model.eval()
    cfg     = MODULES[module_key]
    classes = cfg["classes"]
    save_dir = Path(save_dir) / module_key
    save_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs .extend(probs)
        all_preds .extend(preds)
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Metrics ───────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names = classes,
        digits       = 4,
    )
    try:
        macro_auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro"
        )
    except ValueError:
        macro_auc = 0.0

    print("\n" + "─" * 55)
    print(f"  Test Results — {cfg['name']}")
    print("─" * 55)
    print(report)
    print(f"  Macro AUC : {macro_auc:.4f}")

    # Save report
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nMacro AUC: {macro_auc:.4f}\n")

    # ── Confusion matrix ──────────────────────
    _plot_confusion_matrix(all_labels, all_preds, classes, save_dir)

    # ── ROC curves ────────────────────────────
    _plot_roc_curves(all_labels, all_probs, classes, save_dir)

    return {
        "macro_auc" : macro_auc,
        "report"    : report,
        "preds"     : all_preds,
        "labels"    : all_labels,
        "probs"     : all_probs,
    }


def _plot_confusion_matrix(labels, preds, classes, save_dir):
    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalise

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_n],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = save_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrix → {path}")


def _plot_roc_curves(labels, probs, classes, save_dir):
    n_classes = len(classes)
    fig, ax   = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        binary = (labels == i).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — One vs Rest", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = save_dir / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈  ROC curves       → {path}")