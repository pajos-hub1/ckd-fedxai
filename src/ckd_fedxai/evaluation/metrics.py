"""Evaluation metrics for CKD classification (§3.8.1).

Computes the full metric suite once, so every stage (central, federated,
private) reports results on a consistent basis.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Return accuracy, precision, recall, specificity, F1, and AUC-ROC.

    Args:
        y_true:  ground-truth labels (0/1)
        y_pred:  predicted labels (0/1)
        y_proba: predicted probabilities for the positive class (for AUC)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # sensitivity
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # specificity = TN / (TN + FP), computed from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC-ROC needs probabilities; skip gracefully if not provided
    if y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty one-line-per-metric formatting for printing."""
    order = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]
    lines = []
    for k in order:
        if k in metrics:
            lines.append(f"  {k:<12} {metrics[k]:.4f}")
    return "\n".join(lines)