"""Binary-classification metrics for forgery detection.

Convention: label 1 = FAKE (positive class), label 0 = REAL. `probs` are P(fake).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    labels = np.asarray(labels).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    preds = (probs >= threshold).astype(int)

    out: Dict[str, object] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    # AUC is undefined if only one class is present in `labels`.
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auc"] = float("nan")
    return out


def best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str = "accuracy") -> float:
    """Sweep thresholds and return the one maximising the chosen metric (Phase 3)."""
    labels = np.asarray(labels).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    grid = np.linspace(0.01, 0.99, 99)
    scores = []
    for t in grid:
        m = compute_metrics(labels, probs, threshold=float(t))
        scores.append(m[metric])
    return float(grid[int(np.argmax(scores))])
