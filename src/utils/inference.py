"""Shared inference helper: collect labels and P(fake) for a whole loader."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; return (labels, probs) as 1-D numpy arrays.

    The model emits raw logits (see README "Differences"); probs are sigmoid(logits) = P(fake).
    """
    model.eval()
    all_labels, all_probs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)
