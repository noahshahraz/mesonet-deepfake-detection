"""Device selection. This project targets Apple Silicon, so MPS is the default — NOT cuda."""
from __future__ import annotations

import torch


def get_device(preferred: str = "mps") -> torch.device:
    """Return the best available torch.device.

    Order of preference honours the config (default "mps"). Falls back gracefully so the
    code still runs on machines without MPS.
    """
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
