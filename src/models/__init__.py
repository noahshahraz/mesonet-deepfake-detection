"""Model registry."""
from __future__ import annotations

import torch.nn as nn

from .meso4 import Meso4
from .meso_inception4 import MesoInception4


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "meso4":
        return Meso4(**kwargs)
    if name in {"meso_inception4", "mesoinception4", "meso_inception"}:
        return MesoInception4(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Expected 'meso4' or 'meso_inception4'.")


__all__ = ["Meso4", "MesoInception4", "build_model"]
