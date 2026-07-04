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
    if name == "xception":
        # T19 baseline: ImageNet-pretrained, single logit to match the BCEWithLogits eval path.
        # Fully convolutional + global pooling, so it accepts our 256x256 pipeline unchanged
        # (native size is 299; we keep 256 so the data pipeline is identical across models).
        import timm

        return timm.create_model(
            "legacy_xception",
            pretrained=True,
            num_classes=kwargs.get("num_classes", 1),
            drop_rate=kwargs.get("dropout", 0.0),
        )
    raise ValueError(
        f"Unknown model '{name}'. Expected 'meso4', 'meso_inception4' or 'xception'."
    )


__all__ = ["Meso4", "MesoInception4", "build_model"]
