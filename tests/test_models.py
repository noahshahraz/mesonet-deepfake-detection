"""Smoke tests for the model definitions — these already pass on the scaffold.

    pytest -q
"""
import torch

from src.models import build_model


def test_meso4_forward_shape():
    model = build_model("meso4")
    out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1)


def test_meso_inception4_forward_shape():
    model = build_model("meso_inception4")
    out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1)


def test_param_counts_are_compact():
    # The paper's headline: both nets are tiny (~28k params).
    for name in ("meso4", "meso_inception4"):
        n = sum(p.numel() for p in build_model(name).parameters())
        assert n < 50_000, f"{name} has {n} params, expected < 50k"
