"""Tests for the data pipeline (T5 transforms, T6 dataloaders)."""
import torch
from PIL import Image

from src.data import build_transforms
from src.utils.config import Config


def _cfg(normalize: bool = True) -> Config:
    return Config(
        {
            "data": {
                "image_size": 256,
                "normalize": normalize,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "augment": {
                "horizontal_flip": True,
                "rotation_degrees": 15,
                "zoom": 0.1,
                "brightness": 0.1,
            },
        }
    )


def test_transforms_output_shape_train_and_eval():
    img = Image.new("RGB", (123, 77))  # non-square input must still land at 256x256
    for train in (True, False):
        out = build_transforms(_cfg(), train=train)(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 256, 256)


def test_transforms_no_normalize_stays_in_unit_range():
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    out = build_transforms(_cfg(normalize=False), train=False)(img)
    assert out.max() <= 1.0 and out.min() >= 0.0
    assert torch.allclose(out, torch.ones_like(out))


def test_transforms_eval_is_deterministic():
    img = Image.effect_noise((256, 256), 64).convert("RGB")
    tf = build_transforms(_cfg(), train=False)
    assert torch.equal(tf(img), tf(img))
