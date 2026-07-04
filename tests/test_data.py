"""Tests for the data pipeline (T5 transforms, T6 dataloaders)."""
import pytest
import torch
from PIL import Image

from src.data import build_dataloaders, build_transforms
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


# ---------------------------------------------------------------------------- T6 dataloaders

def _make_fake_dataset(root, counts={"train": 6, "val": 4, "test": 4}):
    """Write a tiny real/fake ImageFolder tree of solid-colour jpgs."""
    for split, n in counts.items():
        for cls, colour in (("real", (0, 255, 0)), ("fake", (255, 0, 0))):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                Image.new("RGB", (64, 64), color=colour).save(d / f"{i}.jpg")


def _loader_cfg(root, **data_overrides) -> Config:
    data = {
        "root": str(root),
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "image_size": 64,
        "normalize": False,
        "max_per_class_train": None,
        "max_per_class_eval": None,
        "num_workers": 0,
    }
    data.update(data_overrides)
    return Config(
        {
            "seed": 42,
            "data": data,
            "augment": {"horizontal_flip": True},
            "train": {"batch_size": 4},
        }
    )


def test_dataloaders_label_convention(tmp_path):
    _make_fake_dataset(tmp_path)
    train, val, test = build_dataloaders(_loader_cfg(tmp_path))
    # real images are solid green, fake solid red -> recover the class from pixels and
    # verify real->0 / fake->1 end to end, not just via class_to_idx.
    for loader in (val, test):
        for x, y in loader:
            red, green = x[:, 0].mean(dim=(1, 2)), x[:, 1].mean(dim=(1, 2))
            is_fake = (red > green).long()
            assert torch.equal(is_fake, y)


def test_dataloaders_reject_wrong_folders(tmp_path):
    (tmp_path / "train" / "cats").mkdir(parents=True)
    (tmp_path / "train" / "dogs").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "train" / "cats" / "0.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "train" / "dogs" / "0.jpg")
    with pytest.raises(RuntimeError, match="real"):
        build_dataloaders(_loader_cfg(tmp_path))


def test_dataloaders_subsample_caps_train_only(tmp_path):
    _make_fake_dataset(tmp_path)
    train, val, test = build_dataloaders(_loader_cfg(tmp_path, max_per_class_train=2))
    assert len(train.dataset) == 4  # 2 per class
    assert len(val.dataset) == 8  # untouched
    assert len(test.dataset) == 8


def test_dataloaders_batch_shape(tmp_path):
    _make_fake_dataset(tmp_path)
    train, _, _ = build_dataloaders(_loader_cfg(tmp_path))
    x, y = next(iter(train))
    assert x.shape == (4, 3, 64, 64)
    assert y.shape == (4,)
