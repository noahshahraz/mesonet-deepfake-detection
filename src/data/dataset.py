"""Data loading for the three supported datasets.

All datasets are normalised to the same on-disk layout so a single ImageFolder-style loader
works everywhere:

    <root>/<split>/real/*.jpg      -> label 0
    <root>/<split>/fake/*.jpg      -> label 1

- OpenForensics (manjilkarki) and 140k already ship pre-cropped faces in real/fake folders.
- FaceForensics++ frames must first be face-cropped into this layout by
  scripts/extract_faces_ffpp.py.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_transforms(cfg, train: bool) -> transforms.Compose:
    """Return the transform pipeline for one split, driven entirely by the config.

    Eval: Resize -> ToTensor [-> Normalize].
    Train adds the paper-style light augmentation (flip, rotation+zoom, brightness) with
    each knob read from cfg.augment; a missing/zero knob disables that augmentation.
    """
    size = cfg.data.image_size
    pipeline = [transforms.Resize((size, size))]

    if train:
        aug = cfg.get_path("augment", {}) or {}
        if aug.get("horizontal_flip", False):
            pipeline.append(transforms.RandomHorizontalFlip())
        degrees = aug.get("rotation_degrees", 0) or 0
        zoom = aug.get("zoom", 0) or 0
        if degrees or zoom:
            # Keras-style zoom_range z means scale in [1-z, 1+z]; one affine covers both knobs.
            scale = (1.0 - zoom, 1.0 + zoom) if zoom else None
            pipeline.append(transforms.RandomAffine(degrees=degrees, scale=scale))
        brightness = aug.get("brightness", 0) or 0
        if brightness:
            pipeline.append(transforms.ColorJitter(brightness=brightness))

    pipeline.append(transforms.ToTensor())
    if cfg.get_path("data.normalize", False):
        pipeline.append(transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std))
    return transforms.Compose(pipeline)


class RealFakeFolder(ImageFolder):
    """ImageFolder that pins real=0, fake=1 — alphabetical order would give the opposite."""

    def find_classes(self, directory) -> Tuple[List[str], Dict[str, int]]:
        classes, _ = super().find_classes(directory)
        if set(classes) != {"real", "fake"}:
            raise RuntimeError(
                f"{directory}: expected exactly 'real' and 'fake' subfolders, got {classes}"
            )
        return ["real", "fake"], {"real": 0, "fake": 1}


def _subsample_per_class(dataset: RealFakeFolder, max_per_class: int | None, seed: int) -> Dataset:
    """Deterministically cap the dataset at max_per_class images per class."""
    if not max_per_class:
        return dataset
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        by_class[target].append(idx)
    rng = random.Random(seed)
    keep: List[int] = []
    for target in sorted(by_class):
        indices = by_class[target]
        if len(indices) > max_per_class:
            indices = rng.sample(indices, max_per_class)
        keep.extend(sorted(indices))
    return Subset(dataset, keep)


def _make_dataset(cfg, root: str | Path, split: str, is_train: bool,
                  max_per_class: int | None) -> Dataset:
    dataset = RealFakeFolder(
        Path(root).expanduser() / split, transform=build_transforms(cfg, train=is_train)
    )
    assert dataset.class_to_idx == {"real": 0, "fake": 1}, dataset.class_to_idx
    return _subsample_per_class(dataset, max_per_class, cfg.seed)


def _wrap_loader(cfg, dataset: Dataset, is_train: bool) -> DataLoader:
    num_workers = cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False,  # no effect on MPS
        persistent_workers=num_workers > 0,
    )


def _make_loader(cfg, split: str, is_train: bool, max_per_class: int | None) -> DataLoader:
    return _wrap_loader(cfg, _make_dataset(cfg, cfg.data.root, split, is_train, max_per_class),
                        is_train)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) DataLoaders for the dataset at cfg.data.root."""
    return (
        _make_loader(cfg, cfg.data.train_split, True, cfg.get_path("data.max_per_class_train")),
        _make_loader(cfg, cfg.data.val_split, False, cfg.get_path("data.max_per_class_eval")),
        _make_loader(cfg, cfg.data.test_split, False, cfg.get_path("data.max_per_class_eval")),
    )


def build_eval_loader(cfg, split: str | None = None) -> DataLoader:
    """Build just one eval-transform loader (default: the test split) — used by src/eval.py."""
    return _make_loader(
        cfg, split or cfg.data.test_split, False, cfg.get_path("data.max_per_class_eval")
    )


def build_multi_dataloaders(cfg, roots: Sequence[str | Path]) -> Tuple[DataLoader, DataLoader,
                                                                       DataLoader]:
    """Multi-source training (Task 7): the union of several dataset roots per split.

    Each root must follow the standard <root>/<split>/{real,fake} layout; the real=0/fake=1
    mapping is asserted per root inside _make_dataset. Per-class subsampling caps apply per
    root. Evaluation on held-out datasets stays single-root via build_eval_loader.
    """
    if not roots:
        raise ValueError("build_multi_dataloaders needs at least one root")
    splits = (
        (cfg.data.train_split, True, cfg.get_path("data.max_per_class_train")),
        (cfg.data.val_split, False, cfg.get_path("data.max_per_class_eval")),
        (cfg.data.test_split, False, cfg.get_path("data.max_per_class_eval")),
    )
    loaders = []
    for split, is_train, cap in splits:
        parts = [_make_dataset(cfg, root, split, is_train, cap) for root in roots]
        loaders.append(_wrap_loader(cfg, ConcatDataset(parts), is_train))
    return tuple(loaders)
