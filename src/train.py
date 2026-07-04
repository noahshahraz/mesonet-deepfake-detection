"""Training entry point.

Usage:
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from src.data import build_dataloaders
from src.models import build_model
from src.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MesoNet")
    p.add_argument("--config", default="configs/default.yaml")
    # Optional CLI overrides of configs/default.yaml
    p.add_argument("--model", default=None, help="meso4 | meso_inception4")
    p.add_argument("--dataset", default=None, help="dataset name; sets data.root to data/<name>")
    p.add_argument("--data-root", default=None, help="explicit data root (overrides --dataset root)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-per-class-train", type=int, default=None,
                   help="cap images/class for fast iterations on the large Kaggle sets")
    return p.parse_args()


def apply_overrides(cfg, args: argparse.Namespace):
    if args.model:
        cfg["model"]["name"] = args.model
    if args.dataset:
        cfg["data"]["name"] = args.dataset
        cfg["data"]["root"] = f"data/{args.dataset}"
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.max_per_class_train is not None:
        cfg["data"]["max_per_class_train"] = args.max_per_class_train
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))
    print(f"[train] device = {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(
        f"[train] dataset = {cfg.data.name} @ {cfg.data.root} | "
        f"train/val/test = {len(train_loader.dataset)}/{len(val_loader.dataset)}"
        f"/{len(test_loader.dataset)}"
    )

    model = build_model(
        cfg.model.name,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        image_size=cfg.data.image_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model = {cfg.model.name} ({n_params:,} params) on {device}")

    # Sanity: one forward pass on a real batch, on-device. (T9 replaces this with the loop.)
    x, y = next(iter(train_loader))
    logits = model(x.to(device))
    assert logits.shape == (x.shape[0], cfg.model.num_classes)
    print(f"[train] sanity forward OK: batch {tuple(x.shape)} -> logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
