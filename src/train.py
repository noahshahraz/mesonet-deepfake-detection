"""Training entry point.

Usage:
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import build_dataloaders
from src.models import build_model
from src.utils import get_device, load_config, predict_probs, set_seed
from src.utils.metrics import compute_metrics
from src.utils.overwrite import guard_overwrite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MesoNet")
    p.add_argument("--config", default="configs/default.yaml")
    # Optional CLI overrides of configs/default.yaml
    p.add_argument("--model", default=None, help="meso4 | meso_inception4")
    p.add_argument("--dataset", default=None, help="dataset name; sets data.root to data/<name>")
    p.add_argument("--data-root", default=None, help="explicit data root (overrides --dataset root)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="override train.batch_size (e.g. smaller for the Xception baseline)")
    p.add_argument("--lr", type=float, default=None,
                   help="override train.lr (e.g. lower for fine-tuning pretrained weights)")
    p.add_argument("--max-per-class-train", type=int, default=None,
                   help="cap images/class for fast iterations on the large Kaggle sets")
    p.add_argument("--run-name", default=None,
                   help="artifact name (default <model>_<dataset>); checkpoints/logs use this")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing an existing run's checkpoint/log")
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
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.max_per_class_train is not None:
        cfg["data"]["max_per_class_train"] = args.max_per_class_train
    return cfg


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower()
    if name != "adam":
        raise ValueError(f"Unsupported optimizer '{name}' — the paper (and this repo) use adam")
    return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss, n_seen = 0.0, 0
    for x, y in tqdm(loader, desc=f"epoch {epoch}", leave=False):
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        n_seen += x.shape[0]
    return total_loss / max(n_seen, 1)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))
    print(f"[train] device = {device}")

    # Resolve artifact paths and guard against clobbering a previous run before any real work.
    run_name = args.run_name or f"{cfg.model.name}_{cfg.data.name}"
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_path = ckpt_dir / f"{run_name}_best.pth"
    log_path = Path(cfg.train.log_dir) / f"{run_name}.jsonl"
    guard_overwrite([ckpt_path, log_path], args.overwrite)

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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(cfg, model)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # fresh log per run

    best_auc, best_epoch, epochs_without_improvement = -math.inf, -1, 0
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        labels, probs = predict_probs(model, val_loader, device)
        val = compute_metrics(labels, probs, threshold=cfg.eval.threshold)
        print(
            f"[epoch {epoch:3d}] train_loss {train_loss:.4f} | "
            f"val acc {val['accuracy']:.4f} auc {val['auc']:.4f} f1 {val['f1']:.4f}"
        )
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss,
                                **{f"val_{k}": v for k, v in val.items()}}) + "\n")

        if not math.isnan(val["auc"]) and val["auc"] > best_auc:
            best_auc, best_epoch, epochs_without_improvement = val["auc"], epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": cfg.model.name,
                    "dataset": cfg.data.name,
                    "epoch": epoch,
                    "val_metrics": val,
                    "config": dict(cfg),
                },
                ckpt_path,
            )
            shutil.copyfile(ckpt_path, ckpt_dir / "best.pth")  # generic name for the README one-liner
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
                print(f"[train] early stop at epoch {epoch} (no val AUC gain for "
                      f"{cfg.train.early_stopping_patience} epochs)")
                break

    print(f"[train] best val AUC {best_auc:.4f} @ epoch {best_epoch} -> {ckpt_path}")


if __name__ == "__main__":
    main()
