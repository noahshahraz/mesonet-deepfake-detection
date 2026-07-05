"""Evaluation entry point.

Usage:
    python -m src.eval --config configs/default.yaml --checkpoint checkpoints/best.pth
    # cross-dataset generalization (Phase 3): evaluate ONE checkpoint on another dataset
    python -m src.eval --checkpoint checkpoints/meso4_faceforensics_best.pth --dataset faces140k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import build_eval_loader
from src.models import build_model
from src.utils import get_device, load_config, predict_probs, set_seed
from src.utils.metrics import compute_metrics
from src.utils.overwrite import guard_overwrite
from src.utils.plots import save_confusion_matrix, save_roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MesoNet")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default=None, help="override cfg.data.name for cross-dataset eval")
    p.add_argument("--data-root", default=None, help="explicit data root (overrides --dataset root)")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-stem", default=None,
                   help="outputs/ filename stem (default <model>_train-<X>_eval-<Y>)")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing existing outputs for this stem")
    return p.parse_args()


def print_metrics_table(metrics: dict, order: list[str]) -> None:
    print(f"{'metric':<12} value")
    print("-" * 22)
    for name in order:
        value = metrics.get(name)
        if name == "confusion_matrix":
            (tn, fp), (fn, tp) = value
            print(f"{name:<12} [[tn {tn}, fp {fp}], [fn {fn}, tp {tp}]]")
        else:
            print(f"{name:<12} {value:.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["data"]["name"] = args.dataset
        cfg["data"]["root"] = f"data/{args.dataset}"
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))
    print(f"[eval] device = {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", cfg.model.name)
    model = build_model(
        model_name,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        image_size=cfg.data.image_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    trained_on = ckpt.get("dataset", "?")
    print(f"[eval] loaded {model_name} (trained on {trained_on}, epoch {ckpt.get('epoch', '?')}) "
          f"from {args.checkpoint}")

    out_dir = Path("outputs")
    stem = args.out_stem or f"{model_name}_train-{trained_on}_eval-{cfg.data.name}"
    guard_overwrite([out_dir / f"{stem}.json"], args.overwrite)

    loader = build_eval_loader(cfg)
    print(f"[eval] test set = {cfg.data.name} @ {cfg.data.root} ({len(loader.dataset)} images)")

    labels, probs = predict_probs(model, loader, device)
    threshold = args.threshold if args.threshold is not None else cfg.eval.threshold
    metrics = compute_metrics(labels, probs, threshold=threshold)
    print(f"\n[eval] {model_name} on {cfg.data.name} (threshold {threshold}):\n")
    print_metrics_table(metrics, cfg.eval.metrics)

    # T11 artefacts: metrics JSON, raw labels/probs (reused by T20 threshold tuning), plots.
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model_name,
        "trained_on": trained_on,
        "eval_dataset": cfg.data.name,
        "checkpoint": str(args.checkpoint),
        "threshold": threshold,
        "n_images": int(labels.shape[0]),
        **metrics,
    }
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(record, f, indent=2)
    np.savez(out_dir / f"{stem}_probs.npz", labels=labels, probs=probs)
    save_roc_curve(labels, probs, metrics["auc"], out_dir / f"{stem}_roc.png")
    save_confusion_matrix(metrics["confusion_matrix"], out_dir / f"{stem}_cm.png")
    print(f"\n[eval] saved metrics JSON, probs and plots to {out_dir}/{stem}*")


if __name__ == "__main__":
    main()
