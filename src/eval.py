"""Evaluation entry point.

Usage:
    python -m src.eval --config configs/default.yaml --checkpoint checkpoints/best.pth
    # cross-dataset generalization (Phase 3): evaluate ONE checkpoint on another dataset
    python -m src.eval --checkpoint checkpoints/ffpp_best.pth --dataset faces140k

STATUS: SKELETON — implement in the build phase (TASKS.md T10–T11).
"""
from __future__ import annotations

import argparse

from src.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MesoNet")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default=None, help="override cfg.data.name for cross-dataset eval")
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))
    print(f"[eval] device = {device}")

    # TODO (T10): load model + checkpoint; build test loader (cfg / --dataset override)
    # TODO (T11): run inference -> probs; compute_metrics(labels, probs, threshold)
    #             print table + save JSON to outputs/ ; save ROC + confusion-matrix plots
    raise NotImplementedError("Implement in build phase — see TASKS.md T10–T11")


if __name__ == "__main__":
    main()
