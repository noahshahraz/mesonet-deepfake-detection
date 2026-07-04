"""Training entry point.

Usage:
    python -m src.train --config configs/default.yaml

STATUS: SKELETON — implement in the build phase (TASKS.md T7–T9).
"""
from __future__ import annotations

import argparse

from src.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MesoNet")
    p.add_argument("--config", default="configs/default.yaml")
    # Optional CLI overrides (implement merge in build phase)
    p.add_argument("--model", default=None, help="meso4 | meso_inception4")
    p.add_argument("--dataset", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-per-class-train", type=int, default=None,
                   help="cap images/class for fast iterations on the large Kaggle sets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))
    print(f"[train] device = {device}")

    # TODO (T7): build_dataloaders(cfg)
    # TODO (T8): build_model(cfg.model.name, ...).to(device)
    # TODO (T9): BCEWithLogitsLoss + Adam(lr=cfg.train.lr); epoch loop with:
    #            - tqdm batches, forward/backward/step
    #            - per-epoch val metrics (src.utils.metrics.compute_metrics)
    #            - checkpoint best val AUC to cfg.train.checkpoint_dir
    #            - early stopping (cfg.train.early_stopping_patience)
    #            - log scalars to cfg.train.log_dir
    raise NotImplementedError("Implement in build phase — see TASKS.md T7–T9")


if __name__ == "__main__":
    main()
