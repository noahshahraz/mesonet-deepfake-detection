"""T20 — decision-threshold tuning, selected on the VALIDATION split only.

For each (model, dataset) run: load the checkpoint, compute val-split probabilities, pick the
accuracy-maximising threshold with utils.metrics.best_threshold, then report TEST accuracy/F1 at
threshold 0.5 vs the val-selected threshold using the test probs already saved by src.eval
(outputs/<model>_train-<ds>_eval-<ds>_probs.npz). The test set is never used for selection.

Usage:
    python scripts/tune_threshold.py --data-root-base ~/mesonet-data \
        --runs meso4:ff_deepfakes meso_inception4:ff_deepfakes \
               meso4:ff_face2face meso_inception4:ff_face2face
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_eval_loader  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import get_device, load_config, predict_probs, set_seed  # noqa: E402
from src.utils.metrics import best_threshold, compute_metrics  # noqa: E402

DEFAULT_RUNS = ["meso4:ff_deepfakes", "meso_inception4:ff_deepfakes",
                "meso4:ff_face2face", "meso_inception4:ff_face2face"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-root-base", default=str(Path.home() / "mesonet-data"))
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="model:dataset[:seed] triples; seed 42 (default) uses the legacy "
                        "un-suffixed artifact names")
    p.add_argument("--metric", default="accuracy", help="metric best_threshold maximises")
    p.add_argument("--out", default="outputs/threshold_tuning.json")
    return p.parse_args()


def run_names(model: str, ds: str, seed: int) -> tuple[str, str]:
    """(checkpoint run name, eval stem) for a seeded run; seed 42 keeps legacy names."""
    suffix = "" if seed == 42 else f"_s{seed}"
    ckpt_run = f"{model}_{ds}{suffix}"
    stem = f"{model}{suffix}_train-{ds}_eval-{ds}"
    return ckpt_run, stem


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))
    print(f"[tune] device = {device}\n")

    rows = []
    for run in args.runs:
        parts = run.split(":")
        model_name, ds = parts[0], parts[1]
        seed = int(parts[2]) if len(parts) > 2 else 42
        ckpt_run, stem = run_names(model_name, ds, seed)
        cfg["data"]["name"] = ds
        cfg["data"]["root"] = str(Path(args.data_root_base).expanduser() / ds)

        ckpt = torch.load(f"checkpoints/{ckpt_run}_best.pth",
                          map_location="cpu", weights_only=False)
        model = build_model(model_name, num_classes=cfg.model.num_classes,
                            dropout=cfg.model.dropout, image_size=cfg.data.image_size).to(device)
        model.load_state_dict(ckpt["model_state"])

        val_labels, val_probs = predict_probs(
            model, build_eval_loader(cfg, cfg.data.val_split), device)
        t_star = best_threshold(val_labels, val_probs, metric=args.metric)

        npz = np.load(f"outputs/{stem}_probs.npz")
        base = compute_metrics(npz["labels"], npz["probs"], threshold=cfg.eval.threshold)
        tuned = compute_metrics(npz["labels"], npz["probs"], threshold=t_star)
        rows.append({"model": model_name, "dataset": ds, "seed": seed, "threshold": t_star,
                     "test_acc_05": base["accuracy"], "test_acc_tuned": tuned["accuracy"],
                     "test_f1_05": base["f1"], "test_f1_tuned": tuned["f1"]})
        print(f"{model_name:16s} {ds:13s} s{seed:<3d} t*={t_star:.2f} (val) | "
              f"test acc {base['accuracy']:.4f} -> {tuned['accuracy']:.4f} "
              f"(+{tuned['accuracy'] - base['accuracy']:+.4f}) | "
              f"f1 {base['f1']:.4f} -> {tuned['f1']:.4f}")

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[tune] saved {out}")


if __name__ == "__main__":
    main()
