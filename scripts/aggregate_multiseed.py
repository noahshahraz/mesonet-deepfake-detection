"""Aggregate multi-seed eval JSONs into mean ± std tables (hardening task 2).

Reads outputs/<model>[_s<seed>]_train-<ds>_eval-<evalds>.json (seed 42 = legacy un-suffixed
stems) plus outputs/threshold_tuning.json, prints README-ready mean±std rows, and saves
outputs/multiseed_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ("meso4", "meso_inception4")
METHODS = ("ff_deepfakes", "ff_face2face")
GEN_EVALS = ("ff_deepfakes", "ff_face2face", "openforensics", "faces140k")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--outputs", default="outputs")
    return p.parse_args()


def stem(model: str, seed: int, train_ds: str, eval_ds: str) -> str:
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"{model}{suffix}_train-{train_ds}_eval-{eval_ds}"


def load(outputs: Path, model: str, seed: int, train_ds: str, eval_ds: str) -> dict:
    return json.loads((outputs / f"{stem(model, seed, train_ds, eval_ds)}.json").read_text())


def ms(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "values": a.tolist()}


def fmt(agg: dict, digits: int = 3) -> str:
    return f"{agg['mean']:.{digits}f} ± {agg['std']:.{digits}f}"


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    summary: dict = {"seeds": args.seeds, "reproduction": {}, "generalization": {}}

    tuning = {}
    tuning_path = outputs / "threshold_tuning.json"
    if tuning_path.exists():
        for row in json.loads(tuning_path.read_text()):
            tuning[(row["model"], row["dataset"], row.get("seed", 42))] = row

    print(f"seeds = {args.seeds}\n\n== Reproduction (FF++ per method, per-image, test) ==")
    for model in MODELS:
        for method in METHODS:
            runs = [load(outputs, model, s, method, method) for s in args.seeds]
            cell = {
                "acc": ms([r["accuracy"] for r in runs]),
                "auc": ms([r["auc"] for r in runs]),
            }
            tuned = [tuning[(model, method, s)]["test_acc_tuned"] for s in args.seeds
                     if (model, method, s) in tuning]
            if len(tuned) == len(args.seeds):
                cell["acc_tuned"] = ms(tuned)
            summary["reproduction"][f"{model}:{method}"] = cell
            tuned_str = fmt(cell["acc_tuned"]) if "acc_tuned" in cell else "n/a"
            print(f"{model:16s} {method:13s} acc {fmt(cell['acc'])} | "
                  f"tuned {tuned_str} | auc {fmt(cell['auc'])}")

    print("\n== Generalization (frozen meso4 ff_deepfakes checkpoints) ==")
    for eval_ds in GEN_EVALS:
        runs = [load(outputs, "meso4", s, "ff_deepfakes", eval_ds) for s in args.seeds]
        cell = {"acc": ms([r["accuracy"] for r in runs]), "auc": ms([r["auc"] for r in runs])}
        summary["generalization"][eval_ds] = cell
        print(f"{eval_ds:14s} acc {fmt(cell['acc'])} | auc {fmt(cell['auc'])}")

    # The two questions multi-seed answers:
    df = summary["reproduction"]
    deltas = [a - b for a, b in zip(df["meso4:ff_deepfakes"]["acc"]["values"],
                                    df["meso_inception4:ff_deepfakes"]["acc"]["values"])]
    summary["meso4_minus_mi4_deepfakes_acc"] = deltas
    print(f"\nMeso4 - MesoInception4 acc deltas on Deepfakes, per seed: "
          f"{[round(d, 4) for d in deltas]}")
    for ds in ("openforensics", "faces140k"):
        aucs = summary["generalization"][ds]["auc"]["values"]
        print(f"cross-dataset AUC per seed on {ds}: {[round(a, 4) for a in aucs]}")

    (outputs / "multiseed_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[aggregate] saved {outputs / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
