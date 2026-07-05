"""Distill outputs/ (gitignored) into results/summary.json (committed) — task 5 infrastructure.

Every number the figure suite needs, in one committed file, so scripts/make_figures.py runs on a
fresh clone without the raw artifacts. Deterministic: content depends only on outputs/*.json.
Sections whose runs don't exist yet are written as null; figures degrade gracefully.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("outputs")
SEEDS = [42, 1, 2]
MODELS = ("meso4", "meso_inception4")
METHODS = ("ff_deepfakes", "ff_face2face")
EVAL_SETS = ("ff_deepfakes", "ff_face2face", "openforensics", "faces140k")

# Reference numbers from Afchar et al. (WIFS 2018), see docs/paper_diff.md for provenance.
PAPER = {
    "headline_video_level": {"deepfakes": 0.98, "face2face": 0.95},
    "video_level": {"deepfakes_their_dataset": 0.984, "face2face_c23": 0.953},
    "per_image_c23_face2face": {"meso4": 0.924, "meso_inception4": 0.934},
    "params": {"meso4": 27977, "meso_inception4": 28615},
}


def stem(model: str, seed: int, train_ds: str, eval_ds: str) -> str:
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"{model}{suffix}_train-{train_ds}_eval-{eval_ds}"


def read(name: str) -> dict | None:
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "n": n}


def per_seed_metric(model: str, train_ds: str, eval_ds: str, key: str) -> dict | None:
    per_seed, values = {}, []
    for s in SEEDS:
        rec = read(stem(model, s, train_ds, eval_ds))
        if rec is not None:
            per_seed[str(s)] = round(rec[key], 4)
            values.append(rec[key])
    if not values:
        return None
    return {"per_seed": per_seed, **stats(values)}


def main() -> None:
    summary: dict = {
        "seeds": SEEDS,
        "params": {"meso4": 27977, "meso_inception4": 28615, "xception": 20809001},
        "paper": PAPER,
        "chance_auc": 0.5,
    }

    # Reproduction: per model x method, per-seed acc/auc.
    repro = {}
    for m in MODELS:
        for ds in METHODS:
            acc = per_seed_metric(m, ds, ds, "accuracy")
            auc = per_seed_metric(m, ds, ds, "auc")
            if acc:
                repro[f"{m}:{ds}"] = {"acc": acc, "auc": auc}
    summary["reproduction"] = repro or None

    # Generalization: frozen meso4 ff_deepfakes across eval sets (multi-seed) + Xception (s42).
    gen: dict = {"meso4": {"train": "ff_deepfakes", "evals": {}},
                 "xception": {"train": "ff_deepfakes", "seeds": [42], "evals": {}}}
    for ev in EVAL_SETS:
        cell = {"acc": per_seed_metric("meso4", "ff_deepfakes", ev, "accuracy"),
                "auc": per_seed_metric("meso4", "ff_deepfakes", ev, "auc")}
        if cell["acc"]:
            gen["meso4"]["evals"][ev] = cell
        x = read(f"xception_train-ff_deepfakes_eval-{ev}")
        if x:
            gen["xception"]["evals"][ev] = {"acc": round(x["accuracy"], 4),
                                            "auc": round(x["auc"], 4)}
    summary["generalization"] = gen

    # Transfer matrix: every meso4 train->eval pair present in outputs/ (seed 42 snapshot).
    matrix = {}
    for train_ds in EVAL_SETS:
        row = {}
        for ev in EVAL_SETS:
            rec = read(stem("meso4", 42, train_ds, ev))
            if rec:
                row[ev] = round(rec["auc"], 4)
        if row:
            matrix[train_ds] = row
    summary["transfer_matrix_auc_seed42"] = matrix or None

    # Threshold tuning rows (12 after multi-seed; 4 legacy before).
    tuning = OUT / "threshold_tuning.json"
    summary["threshold"] = json.loads(tuning.read_text()) if tuning.exists() else None

    # Per-video results (task 3): outputs/pervideo_<model>_s<seed>_<method>.json
    pv = {}
    for m in MODELS:
        for ds in METHODS:
            per_seed_v, vids = {}, []
            for s in SEEDS:
                rec = read(f"pervideo_{m}_s{s}_{ds}")
                if rec:
                    per_seed_v[str(s)] = {"acc": round(rec["per_video"]["accuracy"], 4),
                                          "auc": round(rec["per_video"]["auc"], 4),
                                          "n_videos": rec["n_videos"]}
                    vids.append(rec)
            if vids:
                pv[f"{m}:{ds}"] = {
                    "per_seed": per_seed_v,
                    "acc": stats([r["per_video"]["accuracy"] for r in vids]),
                    "auc": stats([r["per_video"]["auc"] for r in vids]),
                    "frame_acc": stats([r["per_frame"]["accuracy"] for r in vids]),
                }
    summary["per_video"] = pv or None

    # Multi-source training (Task 7): meso4 trained on ff_deepfakes+ff_face2face union,
    # evaluated frozen on the same four test sets. Stems: meso4_multi_s<seed>_train-ff_multi_...
    multi: dict = {"train": "ff_deepfakes+ff_face2face", "evals": {}}
    for ev in EVAL_SETS:
        per_seed, accs, aucs = {}, [], []
        for s in SEEDS:
            rec = read(f"meso4_multi_s{s}_train-ff_multi_eval-{ev}")
            if rec:
                per_seed[str(s)] = {"acc": round(rec["accuracy"], 4),
                                    "auc": round(rec["auc"], 4)}
                accs.append(rec["accuracy"]); aucs.append(rec["auc"])
        if accs:
            multi["evals"][ev] = {"per_seed": per_seed,
                                  "acc": stats(accs), "auc": stats(aucs)}
    summary["multisource"] = multi if multi["evals"] else None

    # OpenForensics baseline (full-data runs, threshold 0.5).
    base = {}
    for m in MODELS:
        rec = read(f"{m}_train-openforensics_eval-openforensics")
        if rec:
            base[m] = {k: round(rec[k], 4) for k in ("accuracy", "auc", "precision",
                                                     "recall", "f1")}
    summary["baseline_openforensics"] = base or None

    # Reverse-direction control observations (two same-seed training runs; see README).
    summary["reverse_control_auc"] = {"openforensics_to_ff_deepfakes": [0.4592, 0.5004]}

    # Stems whose probs npz the distribution/ROC/calibration figures want (when present).
    summary["probs_stems"] = {
        "in_domain": stem("meso4", 42, "ff_deepfakes", "ff_deepfakes"),
        "cross_140k": stem("meso4", 42, "ff_deepfakes", "faces140k"),
        "cross_openforensics": stem("meso4", 42, "ff_deepfakes", "openforensics"),
        "repro_runs": {f"{m}:{ds}": stem(m, 42, ds, ds) for m in MODELS for ds in METHODS},
    }

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    missing = [k for k, v in summary.items() if v is None]
    print(f"[summary] wrote results/summary.json ({(out / 'summary.json').stat().st_size} bytes)")
    print(f"[summary] incomplete sections: {missing or 'none'}")

    # Keep the dashboard's inlined copy in sync (docs/dashboard.html embeds SUMMARY verbatim).
    dash = Path("docs/dashboard.html")
    if dash.exists():
        lines = dash.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.lstrip().startswith("const SUMMARY = "):
                lines[i] = f"const SUMMARY = {json.dumps(summary, separators=(',', ':'))};\n"
                dash.write_text("".join(lines))
                print("[summary] re-injected SUMMARY into docs/dashboard.html")
                break
        else:
            print("[summary] WARNING: docs/dashboard.html has no 'const SUMMARY = ' line")


if __name__ == "__main__":
    main()
