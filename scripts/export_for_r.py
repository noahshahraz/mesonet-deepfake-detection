"""Export per-image predictions to analysis/data/predictions.csv for the R analysis (Task 6).

One row per evaluated image: model, seed, train_source, eval_dataset, method, video_id,
label, prob_fake.

No inference is run: src.eval saved (labels, probs) to outputs/<stem>_probs.npz in dataset
order (eval loaders never shuffle), so paths/video ids are reconstructed by rebuilding the same
sorted listing and verified by asserting the reconstructed labels equal the stored ones.
`video_id` is the crop filename stem up to `_f` (see scripts/eval_per_video.py); empty for the
non-video Kaggle datasets.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import RealFakeFolder  # noqa: E402

SEEDS = (42, 1, 2)
METHOD_NAME = {"ff_deepfakes": "Deepfakes", "ff_face2face": "Face2Face"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root-base", default=str(Path.home() / "mesonet-data"))
    p.add_argument("--out", default="analysis/data/predictions.csv")
    return p.parse_args()


def video_id(path: str) -> str:
    return Path(path).stem.rsplit("_f", 1)[0]


def samples_for(base: Path, eval_ds: str) -> list[tuple[str, int]]:
    return RealFakeFolder(base / eval_ds / "test").samples


def rows_for(stem: str, samples, *, model: str, seed: int, train_source: str,
             eval_ds: str, with_video: bool):
    npz_path = Path("outputs") / f"{stem}_probs.npz"
    if not npz_path.exists():
        print(f"[export] SKIP (no npz): {stem}")
        return
    arr = np.load(npz_path)
    labels, probs = arr["labels"], arr["probs"]
    assert len(samples) == len(labels), f"{stem}: {len(samples)} paths vs {len(labels)} probs"
    recon = np.array([lab for _, lab in samples])
    assert np.array_equal(recon, labels), f"{stem}: reconstructed labels mismatch npz order"
    method = METHOD_NAME.get(eval_ds, "")
    for (path, lab), prob in zip(samples, probs):
        yield {
            "model": model, "seed": seed, "train_source": train_source,
            "eval_dataset": eval_ds, "method": method,
            "video_id": video_id(path) if with_video else "",
            "label": int(lab), "prob_fake": float(prob),
        }


def main() -> None:
    args = parse_args()
    base = Path(args.data_root_base).expanduser()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # cache the (path,label) listings once per eval dataset
    listings = {ds: samples_for(base, ds)
                for ds in ("ff_deepfakes", "ff_face2face", "openforensics", "faces140k")}

    n = 0
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "train_source", "eval_dataset",
                                               "method", "video_id", "label", "prob_fake"])
        writer.writeheader()

        for seed in SEEDS:
            sfx = "" if seed == 42 else f"_s{seed}"
            # in-domain reproduction runs (GLMM rows): trained and evaluated on the same method
            for model in ("meso4", "meso_inception4"):
                for ds in ("ff_deepfakes", "ff_face2face"):
                    stem = f"{model}{sfx}_train-{ds}_eval-{ds}"
                    for row in rows_for(stem, listings[ds], model=model, seed=seed,
                                        train_source=ds, eval_ds=ds, with_video=True):
                        writer.writerow(row); n += 1
            # cross-dataset / cross-method evals of the frozen Deepfakes-trained Meso-4
            for ev in ("ff_face2face", "openforensics", "faces140k"):
                stem = f"meso4{sfx}_train-ff_deepfakes_eval-{ev}"
                for row in rows_for(stem, listings[ev], model="meso4", seed=seed,
                                    train_source="ff_deepfakes", eval_ds=ev,
                                    with_video=ev.startswith("ff_")):
                    writer.writerow(row); n += 1

    print(f"[export] wrote {n:,} rows to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
