"""Per-video evaluation (hardening task 3) — the paper's actual headline protocol.

Frame filenames embed the source video id (`<vid>_f0042.jpg`, `<a>_<b>_f0031.jpg`), so we run
per-frame inference over a split, average P(fake) per video id ("average the network prediction
over the video", Afchar et al. §IV), and score accuracy/AUC on the per-video probabilities.

Usage:
    python scripts/eval_per_video.py --checkpoint checkpoints/meso4_ff_deepfakes_best.pth \
        --dataset ff_deepfakes --data-root ~/mesonet-data/ff_deepfakes
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_eval_loader  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import get_device, load_config, predict_probs, set_seed  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default=None, help="default: cfg test split")
    p.add_argument("--out", default=None, help="JSON output path (default: print only)")
    return p.parse_args()


def video_id(path: str | Path) -> str:
    """`953_f0042.jpg` -> `953`; `953_974_f0031.jpg` -> `953_974`."""
    return Path(path).stem.rsplit("_f", 1)[0]


def dataset_samples(dataset) -> list:
    if isinstance(dataset, Subset):
        return [dataset.dataset.samples[i] for i in dataset.indices]
    return dataset.samples


def per_video_metrics(samples, probs, threshold: float) -> tuple[dict, int]:
    by_video: dict[str, list[float]] = defaultdict(list)
    video_label: dict[str, int] = {}
    for (path, label), prob in zip(samples, probs):
        vid = video_id(path)
        by_video[vid].append(float(prob))
        assert video_label.setdefault(vid, label) == label, f"mixed labels for video {vid}"
    vids = sorted(by_video)
    labels = np.array([video_label[v] for v in vids])
    agg = np.array([np.mean(by_video[v]) for v in vids])
    return compute_metrics(labels, agg, threshold=threshold), len(vids)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["name"] = args.dataset
    cfg["data"]["root"] = args.data_root or f"data/{args.dataset}"
    set_seed(cfg.seed)
    device = get_device(cfg.get_path("device", "mps"))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", cfg.model.name)
    model = build_model(model_name, num_classes=cfg.model.num_classes,
                        dropout=cfg.model.dropout, image_size=cfg.data.image_size).to(device)
    model.load_state_dict(ckpt["model_state"])

    loader = build_eval_loader(cfg, args.split)  # shuffle=False -> probs align with samples
    samples = dataset_samples(loader.dataset)
    labels, probs = predict_probs(model, loader, device)
    assert len(samples) == len(probs)

    frame_metrics = compute_metrics(labels, probs, threshold=cfg.eval.threshold)
    video_metrics, n_videos = per_video_metrics(samples, probs, cfg.eval.threshold)
    print(f"[per-video] {model_name} on {args.dataset} ({len(samples)} frames, "
          f"{n_videos} videos)")
    print(f"  per-frame : acc {frame_metrics['accuracy']:.4f}  auc {frame_metrics['auc']:.4f}")
    print(f"  per-video : acc {video_metrics['accuracy']:.4f}  auc {video_metrics['auc']:.4f}")

    if args.out:
        record = {"checkpoint": args.checkpoint, "model": model_name, "dataset": args.dataset,
                  "n_frames": len(samples), "n_videos": n_videos,
                  "per_frame": frame_metrics, "per_video": video_metrics}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
