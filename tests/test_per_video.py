"""Tests for per-video aggregation (hardening task 3)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_per_video import per_video_metrics, video_id  # noqa: E402


def test_video_id_parsing():
    assert video_id("953_f0042.jpg") == "953"
    assert video_id("953_974_f0031.jpg") == "953_974"
    assert video_id("/some/dir/033_097_f1200.jpg") == "033_097"


def test_per_video_aggregation_flips_noisy_frames():
    # video A (real): frames mostly low prob, one noisy high frame -> mean stays < 0.5
    # video B (fake): frames mostly high prob, one noisy low frame -> mean stays > 0.5
    samples = [("A_f0001.jpg", 0), ("A_f0002.jpg", 0), ("A_f0003.jpg", 0),
               ("B_C_f0001.jpg", 1), ("B_C_f0002.jpg", 1), ("B_C_f0003.jpg", 1)]
    probs = [0.1, 0.9, 0.2, 0.9, 0.2, 0.8]
    metrics, n_videos = per_video_metrics(samples, probs, threshold=0.5)
    assert n_videos == 2
    assert metrics["accuracy"] == 1.0  # per-frame accuracy would be 4/6
    assert metrics["auc"] == 1.0
