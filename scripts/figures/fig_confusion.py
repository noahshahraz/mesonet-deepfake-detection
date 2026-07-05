"""Confusion matrices for Meso-4 on FF++ Deepfakes and Face2Face (threshold 0.5).

Two annotated 2x2 matrices side by side, computed from the per-image score files
(outputs/<stem>_probs.npz) referenced by summary.probs_stems.repro_runs. Counts and
row-percentages in every cell; mistake cells (off-diagonal) get a thin vermillion border.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import figstyle

NAME = "fig_confusion"

# (repro_runs key, display name) — order fixed: Deepfakes left, Face2Face right.
PANELS = [
    ("meso4:ff_deepfakes", "Deepfakes"),
    ("meso4:ff_face2face", "Face2Face"),
]
TICKS = ["Real", "Fake"]  # label convention: real = 0, fake = 1


def _confusion(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """2x2 counts, rows = true label (Real, Fake), cols = predicted label."""
    pred = (np.asarray(probs) >= 0.5).astype(int)
    lab = np.asarray(labels).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = int(((lab == t) & (pred == p)).sum())
    return cm


def _missing_panel(ax, display_name: str, th: dict) -> None:
    ax.set_title(f"{display_name}", fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, "score data not available\n(regenerate with src.eval)",
            ha="center", va="center", fontsize=10, color=th["muted"],
            style="italic", transform=ax.transAxes)


def _matrix_panel(ax, cm: np.ndarray, display_name: str, th: dict) -> None:
    # Single-hue sequential colormap: theme background -> Okabe-Ito blue.
    # Blending from th["bg"] keeps low cells legible on both light and dark themes.
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", [th["bg"], figstyle.OKABE_ITO["blue"]])
    row_totals = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, np.maximum(row_totals, 1), dtype=float)

    ax.imshow(frac, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    acc = 100.0 * np.trace(cm) / max(cm.sum(), 1)
    ax.set_title(f"{display_name}: {acc:.1f}% of decisions correct",
                 fontsize=12, pad=10)
    ax.set_xlabel("Predicted label")
    ax.set_xticks([0, 1], TICKS)
    ax.set_yticks([0, 1], TICKS)
    ax.tick_params(length=0)

    for t in (0, 1):
        for p in (0, 1):
            f = frac[t, p]
            # Cell text: count + row-percentage; switch color for contrast on dark blue.
            color = "#ffffff" if f >= 0.5 else th["fg"]
            ax.text(p, t, f"{cm[t, p]:,}\n{100.0 * f:.1f}%",
                    ha="center", va="center", fontsize=12, color=color,
                    linespacing=1.5)
            if t != p:  # mistake cell -> thin vermillion border
                ax.add_patch(Rectangle((p - 0.46, t - 0.46), 0.92, 0.92,
                                       fill=False, linewidth=1.6,
                                       edgecolor=figstyle.OKABE_ITO["vermillion"]))


def build(data: dict, th: dict):
    import matplotlib.pyplot as plt

    summary = data.get("summary", {}) or {}
    repro_runs = (summary.get("probs_stems") or {}).get("repro_runs") or {}
    probs = data.get("probs", {}) or {}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.7))
    fig.subplots_adjust(wspace=0.35, top=0.78)

    for ax, (key, display_name) in zip(axes, PANELS):
        stem = repro_runs.get(key)
        entry = probs.get(stem) if stem else None
        if entry is None:
            _missing_panel(ax, display_name, th)
            continue
        cm = _confusion(entry["labels"], entry["probs"])
        _matrix_panel(ax, cm, display_name, th)

    axes[0].set_ylabel("True label")

    fig.suptitle("Meso-4 gets more than 9 in 10 test images right on both forgery types",
                 fontsize=figstyle.TITLE_SIZE, fontweight="bold", y=1.04)
    fig.text(0.5, 0.955,
             "Confusion matrices at decision threshold 0.5 on held-out FaceForensics++ "
             "test images —\neach cell shows the number of images and the share of its "
             "true class (its row)",
             ha="center", va="top", fontsize=10.5, color=th["muted"], linespacing=1.4)
    fig.text(0.5, -0.02,
             "Cells outlined in vermillion are mistakes: real images called fake "
             "(false alarms) and fake images called real (missed fakes).",
             ha="center", va="top", fontsize=9, color=th["muted"])
    return fig
