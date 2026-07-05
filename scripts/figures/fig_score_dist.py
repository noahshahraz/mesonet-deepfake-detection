"""Score-distribution figure: what the detector 'sees' in-domain vs. on a new dataset.

Two stacked panels share an x axis of the model's predicted probability that an image is
fake (0 = sure it's real, 1 = sure it's fake). Top panel: the dataset the model was trained
on (FaceForensics++ Deepfakes test split). Bottom panel: the same trained weights scored on
the 140k Real-and-Fake-Faces dataset (StyleGAN-generated fakes) it has never seen.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

import figstyle

NAME = "fig_score_dist"

N_BINS = 40
REAL_COLOR = figstyle.CYCLE[0]   # blue  (identity color 0)
FAKE_COLOR = figstyle.CYCLE[1]   # orange (identity color 1)
FAKE_HATCH = figstyle.HATCHES[1]  # "//"


def _panel(ax, labels, probs, th):
    """Draw density-normalized step histograms for real vs fake; return legend handles."""
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    real_vals = probs[labels == 0]
    fake_vals = probs[labels == 1]

    # Fake images: hatched translucent bars (drawn first, below the real outline).
    ax.hist(fake_vals, bins=bins, density=True, histtype="stepfilled",
            facecolor=to_rgba(FAKE_COLOR, 0.30), edgecolor=FAKE_COLOR,
            linewidth=1.6, hatch=FAKE_HATCH, zorder=3)

    # Real images: bold unfilled step outline above the fake bars, with only a very
    # light fill underneath, so both distributions stay individually identifiable
    # wherever they overlap.
    ax.hist(real_vals, bins=bins, density=True, histtype="stepfilled",
            facecolor=to_rgba(REAL_COLOR, 0.15), edgecolor="none", zorder=3.5)
    ax.hist(real_vals, bins=bins, density=True, histtype="step",
            edgecolor=REAL_COLOR, linewidth=2.5, zorder=4)

    handles = [
        Patch(facecolor=to_rgba(REAL_COLOR, 0.15), edgecolor=REAL_COLOR,
              linewidth=2.5, label=f"Real images (n = {real_vals.size:,})"),
        Patch(facecolor=to_rgba(FAKE_COLOR, 0.30), edgecolor=FAKE_COLOR,
              linewidth=1.6, hatch=FAKE_HATCH,
              label=f"Fake images (n = {fake_vals.size:,})"),
    ]
    return handles


def _threshold_line(ax, th, label=True):
    ax.axvline(0.5, color=th["muted"], linestyle="--", linewidth=1.3, zorder=2)
    if label:
        ax.text(0.5, 0.97, "decision threshold ", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=9, color=th["muted"], style="italic")


def _missing_panel(ax, th):
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "score data not available (regenerate with src.eval)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=th["muted"], style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=th["accent_bg"],
                      edgecolor=th["grid"], linewidth=0.8))
    ax.grid(False)


def build(data: dict, th: dict):
    import matplotlib.pyplot as plt

    summary = data.get("summary", {}) or {}
    stems = summary.get("probs_stems", {}) or {}
    probs = data.get("probs", {}) or {}
    top = probs.get(stems.get("in_domain"))
    bottom = probs.get(stems.get("cross_140k"))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9.6, 7.4),
        gridspec_kw={"hspace": 0.32},
    )

    fig.suptitle("The same detector: confident at home, lost on a new dataset",
                 fontsize=figstyle.TITLE_SIZE, fontweight="bold", x=0.02, y=0.99,
                 ha="left", color=th["fg"])
    fig.text(0.02, 0.945,
             "Meso-4 trained on FaceForensics++ Deepfakes — how often it assigns each "
             "“probability the image is fake” score,\nfor images it should call real vs. fake "
             "(area under each curve = 100% of that group)",
             fontsize=10, color=th["muted"], ha="left", va="top")

    # ---- top panel: in-domain ------------------------------------------------------
    ax_top.set_title("Tested on the data it was trained for (FaceForensics++ Deepfakes)",
                     loc="left", fontsize=11, color=th["fg"], pad=8)
    if top is not None:
        handles = _panel(ax_top, top["labels"], top["probs"], th)
        _threshold_line(ax_top, th)
        ax_top.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.98))
        ax_top.text(0.42, 0.50,
                    "Trained-on data: real images pile up at 0, fakes at 1 —\n"
                    "the two groups barely overlap (93% of images called correctly)",
                    transform=ax_top.transAxes, ha="center", va="center",
                    fontsize=10.5, color=th["fg"],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=th["accent_bg"],
                              edgecolor=th["grid"], linewidth=0.8))
    else:
        _threshold_line(ax_top, th)
        _missing_panel(ax_top, th)

    # ---- bottom panel: cross-dataset 140k ------------------------------------------
    ax_bot.set_title("Same model, tested on a dataset it never saw (140k real-and-fake faces, "
                     "StyleGAN fakes)", loc="left", fontsize=11, color=th["fg"], pad=8)
    if bottom is not None:
        handles = _panel(ax_bot, bottom["labels"], bottom["probs"], th)
        _threshold_line(ax_bot, th)
        ax_bot.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.80))
        ax_bot.text(0.52, 0.48,
                    "New dataset: the model scores almost everything as “real” — the two\n"
                    "distributions sit on top of each other, and fakes actually score slightly\n"
                    "LOWER than real photos. Accuracy collapses to a coin flip (50%).",
                    transform=ax_bot.transAxes, ha="center", va="center",
                    fontsize=10.5, color=th["fg"],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=th["accent_bg"],
                              edgecolor=th["grid"], linewidth=0.8))
    else:
        _threshold_line(ax_bot, th)
        _missing_panel(ax_bot, th)

    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density of images")
        ax.margins(y=0.06)
    ax_bot.set_xlabel("Model's predicted probability that the image is fake  "
                      "(0 = sure it's real, 1 = sure it's fake)")

    fig.subplots_adjust(top=0.855, bottom=0.09, left=0.08, right=0.97)
    return fig
