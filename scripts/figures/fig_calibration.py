"""Reliability diagram tying calibration to the threshold-tuning story.

Single panel. For the two in-domain Deepfakes runs (Meso-4 and MesoInception-4),
predicted P(fake) is split into 10 equal bins; each dot compares the model's
average claimed probability (x) with the fraction of those images that really
are fake (y). Marker area encodes the number of test images per bin. Vertical
lines mark each model's validation-selected decision threshold t* from
summary["threshold"], which compensates for the miscalibration the dots show:
both curves sit BELOW the perfect-calibration diagonal (the models over-claim
"fake"), MesoInception-4 by far the most — hence its threshold rises to 0.69
while Meso-4's stays at 0.50.

Color always paired with a color-free channel: distinct marker shapes for the
two series, distinct linestyles for the two threshold lines, direct labels on
both.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import figstyle

NAME = "fig_calibration"

N_BINS = 10

# (model key, display name, CYCLE index, marker, threshold-line linestyle,
#  side of the vertical line for the t* label: -1 = left, +1 = right)
SERIES = [
    ("meso4", "Meso-4", 0, figstyle.MARKERS[0], (0, (6, 3)), -1),
    ("meso_inception4", "MesoInception-4", 1, figstyle.MARKERS[1], (0, (1, 1.6)), +1),
]


def _stem_for(summary: dict, model_key: str):
    """Stem of the <model> train-Deepfakes eval-Deepfakes run, if recorded."""
    ps = summary.get("probs_stems") or {}
    repro = ps.get("repro_runs") or {}
    stem = repro.get(f"{model_key}:ff_deepfakes")
    if stem is None and model_key == "meso4":
        stem = ps.get("in_domain")  # same run under its other alias
    return stem


def _threshold_for(summary: dict, model_key: str):
    """Val-selected t* for (model, ff_deepfakes); rows carry no seed field, so
    treat a missing seed as the default seed-42 run."""
    for row in summary.get("threshold") or []:
        if (row.get("model") == model_key
                and row.get("dataset") == "ff_deepfakes"
                and row.get("seed", 42) == 42):
            t = row.get("threshold")
            return None if t is None else float(t)
    return None


def _binned(labels: np.ndarray, probs: np.ndarray, n_bins: int = N_BINS):
    """Per-bin (mean claimed P(fake), empirical fraction fake, count) + ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    conf, frac, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue  # skip empty bins rather than plotting NaNs
        conf.append(float(probs[m].mean()))
        frac.append(float(labels[m].mean()))
        count.append(int(m.sum()))
    conf, frac, count = np.array(conf), np.array(frac), np.array(count)
    ece = float(np.sum(count / count.sum() * np.abs(frac - conf)))
    return conf, frac, count, ece


def build(data: dict, th: dict) -> plt.Figure:
    summary = data.get("summary") or {}
    probs = data.get("probs") or {}

    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    ax.set_aspect("equal")

    # Perfect-calibration diagonal: claimed probability == actual fraction fake.
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.4, color=th["muted"],
            zorder=2, label="Perfect calibration (claimed = actual)")
    ax.text(0.30, 0.335, "perfectly honest scores", rotation=45,
            rotation_mode="anchor", ha="center", va="center", fontsize=9.5,
            fontstyle="italic", color=th["muted"], zorder=2)

    # Gather both series first so marker sizes share one count scale.
    curves, missing = {}, []
    for key, label, ci, marker, tls, _side in SERIES:
        stem = _stem_for(summary, key)
        run = probs.get(stem) if stem else None
        if run is None:
            missing.append(label)
            continue
        curves[key] = _binned(run["labels"], run["probs"])
    max_count = max((c[2].max() for c in curves.values()), default=1)

    for key, label, ci, marker, tls, side in SERIES:
        color = figstyle.CYCLE[ci]
        # Okabe-Ito blue fails contrast as TEXT on the dark background; swap the
        # label text to sky blue there while keeping line/marker colors as-is.
        text_color = (figstyle.OKABE_ITO["skyblue"]
                      if th["dark"] and color == figstyle.OKABE_ITO["blue"]
                      else color)

        # Threshold line first (behind the curve), even if scores are missing.
        t_star = _threshold_for(summary, key)
        if t_star is not None:
            ax.axvline(t_star, color=color, linestyle=tls, linewidth=1.6,
                       alpha=0.85, zorder=1)
            # Short horizontal label at the foot of the line (x in data coords,
            # y in axes coords), offset to the series' side so the two labels
            # can never collide with each other or the legend.
            ax.text(t_star + side * 0.015, 0.04, f"t* = {t_star:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="right" if side < 0 else "left", va="bottom",
                    fontsize=8.5, color=text_color, zorder=2)

        if key not in curves:
            continue
        conf, frac, count, ece = curves[key]
        ax.plot(conf, frac, color=color, linewidth=1.7, alpha=0.9, zorder=3)
        ax.scatter(conf, frac, s=25 + 300 * (count / max_count), marker=marker,
                   color=color, edgecolor=th["bg"], linewidth=0.9, zorder=4,
                   label=f"{label} — avg. claimed-vs-actual gap {ece:.2f}")

    # Concrete reading aid on the most miscalibrated bin (MesoInception-4 ~0.55).
    if "meso_inception4" in curves:
        conf, frac, _, _ = curves["meso_inception4"]
        j = int(np.argmin(np.abs(conf - 0.55)))
        ax.annotate(
            f'when MesoInception-4 says "{conf[j]:.0%} fake",\n'
            f"only {frac[j]:.0%} of those images are actually fake",
            xy=(conf[j], frac[j]), xytext=(0.97, 0.06),
            ha="right", va="bottom", fontsize=9, color=th["fg"], zorder=5,
            arrowprops=dict(arrowstyle="->", color=th["muted"], linewidth=1.1,
                            connectionstyle="arc3,rad=-0.18"),
            bbox=dict(boxstyle="round,pad=0.4", facecolor=th["accent_bg"],
                      edgecolor=th["grid"], alpha=0.9))

    if missing:
        pos = (0.5, 0.5) if not curves else (0.24, 0.62)
        ax.text(*pos, "score data not available for:\n"
                + "\n".join(missing) + "\n(regenerate with src.eval)",
                ha="center", va="center", fontsize=9, color=th["muted"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=th["accent_bg"],
                          edgecolor=th["grid"], alpha=0.9))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('Probability the model claims an image is fake (average per bin)')
    ax.set_ylabel("Fraction of those images that actually are fake")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
                    frameon=True, framealpha=0.9)
    leg.get_frame().set_facecolor(th["bg"])
    leg.get_frame().set_edgecolor(th["grid"])

    fig.suptitle('MesoInception-4 overstates how likely images are fake —\n'
                 "its raised decision threshold (t*=0.69) compensates",
                 fontsize=figstyle.TITLE_SIZE, fontweight="bold",
                 color=th["fg"], y=0.995)
    ax.set_title("Reliability diagram, FaceForensics++ Deepfakes test images grouped "
                 "into 10 bins by claimed P(fake);\ndots below the dashed diagonal = "
                 'the model claims "fake" more strongly than reality, and bigger '
                 "markers = more\nimages in the bin. Meso-4 hugs the diagonal, so its "
                 "threshold stays at 0.50; MesoInception-4 sits far\nbelow it, so "
                 "tuning on validation pushes its threshold up to 0.69.",
                 fontsize=9.5, color=th["muted"], pad=10, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return fig
