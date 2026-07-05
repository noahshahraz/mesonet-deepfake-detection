"""Multi-source training comparison (Task 7): does training on two manipulation types
transfer better than training on one?

Grouped bars of AUC per evaluation dataset — single-source Meso-4 (trained on FF++ Deepfakes
only, from summary.generalization) vs multi-source Meso-4 (trained on the Deepfakes+Face2Face
union, summary.multisource). OpenForensics and 140k were held out of BOTH trainings, so those
groups are the honest unseen tests. Whiskers = ±1 std across training seeds; annotated deltas
are multi minus single AUC means.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch

import figstyle

NAME = "fig_multisource"

C_SINGLE = figstyle.CYCLE[0]   # blue = Meso-4 single-source (repo identity)
C_MULTI = figstyle.CYCLE[4]    # purple = the new multi-source variant
EVALS = [
    ("ff_deepfakes", "FF++ Deepfakes\n(in single's training)"),
    ("ff_face2face", "FF++ Face2Face\n(in multi's training only)"),
    ("openforensics", "OpenForensics\nheld out of both"),
    ("faces140k", "140k StyleGAN\nheld out of both"),
]


def _stat(node):
    if not isinstance(node, dict) or not isinstance(node.get("auc"), dict):
        return None
    a = node["auc"]
    return float(a["mean"]), float(a.get("std") or 0.0), int(a.get("n") or 1)


def build(data: dict, th: dict):
    import matplotlib.pyplot as plt

    summary = data.get("summary") or {}
    single = ((summary.get("generalization") or {}).get("meso4") or {}).get("evals") or {}
    multi = (summary.get("multisource") or {}).get("evals") or {}

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    if not multi:
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "multi-source results not yet in results/summary.json\n"
                          "(hardening Task 7 training in progress)",
                ha="center", va="center", fontsize=11, style="italic", color=th["muted"],
                transform=ax.transAxes)
        return fig

    width = 0.36
    handles = [Patch(facecolor=C_SINGLE, label="trained on FF++ Deepfakes only"),
               Patch(facecolor=C_MULTI, hatch="//", edgecolor=th["bg"],
                     label="trained on Deepfakes + Face2Face union")]
    for i, (key, label) in enumerate(EVALS):
        s, m = _stat(single.get(key)), _stat(multi.get(key))
        for j, (st, color, hatch) in enumerate(((s, C_SINGLE, ""), (m, C_MULTI, "//"))):
            if st is None:
                continue
            mean, std, n = st
            x = i + (j - 0.5) * (width + 0.04)
            ax.bar(x, mean, width=width, color=color, hatch=hatch,
                   edgecolor=th["bg"] if hatch else "none",
                   yerr=std if n > 1 else None,
                   error_kw=dict(ecolor=th["fg"], capsize=3, lw=1.1), zorder=3)
            ax.annotate(f"{mean:.2f}", (x, mean), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=th["fg"])
        if s is not None and m is not None:
            delta = m[0] - s[0]
            ax.annotate(f"Δ {delta:+.2f}", (i, 0.04), ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold",
                        color=th["fg"] if abs(delta) >= 0.03 else th["muted"])

    chance = float(summary.get("chance_auc") or 0.5)
    ax.axhline(chance, ls=(0, (5, 3)), lw=1.2, color=th["muted"], zorder=2)
    ax.text(len(EVALS) - 0.52, chance + 0.012, "coin flip (0.5)", ha="right", va="bottom",
            fontsize=9, style="italic", color=th["muted"])

    ax.set_xticks(range(len(EVALS)), labels=[lab for _, lab in EVALS], fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUC on held-out test images\n(1.0 = perfect, 0.5 = coin flip)")
    ax.set_title("Multi-source training: one Meso-4, two manipulation types —\n"
                 "what changes on datasets neither model ever saw?", pad=14)
    ax.text(0, 1.005, "Bars: mean AUC across three training seeds (whiskers ±1 std); "
                      "Δ = multi-source minus single-source.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, style="italic",
            color=th["muted"])
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(axis="y")
    return fig
