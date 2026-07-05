"""ROC overlay for the four in-domain FaceForensics++ reproduction runs.

One curve per (model, forgery-method) run. Color encodes the model (CYCLE identity
mapping), linestyle encodes the forgery method, and sparse distinct markers give a
third, color-free channel so no distinction relies on color alone.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

import figstyle

NAME = "fig_roc"

# Fixed display order (also the marker index per run).
RUN_ORDER = [
    "meso4:ff_deepfakes",
    "meso4:ff_face2face",
    "meso_inception4:ff_deepfakes",
    "meso_inception4:ff_face2face",
]
MODEL_LABEL = {"meso4": "Meso-4", "meso_inception4": "MesoInception-4"}
METHOD_LABEL = {"ff_deepfakes": "Deepfakes", "ff_face2face": "Face2Face"}
MODEL_COLOR = {"meso4": figstyle.CYCLE[0], "meso_inception4": figstyle.CYCLE[1]}
METHOD_LINESTYLE = {"ff_deepfakes": "-", "ff_face2face": "--"}


def _run_label(key: str) -> str:
    model, method = key.split(":", 1)
    return f"{MODEL_LABEL.get(model, model)} - {METHOD_LABEL.get(method, method)}"


def build(data: dict, th: dict) -> plt.Figure:
    summary = data.get("summary") or {}
    probs = data.get("probs") or {}
    repro_stems = (summary.get("probs_stems") or {}).get("repro_runs") or {}

    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    ax.set_aspect("equal")

    # Region below the diagonal: a detector living there is worse than guessing.
    ax.fill_between([0, 1], [0, 1], 0, color=th["muted"], alpha=0.14,
                    linewidth=0, zorder=1)
    ax.plot([0, 1], [0, 1], color=th["muted"], linestyle=":", linewidth=1.4,
            zorder=2, label="Coin flip (AUC 0.500)")
    ax.text(0.67, 0.57, "worse than a coin flip", rotation=45,
            rotation_mode="anchor", ha="center", va="center",
            fontsize=10.5, fontstyle="italic", color=th["muted"], zorder=2)

    missing = []
    n_drawn = 0
    for i, key in enumerate(k for k in RUN_ORDER if k in repro_stems):
        stem = repro_stems[key]
        run = probs.get(stem)
        if run is None:
            missing.append(_run_label(key))
            continue
        fpr, tpr, _ = roc_curve(run["labels"], run["probs"])
        run_auc = auc(fpr, tpr)
        model, method = key.split(":", 1)
        ax.plot(
            fpr, tpr,
            color=MODEL_COLOR.get(model, figstyle.CYCLE[i % len(figstyle.CYCLE)]),
            linestyle=METHOD_LINESTYLE.get(method, "-"),
            linewidth=2.0,
            marker=figstyle.MARKERS[i % len(figstyle.MARKERS)],
            markevery=(0.02 + 0.035 * i, 0.15),  # sparse, offset so markers don't stack
            markersize=7,
            markeredgecolor=th["bg"],
            markeredgewidth=0.8,
            zorder=3 + i,
            label=f"{_run_label(key)} (AUC {run_auc:.3f})",
        )
        n_drawn += 1

    if missing or not repro_stems:
        lines = ["score data not available", "(regenerate with src.eval)"]
        if missing and repro_stems:
            lines = ["score data not available for:"] + missing + ["(regenerate with src.eval)"]
        pos = (0.5, 0.5) if n_drawn == 0 else (0.55, 0.30)
        ax.text(*pos, "\n".join(lines), ha="center", va="center",
                fontsize=9, color=th["muted"], zorder=4,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=th["accent_bg"],
                          edgecolor=th["grid"], alpha=0.9))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("False-alarm rate (real images called fake)")
    ax.set_ylabel("Detection rate (fakes correctly caught)")

    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02),
                    frameon=True, framealpha=0.9)
    leg.get_frame().set_facecolor(th["bg"])
    leg.get_frame().set_edgecolor(th["grid"])

    fig.suptitle("On the forgery type they were trained on,\n"
                 "both MesoNet models rank fakes above reals almost perfectly",
                 fontsize=figstyle.TITLE_SIZE, fontweight="bold",
                 color=th["fg"], y=0.995)
    ax.set_title("ROC curves for the four FaceForensics++ reproduction runs "
                 "(frame-level scores,\nheld-out test images); closer to the "
                 "top-left corner is better",
                 fontsize=10, color=th["muted"], pad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig
