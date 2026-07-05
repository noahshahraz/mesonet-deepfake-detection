"""fig_seed_spread: the per-seed variance hiding behind the mean±std reproduction table.

Two panels (test accuracy, ROC AUC) x four FaceForensics++ runs. Each run shows its
individual per-seed results as jittered points (each labelled "seed N", alternating sides
so near-ties stay legible); runs with two or more seeds additionally get a slim
min-mean-max whisker. Paper per-image c23 Face2Face accuracies are drawn as short dashed
reference segments at the two Face2Face positions, inside a captioned "Face2Face runs" band.

Data: summary.reproduction[*].{acc,auc}.per_seed and summary.paper.per_image_c23_face2face.
Degrades gracefully when runs, metrics, or the paper block are missing.
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import figstyle

NAME = "fig_seed_spread"

# (summary key, tick-label model name, legend model name, dataset, model index into CYCLE/MARKERS)
RUNS = [
    ("meso4:ff_deepfakes", "Meso-4", "Deepfakes", 0),
    ("meso_inception4:ff_deepfakes", "MesoIncep-4", "Deepfakes", 1),
    ("meso4:ff_face2face", "Meso-4", "Face2Face", 0),
    ("meso_inception4:ff_face2face", "MesoIncep-4", "Face2Face", 1),
]
# x position of each model's Face2Face run, for the paper reference segments
PAPER_X = {"meso4": 2, "meso_inception4": 3}

METRICS = [
    ("acc", "Test accuracy", "Share of held-out test images classified correctly"),
    ("auc", "Test ROC AUC (1.0 = perfect real/fake separation)", "Area under the ROC curve"),
]


def _seed_values(rep: dict, key: str, metric: str) -> tuple[list[str], list[float]]:
    """Return (labels, values) for one run+metric; labels are seed numbers (or 'mean')."""
    block = (rep.get(key) or {}).get(metric) or {}
    per_seed = block.get("per_seed") or {}
    if per_seed:
        items = sorted(per_seed.items(), key=lambda kv: int(kv[0]))
        return [k for k, _ in items], [float(v) for _, v in items]
    if block.get("mean") is not None:  # fallback: aggregate only, no per-seed breakdown
        return ["mean"], [float(block["mean"])]
    return [], []


def _title(rep: dict) -> str:
    n_max, spreads = 0, []
    for key, _, _, _ in RUNS:
        for metric, _, _ in METRICS:
            _, vals = _seed_values(rep, key, metric)
            n_max = max(n_max, len(vals))
            if len(vals) >= 2:
                spreads.append(max(vals) - min(vals))
    if n_max <= 1:
        return "One training seed per run so far — seed-to-seed spread is still unknown"
    worst = max(spreads) if spreads else 0.0
    if worst < 0.01:
        return "Different random seeds barely move the results"
    if worst < 0.03:
        return "Different random seeds shift the results only slightly"
    return "Results vary noticeably from one random seed to the next"


def build(data: dict, th: dict):
    summary = data.get("summary") or {}
    rep = summary.get("reproduction") or {}
    paper_ref = (summary.get("paper") or {}).get("per_image_c23_face2face") or {}
    rng = np.random.default_rng(0)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.4))
    fig.subplots_adjust(top=0.745, bottom=0.165, left=0.075, right=0.99, wspace=0.20)

    # Nothing to plot at all -> single muted annotation, never crash.
    have_any = any(_seed_values(rep, key, m)[1] for key, _, _, _ in RUNS for m, _, _ in METRICS)
    if not have_any:
        for ax in axs:
            ax.set_axis_off()
        fig.text(0.5, 0.45, "reproduction results not available in results/summary.json",
                 ha="center", va="center", color=th["muted"], fontsize=12)
        fig.text(0.5, 0.955, "Seed-to-seed spread (no data yet)", ha="center",
                 color=th["fg"], fontsize=figstyle.TITLE_SIZE, fontweight="bold")
        return fig

    # Shared y-limits across both panels so accuracy and AUC are visually comparable.
    all_vals = [v for key, _, _, _ in RUNS for m, _, _ in METRICS
                for v in _seed_values(rep, key, m)[1]]
    all_vals += [float(v) for v in paper_ref.values()]
    lo = float(np.floor((min(all_vals) - 0.025) * 50) / 50)
    hi = float(min(1.0, np.ceil((max(all_vals) + 0.02) * 50) / 50))
    span = hi - lo

    tick_labels = [f"{model}\n{dset}" for _, model, dset, _ in RUNS]

    for ax, (metric, panel_title, y_label) in zip(axs, METRICS):
        # Subtle band grouping the two Face2Face runs, captioned inside the band.
        ax.axvspan(1.5, 3.6, color=th["accent_bg"], zorder=0)
        ax.text((1.5 + 3.6) / 2, 0.975, "Face2Face runs",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=th["muted"], fontsize=8)
        for x, (key, _, _, mi) in enumerate(RUNS):
            color, marker = figstyle.CYCLE[mi], figstyle.MARKERS[mi]
            labels, vals = _seed_values(rep, key, metric)
            if not vals:
                ax.text(x, lo + span / 2, "no\ndata", ha="center", va="center",
                        color=th["muted"], fontsize=8)
                continue
            if len(vals) >= 2:  # slim min-mean-max whisker behind the points
                vmin, vmax, vmean = min(vals), max(vals), float(np.mean(vals))
                ax.plot([x, x], [vmin, vmax], color=color, lw=2, alpha=0.45, zorder=2)
                for v_end in (vmin, vmax):
                    ax.plot([x - 0.06, x + 0.06], [v_end, v_end],
                            color=color, lw=1.5, alpha=0.45, zorder=2)
                ax.plot([x - 0.12, x + 0.12], [vmean, vmean], color=color, lw=2.5, zorder=3)
            offsets = (rng.uniform(-0.09, 0.09, size=len(vals))
                       if len(vals) > 1 else np.zeros(1))
            for j, (off, s_label, v) in enumerate(zip(offsets, labels, vals)):
                ax.scatter(x + off, v, marker=marker, s=46, facecolor=color,
                           edgecolor=th["bg"], linewidth=0.6, zorder=5)
                point_label = s_label if s_label == "mean" else f"seed {s_label}"
                side = -1 if j % 2 else 1  # alternate sides so near-tied labels stay legible
                ax.text(x + off + side * 0.09, v, point_label,
                        ha="right" if side < 0 else "left", va="center",
                        color=th["muted"], fontsize=7)
            if len(vals) == 1:
                ax.text(x, vals[0] - 0.011 * span / 0.12, "1 seed", ha="center", va="top",
                        color=th["muted"], fontsize=8)

        if metric == "acc" and paper_ref:
            for model_key, x in PAPER_X.items():
                v = paper_ref.get(model_key)
                if v is None:
                    continue
                ax.plot([x - 0.3, x + 0.3], [v, v], ls="--", lw=1.6,
                        color=figstyle.CYCLE[3], zorder=4)
                ax.text(x, v + 0.006 * span / 0.12, f"paper {v:.3f}", ha="center",
                        va="bottom", color=figstyle.CYCLE[3], fontsize=7.5)
        if metric == "auc":
            ax.text(0.98, 0.03, "the paper reports no per-image AUC to compare against",
                    transform=ax.transAxes, ha="right", va="bottom",
                    color=th["muted"], fontsize=8, fontstyle="italic")

        ax.set_title(panel_title, fontsize=12, color=th["fg"], pad=8)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(lo, hi)
        ax.set_xticks(range(len(RUNS)))
        ax.set_xticklabels(tick_labels)
        ax.xaxis.grid(False)

    # Figure title + subtitle in plain English.
    fig.text(0.5, 0.965, _title(rep), ha="center", color=th["fg"],
             fontsize=figstyle.TITLE_SIZE, fontweight="bold")
    fig.text(0.5, 0.905,
             "Individual per-seed test results for the four FaceForensics++ (c23) training runs;"
             " dashed segments mark the accuracy the MesoNet paper reports for Face2Face",
             ha="center", color=th["muted"], fontsize=10.5)

    handles = [
        Line2D([], [], marker=figstyle.MARKERS[0], ls="none", markersize=7,
               markerfacecolor=figstyle.CYCLE[0], markeredgecolor=figstyle.CYCLE[0],
               label="Meso-4 (this repo)"),
        Line2D([], [], marker=figstyle.MARKERS[1], ls="none", markersize=7,
               markerfacecolor=figstyle.CYCLE[1], markeredgecolor=figstyle.CYCLE[1],
               label="MesoInception-4 (this repo)"),
    ]
    if paper_ref:
        handles.append(Line2D([], [], ls="--", lw=1.6, color=figstyle.CYCLE[3],
                              label="MesoNet paper, per-image accuracy (Face2Face, c23)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.885),
               ncol=3, fontsize=9, handletextpad=0.5, columnspacing=1.4)

    # Honest footnote: planned vs. completed seeds, and the zoomed axis.
    planned = summary.get("seeds") or []
    done = {s for key, _, _, _ in RUNS for s in _seed_values(rep, key, "acc")[0]}
    n_done_max = max((len(_seed_values(rep, key, "acc")[1]) for key, _, _, _ in RUNS),
                     default=0)
    if n_done_max <= 1 and planned:
        note = (f"Planned seeds: {', '.join(str(s) for s in planned)} — only seed"
                f" {', '.join(sorted(done))} has finished; min-mean-max whiskers will appear"
                " once more seeds complete.")
    else:
        note = "Whiskers span min to max across seeds; the wider tick marks the mean."
    note += f"  Note: the y-axis is zoomed in (starts at {lo:.2f}, not 0)."
    fig.text(0.5, 0.022, note, ha="center", color=th["muted"], fontsize=8.5)
    return fig
