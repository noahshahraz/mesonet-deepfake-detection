"""Dumbbell chart: test accuracy before vs. after decision-threshold tuning (T20).

One row per (model, dataset[, seed]) entry in summary["threshold"]: an open marker at the
accuracy using the default 0.5 cutoff, a filled marker at the accuracy using the threshold
t* chosen on the validation split, and a connecting line colored by model. Deltas are
annotated only when they are big enough to matter (|delta| >= 0.005).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import figstyle

NAME = "fig_threshold_dumbbell"

# Fixed identity mapping (same across the whole figure suite).
MODEL_ORDER = ["meso4", "meso_inception4", "xception"]
MODEL_LABEL = {"meso4": "Meso-4", "meso_inception4": "MesoInception-4", "xception": "Xception"}
MODEL_COLOR = {"meso4": figstyle.CYCLE[0], "meso_inception4": figstyle.CYCLE[1],
               "xception": figstyle.CYCLE[2]}
MODEL_MARKER = {"meso4": figstyle.MARKERS[0], "meso_inception4": figstyle.MARKERS[1],
                "xception": figstyle.MARKERS[2]}
DATASET_ORDER = ["ff_deepfakes", "ff_face2face", "openforensics", "faces140k"]
DATASET_LABEL = {"ff_deepfakes": "Deepfakes", "ff_face2face": "Face2Face",
                 "openforensics": "OpenForensics", "faces140k": "140k real-vs-fake"}

MIN_ANNOT_DELTA = 0.005


def _sort_key(row: dict):
    model = row.get("model", "")
    ds = row.get("dataset", "")
    m = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    d = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER)
    return (m, model, d, ds, row.get("seed", -1))


def _row_label(row: dict) -> str:
    parts = [MODEL_LABEL.get(row.get("model", ""), row.get("model", "?")),
             DATASET_LABEL.get(row.get("dataset", ""), row.get("dataset", "?"))]
    if row.get("seed") is not None:
        parts.append(f"seed {row['seed']}")
    return " · ".join(parts)


def build(data: dict, th: dict) -> plt.Figure:
    rows = data.get("summary", {}).get("threshold") or []
    rows = sorted((r for r in rows if r.get("test_acc_05") is not None
                   and r.get("test_acc_tuned") is not None), key=_sort_key)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, max(3.4, 1.9 + 0.62 * n)))
    ax.set_title("Threshold tuning only mattered for one run", loc="left", pad=30)
    # Subtitle offset in points (not axes fraction) so it clears the title at any height.
    ax.annotate("threshold chosen on validation, applied to test; "
                "t* shown right of each row",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 10),
                textcoords="offset points", fontsize=10, color=th["muted"], va="bottom")

    if not rows:
        ax.text(0.5, 0.5, "threshold data not available in results/summary.json",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=th["muted"], style="italic")
        ax.set_axis_off()
        return fig

    green = figstyle.OKABE_ITO["green"]
    vermillion = figstyle.OKABE_ITO["vermillion"]
    xs = [v for r in rows for v in (r["test_acc_05"], r["test_acc_tuned"])]
    pad = max(0.004, (max(xs) - min(xs)) * 0.12)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)

    # y=0 is the top row; invert so reading order matches the sort order.
    trans_tstar = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, r in enumerate(rows):
        color = MODEL_COLOR.get(r.get("model"), figstyle.CYCLE[4])
        marker = MODEL_MARKER.get(r.get("model"), figstyle.MARKERS[4])
        a05, atuned = r["test_acc_05"], r["test_acc_tuned"]
        ax.plot([a05, atuned], [y, y], color=color, lw=2.4, alpha=0.85, zorder=2)
        # Open marker = default 0.5 cutoff (drawn larger so it stays visible when the
        # tuned point lands on top of it, i.e. delta = 0).
        ax.plot(a05, y, marker=marker, ms=11, mfc="none", mec=color, mew=1.8,
                ls="none", zorder=3)
        # Filled marker = tuned threshold t*.
        ax.plot(atuned, y, marker=marker, ms=7.5, mfc=color, mec=color,
                ls="none", zorder=4)

        delta = atuned - a05
        if abs(delta) >= MIN_ANNOT_DELTA:
            ax.annotate(f"{delta:+.3f}", xy=((a05 + atuned) / 2, y),
                        xytext=(0, 11), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color=green if delta > 0 else vermillion)
        tstar = r.get("threshold")
        if tstar is not None:
            ax.text(1.015, y, f"t* = {tstar:.2f}", transform=trans_tstar,
                    fontsize=9, color=th["muted"], va="center", ha="left")

    ax.set_yticks(range(n))
    ax.set_yticklabels([_row_label(r) for r in rows])
    ax.set_ylim(n - 0.5, -0.85)  # inverted; extra headroom for delta annotations
    ax.set_xlabel("Test accuracy")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    # Legend: what open vs. filled means (neutral color so it names no model).
    handles = [
        plt.Line2D([], [], marker="o", ls="none", ms=10, mfc="none", mec=th["fg"],
                   mew=1.6, label="default cutoff 0.5"),
        plt.Line2D([], [], marker="o", ls="none", ms=7.5, mfc=th["fg"], mec=th["fg"],
                   label="tuned threshold t*"),
    ]
    # Anchor the legend a fixed 0.68 inch below the axes (height-independent, so it hugs
    # the x-axis label whether there are 4 rows or 12).
    below = ax.transAxes + mtransforms.ScaledTranslation(0, -0.68, fig.dpi_scale_trans)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0),
              bbox_transform=below, ncol=2, frameon=False)

    # Honest footnote: the axis is zoomed, so gains look bigger than they are.
    # Everything is computed from the plotted data — no hardcoded limits.
    lo, hi = ax.get_xlim()
    span_pts = (max(xs) - min(xs)) * 100  # spread of all runs, in accuracy points
    note = (f"Note: x-axis is zoomed (spans {lo:.3f}–{hi:.3f}, not 0–1); "
            f"all runs lie within {span_pts:.1f} points of each other.")
    note_below = ax.transAxes + mtransforms.ScaledTranslation(0, -1.02, fig.dpi_scale_trans)
    ax.text(0, 0, note, transform=note_below, fontsize=8.5, color=th["muted"],
            va="top", ha="left")
    return fig
