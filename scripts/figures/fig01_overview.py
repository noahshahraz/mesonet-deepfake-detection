"""Hero overview figure (top of Results): 2x2 composite.

(a) Reproduction vs. the paper's reported numbers (FaceForensics++ accuracy).
(b) Generalization AUC of the frozen Meso-4 Deepfakes checkpoint on four test sets.
(c) Score-distribution mini-panels: in-domain vs. off-distribution.
(d) Parameter count vs. cross-dataset AUC (does capacity buy transfer? no).

All numbers come from results/summary.json; score histograms come from outputs/*_probs.npz
when present and degrade to a muted note otherwise.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

import figstyle

NAME = "fig01_overview"

C_MESO4 = figstyle.CYCLE[0]        # blue
C_INCEPTION = figstyle.CYCLE[1]    # orange
C_XCEPTION = figstyle.CYCLE[2]     # green
C_PAPER = figstyle.CYCLE[3]        # vermillion = paper / reference


# ---------------------------------------------------------------------------- helpers
def _stat(node, key):
    """Return (mean, std, n) for node[key], accepting {'mean':..} dicts or bare floats."""
    if not isinstance(node, dict) or node.get(key) is None:
        return None
    v = node[key]
    if isinstance(v, dict):
        if v.get("mean") is None:
            return None
        return float(v["mean"]), float(v.get("std") or 0.0), int(v.get("n") or 1)
    return float(v), 0.0, 1


def _seed_phrase(nodes, key):
    """Compact seed description for panel subtitles, from the runs' per_seed keys."""
    seeds = sorted({seed for node in nodes
                    if isinstance(node, dict) and isinstance(node.get(key), dict)
                    for seed in (node[key].get("per_seed") or {})}, key=str)
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    return "seed-averaged"


def _seed_line(summary):
    """Truthful seed note for the figure subtitle, derived from summary['reproduction']."""
    repro = summary.get("reproduction") or {}
    stats = [_stat(run, "acc") for run in repro.values()]
    n_seeds = max((s[2] for s in stats if s is not None), default=1)
    if n_seeds > 1:
        return f"Error bars: spread across {n_seeds} training seeds."
    seeds = sorted({seed for run in repro.values() if isinstance(run, dict)
                    for seed in (run.get("acc") or {}).get("per_seed", {})
                    if isinstance(run.get("acc"), dict)})
    seed = seeds[0] if len(seeds) == 1 else "42"
    return f"Single training seed ({seed}); multi-seed runs in progress."


def _header(ax, th, title, subtitle, title_y=1.16, sub_y=1.075):
    ax.text(0.0, title_y, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=th["fg"])
    ax.text(0.0, sub_y, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, style="italic", color=th["muted"])


def _missing(ax, th, msg):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, color=th["muted"], style="italic", wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=th["accent_bg"],
                      edgecolor=th["grid"], linewidth=0.8))


# ---------------------------------------------------------------------------- panel (a)
def _panel_a(ax, summary, th):
    repro = summary.get("reproduction") or {}
    paper = summary.get("paper") or {}
    _header(ax, th,
            "(a) The paper's headline results reproduce within a few points",
            f"Bars: our test accuracy (per-image, {_seed_phrase(repro.values(), 'acc')}). "
            "Dashed lines and open diamonds: numbers\n"
            "reported in the paper. y-axis starts at 50% — coin flip on this balanced test set — not zero.",
            sub_y=1.055)
    methods = [("ff_deepfakes", "Deepfakes"), ("ff_face2face", "Face2Face")]
    models = [("meso4", "Meso-4", C_MESO4, ""),
              ("meso_inception4", "MesoInception-4", C_INCEPTION, figstyle.HATCHES[1])]
    if not repro:
        _missing(ax, th, "reproduction results not available in results/summary.json")
        return

    width = 0.34
    handles = []
    for j, (mkey, mname, color, hatch) in enumerate(models):
        xs, means, stds, ns = [], [], [], []
        for i, (dkey, _) in enumerate(methods):
            s = _stat(repro.get(f"{mkey}:{dkey}"), "acc")
            if s is None:
                continue
            xs.append(i + (j - 0.5) * (width + 0.04))
            means.append(s[0]); stds.append(s[1]); ns.append(s[2])
        if not xs:
            continue
        yerr = [sd if n > 1 else 0.0 for sd, n in zip(stds, ns)]
        ax.bar(xs, means, width=width, color=color, hatch=hatch,
               edgecolor=to_rgba(th["fg"], 0.55), linewidth=0.8,
               yerr=yerr if any(e > 0 for e in yerr) else None,
               error_kw=dict(ecolor=th["fg"], capsize=3, lw=1.2), zorder=3)
        txt_color = "white" if mkey == "meso4" else "#1f2328"
        for x, m in zip(xs, means):
            ax.text(x, m - 0.016, f"{m * 100:.1f}%", ha="center", va="top",
                    fontsize=8.5, fontweight="bold", color=txt_color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=color,
                              edgecolor="none"))
        handles.append(Patch(facecolor=color, hatch=hatch,
                             edgecolor=to_rgba(th["fg"], 0.55), linewidth=0.8,
                             label=f"{mname} (ours)"))

    # paper references: headline per-video accuracy as dashed line segments per group
    headline = (paper.get("headline_video_level") or {})
    drew_line = False
    for i, (dkey, _) in enumerate(methods):
        v = headline.get(dkey.replace("ff_", ""))
        if v is not None:
            ax.hlines(v, i - 0.45, i + 0.45, color=C_PAPER, linestyle=(0, (5, 3)),
                      linewidth=1.7, zorder=4)
            drew_line = True
    if drew_line:
        handles.append(Line2D([], [], color=C_PAPER, linestyle=(0, (5, 3)),
                              linewidth=1.7, label="paper (per-video)"))

    # paper per-image c23 Face2Face refs, one hollow diamond per model bar
    per_image = (paper.get("per_image_c23_face2face") or {})
    i_f2f = next((i for i, (dkey, _) in enumerate(methods) if dkey == "ff_face2face"), None)
    drew_diamond = False
    if i_f2f is not None:
        for j, (mkey, _, _, _) in enumerate(models):
            v = per_image.get(mkey)
            if v is not None:
                ax.plot(i_f2f + (j - 0.5) * (width + 0.04), v, marker="D", ls="none",
                        markersize=7.5, markerfacecolor="none", markeredgecolor=C_PAPER,
                        markeredgewidth=1.7, zorder=6)
                drew_diamond = True
    if drew_diamond:
        handles.append(Line2D([], [], marker="D", ls="none", markersize=7.5,
                              markerfacecolor="none", markeredgecolor=C_PAPER,
                              markeredgewidth=1.7, label="paper (per-image)"))

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f"{name}\n(FaceForensics++)" for _, name in methods], fontsize=9.5)
    ax.set_xlim(-0.55, len(methods) - 0.45)
    ax.set_ylim(0.5, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Accuracy on held-out test images")
    ax.grid(axis="y")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncols=2, fontsize=8.5, handlelength=1.8, columnspacing=1.4)


# ---------------------------------------------------------------------------- panel (b)
def _panel_b(ax, summary, th):
    gen = ((summary.get("generalization") or {}).get("meso4") or {})
    evals = gen.get("evals") or {}
    _header(ax, th,
            "(b) The same frozen Meso-4 fails away from its training data",
            f"AUC of Meso-4 (trained on FF++ Deepfakes, {_seed_phrase(evals.values(), 'auc')}) "
            "on four test sets.\n"
            "Denser hatching = further from the training data. 1.0 = perfect, 0.5 = coin flip.",
            sub_y=1.055)
    order = [
        ("ff_deepfakes", "FF++ Deepfakes\ntrained on this", figstyle.HATCHES[0]),
        ("ff_face2face", "FF++ Face2Face\nnew forgery method", figstyle.HATCHES[1]),
        ("openforensics", "OpenForensics\nnew dataset", figstyle.HATCHES[3]),
        ("faces140k", "140k StyleGAN\nnew dataset", figstyle.HATCHES[3]),
    ]
    bars = [(lbl, _stat(evals.get(k), "auc"), hatch) for k, lbl, hatch in order]
    bars = [(lbl, s, hatch) for lbl, s, hatch in bars if s is not None]
    if not bars:
        _missing(ax, th, "generalization results not available in results/summary.json")
        return

    xs = np.arange(len(bars))
    for x, (lbl, (mean, std, n), hatch) in zip(xs, bars):
        ax.bar(x, mean, width=0.62, color=C_MESO4, hatch=hatch,
               edgecolor=to_rgba(th["fg"], 0.55), linewidth=0.8,
               yerr=std if n > 1 else None,
               error_kw=dict(ecolor=th["fg"], capsize=3, lw=1.2), zorder=3)
        ax.text(x, mean + 0.025, f"{mean:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=th["fg"], zorder=5)

    chance = summary.get("chance_auc", 0.5)
    ax.axhline(chance, color=th["muted"], linestyle="--", linewidth=1.3, zorder=4)
    ax.text(len(bars) - 0.62, chance + 0.02, "coin flip", ha="left", va="bottom",
            fontsize=8.5, style="italic", color=th["muted"])

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _, _ in bars], fontsize=8.8)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.set_ylabel("AUC (ranking quality of scores)")
    ax.grid(axis="y")


# ---------------------------------------------------------------------------- panel (c)
def _hist_pair(ax, labels, probs, th, show_legend):
    bins = np.linspace(0.0, 1.0, 21)
    handles = []
    for lab, color, hatch, word in ((0, C_MESO4, "", "real images"),
                                    (1, C_INCEPTION, figstyle.HATCHES[1], "fake images")):
        vals = probs[labels == lab]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, weights=np.full(vals.size, 1.0 / vals.size),
                histtype="stepfilled", facecolor=to_rgba(color, 0.30),
                edgecolor=color, linewidth=1.5, hatch=hatch, zorder=3)
        handles.append(Patch(facecolor=to_rgba(color, 0.30), edgecolor=color,
                             linewidth=1.5, hatch=hatch, label=word))
    if show_legend and handles:
        ax.legend(handles=handles, loc="upper center", fontsize=8,
                  handlelength=1.4, borderaxespad=0.2)


def _panel_c(ax_l, ax_r, summary, probs, th):
    _header(ax_l, th,
            "(c) Why it fails: the scores collapse on new data",
            "Meso-4's score per image, 0 = “sure it's real”, 1 = “sure it's fake” "
            "(share of each group per bin).",
            sub_y=1.075)
    stems = summary.get("probs_stems") or {}
    meso4_evals = ((summary.get("generalization") or {}).get("meso4") or {}).get("evals") or {}

    def _auc_note(dataset):
        s = _stat(meso4_evals.get(dataset), "auc")
        return f"AUC {s[0]:.2f}" if s is not None else None

    # annotation anchors (x, y, ha) in axes coords: left panel sits in the empty
    # valley between the two score modes (clear of the legend and the tall bar at
    # x≈1.0); right panel keeps its top-right spot.
    panels = [
        (ax_l, stems.get("in_domain"), "in-domain: real vs. fake separate", True,
         _auc_note("ff_deepfakes"), (0.55, 0.72, "right")),
        (ax_r, stems.get("cross_140k"), "new dataset: everything scored “real”", False,
         _auc_note("faces140k"), (0.97, 0.72, "right")),
    ]
    for ax, stem, title, show_legend, auc_note, (nx, ny, ha) in panels:
        ax.set_title(title, loc="left", fontsize=9.5, pad=5, color=th["fg"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        d = probs.get(stem) if stem else None
        if d is None:
            _missing(ax, th, "score data not available\n(regenerate with src.eval)")
            continue
        _hist_pair(ax, d["labels"], d["probs"], th, show_legend)
        ax.set_ylim(0, 1.0)
        if auc_note is not None:
            ax.text(nx, ny, auc_note, transform=ax.transAxes, ha=ha, va="top",
                    fontsize=8.5, style="italic", color=th["muted"])
        ax.set_xlabel("model score P(fake)", fontsize=9)
        ax.grid(axis="y")
    ax_l.set_ylabel("Share of images", fontsize=10)
    ax_r.set_yticklabels([])


# ---------------------------------------------------------------------------- panel (d)
def _panel_d(ax, summary, th):
    gen = summary.get("generalization") or {}
    params = summary.get("params") or {}
    if params.get("xception") and params.get("meso4"):
        ratio = params["xception"] / params["meso4"]
        d_title = f"(d) {ratio:,.0f}× more parameters does not buy generalization"
    else:
        d_title = "(d) More parameters do not buy generalization"
    _header(ax, th, d_title,
            "Faint marker: AUC on the training distribution. Solid marker: mean AUC on the two\n"
            "unseen datasets (OpenForensics + 140k StyleGAN). The vertical line is the drop.",
            sub_y=1.055)
    cross_sets = ("openforensics", "faces140k")
    models = [("meso4", "Meso-4", C_MESO4, figstyle.MARKERS[0]),
              ("meso_inception4", "MesoInception-4", C_INCEPTION, figstyle.MARKERS[1]),
              ("xception", "Xception", C_XCEPTION, figstyle.MARKERS[2])]

    plotted, skipped = [], []
    for mkey, mname, color, marker in models:
        p = params.get(mkey)
        evals = (gen.get(mkey) or {}).get("evals") or {}
        cross = [_stat(evals.get(ds), "auc") for ds in cross_sets]
        cross = [c[0] for c in cross if c is not None]
        train_key = (gen.get(mkey) or {}).get("train")
        indom = _stat(evals.get(train_key), "auc") if train_key else None
        if p is None or len(cross) < len(cross_sets):
            if p is not None:
                skipped.append(mname)
            continue
        y_cross = float(np.mean(cross))
        if indom is not None:
            ax.plot([p, p], [y_cross, indom[0]], color=to_rgba(color, 0.45),
                    linewidth=1.2, zorder=2)
            ax.plot(p, indom[0], marker=marker, ls="none", markersize=9,
                    color=to_rgba(color, 0.35),
                    markeredgecolor=to_rgba(th["fg"], 0.25), zorder=3)
        ax.plot(p, y_cross, marker=marker, ls="none", markersize=10, color=color,
                markeredgecolor=th["fg"], markeredgewidth=0.8, zorder=4)
        plotted.append((mkey, mname, p, y_cross, indom[0] if indom else None))

    if not plotted:
        _missing(ax, th, "cross-dataset generalization results not available in results/summary.json")
        return

    for mkey, mname, p, y_cross, y_in in plotted:
        # keep labels inside the axes: annotate to the left for points near the right edge
        left_side = p > 1e6
        dx, ha = (-12, "right") if left_side else (12, "left")
        ax.annotate(f"{mname} · {p:,} params\nnew data {y_cross:.2f}", (p, y_cross),
                    xytext=(dx, -4), textcoords="offset points", ha=ha, va="top",
                    fontsize=9, color=th["fg"])
        if y_in is not None:
            ax.annotate(f"in-domain {y_in:.3f}", (p, y_in), xytext=(dx, 0),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=8, style="italic", color=th["muted"])

    chance = summary.get("chance_auc", 0.5)
    ax.axhline(chance, color=th["muted"], linestyle="--", linewidth=1.3, zorder=1)
    ax.text(0.02, chance + 0.015, "coin flip (0.5)", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8.5, style="italic", color=th["muted"])

    if skipped:
        ax.text(0.02, 0.02, f"{', '.join(skipped)}: no cross-dataset evals in summary",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, style="italic", color=th["muted"])

    ax.set_xscale("log")
    ax.set_xlim(8e3, 3e8)
    ax.set_ylim(0.2, 1.06)
    ax.set_xlabel("Number of trainable parameters (log scale)")
    ax.set_ylabel("AUC (1.0 = perfect, 0.5 = coin flip)")
    ax.grid(True, which="major")


# ---------------------------------------------------------------------------- build
def build(data: dict, th: dict):
    import matplotlib.pyplot as plt

    summary = data.get("summary", {}) or {}
    probs = data.get("probs", {}) or {}

    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.985, top=0.845, bottom=0.075,
                          hspace=0.72, wspace=0.24)

    fig.suptitle("Deepfake detectors ace their training dataset — "
                 "and fail on data they never saw",
                 x=0.06, y=0.985, ha="left", fontsize=17, fontweight="bold",
                 color=th["fg"])
    fig.text(0.06, 0.945,
             "MesoNet reproduction on FaceForensics++ (c23); the transfer panels (b–d) "
             "stress-test the frozen Meso-4 checkpoint (plus Xception in d)\n"
             f"on datasets it never saw. {_seed_line(summary)}",
             ha="left", va="top", fontsize=10.5, color=th["muted"])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    sub_c = gs[1, 0].subgridspec(1, 2, wspace=0.12)
    ax_c1 = fig.add_subplot(sub_c[0, 0])
    ax_c2 = fig.add_subplot(sub_c[0, 1], sharey=ax_c1)
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, summary, th)
    _panel_b(ax_b, summary, th)
    _panel_c(ax_c1, ax_c2, summary, probs, th)
    _panel_d(ax_d, summary, th)
    return fig
