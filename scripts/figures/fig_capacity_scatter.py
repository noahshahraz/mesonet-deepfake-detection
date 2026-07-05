"""Capacity-vs-generalization scatter: "bigger model, same failure" in one glance.

Two vertical columns of markers on a log-parameter axis — Meso-4 (~28k trainable
parameters) and Xception (~20.8M). Each column shares one x position: the filled
marker is AUC on the dataset the model trained on (FaceForensics++ Deepfakes);
the open and half-filled markers are AUC on two datasets the model never saw
(OpenForensics face swaps, 140k StyleGAN faces). Both columns fall from ~1.0 to
below the 0.5 coin-flip line, so ~744x more parameters buys no generalization.

All numbers come from results/summary.json (generalization + params sections).
Degrades gracefully: missing models/datasets are skipped; if nothing usable is
present, a muted placeholder note is rendered instead of crashing.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle

NAME = "fig_capacity_scatter"

IN_DOMAIN_KEY = "ff_deepfakes"
# (summary dataset key, plain-English label) for the two never-seen datasets.
CROSS = [
    ("openforensics", "OpenForensics"),
    ("faces140k", "140k StyleGAN faces"),
]
# (summary model key, display name, figstyle identity index: 0=Meso-4, 2=Xception)
MODELS = [
    ("meso4", "Meso-4", 0),
    ("xception", "Xception", 2),
]
DODGE = 10 ** 0.085  # tiny symmetric horizontal offset so cross-dataset markers don't overlap


def _auc_stat(entry):
    """Return (mean, std, n) for the AUC of one eval entry, or None.

    Handles both shapes in summary.json: aggregated dicts
    {"auc": {"mean": .., "std": .., "n": ..}} (Meso-4) and plain floats
    {"auc": 0.99} (Xception single-seed).
    """
    if not isinstance(entry, dict):
        return None
    auc = entry.get("auc")
    if isinstance(auc, dict):
        mean = auc.get("mean")
        if mean is None:
            return None
        return float(mean), float(auc.get("std") or 0.0), int(auc.get("n") or 1)
    if isinstance(auc, (int, float)):
        return float(auc), 0.0, 1
    return None


def _human_params(n: float) -> str:
    if n >= 1e6:
        s = f"{n / 1e6:.1f}M"
        return s.replace(".0M", "M")
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def build(data: dict, th: dict):
    summary = data.get("summary") or {}
    params = summary.get("params") or {}
    gen = summary.get("generalization") or {}
    chance = float(summary.get("chance_auc") or 0.5)

    fig, ax = plt.subplots(figsize=(9.4, 5.9))

    # ---- collect one "column" per model -------------------------------------------------
    columns = []
    for key, name, idx in MODELS:
        n_params = params.get(key)
        evals = (gen.get(key) or {}).get("evals") or {}
        in_stat = _auc_stat(evals.get(IN_DOMAIN_KEY))
        cross = [(lab, _auc_stat(evals.get(k))) for k, lab in CROSS]
        cross = [(lab, st) for lab, st in cross if st is not None]
        if n_params and in_stat is not None:
            columns.append({
                "x": float(n_params), "name": name,
                "color": figstyle.CYCLE[idx], "marker": figstyle.MARKERS[idx],
                "in_stat": in_stat, "cross": cross,
            })

    if not columns:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "generalization results not available in results/summary.json",
                ha="center", va="center", color=th["muted"], fontsize=12,
                transform=ax.transAxes)
        return fig

    # ---- axes scaffolding ----------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(6e3, 1.6e8)
    ax.set_ylim(0.05, 1.12)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7], labels=["10k", "100k", "1M", "10M"])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("AUC on held-out test images\n(1.0 = perfect detection, 0.5 = coin flip)")

    # Coin-flip reference: dashed line + shaded "worse than guessing" region.
    ax.axhspan(0.05, chance, color=th["muted"], alpha=0.07, zorder=0)
    ax.axhline(chance, ls=(0, (5, 3)), lw=1.2, color=th["muted"], zorder=1)
    ax.text(1.45e8, chance + 0.015, f"coin flip (AUC {chance:g})", ha="right", va="bottom",
            fontsize=9.5, color=th["muted"])
    ax.text(9.8e5, 0.115, "worse than a coin flip", ha="center", va="center",
            fontsize=9.5, style="italic", color=th["muted"])

    # ---- draw each model column ------------------------------------------------------------
    any_whisker = False
    for col in columns:
        x, color, marker = col["x"], col["color"], col["marker"]
        in_mean, in_std, in_n = col["in_stat"]
        cross_means = [st[0] for _, st in col["cross"]]
        y_low = min([in_mean] + cross_means)

        # thin vertical line: the model "falls" from in-domain to cross-dataset AUC
        ax.plot([x, x], [y_low, in_mean], color=color, lw=1.3, alpha=0.55, zorder=2)

        # in-domain: filled marker on the line
        if in_n > 1 and in_std > 0:
            ax.errorbar([x], [in_mean], yerr=in_std, fmt="none", ecolor=color,
                        elinewidth=1.2, capsize=3.5, zorder=3)
            any_whisker = True
        ax.plot([x], [in_mean], marker=marker, ms=10.5, ls="none",
                mfc=color, mec=th["bg"], mew=0.9, zorder=4)
        # 3 decimals so e.g. 0.9968 doesn't round to "1.00" next to "1.0 = perfect" text.
        # Columns on the left half get the label on the LEFT of the marker so it stays
        # clear of the centered in-plot legend (right-half columns keep it on the right).
        left_half = x < (ax.get_xlim()[0] * ax.get_xlim()[1]) ** 0.5
        dx, ha = (-10, "right") if left_half else (10, "left")
        ax.annotate(f"{in_mean:.3f}", (x, in_mean), xytext=(dx, -3),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=9.5, color=th["fg"])

        # cross-dataset markers, dodged slightly left/right of the line with stub connectors
        for j, (lab, (mean, std, n)) in enumerate(col["cross"]):
            xm = x / DODGE if j == 0 else x * DODGE
            ax.plot([x, xm], [mean, mean], color=color, lw=0.9, alpha=0.5, zorder=2)
            if n > 1 and std > 0:
                ax.errorbar([xm], [mean], yerr=std, fmt="none", ecolor=color,
                            elinewidth=1.2, capsize=3.5, zorder=3)
                any_whisker = True
            if j == 0:  # open marker = OpenForensics
                ax.plot([xm], [mean], marker=marker, ms=10.5, ls="none",
                        mfc=th["bg"], mec=color, mew=1.8, zorder=4)
                ax.annotate(f"{mean:.2f}", (xm, mean), xytext=(-8, -1),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=9.5, color=th["fg"])
            else:  # half-filled marker = 140k StyleGAN faces
                ax.plot([xm], [mean], marker=marker, ms=10.5, ls="none",
                        fillstyle="left", mfc=color, mfcalt=th["bg"], mec=color,
                        mew=1.4, zorder=4)
                ax.annotate(f"{mean:.2f}", (xm, mean), xytext=(8, -1),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=9.5, color=th["fg"])

        # model name + parameter count under the column ("Meso-4 · 28k params").
        # Okabe-Ito blue text fails contrast on the dark background, so theme-switch the
        # TEXT to sky blue there; markers/lines keep the identity color #0072B2.
        text_color = (figstyle.OKABE_ITO["skyblue"]
                      if th["dark"] and color == figstyle.OKABE_ITO["blue"] else color)
        ax.annotate(f"{col['name']} · {_human_params(x)} params", (x, y_low), xytext=(0, -28),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=10.5, fontweight="bold", color=text_color)

    # ---- punchline arrow between the two columns -------------------------------------------
    ratio = None
    if len(columns) == 2:
        x1, x2 = columns[0]["x"], columns[1]["x"]
        ratio = max(x1, x2) / min(x1, x2)
        gx = (x1 * x2) ** 0.5
        ax.annotate("", xy=(x2 / 2.2, 0.64), xytext=(x1 * 2.2, 0.64),
                    arrowprops=dict(arrowstyle="<|-|>", color=th["muted"], lw=1.1))
        ax.text(gx, 0.665, f"{ratio:,.0f}× more trainable parameters",
                ha="center", va="bottom", fontsize=10.5, color=th["fg"])
        ax.text(gx, 0.615, "same collapse on data it never saw",
                ha="center", va="top", fontsize=9.5, style="italic", color=th["muted"])

    # ---- legend: what the marker fill means (dataset), shape+color already = model ----------
    handles = [Line2D([], [], marker="o", ls="none", ms=8.5, mfc=th["fg"], mec=th["fg"],
                      label="Tested on the dataset it trained on (FF++ Deepfakes)")]
    used = {lab for c in columns for lab, _ in c["cross"]}
    if CROSS[0][1] in used:
        handles.append(Line2D([], [], marker="o", ls="none", ms=8.5, mfc="none",
                              mec=th["fg"], mew=1.6,
                              label="Never seen in training: OpenForensics (real-world face swaps)"))
    if CROSS[1][1] in used:
        handles.append(Line2D([], [], marker="o", ls="none", ms=8.5, fillstyle="left",
                              mfc=th["fg"], mfcalt="none", mec=th["fg"], mew=1.2,
                              label="Never seen in training: 140k (StyleGAN-generated faces)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
              handletextpad=0.5, borderaxespad=0.0)

    # The per-model horizontal dodge is purely cosmetic — say so under the x-axis.
    # Parameter counts come from summary["params"] via the collected columns.
    counts = " / ".join(_human_params(c["x"]) for c in columns)
    ax.text(0.5, -0.145, "markers offset horizontally for visibility — "
            f"one parameter count per model ({counts})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=th["muted"])

    # ---- title / subtitle --------------------------------------------------------------------
    if ratio is not None:
        title = f"{ratio:,.0f}× more parameters, the same cross-dataset failure"
    else:
        title = "Bigger model, same cross-dataset failure"
    subtitle = ("ROC AUC of deepfake detectors trained on FaceForensics++ Deepfakes, tested on "
                "their own test set (filled marker)\nand on two datasets never seen in training "
                "(open and half-filled markers). Below 0.5 the ranking is worse than guessing.")
    if any_whisker:
        subtitle += " Whiskers: ±1 std across seeds."
    ax.set_title(title, loc="left", pad=52)
    ax.text(0, 1.06, subtitle, transform=ax.transAxes, fontsize=10,
            color=th["muted"], va="bottom", linespacing=1.45)

    return fig
