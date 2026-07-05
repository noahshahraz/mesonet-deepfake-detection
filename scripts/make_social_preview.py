"""Render the GitHub social-preview card (assets/social_preview.png, exactly 1280x640 px).

Standalone on purpose — NOT a fig_* module, so scripts/make_figures.py never discovers it
(social cards are single-theme, fixed-pixel artifacts; the figure suite is dual-theme + tight-bbox).

    python scripts/make_social_preview.py

Design: dark card (#0d1117 — GitHub renders social previews on both themes, dark reads as
branded), repo title + one-line finding on the left, a mini bar chart of the Meso-4
generalization AUCs (from results/summary.json) on the right. All colors come from
scripts/figstyle.py (Okabe-Ito palette + GitHub dark theme constants).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import figstyle  # noqa: E402

OUT = ROOT / "assets" / "social_preview.png"
W_PX, H_PX = 1280, 640
DPI = 100

# Display order + plain-English labels for the generalization bar chart.
EVAL_ORDER = [
    ("ff_deepfakes", "FF++\nDeepfakes\n(trained here)"),
    ("ff_face2face", "FF++\nFace2Face"),
    ("openforensics", "Open-\nForensics"),
    ("faces140k", "140k\nStyleGAN"),
]


def load_summary() -> dict:
    path = ROOT / "results" / "summary.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def meso4_generalization_aucs(summary: dict) -> list[tuple[str, float]] | None:
    """[(label, auc), ...] in EVAL_ORDER, or None if the section is missing/unusable."""
    evals = ((summary.get("generalization") or {}).get("meso4") or {}).get("evals") or {}
    out = []
    for key, label in EVAL_ORDER:
        entry = evals.get(key)
        auc = None
        if isinstance(entry, dict):
            auc = entry.get("auc")
            if isinstance(auc, dict):
                auc = auc.get("mean")
        if not isinstance(auc, (int, float)):
            return None
        out.append((label, float(auc)))
    return out


def human_params(n: object, fallback: str) -> str:
    if not isinstance(n, (int, float)) or n <= 0:
        return fallback
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{round(n / 1e3)}k"


def finding_lines(summary: dict) -> list[str]:
    """The one-line finding, wrapped for the card. Numbers pulled from summary when present."""
    repro = summary.get("reproduction") or {}
    acc = ((repro.get("meso4:ff_deepfakes") or {}).get("acc") or {}).get("mean")
    acc_txt = f"~{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "~93%"
    params = summary.get("params") or {}
    small = human_params(params.get("meso4"), "28k")
    big = human_params(params.get("xception"), "20.8M")
    return [
        f"{acc_txt} on FaceForensics++ —",
        f"but neither a {small}- nor a {big}-parameter",
        "model survives a dataset change.",
    ]


def draw_chart(fig, th: dict, summary: dict) -> None:
    aucs = meso4_generalization_aucs(summary)
    if aucs is None:
        fig.text(0.765, 0.5, "generalization data not available\n(regenerate results/summary.json)",
                 ha="center", va="center", fontsize=13, color=th["muted"], style="italic")
        return

    ax = fig.add_axes([0.585, 0.175, 0.365, 0.60])
    ax.set_facecolor(th["bg"])

    labels = [lbl for lbl, _ in aucs]
    values = [v for _, v in aucs]
    xs = range(len(values))
    blue = figstyle.CYCLE[0]  # CYCLE[0] = Meso-4 everywhere in the suite

    # All bars are the same model (same blue); in-domain solid, cross-dataset hatched —
    # the distinction is carried by hatch + the "(trained here)" tick label, never color alone.
    for i, v in enumerate(values):
        ax.bar(i, v, width=0.62, facecolor=blue, edgecolor=th["bg"],
               linewidth=1.0, hatch="" if i == 0 else figstyle.HATCHES[1], zorder=3)
        ax.text(i, v + 0.035, f"{v:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=th["fg"])

    chance = summary.get("chance_auc", 0.5)
    ax.axhline(chance, color=th["muted"], linestyle=(0, (5, 4)), linewidth=1.3, zorder=2)
    ax.text(len(values) - 0.55, chance + 0.025, f"coin flip = {chance:.2f}",
            ha="right", va="bottom", fontsize=10, color=th["muted"])

    ax.set_title("Same Meso-4 weights, four test sets", fontsize=14.5,
                 fontweight="bold", color=th["fg"], pad=14)
    ax.set_ylabel("AUC   (1.0 = perfect, 0.5 = coin flip)", fontsize=10.5, color=th["fg"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.tick_params(colors=th["fg"], labelsize=9.5, length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(th["muted"])
    ax.grid(axis="y", color=th["grid"], linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def build_card() -> "plt.Figure":
    th = figstyle.DARK
    summary = load_summary()

    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(th["bg"])

    left = 0.055
    # Title block
    fig.text(left, 0.80, "MesoNet", fontsize=50, fontweight="bold",
             color=th["fg"], va="baseline")
    fig.text(left, 0.705, "Deepfake Detection, Reproduced", fontsize=23,
             fontweight="bold", color=figstyle.OKABE_ITO["skyblue"], va="baseline")

    # One-line finding, wrapped
    for i, line in enumerate(finding_lines(summary)):
        fig.text(left, 0.565 - i * 0.072, line, fontsize=17, color=th["fg"], va="baseline")

    # Small muted tech line, bottom-left
    fig.text(left, 0.075, "PyTorch · Apple MPS · Afchar et al., WIFS 2018 (arXiv:1809.00888)",
             fontsize=12, color=th["muted"], va="baseline")

    draw_chart(fig, th, summary)
    return fig


def main() -> None:
    fig = build_card()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches='tight' — it would resize the canvas away from exactly 1280x640.
    fig.savefig(OUT, dpi=DPI, facecolor=figstyle.DARK["bg"])
    plt.close(fig)

    size = Image.open(OUT).size
    assert size == (W_PX, H_PX), f"social preview is {size}, expected {(W_PX, H_PX)}"
    print(f"[social] wrote {OUT.relative_to(ROOT)} ({size[0]}x{size[1]} px)")


if __name__ == "__main__":
    main()
