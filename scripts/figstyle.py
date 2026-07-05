"""Shared figure style for the publication-quality figure suite (task 5).

One visual language for every figure:
- Okabe-Ito colorblind-safe palette, and never color alone (pair with MARKERS/HATCHES/labels).
- Plain-English titles/subtitles a non-ML reader can follow.
- Every figure renders twice: assets/<name>.png (GitHub light) and assets/<name>.dark.png
  (GitHub dark, bg #0d1117) so the README can theme-switch via <picture>.

Figure modules (scripts/figures/fig_*.py) expose `NAME: str` and
`build(data: dict, th: dict) -> matplotlib.figure.Figure`; they must not call savefig —
`render()` here owns theming and output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Okabe-Ito palette (colorblind-safe) -------------------------------------------------
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
# Cycle order used across every figure; same index = same meaning wherever possible:
# 0 blue = Meso-4, 1 orange = MesoInception-4, 2 green = Xception, 3 vermillion = paper/reference
CYCLE = [OKABE_ITO[k] for k in ("blue", "orange", "green", "vermillion", "purple", "skyblue")]
MARKERS = ["o", "s", "^", "D", "v", "P"]
HATCHES = ["", "//", "..", "xx", "\\\\", "++"]

LIGHT = {"dark": False, "bg": "#ffffff", "fg": "#1f2328", "grid": "#d0d7de",
         "muted": "#57606a", "accent_bg": "#f6f8fa"}
DARK = {"dark": True, "bg": "#0d1117", "fg": "#e6edf3", "grid": "#30363d",
        "muted": "#8b949e", "accent_bg": "#161b22"}

TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 15, 12, 10


def theme(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def apply(dark: bool = False) -> dict:
    """Set global rcParams for one theme and return the theme dict."""
    th = theme(dark)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": th["bg"],
        "axes.facecolor": th["bg"],
        "savefig.facecolor": th["bg"],
        "text.color": th["fg"],
        "axes.labelcolor": th["fg"],
        "axes.edgecolor": th["muted"],
        "xtick.color": th["fg"],
        "ytick.color": th["fg"],
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": th["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
    })
    return th


def save(fig, name: str, dark: bool, assets_dir: str | Path = "assets") -> Path:
    out = Path(assets_dir) / (f"{name}.dark.png" if dark else f"{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=theme(dark)["bg"])
    plt.close(fig)
    return out


def render(build: Callable[[dict, dict], "plt.Figure"], name: str,
           data: dict, assets_dir: str | Path = "assets") -> list[Path]:
    """Build + save a figure in both GitHub themes; returns the two paths."""
    paths = []
    for dark in (False, True):
        th = apply(dark)
        fig = build(data, th)
        paths.append(save(fig, name, dark, assets_dir))
    return paths
