"""Cross-dataset transfer heatmap: AUC of every trained model on every dataset.

One annotated matrix from summary.transfer_matrix_auc_seed42 (single seed-42 run).
Rows = dataset the model was trained on, columns = dataset it is evaluated on.
Diverging colorblind-safe colormap (PuOr) centered at 0.5 so below-chance cells
read as a categorically different hue; combinations that were never run are
hatched and labeled "not run"; in-domain (diagonal) cells get a thin border.
"""
from __future__ import annotations

import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import figstyle

NAME = "fig_transfer_matrix"

# Fixed dataset order shared by rows and columns (union of keys in the matrix).
FIXED_ORDER = ["ff_deepfakes", "ff_face2face", "openforensics", "faces140k"]
DISPLAY = {
    "ff_deepfakes": "FF++ Deepfakes",
    "ff_face2face": "FF++ Face2Face",
    "openforensics": "OpenForensics",
    "faces140k": "140k Faces (StyleGAN)",
}
# Two-line variants keep the column labels from crowding each other.
DISPLAY_COL = {
    "ff_deepfakes": "FF++\nDeepfakes",
    "ff_face2face": "FF++\nFace2Face",
    "openforensics": "OpenForensics",
    "faces140k": "140k Faces\n(StyleGAN)",
}


def _ordered_union(matrix: dict) -> list[str]:
    """All dataset keys (rows and columns) in FIXED_ORDER, extras appended sorted."""
    seen = set(matrix.keys())
    for row in matrix.values():
        if isinstance(row, dict):
            seen.update(row.keys())
    ordered = [k for k in FIXED_ORDER if k in seen]
    ordered += sorted(seen - set(FIXED_ORDER))
    return ordered


def _cell_text_color(cmap, norm, value: float) -> str:
    """Black-or-white annotation color chosen from the cell's own luminance."""
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#1f2328" if luminance > 0.55 else "#ffffff"


def _missing_figure(th: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, "transfer-matrix data not available in results/summary.json",
            ha="center", va="center", fontsize=11, color=th["muted"],
            style="italic", transform=ax.transAxes)
    fig.suptitle("Cross-dataset transfer matrix", fontsize=figstyle.TITLE_SIZE,
                 fontweight="bold")
    return fig


def build(data: dict, th: dict):
    import matplotlib.pyplot as plt

    summary = data.get("summary", {}) or {}
    matrix = summary.get("transfer_matrix_auc_seed42")
    if not isinstance(matrix, dict) or not matrix:
        return _missing_figure(th)

    keys = _ordered_union(matrix)
    n = len(keys)
    grid = np.full((n, n), np.nan)
    for i, row_key in enumerate(keys):
        row = matrix.get(row_key) or {}
        for j, col_key in enumerate(keys):
            if isinstance(row, dict) and col_key in row and row[col_key] is not None:
                grid[i, j] = float(row[col_key])

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.22, right=0.98)

    # Diverging colorblind-safe map centered on chance (AUC 0.5): below-chance
    # cells land on the orange side, above-chance on the purple side.
    cmap = plt.get_cmap("PuOr").copy()
    cmap.set_bad(alpha=0.0)  # missing cells are drawn as hatched patches instead
    finite = grid[np.isfinite(grid)]
    vmin = min(0.35, float(finite.min()) - 0.02) if finite.size else 0.35
    vmax = max(1.0, float(finite.max())) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)

    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell annotations: the number itself for every run, "not run" for the rest.
    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12,
                        fontweight="bold", color=_cell_text_color(cmap, norm, v))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                       facecolor=th["accent_bg"],
                                       edgecolor=th["grid"], linewidth=0.0,
                                       hatch=figstyle.HATCHES[1], zorder=2))
                ax.text(j, i, "not run", ha="center", va="center", fontsize=10,
                        style="italic", color=th["muted"], zorder=3)

    # Thin background-colored separators keep the cells visually distinct.
    for k in range(n + 1):
        ax.axhline(k - 0.5, color=th["bg"], linewidth=2.0, zorder=4)
        ax.axvline(k - 0.5, color=th["bg"], linewidth=2.0, zorder=4)

    # Thin border marks the in-domain (trained and tested on the same data) cells.
    for i in range(n):
        if np.isfinite(grid[i, i]):
            ax.add_patch(Rectangle((i - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                   edgecolor=th["fg"], linewidth=1.6, zorder=5))

    ax.set_xticks(range(n), [DISPLAY_COL.get(k, k) for k in keys])
    ax.set_yticks(range(n), [DISPLAY.get(k, k) for k in keys])
    ax.tick_params(length=0)
    ax.set_xlabel("Evaluated on (held-out test set)")
    ax.set_ylabel("Trained on")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC (0.5 = chance)", color=th["fg"], fontsize=10.5)
    cbar.outline.set_edgecolor(th["grid"])
    cbar.ax.tick_params(color=th["muted"], labelsize=9)
    # Mark the chance line. The PuOr midpoint is near-white in BOTH themes, so the
    # marker must be a fixed dark color (theme fg would vanish on dark); a thin
    # light halo keeps it legible over the surrounding colormap shades too.
    cbar.ax.axhline(0.5, color="#1f2328", linewidth=1.2, path_effects=[
        path_effects.withStroke(linewidth=2.6, foreground="#ffffff", alpha=0.8)])

    fig.suptitle("Detectors ace their home dataset but transfer poorly —\n"
                 "sometimes scoring worse than a coin flip",
                 fontsize=figstyle.TITLE_SIZE, fontweight="bold",
                 x=0.05, y=0.985, ha="left", linespacing=1.25)
    fig.text(0.05, 0.855,
             "Area under the ROC curve (AUC) for each Meso-4 model, single-run "
             "seed-42 snapshot. Reading a row: how a model trained\non that dataset "
             "scores everywhere else. Below 0.5 means its ranking actually inverts "
             "— it rates fakes as more real than real images.",
             ha="left", va="top", fontsize=10, color=th["muted"], linespacing=1.45)
    fig.text(0.05, 0.035,
             "Outlined diagonal cells are in-domain (trained and tested on the same "
             "dataset). Hatched cells are train/test combinations that were not run.",
             ha="left", va="top", fontsize=9, color=th["muted"])
    return fig
