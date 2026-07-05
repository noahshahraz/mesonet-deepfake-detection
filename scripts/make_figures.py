"""Regenerate every figure in the suite from committed data (task 5 entry point).

    python scripts/make_figures.py            # all figures, light + dark
    python scripts/make_figures.py fig_roc    # just one

Data: results/summary.json (committed) plus outputs/<stem>_probs.npz when present (gitignored;
distribution/ROC/calibration figures degrade gracefully without them). Deterministic given those
inputs. Figure modules live in scripts/figures/fig_*.py and expose NAME and build(data, th).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import figstyle  # noqa: E402


def load_data() -> dict:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    probs: dict = {}
    stems = set()
    ps = summary.get("probs_stems", {})
    for v in ps.values():
        stems.update(v.values() if isinstance(v, dict) else [v])
    for s in stems:
        npz_path = ROOT / "outputs" / f"{s}_probs.npz"
        if npz_path.exists():
            arr = np.load(npz_path)
            probs[s] = {"labels": arr["labels"], "probs": arr["probs"]}
    return {"summary": summary, "probs": probs}


def figure_modules() -> list:
    mods = []
    for path in sorted((ROOT / "scripts" / "figures").glob("fig*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_data()
    print(f"[figures] probs npz available for {len(data['probs'])} stems")
    made = []
    for mod in figure_modules():
        if only and mod.NAME != only:
            continue
        paths = figstyle.render(mod.build, mod.NAME, data, assets_dir=ROOT / "assets")
        made.extend(paths)
        print(f"[figures] {mod.NAME}: {', '.join(p.name for p in paths)}")
    if not made:
        raise SystemExit(f"no figure named {only!r}")
    print(f"[figures] wrote {len(made)} files to assets/")


if __name__ == "__main__":
    main()
