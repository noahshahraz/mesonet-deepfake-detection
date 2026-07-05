"""Refuse to clobber prior run artifacts unless explicitly allowed.

Run names default to <model>_<dataset>, so a re-run would silently replace earlier
checkpoints/logs/metrics. Callers collect the artifact paths a run is about to write and pass
them here before doing any work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def guard_overwrite(paths: Iterable[str | Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise SystemExit(
            "[guard] refusing to overwrite existing artifacts:\n  "
            + "\n  ".join(existing)
            + "\nPass --overwrite to replace them, or choose a unique --run-name / --out-stem."
        )
