"""Tests for the artifact overwrite guard (hardening task 1)."""
import pytest

from src.utils.overwrite import guard_overwrite


def test_guard_passes_when_nothing_exists(tmp_path):
    guard_overwrite([tmp_path / "nope.pth", tmp_path / "nope.jsonl"], overwrite=False)


def test_guard_refuses_existing_without_overwrite(tmp_path):
    ckpt = tmp_path / "run_best.pth"
    ckpt.write_text("x")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        guard_overwrite([ckpt, tmp_path / "missing.jsonl"], overwrite=False)


def test_guard_allows_existing_with_overwrite(tmp_path):
    ckpt = tmp_path / "run_best.pth"
    ckpt.write_text("x")
    guard_overwrite([ckpt], overwrite=True)
