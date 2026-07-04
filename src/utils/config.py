"""Tiny YAML config loader with dotted access and CLI override support."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict that also supports attribute access and nested `get_path('a.b.c')`."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        return Config(yaml.safe_load(f))
