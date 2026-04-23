"""
Config loader.

Reads `config.yaml` once and exposes it as a nested `Config` object with both
attribute access (`cfg.unet.lr`) and dict access (`cfg["unet"]["lr"]`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class Config(dict):
    """Dict subclass that supports attribute-style access recursively."""

    def __init__(self, data: dict | None = None) -> None:
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path | None = None) -> Config:
    """Load `config.yaml` from disk. Defaults to the repo-root config."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)


def resolve_path(cfg_path_value: str | Path) -> Path:
    """Resolve a path from config: absolute stays absolute, relative → repo root."""
    p = Path(cfg_path_value)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()
