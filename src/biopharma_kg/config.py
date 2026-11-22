"""Configuration loading: packaged defaults < repo configs/default.yaml < user --config file."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    """Thin wrapper over the nested config dict with path helpers."""

    raw: dict
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    # ---- paths -----------------------------------------------------------------
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self.path(self.raw["paths"]["data_dir"])

    @property
    def reports_dir(self) -> Path:
        return self.path(self.raw["paths"]["reports_dir"])

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "processed" / "biopharma.sqlite"

    @property
    def kg_nquads(self) -> Path:
        return self.data_dir / "processed" / "kg.nq"

    @property
    def kg_inferred_nquads(self) -> Path:
        return self.data_dir / "processed" / "kg_inferred.nq"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.cache_dir, self.db_path.parent, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Load config. ``root`` is the project root (default: current working directory)."""
    root_p = Path(root) if root else Path.cwd()
    cfg: dict = {}
    default_file = root_p / DEFAULT_CONFIG_PATH
    if default_file.exists():
        cfg = yaml.safe_load(default_file.read_text()) or {}
    if path:
        user = Path(path)
        if not user.exists():
            raise FileNotFoundError(f"Config file not found: {user}")
        cfg = _deep_merge(cfg, yaml.safe_load(user.read_text()) or {})
    if not cfg:
        raise FileNotFoundError(
            f"No configuration found. Run from the repository root (expected {default_file}) "
            "or pass --config."
        )
    return Config(raw=cfg, root=root_p)
