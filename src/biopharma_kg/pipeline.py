"""End-to-end pipeline orchestration."""
from __future__ import annotations

import json
import logging
import time

from . import db
from .config import Config
from .extract.run import run_extraction
from .ingest.fetch import SOURCES, fetch_all
from .kg.build import build_kg
from .kg.query import run_queries
from .kg.reason import reason_kg
from .kg.validate import validate_kg
from .load import load_all
from .resolve.entities import resolve_all

log = logging.getLogger(__name__)


def _timed(name: str, fn, summary: dict, *args, **kw):
    t0 = time.time()
    log.info("=== %s ===", name)
    out = fn(*args, **kw)
    summary[name] = {"seconds": round(time.time() - t0, 1), "result": out}
    return out


def run_all(cfg: Config, quick: bool = False, skip_fetch: bool = False, skip_ml: bool = False,
            backend: str | None = None) -> dict:
    summary: dict = {}
    if not skip_fetch:
        _timed("fetch", fetch_all, summary, cfg, list(SOURCES))
    _timed("load", load_all, summary, cfg)
    _timed("resolve", resolve_all, summary, cfg)
    _timed("extract", run_extraction, summary, cfg, backend)
    _timed("build-kg", build_kg, summary, cfg)
    val = _timed("validate", validate_kg, summary, cfg)
    _timed("reason", reason_kg, summary, cfg)
    _timed("query", run_queries, summary, cfg)
    if not skip_ml:
        from .ml.run import run_ml  # optional dependency group
        _timed("ml", run_ml, summary, cfg, quick=quick)
    summary["shacl_violations"] = val["violations"]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def status(cfg: Config) -> dict:
    out: dict = {"db_exists": cfg.db_path.exists(), "kg_exists": cfg.kg_nquads.exists(),
                 "inferred_exists": cfg.kg_inferred_nquads.exists()}
    if cfg.db_path.exists():
        conn = db.connect(cfg.db_path)
        out["table_counts"] = db.table_counts(conn)
        conn.close()
    out["reports"] = sorted(p.name for p in cfg.reports_dir.glob("*")) if cfg.reports_dir.exists() else []
    return out
