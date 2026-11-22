"""Orchestrate link-prediction experiments: tasks x models x seeds -> reports/ml_results.{csv,md,json}."""
from __future__ import annotations

import json
import logging
from importlib import metadata

import numpy as np
import pandas as pd

from .. import db
from ..config import Config
from .dataset import TaskTooSmall, assert_no_leakage, build_task, load_edges, shortcut_report
from .evaluate import evaluate
from .models import popularity_baseline, train_pykeen, train_rgcn

log = logging.getLogger(__name__)

TASKS = {
    "drug_condition": dict(target="investigatedFor", temporal=True),
    "drug_sensitivity": dict(target="sensitiveIn", temporal=False),
    "drug_adverse_event": dict(target="hasReportedReaction", temporal=False),
}


def _versions() -> dict:
    out = {}
    for p in ("torch", "torch_geometric", "pykeen", "numpy", "pandas"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def run_ml(cfg: Config, quick: bool = False, tasks: list[str] | None = None, models: list[str] | None = None,
           seeds: list[int] | None = None) -> dict:
    ml = cfg.get("ml", {})
    task_names = tasks or ml.get("tasks", list(TASKS))
    model_names = models if models is not None else ml.get("models", ["TransE"])
    if not model_names:
        raise ValueError("ml.models must list at least one PyKEEN embedding model (e.g. TransE)")
    seeds = seeds or ml.get("seeds", [0])
    dim = int(ml.get("embedding_dim", 64))
    epochs = 5 if quick else int(ml.get("epochs", 100))
    gnn_epochs = 5 if quick else int(ml.get("gnn_epochs", 60))
    bs = int(ml.get("batch_size", 512))
    lr = float(ml.get("lr", 0.01))
    device = ml.get("device", "cpu")
    if quick:
        seeds = seeds[:1]

    conn = db.connect(cfg.db_path)
    edges = load_edges(conn, literature=True)
    conn.close()

    rows: list[dict] = []
    skipped: dict[str, str] = {}
    task_info: dict[str, dict] = {}
    for tname in task_names:
        spec = TASKS[tname]
        for seed in seeds:
            try:
                task = build_task(tname, spec["target"], edges, seed=seed, temporal=spec["temporal"],
                                  temporal_test_fraction=float(ml.get("temporal_test_fraction", 0.2)))
            except TaskTooSmall as e:
                skipped[tname] = str(e)
                log.warning("Skipping task %s: %s", tname, e)
                break
            assert_no_leakage(task)
            task_info.setdefault(tname, {**task.info, "shortcuts": shortcut_report(task)})
            runs = [popularity_baseline(task)]
            for m in model_names:
                runs.append(train_pykeen(task, m, dim, epochs, bs, seed, device, lr=lr))
            if ml.get("gnn", True):
                runs.append(train_rgcn(task, dim, gnn_epochs, seed, device=device))
            for tr in runs:
                met = evaluate(task, tr.score_fn, "test", device=device if tr.name != "baseline:popularity" else "cpu")
                rows.append({"task": tname, "model": tr.name, "seed": seed, "split": task.info["split"],
                             "train_seconds": round(tr.train_seconds, 2), "final_loss": tr.final_loss, **met})
                log.info("%s | %s | seed %d | MRR %.4f Hits@10 %.4f (n=%d)", tname, tr.name, seed, met["mrr"], met["hits@10"], met["n"])
            if spec["temporal"]:
                pass  # temporal split is deterministic; seeds only change model initialisation

    df = pd.DataFrame(rows)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"versions": _versions(), "quick_mode": quick, "skipped_tasks": skipped, "task_info": task_info,
                    "config": {"dim": dim, "epochs": epochs, "gnn_epochs": gnn_epochs, "batch_size": bs, "lr": lr, "seeds": seeds, "models": model_names}}
    if df.empty:
        result["note"] = "No task had enough edges. See skipped_tasks."
    else:
        df.to_csv(cfg.reports_dir / "ml_results.csv", index=False)
        agg = (df.groupby(["task", "model"])[["mrr", "hits@1", "hits@3", "hits@10", "mean_rank"]]
               .agg(["mean", "std"]).round(4))
        md = ["# Link-prediction results\n",
              f"Quick mode: {quick}. Seeds: {seeds}. Metrics are filtered, type-constrained, test split; mean (std) over seeds.\n",
              "| task | model | MRR | Hits@1 | Hits@3 | Hits@10 | n_test |", "|---|---|---|---|---|---|---|"]
        n_by = df.groupby(["task", "model"])["n"].first()
        for (t, m), r in agg.iterrows():
            f = lambda c: f"{r[(c, 'mean')]:.3f} ({0 if np.isnan(r[(c, 'std')]) else r[(c, 'std')]:.3f})"  # noqa: E731
            md.append(f"| {t} | {m} | {f('mrr')} | {f('hits@1')} | {f('hits@3')} | {f('hits@10')} | {n_by[(t, m)]} |")
        (cfg.reports_dir / "ml_results.md").write_text("\n".join(md) + "\n")
        result["summary_markdown"] = str(cfg.reports_dir / "ml_results.md")
    (cfg.reports_dir / "ml_results.json").write_text(json.dumps(result, indent=2, default=str))
    return result
