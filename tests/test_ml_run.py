"""run_ml end to end on a synthetic database (mechanics only; the numbers are not results)."""
import pytest
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pykeen")

import json

import pandas as pd

from biopharma_kg.ml.run import run_ml
from synth_db import make_synthetic_db


def test_run_ml_quick_all_tasks(cfg):
    make_synthetic_db(cfg.db_path)
    cfg.raw["ml"].update(models=["TransE", "ComplEx"], gnn=True, seeds=[0], embedding_dim=16, batch_size=128)
    res = run_ml(cfg, quick=True)
    assert res["skipped_tasks"] == {}, res["skipped_tasks"]
    df = pd.read_csv(cfg.reports_dir / "ml_results.csv")
    assert set(df["task"]) == {"drug_condition", "drug_sensitivity", "drug_adverse_event"}
    assert set(df["model"]) == {"baseline:popularity", "pykeen:TransE", "pykeen:ComplEx", "pyg:RGCN-DistMult"}
    assert df["mrr"].between(0, 1).all() and (df["n"] > 0).all()
    md = (cfg.reports_dir / "ml_results.md").read_text()
    assert "| drug_condition | pykeen:TransE |" in md
    info = json.loads((cfg.reports_dir / "ml_results.json").read_text())
    assert info["task_info"]["drug_condition"]["split"] == "temporal"
    assert info["config"]["lr"] == 0.01 and info["quick_mode"] is True


def test_run_ml_skips_tiny_graph_with_reason(cfg):
    make_synthetic_db(cfg.db_path, n_drugs=4, n_cond=8, n_cl=8, n_ae=8, trials_per_drug=2)
    res = run_ml(cfg, quick=True, models=["TransE"])
    assert set(res["skipped_tasks"]) == {"drug_condition", "drug_sensitivity", "drug_adverse_event"}
    assert all("need" in v for v in res["skipped_tasks"].values())
