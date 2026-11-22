"""The `smoke` command itself, run against the local mock servers (so its own code path is exercised)."""
import json

import pytest

from biopharma_kg import smoke as smoke_mod
from biopharma_kg.config import Config


def _redirect(monkeypatch, tmp_path):
    orig = smoke_mod.smoke_config

    def patched(cfg):
        s = orig(cfg)
        s.raw["paths"] = {"data_dir": str(tmp_path / "smoke_data"), "reports_dir": str(tmp_path / "smoke_reports")}
        s.raw["ml"]["gnn"] = False
        return s
    monkeypatch.setattr(smoke_mod, "smoke_config", patched)


def test_smoke_all_sources_ok_and_pipeline_runs(cfg, tmp_path, monkeypatch):
    pytest.importorskip("torch"); pytest.importorskip("pykeen"); pytest.importorskip("torch_geometric")
    _redirect(monkeypatch, tmp_path)
    rep = smoke_mod.run_smoke(cfg)
    assert rep["all_sources_ok"], [c for c in rep["source_checks"] if not c["ok"]]
    assert isinstance(rep["pipeline"], dict) and "error" not in rep["pipeline"], rep["pipeline"]
    assert rep["pipeline"]["shacl_violations"] == 0
    assert (cfg.reports_dir / "smoke_report.json").exists()


def test_smoke_reports_failing_source_without_crashing(cfg, tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    cfg.raw["ctgov"]["base_url"] = cfg.raw["ctgov"]["base_url"].replace("/ctgov", "/badctgov")  # 200 but wrong shape
    rep = smoke_mod.run_smoke(cfg)
    bad = [c for c in rep["source_checks"] if not c["ok"]]
    assert len(bad) == 1 and "ClinicalTrials.gov" in bad[0]["check"] and "studies" in bad[0]["detail"].lower()
    assert rep["all_sources_ok"] is False and str(rep["pipeline"]).startswith("not run")


def test_smoke_flags_missing_gdsc_with_instructions(cfg, tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    cfg.raw["gdsc"].update(file=None, url=None)
    rep = smoke_mod.run_smoke(cfg)
    g = [c for c in rep["source_checks"] if c["check"].startswith("GDSC2")][0]
    assert not g["ok"] and "cancerrxgene" in g["detail"]
    assert str(rep["pipeline"]).startswith("not run")           # required by default
