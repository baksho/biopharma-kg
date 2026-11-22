"""CLI-level tests run as real subprocesses, the way a user would invoke them."""
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from conftest import FIX, REPO


def _override(tmp_path, mock_server):
    (tmp_path / "seeds.txt").write_text("erlotinib\ngefitinib\ncisplatin\n")
    p = tmp_path / "override.yaml"
    p.write_text(yaml.safe_dump({
        "paths": {"data_dir": str(tmp_path / "data"), "reports_dir": str(tmp_path / "reports")},
        "seed": {"drug_list": str(tmp_path / "seeds.txt"), "max_drugs": None, "use_gdsc_drugs": True},
        "gdsc": {"file": str(FIX / "gdsc_mini.csv")},
        "ctgov": {"base_url": mock_server + "/ctgov", "page_size": 3},
        "openfda": {"base_url": mock_server + "/fda"},
        "pubmed": {"base_url": mock_server + "/eutils"},
        "http": {"max_retries": 2, "backoff_s": 0.01},
        "extraction": {"backend": "rules", "min_condition_frequency": 1},
    }))
    return p


def run(args, cwd, env=None, check=True):
    e = {**os.environ, **(env or {})}
    r = subprocess.run([sys.executable, "-m", "biopharma_kg", *args], cwd=cwd, capture_output=True, text=True, env=e, timeout=600)
    if check:
        assert r.returncode == 0, f"{args}\nSTDOUT:{r.stdout[-1500:]}\nSTDERR:{r.stderr[-2500:]}"
    return r


def test_cli_run_all_offline_then_strict_validate_and_status(tmp_path, mock_server):
    ov = _override(tmp_path, mock_server)
    r = run(["--root", str(REPO), "--config", str(ov), "run-all", "--skip-ml"], cwd=tmp_path)
    summary = json.loads((tmp_path / "reports" / "run_summary.json").read_text())
    assert summary["shacl_violations"] == 0
    assert (tmp_path / "reports" / "queries" / "cq01_sponsors_of_screened_drugs.csv").exists()
    run(["--root", str(REPO), "--config", str(ov), "validate", "--strict"], cwd=tmp_path)
    st = json.loads(run(["--root", str(REPO), "--config", str(ov), "status"], cwd=tmp_path).stdout)
    assert st["db_exists"] and st["kg_exists"] and st["table_counts"]["trial"] == 9


def test_cli_missing_prerequisite_gives_clean_error(tmp_path, mock_server):
    ov = _override(tmp_path, mock_server)
    r = run(["--root", str(REPO), "--config", str(ov), "build-kg"], cwd=tmp_path, check=False)
    assert r.returncode == 1 and "ERROR" in r.stderr and "Traceback" not in r.stderr


def test_cli_ml_without_enough_data_reports_skip(tmp_path, mock_server):
    pytest.importorskip("torch"); pytest.importorskip("pykeen")
    ov = _override(tmp_path, mock_server)
    run(["--root", str(REPO), "--config", str(ov), "run-all", "--skip-ml"], cwd=tmp_path)
    r = run(["--root", str(REPO), "--config", str(ov), "ml", "--quick", "--models", "TransE"], cwd=tmp_path)
    assert "skipped_tasks" in r.stdout and "need" in r.stdout


def test_rebase_produces_a_working_graph(tmp_path, mock_server):
    clone = tmp_path / "clone"
    shutil.copytree(REPO, clone, ignore=shutil.ignore_patterns("data", "reports", "__pycache__", ".git", "*.egg-info", ".pytest_cache"))
    new = "https://example.org/mykg/"
    run(["--root", str(clone), "rebase", "--iri", new], cwd=clone)
    assert "biopharma-kg.example" not in "".join(p.read_text() for p in (clone / "mappings").glob("*.ttl"))
    ov = _override(tmp_path, mock_server)
    env = {"PYTHONPATH": str(clone / "src")}
    chk = subprocess.run([sys.executable, "-c", "import biopharma_kg; print(biopharma_kg.__file__)"], env={**os.environ, **env},
                         capture_output=True, text=True, cwd=tmp_path)
    assert str(clone) in chk.stdout, f"test would import the original package, not the rebased copy: {chk.stdout}"
    run(["--root", str(clone), "--config", str(ov), "run-all", "--skip-ml"], cwd=tmp_path, env=env)
    nq = (tmp_path / "data" / "processed" / "kg.nq").read_text()
    assert new in nq and "biopharma-kg.example" not in nq
    assert json.loads((tmp_path / "reports" / "shacl_summary.json").read_text())["violations"] == 0
