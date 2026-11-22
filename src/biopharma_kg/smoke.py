"""`biopharma-kg smoke`: verify, against the LIVE services, every external-schema assumption the
connectors make, then run a small end-to-end pipeline in an isolated data directory.

This exists because the connectors were written from public documentation in an environment without
access to those services; this command is the first thing to run on a machine that has it.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path

from .config import Config
from .ingest import ctgov, openfda, pubmed
from .ingest.fetch import _client
from .ingest.gdsc import MANUAL_DOWNLOAD_HELP, find_gdsc_file, read_gdsc
from .ingest.http import FetchError

log = logging.getLogger(__name__)
PROBE_DRUG = "erlotinib"


def _check(name: str, fn) -> dict:
    t0 = time.time()
    try:
        detail = fn()
        return {"check": name, "ok": True, "detail": detail, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 - we want every failure reported, not raised
        return {"check": name, "ok": False, "detail": f"{type(e).__name__}: {e}"[:600], "seconds": round(time.time() - t0, 1)}


def check_sources(cfg: Config) -> list[dict]:
    client = _client(cfg)
    client.use_cache = False
    checks: list[dict] = []

    def ct():
        studies = ctgov.fetch_studies_for_drug(client, cfg.get("ctgov.base_url"), PROBE_DRUG, 5, 5)
        assert studies, "no studies returned for the probe drug"
        parsed = [ctgov.parse_study(s) for s in studies]
        present = {k: sum(1 for p in parsed if p[k]) for k in ("title", "start_date", "phase", "lead_sponsor", "conditions", "interventions")}
        assert present["lead_sponsor"] and present["conditions"] and present["interventions"], f"core fields empty: {present}"
        itypes = {i["type"] for p in parsed for i in p["interventions"]}
        return {"studies": len(parsed), "fields_present_counts": present, "intervention_types_seen": sorted(map(str, itypes))}

    def ndc():
        recs = openfda.fetch_ndc(client, cfg.get("openfda.base_url"), PROBE_DRUG, 5, os.environ.get("OPENFDA_API_KEY"))
        assert recs, "no NDC records returned"
        parsed = [openfda.parse_ndc(r) for r in recs]
        parsed = [p for p in parsed if p]
        assert parsed, "no record had product_ndc"
        return {"records": len(parsed), "labeler_present": sum(1 for p in parsed if p["labeler"]),
                "pharm_class_present": sum(1 for p in parsed if p["pharm_class"]),
                "active_ingredients_present": sum(1 for p in parsed if p["active_ingredients"]),
                "top_level_keys_seen": sorted(recs[0].keys())}

    def faers():
        res = openfda.fetch_faers_counts(client, cfg.get("openfda.base_url"), PROBE_DRUG, 5, os.environ.get("OPENFDA_API_KEY"))
        parsed = openfda.parse_faers_counts(res)
        assert parsed, "FAERS count query returned no (term, count) pairs"
        return {"top_terms": parsed[:3]}

    def pm():
        key, mail = os.environ.get("NCBI_API_KEY"), os.environ.get("NCBI_EMAIL")
        base = cfg.get("pubmed.base_url")
        ids = pubmed.fetch_pmids(client, base, PROBE_DRUG, cfg.get("pubmed.query_suffix", ""), 3, key, mail)
        assert ids, "esearch returned no PMIDs"
        arts = pubmed.parse_pubmed_xml(pubmed.fetch_articles_xml(client, base, ids, key, mail))
        assert arts, "efetch XML parsed to no articles"
        return {"pmids": ids, "with_abstract": sum(1 for a in arts if a["abstract"]), "with_year": sum(1 for a in arts if a["year"])}

    def gd():
        from .ingest.fetch import fetch_gdsc  # same path `fetch` uses: local file, else best-effort download
        p = fetch_gdsc(cfg, client)
        df = read_gdsc(p)
        return {"file": str(p), "rows": len(df), "drugs": int(df["DRUG_NAME"].nunique()), "cell_lines": int(df["CELL_LINE_NAME"].nunique()),
                "has_TCGA_DESC": bool(df["TCGA_DESC"].notna().any()), "has_PUTATIVE_TARGET": bool(df["PUTATIVE_TARGET"].notna().any())}

    for name, fn in (("ClinicalTrials.gov v2 (studies, parse)", ct), ("openFDA NDC (records, parse)", ndc),
                     ("openFDA FAERS counts", faers), ("PubMed E-utilities (esearch+efetch, parse)", pm),
                     ("GDSC2 file present & readable", gd)):
        checks.append(_check(name, fn))
    return checks


def smoke_config(cfg: Config) -> Config:
    raw = copy.deepcopy(cfg.raw)
    raw["paths"] = {"data_dir": "data/smoke", "reports_dir": "reports/smoke"}
    raw.setdefault("seed", {})["max_drugs"] = 8
    raw["ctgov"]["max_studies_per_drug"] = 30
    raw["ctgov"]["page_size"] = 30
    raw["openfda"]["ndc_limit_per_drug"] = 10
    raw["openfda"]["faers_top_reactions"] = 15
    raw["pubmed"]["max_abstracts_per_drug"] = 5
    raw["ml"]["models"] = [raw["ml"]["models"][0]]
    raw["ml"]["seeds"] = [0]
    return Config(raw=raw, root=cfg.root)


def run_smoke(cfg: Config) -> dict:
    from .pipeline import run_all

    checks = check_sources(cfg)
    report: dict = {"source_checks": checks, "all_sources_ok": all(c["ok"] for c in checks)}
    # Every source is required, GDSC2 included unless the config says `gdsc: {required: false}`.
    gdsc_required = bool(cfg.get("gdsc.required", True))
    required_ok = all(c["ok"] for c in checks if gdsc_required or not c["check"].startswith("GDSC2"))
    if not required_ok:
        report["pipeline"] = "not run: fix the failing source checks above first"
    else:
        scfg = smoke_config(cfg)
        # share the main config's GDSC file location
        if find_gdsc_file(cfg.get("gdsc.file"), cfg.raw_dir) is not None and not scfg.get("gdsc.file"):
            scfg.raw["gdsc"]["file"] = str(find_gdsc_file(cfg.get("gdsc.file"), cfg.raw_dir))
        try:
            report["pipeline"] = run_all(scfg, quick=True, skip_fetch=False)
        except Exception as e:  # noqa: BLE001
            report["pipeline"] = {"error": f"{type(e).__name__}: {e}"}
    out = cfg.reports_dir / "smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    report["report_file"] = str(out)
    return report
