"""Load raw files (data/raw/**) into the SQLite staging/canonical tables.

Idempotent: all data tables are cleared first, then re-populated from the raw files, so
``load`` can be re-run safely after a re-fetch.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from importlib import resources
from pathlib import Path

import pandas as pd

from . import db
from .config import Config
from .ingest import ctgov, openfda
from .ingest.gdsc import find_gdsc_file, read_gdsc
from .resolve.names import contains_whole_phrase, drug_key, normalize_drug, org_key, slug

log = logging.getLogger(__name__)

DATA_TABLES = [
    "gdsc_response", "cell_line", "drug", "drug_alias", "target", "drug_target", "organization", "org_alias",
    "condition", "condition_cancer_type", "cancer_type", "adverse_event", "pharm_class", "trial",
    "trial_condition", "trial_drug", "trial_intervention", "trial_query", "ndc_product", "ndc_pharm_class",
    "faers_reaction", "pubmed_article", "extracted_relation",
]


def _read_json_dir(d: Path) -> list[tuple[str, dict]]:
    if not d.exists():
        return []
    return [(p.stem, json.loads(p.read_text())) for p in sorted(d.glob("*.json"))]


def load_cancer_types(conn: sqlite3.Connection) -> None:
    with resources.files("biopharma_kg.resources").joinpath("cancer_types.csv").open("r", encoding="utf-8") as fh:
        df = pd.read_csv(fh).fillna("")
    conn.executemany(
        "INSERT OR REPLACE INTO cancer_type(code,label,parent_code,kind) VALUES (?,?,?,?)",
        [(r.code, r.label, r.parent_code or None, r.kind) for r in df.itertuples()],
    )


def load_gdsc(conn: sqlite3.Connection, path: Path, sensitive_quantile: float) -> dict:
    df = read_gdsc(path)
    df["drug_key"] = df["DRUG_NAME"].map(drug_key)
    cosmic = df["COSMIC_ID"]
    df["cell_line_key"] = [
        f"cl{int(c)}" if pd.notna(c) else "cl_" + slug(n) for c, n in zip(cosmic, df["CELL_LINE_NAME"])
    ]
    # Per-drug *relative* sensitivity: lowest `q` fraction of LN_IC50 for drugs with >= 10 responses.
    grp = df.groupby("drug_key")["LN_IC50"]
    thresh = grp.transform(lambda s: s.quantile(sensitive_quantile))
    n = grp.transform("size")
    df["is_sensitive"] = ((df["LN_IC50"] <= thresh) & (n >= 10)).astype(int)
    out = pd.DataFrame(
        {
            "cosmic_id": df["COSMIC_ID"], "cell_line_key": df["cell_line_key"], "cell_line_name": df["CELL_LINE_NAME"],
            "gdsc_drug_id": df["DRUG_ID"], "drug_key": df["drug_key"], "drug_name": df["DRUG_NAME"],
            "putative_target": df["PUTATIVE_TARGET"], "pathway_name": df["PATHWAY_NAME"], "tcga_desc": df["TCGA_DESC"],
            "ln_ic50": df["LN_IC50"], "auc": df["AUC"], "z_score": df["Z_SCORE"], "is_sensitive": df["is_sensitive"],
        }
    )
    out.to_sql("gdsc_response", conn, if_exists="append", index=False)
    cl = (
        out.dropna(subset=["cell_line_key"])
        .groupby("cell_line_key")
        .agg(cosmic_id=("cosmic_id", "first"), name=("cell_line_name", "first"), tcga_desc=("tcga_desc", "first"))
        .reset_index()
    )
    cl["cancer_type_code"] = None
    cl.to_sql("cell_line", conn, if_exists="append", index=False)
    return {"rows": len(out), "drugs": out["drug_key"].nunique(), "cell_lines": len(cl),
            "sensitive_pairs": int(out["is_sensitive"].sum()), "file": str(path)}


def load_ctgov(conn: sqlite3.Connection, raw_dir: Path) -> dict:
    n_trials = 0
    for key, payload in _read_json_dir(raw_dir / "ctgov"):
        for study in payload.get("studies", []):
            t = ctgov.parse_study(study)
            conn.execute(
                "INSERT OR IGNORE INTO trial(nct_id,title,overall_status,start_date,phase,lead_sponsor,org_key) VALUES (?,?,?,?,?,?,?)",
                (t["nct_id"], t["title"], t["overall_status"], t["start_date"], t["phase"], t["lead_sponsor"],
                 org_key(t["lead_sponsor"]) or None),
            )
            conn.execute("INSERT OR IGNORE INTO trial_query(nct_id,drug_key) VALUES (?,?)", (t["nct_id"], key))
            for c in t["conditions"]:
                ck = slug(c)
                if not ck:
                    continue
                conn.execute("INSERT OR IGNORE INTO condition(condition_key,label) VALUES (?,?)", (ck, c.strip()))
                conn.execute("INSERT OR IGNORE INTO trial_condition(nct_id,condition_key) VALUES (?,?)", (t["nct_id"], ck))
            existing = {r[0] for r in conn.execute("SELECT name FROM trial_intervention WHERE nct_id=?", (t["nct_id"],))}
            for iv in t["interventions"]:
                if iv["name"] not in existing:
                    conn.execute("INSERT INTO trial_intervention(nct_id,name,type) VALUES (?,?,?)",
                                 (t["nct_id"], iv["name"], iv["type"]))
                    existing.add(iv["name"])
            n_trials += 1
    return {"studies_read": n_trials, "distinct_trials": conn.execute("SELECT COUNT(*) FROM trial").fetchone()[0]}


def load_ndc(conn: sqlite3.Connection, raw_dir: Path) -> dict:
    n = linked = 0
    for key, payload in _read_json_dir(raw_dir / "openfda_ndc"):
        q = normalize_drug(payload.get("query_drug", ""))
        for rec in payload.get("results", []):
            p = openfda.parse_ndc(rec)
            if p is None:
                continue
            names = [normalize_drug(p["generic_name"])] + [normalize_drug(i["name"]) for i in p["active_ingredients"]]
            dk = key if any(contains_whole_phrase(nm, q) for nm in names if nm) else None
            conn.execute(
                "INSERT OR IGNORE INTO ndc_product(product_ndc,brand_name,generic_name,dosage_form,route,marketing_category,labeler,drug_key) VALUES (?,?,?,?,?,?,?,?)",
                (p["product_ndc"], p["brand_name"], p["generic_name"], p["dosage_form"], p["route"],
                 p["marketing_category"], p["labeler"], dk),
            )
            for cls in p["pharm_class"]:
                ck = slug(cls)
                if ck:
                    conn.execute("INSERT OR IGNORE INTO pharm_class(class_key,label) VALUES (?,?)", (ck, cls))
                    conn.execute("INSERT OR IGNORE INTO ndc_pharm_class(product_ndc,class_key) VALUES (?,?)", (p["product_ndc"], ck))
            n += 1
            linked += dk is not None
    return {"products": n, "linked_to_seed_drug": linked}


def load_faers(conn: sqlite3.Connection, raw_dir: Path) -> dict:
    n = 0
    for key, payload in _read_json_dir(raw_dir / "openfda_faers"):
        for term, count in openfda.parse_faers_counts(payload.get("results", [])):
            ak = slug(term)
            if not ak:
                continue
            conn.execute("INSERT OR IGNORE INTO adverse_event(ae_key,term) VALUES (?,?)", (ak, term))
            conn.execute("INSERT OR REPLACE INTO faers_reaction(drug_key,ae_key,report_count) VALUES (?,?,?)", (key, ak, count))
            n += 1
    return {"drug_reaction_pairs": n}


def load_pubmed(conn: sqlite3.Connection, raw_dir: Path) -> dict:
    n = 0
    for key, payload in _read_json_dir(raw_dir / "pubmed"):
        for a in payload.get("articles", []):
            conn.execute(
                "INSERT OR IGNORE INTO pubmed_article(pmid,title,abstract,pub_year,source_drug_key) VALUES (?,?,?,?,?)",
                (a["pmid"], a.get("title"), a.get("abstract"), a.get("year"), key),
            )
            n += 1
    return {"articles": conn.execute("SELECT COUNT(*) FROM pubmed_article").fetchone()[0],
            "with_abstract": conn.execute("SELECT COUNT(*) FROM pubmed_article WHERE abstract <> ''").fetchone()[0]}


def load_all(cfg: Config) -> dict:
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    db.reset_tables(conn, DATA_TABLES)
    stats: dict = {}
    load_cancer_types(conn)
    gdsc_file = find_gdsc_file(cfg.get("gdsc.file"), cfg.raw_dir)
    if gdsc_file:
        stats["gdsc"] = load_gdsc(conn, gdsc_file, float(cfg.get("gdsc.sensitive_quantile", 0.25)))
    else:
        log.warning("No GDSC2 file found: the graph will have no drug-sensitivity edges.")
        stats["gdsc"] = {"rows": 0}
    stats["ctgov"] = load_ctgov(conn, cfg.raw_dir)
    stats["ndc"] = load_ndc(conn, cfg.raw_dir)
    stats["faers"] = load_faers(conn, cfg.raw_dir)
    stats["pubmed"] = load_pubmed(conn, cfg.raw_dir)
    conn.commit()
    db.set_meta(conn, "load_stats", json.dumps(stats))
    conn.close()
    return stats
