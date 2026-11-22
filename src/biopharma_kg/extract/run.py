"""Run relation extraction over loaded PubMed abstracts and store results with provenance."""
from __future__ import annotations

import csv
import json
import logging
import random
import sqlite3
from collections import Counter

from tqdm import tqdm

from .. import db
from ..config import Config
from . import rules
from .base import Relation, build_dictionaries
from .llm import EXTRACTOR_NAME as LLM_NAME
from .llm import LLMExtractor, llm_available

log = logging.getLogger(__name__)


def choose_backends(requested: str) -> tuple[list[str], list[str]]:
    """Return (backends_to_run, warnings)."""
    ok, why = llm_available()
    warns: list[str] = []
    if requested == "auto":
        if ok:
            return ["llm"], warns
        warns.append(f"LLM backend NOT used ({why}); falling back to the offline rules baseline.")
        return ["rules"], warns
    if requested in ("llm", "both") and not ok:
        raise RuntimeError(f"extraction.backend={requested} but {why}. Use --backend rules for the offline baseline.")
    return (["rules", "llm"] if requested == "both" else [requested]), warns


def _insert(conn: sqlite3.Connection, rels: list[Relation]) -> int:
    n = 0
    for r in rels:
        cur = conn.execute(
            "INSERT OR IGNORE INTO extracted_relation(pmid,drug_key,predicate,object_text,object_key,object_kind,confidence,evidence,extractor,model) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (r.pmid, r.drug_key, r.predicate, r.object_text, r.object_key, r.object_kind, r.confidence, r.evidence, r.extractor, r.model),
        )
        n += cur.rowcount
    return n


def run_extraction(cfg: Config, backend: str | None = None, llm_client=None, max_articles: int | None = None) -> dict:
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    requested = backend or cfg.get("extraction.backend", "auto")
    if llm_client is not None and requested in ("auto", "llm", "both"):
        backends = ["rules", "llm"] if requested == "both" else ["llm"]
        warns: list[str] = []
    else:
        backends, warns = choose_backends(requested)
    for w in warns:
        log.warning(w)

    d = build_dictionaries(conn, int(cfg.get("extraction.min_condition_frequency", 2)))
    articles = conn.execute("SELECT pmid, title, abstract FROM pubmed_article WHERE abstract IS NOT NULL AND abstract <> '' ORDER BY pmid").fetchall()
    if max_articles:
        articles = articles[:max_articles]
    conn.execute("DELETE FROM extracted_relation WHERE extractor IN (%s)" % ",".join("?" * len(backends)),
                 [rules.EXTRACTOR_NAME if b == "rules" else LLM_NAME for b in backends])
    report: dict = {"articles_with_abstract": len(articles), "backends": backends, "warnings": warns, "inserted": {}}

    if "rules" in backends:
        n = 0
        for a in tqdm(articles, desc="extract (rules)", unit="abs"):
            n += _insert(conn, rules.extract_article(a["pmid"], f"{a['title']}. {a['abstract']}", d))
        report["inserted"][rules.EXTRACTOR_NAME] = n
    if "llm" in backends:
        ex = LLMExtractor(cfg.get("extraction.model", "claude-sonnet-5-5"), int(cfg.get("extraction.max_tokens", 1500)),
                          cfg.cache_dir, client=llm_client)
        n = 0
        for a in tqdm(articles, desc=f"extract (LLM {ex.model})", unit="abs"):
            n += _insert(conn, ex.extract_article(a["pmid"], a["title"] or "", a["abstract"], d))
        report["inserted"][LLM_NAME] = n
        report["llm_stats"] = dict(ex.stats)
        report["llm_model"] = ex.model
    conn.commit()

    by = conn.execute("SELECT extractor, predicate, COUNT(*) n, SUM(object_key IS NOT NULL) linked FROM extracted_relation GROUP BY 1,2").fetchall()
    report["table_counts"] = [dict(r) for r in by]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "extraction_report.json").write_text(json.dumps(report, indent=2))
    conn.close()
    return report


def audit_sample(cfg: Config, n: int = 50, seed: int = 0) -> dict:
    """Write a random sample of extracted relations to CSV for *manual* precision review, and, if
    several extractors ran, their agreement on (pmid, drug, predicate, object_key)."""
    conn = db.connect(cfg.db_path)
    rows = conn.execute("SELECT * FROM extracted_relation ORDER BY rel_id").fetchall()
    rnd = random.Random(seed)
    sample = rnd.sample(rows, min(n, len(rows)))
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.reports_dir / "extraction_audit_sample.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_id", "pmid", "extractor", "drug_key", "predicate", "object_text", "object_key", "confidence", "evidence", "correct (y/n, fill in)"])
        for r in sample:
            w.writerow([r["rel_id"], r["pmid"], r["extractor"], r["drug_key"], r["predicate"], r["object_text"],
                        r["object_key"], r["confidence"], r["evidence"], ""])
    sets: dict[str, set] = {}
    for r in rows:
        if r["object_key"]:
            sets.setdefault(r["extractor"], set()).add((r["pmid"], r["drug_key"], r["predicate"], r["object_key"]))
    agreement = None
    if len(sets) >= 2:
        names = sorted(sets)
        a, b = sets[names[0]], sets[names[1]]
        agreement = {"extractors": names, "intersection": len(a & b), "union": len(a | b),
                     "jaccard": round(len(a & b) / len(a | b), 4) if (a | b) else None}
    conn.close()
    return {"sample_file": str(path), "sample_size": len(sample), "total_relations": len(rows),
            "relations_per_extractor": dict(Counter(r["extractor"] for r in rows)), "agreement": agreement}
