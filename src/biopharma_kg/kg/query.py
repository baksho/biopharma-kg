"""Run the competency-question SPARQL files in queries/ and write CSVs + a summary.

A query file may start with header comments:
    # title: ...
    # question: ...
    # inference: none | required | compare
``compare`` runs the query on the asserted graph AND on asserted+inferred, so the summary shows
exactly what reasoning adds for that question.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from rdflib import Graph

from ..config import Config
from .build import load_union_graph

log = logging.getLogger(__name__)
PREFIXES = """
PREFIX bp: <https://biopharma-kg.example/ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX prov: <http://www.w3.org/ns/prov#>
"""


def parse_header(text: str) -> dict:
    meta = {"title": "", "question": "", "inference": "none"}
    for line in text.splitlines():
        m = re.match(r"#\s*(title|question|inference):\s*(.*)", line)
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return meta


def run_select(g: Graph, query_text: str) -> tuple[list[str], list[list[str]]]:
    res = g.query(PREFIXES + query_text)
    cols = [str(v) for v in res.vars]
    rows = [[("" if v is None else str(v)) for v in row] for row in res]
    return cols, rows


def run_queries(cfg: Config, only: list[str] | None = None) -> dict:
    if not cfg.kg_nquads.exists():
        raise FileNotFoundError(f"{cfg.kg_nquads} not found: run `biopharma-kg build-kg` first.")
    base = load_union_graph(cfg.kg_nquads)
    have_inferred = cfg.kg_inferred_nquads.exists() and cfg.kg_inferred_nquads.stat().st_size > 0
    full = load_union_graph(cfg.kg_nquads, cfg.kg_inferred_nquads) if have_inferred else None
    out_dir = cfg.reports_dir / "queries"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for qf in sorted(cfg.path("queries").glob("*.rq")):
        if only and qf.stem not in only:
            continue
        text = qf.read_text()
        meta = parse_header(text)
        entry = {"query": qf.stem, "title": meta["title"], "question": meta["question"], "inference": meta["inference"]}
        mode = meta["inference"]
        if mode == "required" and full is None:
            entry.update(status="skipped", note="needs inferred triples: run `biopharma-kg reason` first (or reasoning was skipped)")
            results.append(entry)
            continue
        target = full if (mode == "required") else base
        cols, rows = run_select(target, text)
        with open(out_dir / f"{qf.stem}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows[:5000])
        entry.update(status="ok", rows=len(rows), columns=cols)
        if mode == "compare":
            if full is None:
                entry["with_inference_rows"] = None
                entry["note"] = "run `reason` to see the with-inference result"
            else:
                _, rows_inf = run_select(full, text)
                entry["with_inference_rows"] = len(rows_inf)
                with open(out_dir / f"{qf.stem}__with_inference.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(cols)
                    w.writerows(rows_inf[:5000])
        results.append(entry)
    summary = {"queries": results}
    (cfg.reports_dir / "query_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
