"""SHACL validation of the materialised KG (pySHACL, no inference)."""
from __future__ import annotations

import json
from collections import Counter

import pyshacl
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

from ..config import Config
from .build import load_union_graph

SH = Namespace("http://www.w3.org/ns/shacl#")


def validate_kg(cfg: Config) -> dict:
    if not cfg.kg_nquads.exists():
        raise FileNotFoundError(f"{cfg.kg_nquads} not found: run `biopharma-kg build-kg` first.")
    data = load_union_graph(cfg.kg_nquads)
    shapes = Graph().parse(str(cfg.path("shapes/biopharma-shapes.ttl")), format="turtle")
    _conforms_any, report_graph, _text = pyshacl.validate(
        data, shacl_graph=shapes, inference="none", advanced=False, abort_on_first=False
    )
    sev = Counter()
    by_shape: Counter = Counter()
    examples: dict[str, str] = {}
    for res in report_graph.subjects(RDF.type, SH.ValidationResult):
        severity = str(report_graph.value(res, SH.resultSeverity)).rsplit("#", 1)[-1]
        sev[severity] += 1
        shape = str(report_graph.value(res, SH.sourceShape)).rsplit("#", 1)[-1]
        path = str(report_graph.value(res, SH.resultPath)).rsplit("#", 1)[-1]
        key = f"{severity}: {shape} / {path}"
        by_shape[key] += 1
        if key not in examples:
            msg = report_graph.value(res, SH.resultMessage)
            focus = report_graph.value(res, SH.focusNode)
            examples[key] = f"{focus} :: {msg}"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_graph.serialize(destination=str(cfg.reports_dir / "shacl_report.ttl"), format="turtle")
    summary = {
        "data_triples": len(data),
        "violations": sev.get("Violation", 0),
        "warnings": sev.get("Warning", 0),
        "conforms_ignoring_warnings": sev.get("Violation", 0) == 0,
        "results_by_shape_and_path": dict(by_shape.most_common()),
        "first_example_per_result_type": examples,
    }
    (cfg.reports_dir / "shacl_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
