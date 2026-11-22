"""Materialise the RDF knowledge graph: SQLite --(R2RML via morph-kgc)--> N-Quads, plus the ontology.

Named graphs (one per source) keep provenance queryable:
  reference, resolution, ctgov, gdsc, openfda-ndc, openfda-faers, pubmed, extraction, ontology
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from rdflib import Dataset, Graph, URIRef

from ..config import Config

log = logging.getLogger(__name__)

GRAPH_BASE = "https://biopharma-kg.example/graph/"
CORE_MAPPINGS = [
    "mappings/reference.r2rml.ttl",
    "mappings/drugs.r2rml.ttl",
    "mappings/ctgov.r2rml.ttl",
    "mappings/openfda.r2rml.ttl",
    "mappings/gdsc.r2rml.ttl",
    "mappings/pubmed_extraction.r2rml.ttl",
]
RESPONSES_MAPPING = "mappings/gdsc_responses.r2rml.ttl"


def mapping_files(cfg: Config) -> list[Path]:
    files = list(CORE_MAPPINGS)
    if cfg.get("gdsc.materialize_all_responses", False):
        files.append(RESPONSES_MAPPING)
    paths = [cfg.path(f) for f in files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Mapping file(s) not found: {missing}. Run from the repository root.")
    return paths


def write_morph_config(cfg: Config, out_nq: Path) -> Path:
    ini = cfg.data_dir / "processed" / "morph-kgc.ini"
    ini.parent.mkdir(parents=True, exist_ok=True)
    mappings = ",".join(str(p.resolve()) for p in mapping_files(cfg))
    ini.write_text(
        "[CONFIGURATION]\n"
        "output_format=N-QUADS\n"
        f"output_file={out_nq.resolve()}\n"
        "logging_level=WARNING\n"
        # morph-kgc 2.10 stringifies SQL NULL to "None" before its NA check (default na_values is ",nan"),
        # which otherwise emits "None"^^xsd:date literals. Consequence: a genuine value spelled exactly
        # "None" is also dropped. Verified against the e2e output (see tests/test_pipeline.py).
        "na_values=,nan,None\n"
        "[DataSource1]\n"
        f"mappings={mappings}\n"
        f"db_url=sqlite:///{cfg.db_path.resolve()}\n"
    )
    return ini


def build_kg(cfg: Config) -> dict:
    if not cfg.db_path.exists():
        raise FileNotFoundError(f"{cfg.db_path} not found: run `biopharma-kg load` and `resolve` first.")
    out = cfg.kg_nquads
    out.parent.mkdir(parents=True, exist_ok=True)
    data_nq = out.with_name("kg_data.nq")
    if data_nq.exists():
        data_nq.unlink()
    ini = write_morph_config(cfg, data_nq)
    # Subprocess: morph-kgc's file output path is its CLI; it also keeps its rdflib pin isolated.
    proc = subprocess.run([sys.executable, "-m", "morph_kgc", str(ini)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"morph-kgc failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    if not data_nq.exists():
        # morph-kgc writes no file when every mapping yields zero triples
        data_nq.write_text("")

    onto = Graph()
    onto.parse(str(cfg.path("ontology/biopharma.ttl")), format="turtle")
    ds = Dataset()
    g_onto = ds.graph(URIRef(GRAPH_BASE + "ontology"))
    for t in onto:
        g_onto.add(t)
    with open(out, "w", encoding="utf-8") as fh:
        with open(data_nq, encoding="utf-8") as src:
            for line in src:
                fh.write(line)
        fh.write(ds.serialize(format="nquads"))
    data_nq.unlink()

    stats = quad_stats(out)
    return stats


def quad_stats(nq: Path) -> dict:
    """Cheap per-graph quad counts without loading the file into rdflib."""
    per_graph: dict[str, int] = {}
    total = 0
    with open(nq, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            total += 1
            # N-Quads: the graph IRI is the last <...> before the terminating ' .'
            end = line.rfind(">")
            start = line.rfind("<", 0, end)
            g = line[start + 1 : end].replace(GRAPH_BASE, "")
            per_graph[g] = per_graph.get(g, 0) + 1
    return {"quads": total, "per_graph": dict(sorted(per_graph.items())), "file": str(nq)}


def load_union_graph(nq: Path, include_inferred: Path | None = None) -> Graph:
    """Load N-Quads into a single (union) rdflib Graph; optionally add inferred quads."""
    ds = Dataset()
    ds.parse(str(nq), format="nquads")
    if include_inferred and include_inferred.exists():
        ds.parse(str(include_inferred), format="nquads")
    g = Graph()
    for s, p, o, _ in ds.quads((None, None, None, None)):
        g.add((s, p, o))
    return g
