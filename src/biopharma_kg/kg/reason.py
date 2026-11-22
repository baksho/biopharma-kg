"""OWL 2 RL materialisation with owlrl, restricted to what is useful and tractable.

* Input: the union of all named graphs *minus triples whose object is a literal* (RL rules about
  literals are irrelevant here and would dominate the cost).
* Output: only the inferred triples that mention the bp: vocabulary (property or class), written to
  ``data/processed/kg_inferred.nq`` in the graph <.../graph/inferred>. Axiomatic noise
  (owl:Thing typing, reflexive owl:sameAs, ...) is dropped.
* If the graph exceeds ``kg.reasoning_max_triples`` the step is skipped with a clear warning,
  because owlrl is a pure-Python reasoner: run it on a smaller seed set, or load kg.nq into a
  triple store with a native RL reasoner.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter

import owlrl
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

from .. import NS
from ..config import Config
from .build import GRAPH_BASE, load_union_graph

log = logging.getLogger(__name__)
BP = Namespace(NS)


def reason_kg(cfg: Config) -> dict:
    if not cfg.kg_nquads.exists():
        raise FileNotFoundError(f"{cfg.kg_nquads} not found: run `biopharma-kg build-kg` first.")
    union = load_union_graph(cfg.kg_nquads)
    base = Graph()
    for s, p, o in union:
        if not isinstance(o, Literal):
            base.add((s, p, o))
    limit = int(cfg.get("kg.reasoning_max_triples", 250000))
    stats: dict = {"input_triples_total": len(union), "input_triples_for_reasoning": len(base), "limit": limit}
    if len(base) > limit:
        msg = (f"Skipped OWL-RL reasoning: {len(base)} resource triples exceed kg.reasoning_max_triples={limit}. "
               "Raise the limit (needs lots of RAM/time) or reduce the seed set.")
        log.warning(msg)
        stats.update(skipped=True, reason=msg)
        _write(cfg, stats)
        if cfg.kg_inferred_nquads.exists():
            cfg.kg_inferred_nquads.unlink()
        return stats

    closure = Graph()
    for t in base:
        closure.add(t)
    t0 = time.time()
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics, rdfs_closure=False, axiomatic_triples=False,
                           datatype_axioms=False).expand(closure)
    stats["reasoning_seconds"] = round(time.time() - t0, 2)

    ds = Dataset()
    g = ds.graph(URIRef(GRAPH_BASE + "inferred"))
    by_pred: Counter = Counter()
    for s, p, o in closure:
        if (s, p, o) in base:
            continue
        if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
            continue
        is_bp_pred = str(p).startswith(NS)
        is_bp_type = p == RDF.type and str(o).startswith(NS)
        if not (is_bp_pred or is_bp_type):
            continue
        if p == RDF.type and str(s).startswith(NS):
            continue  # typing of ontology terms themselves
        g.add((s, p, o))
        by_pred[str(p).replace(NS, "bp:") if is_bp_pred else "rdf:type " + str(o).replace(NS, "bp:")] += 1
    cfg.kg_inferred_nquads.parent.mkdir(parents=True, exist_ok=True)
    cfg.kg_inferred_nquads.write_text(ds.serialize(format="nquads"))
    stats.update(skipped=False, inferred_triples_kept=len(g), inferred_by_predicate=dict(by_pred.most_common()))
    _write(cfg, stats)
    return stats


def _write(cfg: Config, stats: dict) -> None:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "reasoning_report.json").write_text(json.dumps(stats, indent=2))
