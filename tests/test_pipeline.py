"""Offline end-to-end tests: fetch(mock) -> load -> resolve -> extract -> R2RML -> SHACL -> OWL-RL -> SPARQL.

SPARQL answers are compared with independent SQL/Python ground truth computed from the same database, so
a passing test means the mappings + queries agree with the relational layer, not merely that they ran.
"""
import json
import sqlite3
import warnings

import pytest
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

from biopharma_kg import db
from biopharma_kg.extract.run import run_extraction
from biopharma_kg.ingest.fetch import SOURCES, fetch_all
from biopharma_kg.kg.build import build_kg, load_union_graph
from biopharma_kg.kg.query import PREFIXES, run_queries, run_select
from biopharma_kg.kg.reason import reason_kg
from biopharma_kg.kg.validate import validate_kg
from biopharma_kg.load import load_all
from biopharma_kg.resolve.entities import resolve_all

BP = Namespace("https://biopharma-kg.example/ns#")


@pytest.fixture(scope="module")
def built(tmp_path_factory, mock_server):
    # module-scoped copy of the `cfg` fixture (function-scoped there) so the pipeline runs once
    from conftest import FIX, REPO
    from biopharma_kg.config import load_config
    tmp = tmp_path_factory.mktemp("e2e")
    c = load_config(root=REPO)
    r = c.raw
    r["paths"] = {"data_dir": str(tmp / "data"), "reports_dir": str(tmp / "reports")}
    (tmp / "seeds.txt").write_text("erlotinib\ngefitinib\ncisplatin\n")
    r["seed"] = {"drug_list": str(tmp / "seeds.txt"), "max_drugs": None, "use_gdsc_drugs": True}
    r["gdsc"]["file"] = str(FIX / "gdsc_mini.csv")
    r["ctgov"].update(base_url=mock_server + "/ctgov", page_size=3, max_studies_per_drug=50)
    r["openfda"].update(base_url=mock_server + "/fda")
    r["pubmed"].update(base_url=mock_server + "/eutils")
    r["http"].update(max_retries=2, backoff_s=0.01, timeout_s=5)
    r["extraction"].update(backend="rules", min_condition_frequency=1)
    fetch_all(c, list(SOURCES))
    load_all(c)
    resolve_all(c)
    run_extraction(c, "rules")
    build_kg(c)
    return c


def _sql(cfg, q, *a):
    conn = sqlite3.connect(cfg.db_path)
    try:
        return conn.execute(q, a).fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------------ relational layer
def test_counts_and_resolution(built):
    c = built
    assert _sql(c, "SELECT COUNT(*) FROM trial")[0][0] == 9
    assert _sql(c, "SELECT COUNT(*) FROM drug")[0][0] == 3
    # 'Erlotinib Hydrochloride' / 'Tarceva (erlotinib)' style variants must not create extra drugs
    assert {r[0] for r in _sql(c, "SELECT drug_key FROM drug")} == {"erlotinib", "gefitinib", "cisplatin"}
    assert _sql(c, "SELECT COUNT(*) FROM trial WHERE start_date IS NULL")[0][0] >= 1  # missing dates stay NULL


def test_org_merging(built):
    keys = {r[0] for r in _sql(built, "SELECT org_key FROM organization")}
    assert "genentech" in keys and "genentech_inc" not in keys
    assert len([k for k in keys if k.startswith("genentech")]) == 1


# ------------------------------------------------------------------ KG build
def test_kg_has_named_graphs_and_no_null_literals(built):
    g = load_union_graph(built.kg_nquads)
    text = built.kg_nquads.read_text()
    assert '"None"' not in text and '"nan"' not in text and '"NaN"' not in text
    from rdflib import Dataset
    ds = Dataset()
    ds.parse(str(built.kg_nquads), format="nquads")
    graphs = {str(c.identifier).rsplit("/", 1)[-1] for c in ds.graphs() if len(c)}
    assert {"ctgov", "gdsc", "openfda-ndc", "openfda-faers", "pubmed", "extraction", "ontology", "reference"} <= graphs
    # every typed date / decimal literal is well-formed (rdflib returns a python value only when it parses)
    for lit in g.objects():
        if isinstance(lit, Literal) and lit.datatype in (XSD.date, XSD.decimal, XSD.integer):
            assert lit.value is not None, f"ill-typed literal {lit!r}"


def test_kg_matches_sql_counts(built):
    g = load_union_graph(built.kg_nquads)
    n_trials = len(set(g.subjects(RDF.type, BP.ClinicalTrial)))
    assert n_trials == _sql(built, "SELECT COUNT(*) FROM trial")[0][0]
    n_drugs = len(set(g.subjects(RDF.type, BP.Drug)))
    assert n_drugs == _sql(built, "SELECT COUNT(*) FROM drug")[0][0]
    n_assert = len(set(g.subjects(RDF.type, BP.ExtractedAssertion)))
    assert n_assert == _sql(built, "SELECT COUNT(*) FROM extracted_relation")[0][0] > 0


def test_ml_edges_equal_kg_triples(built):
    """The graph-ML edge sets and the RDF graph must come from the same data (both read the same SQL views)."""
    from urllib.parse import unquote
    from biopharma_kg.ml.dataset import LITERATURE, RELATIONS, load_edges
    conn = sqlite3.connect(built.db_path)
    edges = load_edges(conn, literature=True)
    conn.close()
    g = load_union_graph(built.kg_nquads)  # base graph only: no inferred triples

    def local(x):
        return unquote(str(x).rstrip("/").rsplit("/", 1)[-1])

    checked = 0
    for rel in list(RELATIONS) + list(LITERATURE):
        kg = {(local(s), local(o)) for s, o in g.subject_objects(BP[rel]) if isinstance(o, URIRef)}
        ml = {(h.split(":", 1)[1], t.split(":", 1)[1]) for h, t in zip(edges[rel]["head"], edges[rel]["tail"])} if rel in edges else set()
        assert kg == ml, f"{rel}: only in KG {sorted(kg - ml)[:5]} | only in ML {sorted(ml - kg)[:5]}"
        checked += bool(kg)
    assert checked >= 6  # the fixture populates most relations; guards against an all-empty vacuous pass


# ------------------------------------------------------------------ SHACL
def test_shacl_conforms(built):
    s = validate_kg(built)
    assert s["violations"] == 0, s["first_example_per_result_type"]


def test_shacl_catches_bad_data(built, tmp_path):
    """A validator that never fails is worthless: corrupt a copy of the KG and require a violation."""
    import shutil
    c = built
    backup = c.kg_nquads.read_text()
    try:
        bad = backup + (
            '<https://biopharma-kg.example/id/assertion/9999> <https://biopharma-kg.example/ns#assertionPredicate> '
            '"cures" <https://biopharma-kg.example/graph/extraction> .\n'
            '<https://biopharma-kg.example/id/assertion/9999> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> '
            '<https://biopharma-kg.example/ns#ExtractedAssertion> <https://biopharma-kg.example/graph/extraction> .\n')
        c.kg_nquads.write_text(bad)
        assert validate_kg(c)["violations"] >= 1
    finally:
        c.kg_nquads.write_text(backup)


# ------------------------------------------------------------------ reasoning
def test_reasoning_adds_only_bp_inferences_and_is_sound(built):
    rep = reason_kg(built)
    assert not rep["skipped"] and rep["inferred_triples_kept"] > 0
    base = load_union_graph(built.kg_nquads)
    full = load_union_graph(built.kg_nquads, built.kg_inferred_nquads)
    new = set(full) - set(base)
    assert new and all(str(p).startswith(str(BP)) for _, p, _ in new)
    # transitive closure: every direct broaderType parent's parent must now be a broader type too
    for t, _, p in list(base.triples((None, BP.broaderType, None))):
        for _, _, gp in base.triples((p, BP.broaderType, None)):
            assert (t, BP.broaderType, gp) in full
    # subPropertyOf: every investigatesDrug-based association is lifted
    assert any(full.triples((None, BP.associatedWithCondition, None)))


def test_reasoning_skips_above_limit(built):
    built.raw["kg"]["reasoning_max_triples"] = 10
    try:
        rep = reason_kg(built)
        assert rep["skipped"] is True
    finally:
        built.raw["kg"]["reasoning_max_triples"] = 250000
        reason_kg(built)


# ------------------------------------------------------------------ SPARQL vs SQL ground truth
def _gap_truth(cfg):
    """Python/SQL re-implementation of cq03: (drug, group) pairs trialled but with no sensitive cell line."""
    parent = dict(_sql(cfg, "SELECT code, parent_code FROM cancer_type"))
    root = next(c for c, p in parent.items() if p is None)

    def anc_or_self(code):
        out = []
        while code is not None:
            out.append(code)
            code = parent.get(code)
        return out

    def groups(code):
        return {a for a in anc_or_self(code) if parent.get(a) == root}

    trials = {}  # (drug, group) -> set(nct)
    for nct, drug, cond_code in _sql(cfg, """SELECT td.nct_id, td.drug_key, cc.code FROM trial_drug td
        JOIN trial_condition tc ON tc.nct_id = td.nct_id JOIN condition_cancer_type cc ON cc.condition_key = tc.condition_key
        WHERE td.drug_key IS NOT NULL"""):
        for gp in groups(cond_code):
            trials.setdefault((drug, gp), set()).add(nct)
    screened = set()
    for drug, code in _sql(cfg, """SELECT DISTINCT r.drug_key, cl.cancer_type_code FROM gdsc_response r
        JOIN cell_line cl ON cl.cell_line_key = r.cell_line_key WHERE r.is_sensitive = 1 AND cl.cancer_type_code IS NOT NULL"""):
        for gp in groups(code):
            screened.add((drug, gp))
    labels = dict(_sql(cfg, "SELECT code, label FROM cancer_type"))
    dl = dict(_sql(cfg, "SELECT drug_key, label FROM drug"))
    return {(dl[d], labels[gp]): len(n) for (d, gp), n in trials.items() if (d, gp) not in screened}


def _gap_sparql(cfg):
    g = load_union_graph(cfg.kg_nquads)
    text = (cfg.path("queries") / "cq03_screening_gaps.rq").read_text()
    cols, rows = run_select(g, text)
    return {(r[0], r[1]): int(r[2]) for r in rows}


def test_cq03_matches_sql_ground_truth_nonvacuously(built):
    assert _gap_sparql(built) == _gap_truth(built)
    # make the check non-vacuous: remove one drug's sensitivity, rebuild, and require gaps to appear
    conn = sqlite3.connect(built.db_path)
    conn.execute("UPDATE gdsc_response SET is_sensitive = 0 WHERE drug_key = 'cisplatin'")
    conn.commit(); conn.close()
    build_kg(built)
    truth = _gap_truth(built)
    assert any(d == "cisplatin" for d, _ in truth) or any(truth), "variant produced no gaps; test is vacuous"
    assert _gap_sparql(built) == truth


def test_all_queries_run_and_write_csv(built):
    build_kg(built)  # restore consistent KG (db was modified above; rows differ but must still run)
    reason_kg(built)
    res = run_queries(built)
    statuses = {q["query"]: q["status"] for q in res["queries"]}
    assert len(statuses) == 10 and set(statuses.values()) <= {"ok"}, statuses
    for q in res["queries"]:
        assert (built.reports_dir / "queries" / f"{q['query']}.csv").exists()
