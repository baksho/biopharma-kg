"""Extraction tests: rules baseline, and the LLM backend's grounding filters via a fake client (no network)."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from biopharma_kg import db
from biopharma_kg.extract import llm as llm_mod
from biopharma_kg.extract.base import build_dictionaries
from biopharma_kg.extract.llm import LLMExtractor, _parse_json_array
from biopharma_kg.extract.rules import extract_article
from biopharma_kg.extract.run import audit_sample, choose_backends, run_extraction
from biopharma_kg.ingest.fetch import SOURCES, fetch_all
from biopharma_kg.load import load_all
from biopharma_kg.resolve.entities import resolve_all

ABS_111 = ("Erlotinib was effective in the treatment of non-small cell lung cancer patients. "
           "Rash was a common adverse effect of erlotinib. Erlotinib inhibits EGFR signalling in vitro.")
TITLE_111 = "Erlotinib treatment in non-small cell lung cancer."


@pytest.fixture()
def resolved(cfg):
    fetch_all(cfg, list(SOURCES)); load_all(cfg); resolve_all(cfg)
    conn = db.connect(cfg.db_path)
    d = build_dictionaries(conn, 1)
    yield cfg, conn, d
    conn.close()


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        assert kw["system"] and kw["model"]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def _item(**kw):
    base = dict(drug="erlotinib", predicate="treats", object="non-small cell lung cancer",
                evidence="Erlotinib was effective in the treatment of non-small cell lung cancer patients.", confidence=0.9)
    base.update(kw)
    return base


def test_rules_extracts_and_skips_negation(resolved):
    _, conn, d = resolved
    rels = extract_article("111", f"{TITLE_111} {ABS_111}", d)
    got = {(r.predicate, r.object_kind) for r in rels}
    assert {("treats", "condition"), ("adverse_effect", "adverse_event"), ("inhibits", "target")} <= got
    assert all(r.drug_key == "erlotinib" and r.confidence is None for r in rels)
    neg = extract_article("222", "Gefitinib did not improve survival in breast cancer patients.", d)
    assert neg == []


def test_rules_run_is_idempotent(resolved):
    cfg, conn, _ = resolved
    a = run_extraction(cfg, "rules")
    n1 = conn.execute("SELECT COUNT(*) FROM extracted_relation").fetchone()[0]
    run_extraction(cfg, "rules")
    n2 = sqlite3.connect(cfg.db_path).execute("SELECT COUNT(*) FROM extracted_relation").fetchone()[0]
    assert n1 == n2 > 0 and a


def test_llm_grounding_filters(resolved, tmp_path):
    _, _, d = resolved
    items = [
        _item(),                                                                          # valid, linked
        _item(predicate="adverse_effect", object="Rash",
              evidence="Rash was a common adverse effect of erlotinib.", confidence=7),   # valid; confidence out of range -> None
        _item(evidence="Erlotinib cures everything."),                                    # evidence not verbatim -> drop
        _item(object="pancreatic cancer"),                                                # object not in evidence -> drop
        _item(drug="nonexistentinib"),                                                    # unknown drug -> drop
        _item(predicate="cures"),                                                         # bad predicate -> drop
        "garbage",                                                                        # malformed -> drop
        _item(predicate="inhibits", object="EGFR", evidence="Erlotinib inhibits EGFR signalling in vitro.",
              confidence="high"),                                                         # valid; non-numeric confidence -> None
    ]
    reply = "Here you go:\n```json\n" + json.dumps(items) + "\n```"
    ex = LLMExtractor("test-model", 512, tmp_path / "c", client=FakeClient(reply))
    rels = ex.extract_article("111", TITLE_111, ABS_111, d)
    assert [(r.predicate, r.object_text) for r in rels] == [
        ("treats", "non-small cell lung cancer"), ("adverse_effect", "Rash"), ("inhibits", "EGFR")]
    assert rels[0].object_key == "non_small_cell_lung_cancer"
    assert rels[1].confidence is None and rels[2].confidence is None
    assert rels[0].confidence == 0.9 and rels[0].model == "test-model"
    s = ex.stats
    assert s["items_returned"] == 8 and s["kept"] == 3
    assert s["dropped_evidence_not_verbatim"] == 1 and s["dropped_object_not_in_evidence"] == 1
    assert s["dropped_unknown_drug"] == 1 and s["dropped_malformed"] == 2


def test_llm_cache_and_unparseable_and_no_candidate(resolved, tmp_path):
    _, _, d = resolved
    client = FakeClient(json.dumps([_item()]))
    ex = LLMExtractor("m", 512, tmp_path / "c", client=client)
    ex.extract_article("111", TITLE_111, ABS_111, d)
    ex.extract_article("111", TITLE_111, ABS_111, d)
    assert client.calls == 1 and ex.stats["cache_hits"] == 1
    bad = LLMExtractor("m", 512, tmp_path / "c2", client=FakeClient("I cannot help with that"))
    assert bad.extract_article("111", TITLE_111, ABS_111, d) == [] and bad.stats["unparseable_responses"] == 1
    assert ex.extract_article("9", "Nothing relevant", "No drug is mentioned here.", d) == []
    assert ex.stats["skipped_no_drug_candidate"] == 1 and client.calls == 1  # no API call wasted


def test_parse_json_array_edge_cases():
    assert _parse_json_array("[]") == []
    assert _parse_json_array("```json\n[{\"a\":1}]\n```") == [{"a": 1}]
    with pytest.raises(ValueError):
        _parse_json_array("{\"a\": 1}")


def test_run_with_injected_llm_client_and_audit(resolved):
    cfg, conn, _ = resolved
    run_extraction(cfg, "rules")
    run_extraction(cfg, "llm", llm_client=FakeClient(json.dumps([_item()])))
    c = sqlite3.connect(cfg.db_path)
    by = dict(c.execute("SELECT extractor, COUNT(*) FROM extracted_relation GROUP BY 1").fetchall())
    assert set(by) == {llm_mod.EXTRACTOR_NAME, "rules-cooccurrence-v1"} and by[llm_mod.EXTRACTOR_NAME] >= 1
    a = audit_sample(cfg, n=5)
    import csv
    rows = list(csv.reader(open(a["sample_file"], encoding="utf-8")))
    assert rows[0][-1].startswith("correct") and len(rows) - 1 == a["sample_size"] == min(5, a["total_relations"])
    assert a["agreement"] and a["agreement"]["intersection"] >= 1   # both found (111, erlotinib, treats, NSCLC)
    assert 0 < a["agreement"]["jaccard"] <= 1


def test_backend_selection_is_loud_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backends, warnings = choose_backends("auto")
    assert backends == ["rules"] and warnings and "ANTHROPIC_API_KEY" in " ".join(warnings)
    with pytest.raises(RuntimeError):
        choose_backends("llm")
