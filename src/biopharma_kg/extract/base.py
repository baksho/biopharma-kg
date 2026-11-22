"""Shared types and dictionaries for relation extraction."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from ..resolve.index import PhraseIndex
from ..resolve.names import normalize_drug, slug

# predicate -> kind of object
PREDICATES = {
    "treats": "condition",
    "adverse_effect": "adverse_event",
    "inhibits": "target",
}

_GENERIC_CONDITIONS = {
    "cancer", "cancers", "tumor", "tumors", "tumour", "tumours", "neoplasm", "neoplasms", "carcinoma",
    "malignancy", "malignancies", "metastasis", "metastases", "solid tumor", "solid tumors", "advanced cancer",
    "healthy", "healthy volunteers", "pain", "infection",
}


@dataclass
class Relation:
    pmid: str
    drug_key: str
    predicate: str
    object_text: str
    object_key: str | None
    object_kind: str
    evidence: str
    extractor: str
    model: str | None = None
    confidence: float | None = None  # None for the rules backend: it has no calibrated confidence


@dataclass
class Dictionaries:
    """Dictionaries built from the already-loaded & resolved tables (no external vocabulary)."""

    drugs: PhraseIndex
    conditions: PhraseIndex
    adverse_events: PhraseIndex
    target_symbols: dict[str, str] = field(default_factory=dict)  # upper-case symbol -> symbol
    target_regex: re.Pattern | None = None
    drug_labels: dict[str, str] = field(default_factory=dict)


def build_dictionaries(conn: sqlite3.Connection, min_condition_frequency: int = 2) -> Dictionaries:
    drugs = PhraseIndex(max_len=6, min_chars=3)
    labels: dict[str, str] = {}
    for r in conn.execute("SELECT drug_key, alias_norm FROM drug_alias"):
        drugs.add(r["alias_norm"], r["drug_key"])
    for r in conn.execute("SELECT drug_key, label FROM drug"):
        labels[r["drug_key"]] = r["label"]
        drugs.add(normalize_drug(r["label"]), r["drug_key"])

    conditions = PhraseIndex(max_len=8, min_chars=4)
    for r in conn.execute(
        "SELECT c.condition_key, c.label, COUNT(*) AS n FROM condition c "
        "JOIN trial_condition tc ON tc.condition_key=c.condition_key GROUP BY c.condition_key"
    ):
        if r["n"] >= min_condition_frequency and r["label"].lower().strip() not in _GENERIC_CONDITIONS:
            conditions.add(r["label"].lower(), r["condition_key"])

    aes = PhraseIndex(max_len=6, min_chars=4)
    for r in conn.execute("SELECT ae_key, term FROM adverse_event"):
        aes.add(r["term"].lower(), r["ae_key"])

    syms = {r["symbol"].upper(): r["symbol"] for r in conn.execute("SELECT symbol FROM target") if len(r["symbol"]) >= 3}
    rx = None
    if syms:
        rx = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, syms.values()), key=len, reverse=True)) + r")(?![A-Za-z0-9])")
    return Dictionaries(drugs, conditions, aes, syms, rx, labels)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text) if s.strip()]


def exact_match(index: PhraseIndex, text: str) -> str | None:
    """Key of ``index`` if the *whole* ``text`` (all tokens) is one known phrase, else None."""
    from ..resolve.index import tokenize

    n = len(tokenize(text))
    hits = index.find(text.lower())
    if len(hits) == 1 and hits[0][2] - hits[0][1] == n:
        return hits[0][0]
    return None


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


__all__ = ["PREDICATES", "Relation", "Dictionaries", "build_dictionaries", "split_sentences", "norm_ws", "exact_match", "slug"]
