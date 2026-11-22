"""Offline, deterministic baseline extractor: sentence-level dictionary co-occurrence + trigger words.

This is a *baseline*, deliberately conservative:
  * a sentence is used only if it mentions a known drug AND a known object (condition / adverse
    event / target) AND a predicate trigger word;
  * sentences containing a negation cue are skipped entirely.
It has no confidence score (``confidence=None``). Its precision is NOT measured here; use
``biopharma-kg audit-extraction`` to draw a sample for manual review.
"""
from __future__ import annotations

import re

from ..resolve.index import tokenize
from ..resolve.names import normalize_drug
from .base import Dictionaries, Relation, split_sentences

EXTRACTOR_NAME = "rules-cooccurrence-v1"

_TRIGGERS = {
    "treats": re.compile(
        r"\b(treat(?:ed|ment|ments|ing|s)?|therap(?:y|ies|eutic)|efficacy|effective(?:ness)?|"
        r"respon(?:se|ses|ded|ders?)|survival|activity against|patients? with)\b", re.I),
    "adverse_effect": re.compile(
        r"\b(adverse|toxicit(?:y|ies)|side[- ]effects?|induced|caused|grade [2-5]|associated with)\b", re.I),
    "inhibits": re.compile(r"\b(inhibit(?:s|ed|or|ors|ion)?|block(?:s|ed|ade)?|antagonis[mt]s?|target(?:s|ed|ing)?)\b", re.I),
}
_NEGATION = re.compile(r"\b(no|not|without|failed|fail|lack(?:ed|s)?|absence|neither|nor|unlikely|did not|cannot)\b", re.I)


def extract_article(pmid: str, text: str, d: Dictionaries) -> list[Relation]:
    out: list[Relation] = []
    seen: set[tuple] = set()
    for sent in split_sentences(text):
        if _NEGATION.search(sent):
            continue
        drugs = d.drugs.find_keys(normalize_drug(sent))
        if not drugs:
            continue
        low = sent.lower()
        objects: dict[str, list[tuple[str, str]]] = {"treats": [], "adverse_effect": [], "inhibits": []}
        if _TRIGGERS["treats"].search(sent):
            toks = tokenize(low)
            for key, a, b in d.conditions.find(low):
                objects["treats"].append((key, " ".join(toks[a:b])))
        if _TRIGGERS["adverse_effect"].search(sent):
            toks = tokenize(low)
            for key, a, b in d.adverse_events.find(low):
                objects["adverse_effect"].append((key, " ".join(toks[a:b])))
        if _TRIGGERS["inhibits"].search(sent) and d.target_regex is not None:
            for m in d.target_regex.finditer(sent):
                sym = d.target_symbols.get(m.group(1).upper(), m.group(1))
                objects["inhibits"].append((sym, sym))
        for pred, objs in objects.items():
            for okey, otext in objs:
                for dk in drugs:
                    sig = (pmid, dk, pred, okey)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    kind = {"treats": "condition", "adverse_effect": "adverse_event", "inhibits": "target"}[pred]
                    out.append(Relation(pmid=pmid, drug_key=dk, predicate=pred, object_text=otext, object_key=okey,
                                        object_kind=kind, evidence=sent, extractor=EXTRACTOR_NAME))
    return out
