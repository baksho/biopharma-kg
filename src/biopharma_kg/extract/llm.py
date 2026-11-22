"""LLM relation extractor (Anthropic API) with strict grounding and validation.

The model is asked for a JSON array of relations. Nothing it returns is trusted: every item is
validated and **dropped** unless
  1. the predicate is one of the allowed ones,
  2. the drug resolves to a known canonical drug (via the alias index),
  3. the ``evidence`` sentence occurs verbatim (case/whitespace-insensitive) in the abstract/title,
  4. the object string occurs in that evidence sentence.
Objects are then *linked* to known conditions / adverse events / targets when they match
exactly (after normalisation); unlinked objects are kept in the table (object_key NULL) for
provenance but are not turned into graph edges.

``confidence`` is the model's self-reported number. It is NOT calibrated; treat it as a coarse
ordering signal only.

Responses are cached on disk (``data/cache/llm``) keyed by model + prompt, so reruns are free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..resolve.names import normalize_drug, slug
from .base import PREDICATES, Dictionaries, Relation, exact_match, norm_ws

log = logging.getLogger(__name__)

EXTRACTOR_NAME = "llm-anthropic-v1"

SYSTEM_PROMPT = """You extract explicitly stated biomedical relations from one PubMed abstract.
Return ONLY a JSON array (no prose, no markdown fences). Each element must be an object:
  {"drug": <drug name as written in the text>,
   "predicate": one of "treats" | "adverse_effect" | "inhibits",
   "object": <condition / adverse event / molecular target, as written in the text>,
   "evidence": <ONE sentence copied VERBATIM from the abstract that states the relation>,
   "confidence": <number between 0 and 1>}
Definitions:
  treats         - the drug is used to treat, or is reported to be active/effective against, the condition.
  adverse_effect - the abstract reports the object as an adverse event / toxicity of the drug.
  inhibits       - the drug inhibits / targets the molecular target.
Rules:
  - Use only drugs from the provided "drug_candidates" list.
  - Include a relation only if the abstract states it as a finding or established fact.
  - Do NOT include negated, hypothetical, or merely background statements, or relations of other drugs.
  - If nothing qualifies, return [].
"""


def make_client() -> Any:
    try:
        import anthropic  # noqa: WPS433
    except ImportError as e:  # pragma: no cover - exercised only without the SDK
        raise RuntimeError("The llm backend needs the Anthropic SDK: pip install 'biopharma-kg[llm]'") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic()


def llm_available() -> tuple[bool, str]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY is not set"
    try:
        import anthropic  # noqa: F401, WPS433
    except ImportError:
        return False, "the 'anthropic' package is not installed (pip install 'biopharma-kg[llm]')"
    return True, ""


def _parse_json_array(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("model output is not a JSON array")
    return data


class LLMExtractor:
    def __init__(self, model: str, max_tokens: int, cache_dir: Path, client: Any | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client if client is not None else make_client()
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ model call
    def _call(self, pmid: str, user_payload: str) -> str:
        key = hashlib.sha1((self.model + "\n" + SYSTEM_PROMPT + "\n" + user_payload).encode()).hexdigest()
        cache = self.cache_dir / f"{key}.txt"
        if cache.exists():
            self.stats["cache_hits"] += 1
            return cache.read_text()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "text") == "text")
        cache.write_text(text)
        self.stats["api_calls"] += 1
        return text

    # ------------------------------------------------------------------ public
    def extract_article(self, pmid: str, title: str, abstract: str, d: Dictionaries) -> list[Relation]:
        full = f"{title}. {abstract}".strip()
        candidates = sorted({d.drug_labels.get(k, k) for k in d.drugs.find_keys(normalize_drug(full))})
        if not candidates:
            self.stats["skipped_no_drug_candidate"] += 1
            return []
        payload = json.dumps({"pmid": pmid, "title": title, "abstract": abstract, "drug_candidates": candidates}, ensure_ascii=False)
        try:
            raw = self._call(pmid, payload)
            items = _parse_json_array(raw)
        except (ValueError, json.JSONDecodeError) as e:
            self.stats["unparseable_responses"] += 1
            log.warning("PMID %s: unparseable LLM output (%s)", pmid, e)
            return []
        return self._validate(pmid, full, items, d)

    # ------------------------------------------------------------------ validation / grounding
    def _validate(self, pmid: str, full_text: str, items: list, d: Dictionaries) -> list[Relation]:
        out: list[Relation] = []
        haystack = norm_ws(full_text)
        for it in items:
            self.stats["items_returned"] += 1
            if not isinstance(it, dict):
                self.stats["dropped_malformed"] += 1
                continue
            pred = it.get("predicate")
            drug, obj, ev = it.get("drug"), it.get("object"), it.get("evidence")
            if pred not in PREDICATES or not all(isinstance(x, str) and x.strip() for x in (drug, obj, ev)):
                self.stats["dropped_malformed"] += 1
                continue
            keys = d.drugs.find_keys(normalize_drug(drug))
            if len(keys) != 1:
                self.stats["dropped_unknown_drug"] += 1
                continue
            if norm_ws(ev) not in haystack:
                self.stats["dropped_evidence_not_verbatim"] += 1
                continue
            if norm_ws(obj) not in norm_ws(ev):
                self.stats["dropped_object_not_in_evidence"] += 1
                continue
            kind = PREDICATES[pred]
            okey = self._link_object(kind, obj, d)
            try:
                conf = float(it["confidence"]) if it.get("confidence") is not None else None
                if conf is not None and not (0.0 <= conf <= 1.0):
                    conf = None
            except (TypeError, ValueError):
                conf = None
            out.append(Relation(pmid=pmid, drug_key=next(iter(keys)), predicate=pred, object_text=obj.strip(),
                                object_key=okey, object_kind=kind, evidence=ev.strip(), extractor=EXTRACTOR_NAME,
                                model=self.model, confidence=conf))
            self.stats["kept"] += 1
            self.stats["kept_linked" if okey else "kept_unlinked"] += 1
        return out

    @staticmethod
    def _link_object(kind: str, obj: str, d: Dictionaries) -> str | None:
        if kind == "condition":
            return exact_match(d.conditions, obj)
        if kind == "adverse_event":
            return exact_match(d.adverse_events, obj)
        if kind == "target":
            return d.target_symbols.get(obj.strip().upper())
        return None


__all__ = ["LLMExtractor", "llm_available", "EXTRACTOR_NAME", "SYSTEM_PROMPT", "slug"]
