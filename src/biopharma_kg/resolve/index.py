"""Longest-match phrase index over normalised text (used for drug / condition / gene mentions)."""
from __future__ import annotations

import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tokens that must never be dictionary terms on their own (too generic / placebo-like).
GENERIC_STOPLIST = {
    "placebo", "saline", "water", "vitamin", "glucose", "control", "standard", "care", "therapy", "drug",
    "chemotherapy", "treatment", "radiation", "surgery", "normal", "sugar", "oral", "dose", "none",
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class PhraseIndex:
    """Maps normalised phrases (1..max_len tokens) to keys; ``find`` returns longest, non-overlapping matches."""

    def __init__(self, max_len: int = 6, min_chars: int = 3):
        self.max_len = max_len
        self.min_chars = min_chars
        self._map: dict[tuple[str, ...], set[str]] = defaultdict(set)

    def add(self, phrase_norm: str, key: str) -> bool:
        toks = tuple(tokenize(phrase_norm))
        if not toks or len(toks) > self.max_len:
            return False
        if len(" ".join(toks)) < self.min_chars:
            return False
        if len(toks) == 1 and toks[0] in GENERIC_STOPLIST:
            return False
        self._map[toks].add(key)
        return True

    def __len__(self) -> int:
        return len(self._map)

    def find(self, text_norm: str) -> list[tuple[str, int, int]]:
        """Return ``[(key, start_token, end_token_exclusive)]``; ambiguous phrases (several keys) are skipped."""
        toks = tokenize(text_norm)
        out: list[tuple[str, int, int]] = []
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                keys = self._map.get(tuple(toks[i : i + n]))
                if keys:
                    hit = (keys, n)
                    break
            if hit:
                keys, n = hit
                if len(keys) == 1:
                    out.append((next(iter(keys)), i, i + n))
                i += n
            else:
                i += 1
        return out

    def find_keys(self, text_norm: str) -> set[str]:
        return {k for k, _, _ in self.find(text_norm)}
