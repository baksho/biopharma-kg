"""Deterministic name normalisation for drugs and organisations.

These are *string* normalisers, not a terminology service. They are intentionally simple,
documented and unit-tested; their known limits (brand names, code names, combination
products) are listed in the README ("Entity resolution").
"""
from __future__ import annotations

import re
import unicodedata

# Salt / counter-ion / dosage-form words that do not change which drug it is.
_SALT_WORDS = {
    "hydrochloride", "hcl", "dihydrochloride", "sodium", "potassium", "calcium", "mesylate", "mesilate",
    "tosylate", "maleate", "citrate", "sulfate", "sulphate", "phosphate", "acetate", "besylate", "succinate",
    "tartrate", "fumarate", "bromide", "chloride", "hydrobromide", "malate", "lactate", "disodium",
    "tablet", "tablets", "capsule", "capsules", "injection", "oral", "solution", "infusion", "intravenous",
    "iv", "po", "film", "coated", "extended", "release", "er", "xr", "sr", "for", "and",
}
_DOSE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|iu|units?|mg/m2|mg/kg|%)\b", re.I)
_PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")

_ORG_SUFFIXES = {
    "inc", "incorporated", "ltd", "limited", "llc", "lp", "llp", "corp", "corporation", "co", "company",
    "gmbh", "ag", "sa", "nv", "bv", "plc", "kg", "kgaa", "spa", "srl", "ab", "as", "oy", "pty", "pte",
    "kk", "the", "de", "cv", "sas",
}


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_drug(name: str | None) -> str:
    """Lower-case, strip accents / parentheticals / doses / salt words, collapse punctuation.

    >>> normalize_drug("Erlotinib Hydrochloride 150 mg Tablets")
    'erlotinib'
    >>> normalize_drug("5-Fluorouracil")
    '5 fluorouracil'
    """
    if not name:
        return ""
    s = _ascii(name).lower()
    s = _PAREN_RE.sub(" ", s)
    s = _DOSE_RE.sub(" ", s)
    s = _NONALNUM_RE.sub(" ", s).strip()
    toks = [t for t in s.split() if t not in _SALT_WORDS]
    return " ".join(toks)


def drug_key(name: str | None) -> str:
    """URL/IRI-safe slug for a normalised drug name ('' if nothing is left)."""
    return normalize_drug(name).replace(" ", "_")


def normalize_org(name: str | None) -> str:
    """Normalise an organisation name by stripping legal-form suffixes and punctuation.

    >>> normalize_org("Pfizer Inc.")
    'pfizer'
    >>> normalize_org("F. Hoffmann-La Roche AG")
    'f hoffmann la roche'
    """
    if not name:
        return ""
    s = _ascii(name).lower().replace("&", " and ")
    s = _PAREN_RE.sub(" ", s)
    s = _NONALNUM_RE.sub(" ", s).strip()
    toks = [t for t in s.split() if t not in _ORG_SUFFIXES]
    return " ".join(toks)


def org_key(name: str | None) -> str:
    return normalize_org(name).replace(" ", "_")


def slug(text: str | None) -> str:
    """Generic IRI-safe slug (lower-case alnum + underscore)."""
    if not text:
        return ""
    return _NONALNUM_RE.sub("_", _ascii(text).lower()).strip("_")


def contains_whole_phrase(haystack_norm: str, needle_norm: str) -> bool:
    """True if ``needle_norm`` occurs in ``haystack_norm`` on token boundaries."""
    if not needle_norm or not haystack_norm:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(needle_norm) + r"(?![a-z0-9])", haystack_norm) is not None
