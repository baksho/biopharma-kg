"""openFDA connector: NDC directory (products) and FAERS adverse-event *counts*.

NDC top-level fields used (from the openFDA NDC "searchable fields" page):
  product_ndc, brand_name, generic_name, dosage_form, route, marketing_category,
  pharm_class (list of strings), active_ingredients (list of objects), openfda (object)
The labeler is read defensively from ``labeler_name`` or ``openfda.manufacturer_name`` because the
public field list does not guarantee either; it is stored as NULL when neither is present.

FAERS: ``/drug/event.json`` with ``count=patient.reaction.reactionmeddrapt.exact`` returns
``results: [{"term": ..., "count": ...}]``. These are *raw report counts* for reports that mention
the drug, dominated by common reaction terms; they are NOT incidence rates and do not establish
causality. openFDA signals "no matches" with HTTP 404, which is treated as an empty result.
"""
from __future__ import annotations

from .http import HttpClient


def _auth(params: dict, api_key: str | None) -> dict:
    if api_key:
        params["api_key"] = api_key
    return params


def fetch_ndc(client: HttpClient, base_url: str, drug_name: str, limit: int, api_key: str | None = None) -> list[dict]:
    params = _auth({"search": f'generic_name:"{drug_name}"', "limit": limit}, api_key)
    data = client.get_json(f"{base_url}/drug/ndc.json", params, accept_404=True)
    if data is None:
        return []
    return data.get("results", []) if isinstance(data, dict) else []


def fetch_faers_counts(
    client: HttpClient, base_url: str, drug_name: str, top_n: int, api_key: str | None = None
) -> list[dict]:
    params = _auth(
        {
            "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": top_n,
        },
        api_key,
    )
    data = client.get_json(f"{base_url}/drug/event.json", params, accept_404=True)
    if data is None:
        return []
    return data.get("results", []) if isinstance(data, dict) else []


def parse_ndc(rec: dict) -> dict | None:
    ndc = rec.get("product_ndc")
    if not ndc:
        return None
    openfda = rec.get("openfda") or {}
    labeler = rec.get("labeler_name")
    if not labeler:
        mfr = openfda.get("manufacturer_name")
        labeler = mfr[0] if isinstance(mfr, list) and mfr else (mfr if isinstance(mfr, str) else None)
    route = rec.get("route")
    ingredients = []
    for ing in rec.get("active_ingredients") or []:
        if isinstance(ing, dict) and ing.get("name"):
            ingredients.append({"name": ing["name"], "strength": ing.get("strength")})
    classes = rec.get("pharm_class") or []
    return {
        "product_ndc": ndc,
        "brand_name": rec.get("brand_name"),
        "generic_name": rec.get("generic_name"),
        "dosage_form": rec.get("dosage_form"),
        "route": ",".join(route) if isinstance(route, list) else route,
        "marketing_category": rec.get("marketing_category"),
        "labeler": labeler,
        "active_ingredients": ingredients,
        "pharm_class": [c for c in classes if isinstance(c, str)],
    }


def parse_faers_counts(results: list[dict]) -> list[tuple[str, int]]:
    out = []
    for r in results:
        if isinstance(r, dict) and r.get("term") and isinstance(r.get("count"), int):
            out.append((str(r["term"]), int(r["count"])))
    return out
