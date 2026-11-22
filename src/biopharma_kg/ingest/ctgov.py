"""ClinicalTrials.gov API v2 connector.

Field paths used (checked against public documentation / examples of API v2):
  protocolSection.identificationModule.nctId
  protocolSection.statusModule.overallStatus
  protocolSection.statusModule.startDateStruct.date
  protocolSection.designModule.phases                       (list)
  protocolSection.sponsorCollaboratorsModule.leadSponsor.name
  protocolSection.conditionsModule.conditions               (list)
  protocolSection.armsInterventionsModule.interventions[].name / .type
Paging is cursor based: the response carries ``nextPageToken`` while more pages exist.

Only ``nctId`` is mandatory; everything else is read defensively (a trial without a
sponsor is still a trial). A study without ``protocolSection`` raises ``SchemaError`` so a
changed API cannot silently produce an empty graph.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .http import FetchError, HttpClient

log = logging.getLogger(__name__)


class SchemaError(RuntimeError):
    """The response does not have the shape this connector was written for."""


def normalize_date(s: str | None) -> str | None:
    """'2019' -> '2019-01-01', '2019-03' -> '2019-03-01', '2019-03-15' unchanged, else None."""
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return s + "-01"
    if re.fullmatch(r"\d{4}", s):
        return s + "-01-01"
    return None


def fetch_studies_for_drug(
    client: HttpClient, base_url: str, drug_name: str, page_size: int = 100, max_studies: int = 100
) -> list[dict]:
    """Return raw study dicts whose *intervention* matches ``drug_name`` (API: query.intr)."""
    studies: list[dict] = []
    token: str | None = None
    while len(studies) < max_studies:
        params: dict[str, Any] = {
            "query.intr": drug_name,
            "pageSize": min(page_size, max_studies - len(studies)),
            "format": "json",
        }
        if token:
            params["pageToken"] = token
        data = client.get_json(f"{base_url}/studies", params)
        if not isinstance(data, dict) or "studies" not in data:
            raise SchemaError(
                f"ClinicalTrials.gov response for {drug_name!r} has no 'studies' key "
                f"(got {type(data).__name__}: {str(data)[:200]!r}). The v2 API may have changed."
            )
        studies.extend(data["studies"])
        token = data.get("nextPageToken")
        if not token:
            break
    return studies[:max_studies]


def parse_study(study: dict) -> dict:
    ps = study.get("protocolSection")
    if not isinstance(ps, dict):
        raise SchemaError("Study record without 'protocolSection'; the ClinicalTrials.gov v2 schema may have changed.")
    ident = ps.get("identificationModule") or {}
    nct = ident.get("nctId")
    if not nct:
        raise SchemaError("Study record without protocolSection.identificationModule.nctId")
    status = ps.get("statusModule") or {}
    design = ps.get("designModule") or {}
    sponsor = (ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}
    conditions = (ps.get("conditionsModule") or {}).get("conditions") or []
    interventions = (ps.get("armsInterventionsModule") or {}).get("interventions") or []
    return {
        "nct_id": nct,
        "title": ident.get("briefTitle") or ident.get("officialTitle"),
        "overall_status": status.get("overallStatus"),
        "start_date": normalize_date((status.get("startDateStruct") or {}).get("date")),
        "phase": ",".join(design.get("phases") or []) or None,
        "lead_sponsor": sponsor.get("name"),
        "conditions": [c for c in conditions if isinstance(c, str) and c.strip()],
        "interventions": [
            {"name": i.get("name"), "type": i.get("type")}
            for i in interventions
            if isinstance(i, dict) and i.get("name")
        ],
    }


__all__ = ["fetch_studies_for_drug", "parse_study", "normalize_date", "SchemaError", "FetchError"]
