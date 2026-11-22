"""PubMed connector (NCBI E-utilities esearch + efetch).

Rate limit: 3 requests/s without an API key, 10/s with one (NCBI policy). The HTTP client is
configured with a matching minimum interval by ``ingest.fetch``.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .http import FetchError, HttpClient


def fetch_pmids(
    client: HttpClient,
    base_url: str,
    drug_name: str,
    suffix: str,
    retmax: int,
    api_key: str | None = None,
    email: str | None = None,
) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f'"{drug_name}"[Title/Abstract] {suffix}'.strip(),
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    data = client.get_json(f"{base_url}/esearch.fcgi", params)
    try:
        return [str(x) for x in data["esearchresult"]["idlist"]]
    except (KeyError, TypeError) as e:
        raise FetchError(f"Unexpected esearch response for {drug_name!r}: {str(data)[:200]!r}") from e


def fetch_articles_xml(
    client: HttpClient,
    base_url: str,
    pmids: list[str],
    api_key: str | None = None,
    email: str | None = None,
) -> str:
    if not pmids:
        return ""
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    _, text = client.get(f"{base_url}/efetch.fcgi", params)
    return text


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse efetch XML into ``[{pmid, title, abstract, year}]``; articles without abstract are kept
    with an empty abstract (they are skipped later by the extractor)."""
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise FetchError(f"Could not parse PubMed XML: {e}") from e
    out: list[dict] = []
    for art in root.iter("PubmedArticle"):
        pmid_el = art.find("./MedlineCitation/PMID")
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        title_el = art.find("./MedlineCitation/Article/ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        parts = []
        for ab in art.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            txt = "".join(ab.itertext()).strip()
            if txt:
                label = ab.get("Label")
                parts.append(f"{label}: {txt}" if label else txt)
        year = None
        for path in (
            "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
            "./MedlineCitation/Article/ArticleDate/Year",
            "./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate",
        ):
            el = art.find(path)
            if el is not None and el.text:
                m = re.search(r"(19|20)\d{2}", el.text)
                if m:
                    year = int(m.group(0))
                    break
        out.append({"pmid": pmid_el.text.strip(), "title": title, "abstract": " ".join(parts), "year": year})
    return out
