"""Fetch orchestration: seed drugs -> per-drug raw JSON files under data/raw/<source>/.

Resumable: an existing raw file is skipped unless ``force=True``. Per-drug failures are logged
and collected; if *every* request for a source fails (typically: no network access) a
``FetchError`` with an actionable message is raised instead of silently producing an empty graph.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..resolve.names import drug_key, normalize_drug
from . import ctgov, openfda, pubmed
from .gdsc import MANUAL_DOWNLOAD_HELP, download_gdsc, find_gdsc_file, read_gdsc
from .http import FetchError, HttpClient

log = logging.getLogger(__name__)

SOURCES = ("gdsc", "ctgov", "openfda", "pubmed")


@dataclass
class Seed:
    name: str   # normalised name used to query the APIs
    key: str    # IRI-safe drug key
    origin: str  # 'seed_list' | 'gdsc'
    label: str = ""  # human-readable spelling as given in the source


def read_seed_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def build_seeds(cfg: Config, gdsc_file: Path | None) -> list[Seed]:
    """Seed list first (well-known drugs), then GDSC2 drug names (alphabetical), capped at max_drugs."""
    seeds: dict[str, Seed] = {}
    for name in read_seed_list(cfg.path(cfg.get("seed.drug_list", "configs/seed_drugs.txt"))):
        k = drug_key(name)
        if k and k not in seeds:
            seeds[k] = Seed(name=normalize_drug(name), key=k, origin="seed_list", label=name.strip())
    if gdsc_file is not None and cfg.get("seed.use_gdsc_drugs", True):
        df = read_gdsc(gdsc_file)
        for name in sorted(df["DRUG_NAME"].astype(str).unique(), key=str.lower):
            k = drug_key(name)
            if k and k not in seeds:
                seeds[k] = Seed(name=normalize_drug(name), key=k, origin="gdsc", label=str(name).strip())
    out = list(seeds.values())
    cap = cfg.get("seed.max_drugs")
    return out[: int(cap)] if cap else out


def _client(cfg: Config) -> HttpClient:
    ncbi_key = os.environ.get(cfg.get("pubmed.api_key_env", "NCBI_API_KEY"))
    email = os.environ.get(cfg.get("pubmed.email_env", "NCBI_EMAIL"))
    ua = cfg.get("http.user_agent", "biopharma-kg")
    if email:
        ua = f"{ua} contact:{email}"
    return HttpClient(
        cache_dir=cfg.cache_dir,
        timeout_s=cfg.get("http.timeout_s", 60),
        max_retries=cfg.get("http.max_retries", 5),
        backoff_s=cfg.get("http.backoff_s", 2.0),
        user_agent=ua,
        min_interval_s={
            "clinicaltrials.gov": 0.25,
            "api.fda.gov": 0.3,
            "eutils.ncbi.nlm.nih.gov": 0.11 if ncbi_key else 0.36,
        },
    )


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def fetch_gdsc(cfg: Config, client: HttpClient) -> Path:
    path = find_gdsc_file(cfg.get("gdsc.file"), cfg.raw_dir)
    if path is None:
        url = cfg.get("gdsc.url")
        if not url:
            raise FileNotFoundError(MANUAL_DOWNLOAD_HELP)
        log.info("Trying to download GDSC2 from %s", url)
        path = download_gdsc(client, url, cfg.raw_dir)
    log.info("Using GDSC2 file: %s", path)
    return path


def fetch_all(cfg: Config, sources: list[str], force: bool = False, limit_drugs: int | None = None) -> dict:
    cfg.ensure_dirs()
    client = _client(cfg)
    summary: dict = {"sources": {}, "errors": []}

    gdsc_file = None
    if "gdsc" in sources:
        try:
            gdsc_file = fetch_gdsc(cfg, client)
            summary["sources"]["gdsc"] = {"file": str(gdsc_file)}
        except (FetchError, FileNotFoundError) as e:
            if cfg.get("gdsc.required", True):
                raise FileNotFoundError(f"{e}\n(GDSC2 is required by default. To run without drug-sensitivity data, "
                                        "set `gdsc: {required: false}` in your config; the sensitivity queries and the "
                                        "drug_sensitivity ML task will then be empty.)") from e
            log.warning("GDSC2 unavailable, continuing WITHOUT sensitivity data: %s", e)
            summary["sources"]["gdsc"] = {"file": None, "warning": str(e)}
    else:
        gdsc_file = find_gdsc_file(cfg.get("gdsc.file"), cfg.raw_dir)

    seeds = build_seeds(cfg, gdsc_file)
    if limit_drugs:
        seeds = seeds[:limit_drugs]
    if not seeds:
        raise RuntimeError("No seed drugs: provide GDSC2 or a non-empty configs/seed_drugs.txt")
    _write(cfg.raw_dir / "seeds.json", {"seeds": [s.__dict__ for s in seeds]})
    log.info("%d seed drugs (%d from seed list, %d from GDSC2)", len(seeds),
             sum(s.origin == "seed_list" for s in seeds), sum(s.origin == "gdsc" for s in seeds))

    openfda_key = os.environ.get(cfg.get("openfda.api_key_env", "OPENFDA_API_KEY"))
    ncbi_key = os.environ.get(cfg.get("pubmed.api_key_env", "NCBI_API_KEY"))
    ncbi_email = os.environ.get(cfg.get("pubmed.email_env", "NCBI_EMAIL"))

    def run(source: str, subdir: str, fn) -> None:
        ok = fail = skipped = 0
        first_err: str | None = None
        for s in tqdm(seeds, desc=f"fetch {source}", unit="drug"):
            out = cfg.raw_dir / subdir / f"{s.key}.json"
            if out.exists() and not force:
                skipped += 1
                continue
            try:
                _write(out, {"query_drug": s.name, **fn(s)})
                ok += 1
            except (FetchError, ctgov.SchemaError) as e:
                fail += 1
                first_err = first_err or str(e)
                summary["errors"].append({"source": source, "drug": s.name, "error": str(e)[:300]})
                log.error("%s / %s failed: %s", source, s.name, e)
        summary["sources"][source] = {"fetched": ok, "failed": fail, "skipped_existing": skipped}
        if ok == 0 and skipped == 0 and fail > 0:
            raise FetchError(f"All {fail} {source} requests failed. First error: {first_err}")

    if "ctgov" in sources:
        run("ctgov", "ctgov", lambda s: {"studies": ctgov.fetch_studies_for_drug(
            client, cfg.get("ctgov.base_url"), s.name, cfg.get("ctgov.page_size", 100), cfg.get("ctgov.max_studies_per_drug", 100))})
    if "openfda" in sources:
        base = cfg.get("openfda.base_url")
        run("openfda_ndc", "openfda_ndc", lambda s: {"results": openfda.fetch_ndc(
            client, base, s.name, cfg.get("openfda.ndc_limit_per_drug", 50), openfda_key)})
        run("openfda_faers", "openfda_faers", lambda s: {"results": openfda.fetch_faers_counts(
            client, base, s.name, cfg.get("openfda.faers_top_reactions", 25), openfda_key)})
    if "pubmed" in sources:
        base = cfg.get("pubmed.base_url")

        def pm(s: Seed) -> dict:
            pmids = pubmed.fetch_pmids(client, base, s.name, cfg.get("pubmed.query_suffix", ""),
                                       cfg.get("pubmed.max_abstracts_per_drug", 15), ncbi_key, ncbi_email)
            xml = pubmed.fetch_articles_xml(client, base, pmids, ncbi_key, ncbi_email)
            return {"pmids": pmids, "articles": pubmed.parse_pubmed_xml(xml)}

        run("pubmed", "pubmed", pm)

    _write(cfg.raw_dir / "fetch_summary.json", summary)
    return summary
