"""Entity resolution and canonicalisation ("master data" step).

What it does (all deterministic, all string-based; limits are documented in the README):

* **Drugs**: one canonical ``drug`` per normalised generic name. Aliases come from the seed list,
  GDSC drug names and openFDA NDC generic/brand names. Trial interventions are linked to drugs by
  whole-phrase longest match against that alias index (so "Erlotinib + Bevacizumab" links to both).
* **Organisations**: sponsor strings are merged on a normalised key (legal-form suffixes removed);
  the most frequent raw spelling becomes the label and every raw variant is kept as an alias.
* **Conditions -> cancer types**: curated keyword table (``resources/condition_keywords.csv``).
* **Cell lines -> cancer types**: GDSC's TCGA-style code, mapped onto the curated hierarchy.
* **Targets**: GDSC "putative target" strings split into tokens that look like target symbols.

It also writes ``reports/resolution_report.json`` with link rates and cross-source overlap, i.e.
the numbers needed to judge how well the sources actually connect.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from importlib import resources

from .. import db
from ..config import Config
from .index import PhraseIndex
from .names import drug_key, normalize_drug, org_key

log = logging.getLogger(__name__)

NON_DRUG_TYPES = {"PROCEDURE", "DEVICE", "BEHAVIORAL", "RADIATION", "DIETARY_SUPPLEMENT", "DIAGNOSTIC_TEST", "GENETIC"}
_TARGET_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-._]{1,19}$")
_GENERIC_CANCER = re.compile(r"cancer|carcinoma|neoplasm|tumou?r|malignan|metasta|oncolog|adenocarcinoma|blastoma")


def load_condition_rules() -> list[tuple[re.Pattern, str]]:
    with resources.files("biopharma_kg.resources").joinpath("condition_keywords.csv").open("r", encoding="utf-8") as fh:
        return [(re.compile(r["pattern"], re.I), r["code"]) for r in csv.DictReader(fh)]


def map_condition_to_cancer_types(label: str, rules: list[tuple[re.Pattern, str]]) -> list[str]:
    text = label.lower()
    codes: list[str] = []
    for pat, code in rules:
        if pat.search(text) and code not in codes:
            codes.append(code)
    if re.search(r"non.?small", text) and "SCLC" in codes:
        codes.remove("SCLC")  # 'non-small cell lung' contains 'small cell lung'
    if not codes and _GENERIC_CANCER.search(text):
        codes.append("CANCER")
    return codes


def normalize_tcga_code(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or s in {"NAN", "NONE", "NA", "N/A"}:
        return None
    return re.sub(r"[\s/]+", "_", s)


def parse_targets(raw: str | None) -> list[str]:
    if not raw or str(raw).lower() == "nan":
        return []
    out = []
    for tok in re.split(r"[,;/]", str(raw)):
        tok = tok.strip()
        if _TARGET_TOKEN.match(tok):
            out.append(tok)
    return out


def build_drug_alias_index(conn) -> PhraseIndex:
    idx = PhraseIndex(max_len=6, min_chars=3)
    for r in conn.execute("SELECT alias_norm, drug_key FROM drug_alias WHERE source IN ('seed','GDSC','NDC')"):
        idx.add(r["alias_norm"], r["drug_key"])
    return idx


def resolve_all(cfg: Config) -> dict:
    conn = db.connect(cfg.db_path)
    db.init_schema(conn)
    seeds_file = cfg.raw_dir / "seeds.json"
    if not seeds_file.exists():
        raise FileNotFoundError(f"{seeds_file} not found: run `biopharma-kg fetch` (and `load`) first.")
    seeds = json.loads(seeds_file.read_text())["seeds"]
    for t in ("drug", "drug_alias", "target", "drug_target", "organization", "org_alias", "condition_cancer_type", "trial_drug"):
        conn.execute(f"DELETE FROM {t}")
    conn.execute("UPDATE cell_line SET cancer_type_code = NULL")

    # ---- drugs + aliases ------------------------------------------------------------------
    for s in seeds:
        conn.execute("INSERT OR IGNORE INTO drug(drug_key,label) VALUES (?,?)", (s["key"], s.get("label") or s["name"]))
        conn.execute("INSERT OR IGNORE INTO drug_alias(drug_key,alias_norm,alias_raw,source) VALUES (?,?,?,?)",
                     (s["key"], s["name"], s.get("label"), "seed"))
    gdsc_drugs = conn.execute(
        "SELECT drug_key, MIN(drug_name) AS name, MIN(gdsc_drug_id) AS gid, "
        "MAX(pathway_name) AS pathway, COUNT(*) AS n FROM gdsc_response WHERE drug_key <> '' GROUP BY drug_key"
    ).fetchall()
    for r in gdsc_drugs:
        conn.execute("INSERT OR IGNORE INTO drug(drug_key,label) VALUES (?,?)", (r["drug_key"], r["name"]))
        pathway = r["pathway"] if r["pathway"] and str(r["pathway"]).lower() != "nan" else None
        conn.execute("UPDATE drug SET gdsc_drug_id=?, pathway_name=?, in_gdsc=1 WHERE drug_key=?",
                     (None if r["gid"] is None else int(r["gid"]), pathway, r["drug_key"]))
        conn.execute("INSERT OR IGNORE INTO drug_alias(drug_key,alias_norm,alias_raw,source) VALUES (?,?,?,?)",
                     (r["drug_key"], normalize_drug(r["name"]), r["name"], "GDSC"))
    for r in conn.execute("SELECT drug_key, generic_name, brand_name FROM ndc_product WHERE drug_key IS NOT NULL").fetchall():
        for raw in (r["generic_name"], r["brand_name"]):
            n = normalize_drug(raw)
            if n:
                conn.execute("INSERT OR IGNORE INTO drug_alias(drug_key,alias_norm,alias_raw,source) VALUES (?,?,?,?)",
                             (r["drug_key"], n, raw, "NDC"))

    # ---- targets ----------------------------------------------------------------------------
    tok_ok = tok_bad = 0
    for r in conn.execute("SELECT DISTINCT drug_key, putative_target FROM gdsc_response WHERE putative_target IS NOT NULL").fetchall():
        raw = str(r["putative_target"])
        toks = parse_targets(raw)
        n_raw = len([t for t in re.split(r"[,;/]", raw) if t.strip()])
        tok_ok += len(toks)
        tok_bad += max(0, n_raw - len(toks))
        for tok in toks:
            conn.execute("INSERT OR IGNORE INTO target(symbol) VALUES (?)", (tok,))
            conn.execute("INSERT OR IGNORE INTO drug_target(drug_key,target_symbol) VALUES (?,?)", (r["drug_key"], tok))

    # ---- trials -> drugs ---------------------------------------------------------------------
    idx = build_drug_alias_index(conn)
    linked_rows = 0
    for r in conn.execute("SELECT nct_id, name, type FROM trial_intervention").fetchall():
        if r["type"] and str(r["type"]).upper() in NON_DRUG_TYPES:
            continue
        for key in idx.find_keys(normalize_drug(r["name"])):
            cur = conn.execute("INSERT OR IGNORE INTO trial_drug(nct_id,drug_key,intervention_name,intervention_type) VALUES (?,?,?,?)",
                               (r["nct_id"], key, r["name"], r["type"]))
            linked_rows += cur.rowcount
    q_total = conn.execute("SELECT COUNT(*) FROM trial_query").fetchone()[0]
    q_linked = conn.execute("SELECT COUNT(*) FROM trial_query q JOIN trial_drug d ON d.nct_id=q.nct_id AND d.drug_key=q.drug_key").fetchone()[0]

    # ---- organisations -----------------------------------------------------------------------
    variants: dict[str, Counter] = defaultdict(Counter)
    for r in conn.execute("SELECT lead_sponsor FROM trial WHERE lead_sponsor IS NOT NULL AND lead_sponsor <> ''"):
        k = org_key(r[0])
        if k:
            variants[k][r[0].strip()] += 1
    for k, c in variants.items():
        conn.execute("INSERT INTO organization(org_key,label) VALUES (?,?)", (k, c.most_common(1)[0][0]))
        for raw in c:
            conn.execute("INSERT OR IGNORE INTO org_alias(org_key,raw_name) VALUES (?,?)", (k, raw))
    n_raw_orgs = sum(len(c) for c in variants.values())
    conn.execute("UPDATE trial SET org_key = NULL WHERE org_key IS NOT NULL AND org_key NOT IN (SELECT org_key FROM organization)")

    # ---- conditions -> cancer types ------------------------------------------------------------
    rules = load_condition_rules()
    n_cond = n_cond_mapped = 0
    for r in conn.execute("SELECT condition_key, label FROM condition").fetchall():
        n_cond += 1
        codes = map_condition_to_cancer_types(r["label"], rules)
        n_cond_mapped += bool(codes)
        for code in codes:
            conn.execute("INSERT OR IGNORE INTO condition_cancer_type(condition_key,code) VALUES (?,?)", (r["condition_key"], code))

    # ---- cell lines -> cancer types ----------------------------------------------------------------
    known = {r[0] for r in conn.execute("SELECT code FROM cancer_type")}
    n_cl = n_cl_typed = 0
    unmapped: Counter = Counter()
    for r in conn.execute("SELECT cell_line_key, tcga_desc FROM cell_line").fetchall():
        n_cl += 1
        code = normalize_tcga_code(r["tcga_desc"])
        if code is None:
            continue
        if code not in known:
            conn.execute("INSERT INTO cancer_type(code,label,parent_code,kind) VALUES (?,?,?,?)", (code, code, "CANCER", "unmapped"))
            known.add(code)
            unmapped[code] += 1
        conn.execute("UPDATE cell_line SET cancer_type_code=? WHERE cell_line_key=?", (code, r["cell_line_key"]))
        n_cl_typed += 1

    # ---- source flags -------------------------------------------------------------------------------
    conn.execute("UPDATE drug SET in_ctgov = (drug_key IN (SELECT drug_key FROM trial_drug))")
    conn.execute("UPDATE drug SET in_ndc = (drug_key IN (SELECT drug_key FROM ndc_product WHERE drug_key IS NOT NULL))")
    conn.execute("UPDATE drug SET in_faers = (drug_key IN (SELECT drug_key FROM faers_reaction))")
    conn.execute("UPDATE drug SET in_pubmed = (drug_key IN (SELECT source_drug_key FROM pubmed_article))")
    conn.commit()

    one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
    report = {
        "drugs_total": one("SELECT COUNT(*) FROM drug"),
        "drugs_per_source": {s: one(f"SELECT COUNT(*) FROM drug WHERE in_{s}=1") for s in ("gdsc", "ctgov", "ndc", "faers", "pubmed")},
        "drugs_in_gdsc_and_ctgov": one("SELECT COUNT(*) FROM drug WHERE in_gdsc=1 AND in_ctgov=1"),
        "drugs_in_gdsc_and_any_other": one("SELECT COUNT(*) FROM drug WHERE in_gdsc=1 AND (in_ctgov+in_ndc+in_faers+in_pubmed)>0"),
        "drugs_in_ctgov_ndc_faers": one("SELECT COUNT(*) FROM drug WHERE in_ctgov=1 AND in_ndc=1 AND in_faers=1"),
        "drugs_in_all_sources": one("SELECT COUNT(*) FROM drug WHERE in_gdsc=1 AND in_ctgov=1 AND in_ndc=1 AND in_faers=1 AND in_pubmed=1"),
        "trials": one("SELECT COUNT(*) FROM trial"),
        "trial_drug_links": one("SELECT COUNT(*) FROM trial_drug"),
        "trial_query_pairs": q_total,
        "trial_query_pairs_with_intervention_name_match": q_linked,
        "trial_query_name_match_rate": round(q_linked / q_total, 4) if q_total else None,
        "organizations_raw_variants": n_raw_orgs,
        "organizations_canonical": len(variants),
        "conditions": n_cond,
        "conditions_mapped_to_cancer_type": n_cond_mapped,
        "cell_lines": n_cl,
        "cell_lines_with_cancer_type": n_cl_typed,
        "cancer_type_codes_not_in_curated_table": dict(unmapped),
        "target_tokens_accepted": tok_ok,
        "target_tokens_rejected_as_non_symbols": tok_bad,
        "ndc_products": one("SELECT COUNT(*) FROM ndc_product"),
        "ndc_products_linked_to_drug": one("SELECT COUNT(*) FROM ndc_product WHERE drug_key IS NOT NULL"),
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "resolution_report.json").write_text(json.dumps(report, indent=2))
    conn.close()
    return report
