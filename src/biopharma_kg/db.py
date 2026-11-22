"""SQLite schema. SQLite is the relational layer that the R2RML mappings read from.

Design notes
------------
* ``drug`` / ``organization`` / ``condition`` / ``cell_line`` / ``adverse_event`` / ``gene`` /
  ``pharm_class`` / ``cancer_type`` are *canonical entity* tables (the "master data").
* ``v_*`` views expose the exact relations that become RDF edges. The R2RML mappings *and* the
  ML dataset exporter both read these views, so the KG and the ML graph cannot silently diverge
  (``tests/test_consistency.py`` asserts the counts match).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- ---------------------------------------------------------------- GDSC2
CREATE TABLE IF NOT EXISTS gdsc_response (
    cosmic_id INTEGER, cell_line_key TEXT, cell_line_name TEXT,
    gdsc_drug_id INTEGER, drug_key TEXT, drug_name TEXT,
    putative_target TEXT, pathway_name TEXT, tcga_desc TEXT,
    ln_ic50 REAL, auc REAL, z_score REAL, is_sensitive INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_gdsc_drug ON gdsc_response(drug_key);
CREATE INDEX IF NOT EXISTS ix_gdsc_cl ON gdsc_response(cell_line_key);

CREATE TABLE IF NOT EXISTS cell_line (
    cell_line_key TEXT PRIMARY KEY, cosmic_id INTEGER, name TEXT,
    tcga_desc TEXT, cancer_type_code TEXT
);

-- ---------------------------------------------------------------- canonical entities
CREATE TABLE IF NOT EXISTS drug (
    drug_key TEXT PRIMARY KEY, label TEXT NOT NULL,
    gdsc_drug_id INTEGER, pathway_name TEXT,
    in_gdsc INTEGER DEFAULT 0, in_ctgov INTEGER DEFAULT 0, in_ndc INTEGER DEFAULT 0,
    in_faers INTEGER DEFAULT 0, in_pubmed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS drug_alias (
    drug_key TEXT NOT NULL, alias_norm TEXT NOT NULL, alias_raw TEXT, source TEXT NOT NULL,
    PRIMARY KEY (drug_key, alias_norm, source)
);
CREATE TABLE IF NOT EXISTS target (symbol TEXT PRIMARY KEY);  -- molecular target label as named by GDSC (usually, not always, an HGNC symbol)
CREATE TABLE IF NOT EXISTS drug_target (
    drug_key TEXT NOT NULL, target_symbol TEXT NOT NULL, source TEXT DEFAULT 'GDSC',
    PRIMARY KEY (drug_key, target_symbol)
);
CREATE TABLE IF NOT EXISTS organization (org_key TEXT PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS org_alias (org_key TEXT NOT NULL, raw_name TEXT NOT NULL, PRIMARY KEY (org_key, raw_name));
CREATE TABLE IF NOT EXISTS condition (
    condition_key TEXT PRIMARY KEY, label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS condition_cancer_type (
    condition_key TEXT NOT NULL, code TEXT NOT NULL, PRIMARY KEY (condition_key, code)
);
CREATE TABLE IF NOT EXISTS cancer_type (
    code TEXT PRIMARY KEY, label TEXT NOT NULL, parent_code TEXT, kind TEXT
);
CREATE TABLE IF NOT EXISTS adverse_event (ae_key TEXT PRIMARY KEY, term TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pharm_class (class_key TEXT PRIMARY KEY, label TEXT NOT NULL);

-- ---------------------------------------------------------------- ClinicalTrials.gov
CREATE TABLE IF NOT EXISTS trial (
    nct_id TEXT PRIMARY KEY, title TEXT, overall_status TEXT,
    start_date TEXT, phase TEXT, lead_sponsor TEXT, org_key TEXT
);
CREATE TABLE IF NOT EXISTS trial_condition (
    nct_id TEXT NOT NULL, condition_key TEXT NOT NULL, PRIMARY KEY (nct_id, condition_key)
);
CREATE TABLE IF NOT EXISTS trial_drug (
    nct_id TEXT NOT NULL, drug_key TEXT NOT NULL, intervention_name TEXT, intervention_type TEXT,
    PRIMARY KEY (nct_id, drug_key)
);


CREATE TABLE IF NOT EXISTS trial_intervention (
    nct_id TEXT NOT NULL, name TEXT NOT NULL, type TEXT
);
CREATE INDEX IF NOT EXISTS ix_trial_int ON trial_intervention(nct_id);
CREATE TABLE IF NOT EXISTS trial_query (
    nct_id TEXT NOT NULL, drug_key TEXT NOT NULL, PRIMARY KEY (nct_id, drug_key)
);

-- ---------------------------------------------------------------- openFDA
CREATE TABLE IF NOT EXISTS ndc_product (
    product_ndc TEXT PRIMARY KEY, brand_name TEXT, generic_name TEXT, dosage_form TEXT,
    route TEXT, marketing_category TEXT, labeler TEXT, drug_key TEXT
);
CREATE TABLE IF NOT EXISTS ndc_pharm_class (
    product_ndc TEXT NOT NULL, class_key TEXT NOT NULL, PRIMARY KEY (product_ndc, class_key)
);
CREATE TABLE IF NOT EXISTS faers_reaction (
    drug_key TEXT NOT NULL, ae_key TEXT NOT NULL, report_count INTEGER NOT NULL,
    PRIMARY KEY (drug_key, ae_key)
);

-- ---------------------------------------------------------------- PubMed + extraction
CREATE TABLE IF NOT EXISTS pubmed_article (
    pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT, pub_year INTEGER, source_drug_key TEXT
);
CREATE TABLE IF NOT EXISTS extracted_relation (
    rel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT NOT NULL, drug_key TEXT NOT NULL, predicate TEXT NOT NULL,
    object_text TEXT NOT NULL, object_key TEXT, object_kind TEXT,
    confidence REAL, evidence TEXT NOT NULL, extractor TEXT NOT NULL, model TEXT,
    UNIQUE (pmid, drug_key, predicate, object_text, extractor)
);

-- ---------------------------------------------------------------- views = the KG's edges
DROP VIEW IF EXISTS v_drug_condition;
CREATE VIEW v_drug_condition AS
SELECT td.drug_key AS drug_key, tc.condition_key AS condition_key,
       MIN(t.start_date) AS first_start, COUNT(DISTINCT t.nct_id) AS n_trials
FROM trial_drug td
JOIN trial t ON t.nct_id = td.nct_id
JOIN trial_condition tc ON tc.nct_id = td.nct_id
GROUP BY td.drug_key, tc.condition_key;

DROP VIEW IF EXISTS v_drug_org;
CREATE VIEW v_drug_org AS
SELECT td.drug_key AS drug_key, t.org_key AS org_key,
       MIN(t.start_date) AS first_start, COUNT(DISTINCT t.nct_id) AS n_trials
FROM trial_drug td JOIN trial t ON t.nct_id = td.nct_id
WHERE t.org_key IS NOT NULL AND t.org_key <> ''
GROUP BY td.drug_key, t.org_key;

DROP VIEW IF EXISTS v_drug_sensitive_cell_line;
CREATE VIEW v_drug_sensitive_cell_line AS
SELECT DISTINCT drug_key, cell_line_key FROM gdsc_response
WHERE is_sensitive = 1 AND drug_key IS NOT NULL AND drug_key <> '';

DROP VIEW IF EXISTS v_drug_pharm_class;
CREATE VIEW v_drug_pharm_class AS
SELECT DISTINCT p.drug_key AS drug_key, c.class_key AS class_key
FROM ndc_product p JOIN ndc_pharm_class c ON c.product_ndc = p.product_ndc
WHERE p.drug_key IS NOT NULL AND p.drug_key <> '';

DROP VIEW IF EXISTS v_drug_ae;
CREATE VIEW v_drug_ae AS
SELECT drug_key, ae_key, report_count FROM faers_reaction;

DROP VIEW IF EXISTS v_drug_extracted;
CREATE VIEW v_drug_extracted AS
SELECT DISTINCT drug_key, object_key, predicate
FROM extracted_relation WHERE object_key IS NOT NULL;
"""


def connect(path: Path | str) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def reset_tables(conn: sqlite3.Connection, tables: list[str]) -> None:
    for t in tables:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    return {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in sorted(names)}
