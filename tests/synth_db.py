"""TEST-ONLY generator of a synthetic relational database with the project's real schema and planted cluster
structure. It exists to exercise run_ml / the KG build at sizes the tiny fixtures cannot reach (and for the scale
check in scripts/scale_check.py). Nothing generated here is ever part of the project's data or reported results."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from biopharma_kg import db
from biopharma_kg.load import load_cancer_types


def make_synthetic_db(path: Path, n_drugs=80, n_cond=60, n_cl=120, n_ae=50, trials_per_drug=8, k=4, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    conn = db.connect(path)
    db.init_schema(conn)
    load_cancer_types(conn)
    codes = [r[0] for r in conn.execute("SELECT code FROM cancer_type WHERE kind NOT IN ('root','group') ORDER BY code")][:k]
    assert len(codes) == k, "need >= k leaf cancer types"
    conn.executemany("INSERT INTO drug(drug_key,label,in_gdsc,in_ctgov) VALUES (?,?,1,1)", [(f"drug{i}", f"Drug{i}") for i in range(n_drugs)])
    conn.executemany("INSERT INTO condition VALUES (?,?)", [(f"cond{c}", f"Condition {c}") for c in range(n_cond)])
    conn.executemany("INSERT INTO condition_cancer_type VALUES (?,?)", [(f"cond{c}", codes[c % k]) for c in range(n_cond)])
    conn.executemany("INSERT INTO organization VALUES (?,?)", [(f"org{j}", f"Org {j}") for j in range(k * 3)])
    conn.executemany("INSERT INTO adverse_event VALUES (?,?)", [(f"ae{a}", f"event {a}") for a in range(n_ae)])
    conn.executemany("INSERT INTO target VALUES (?)", [(f"T{j}",) for j in range(k * 2)])
    conn.executemany("INSERT INTO drug_target VALUES (?,?,'GDSC')", [(f"drug{i}", f"T{(i % k) * 2 + int(rng.integers(0, 2))}") for i in range(n_drugs)])
    n_tr = 0
    for i in range(n_drugs):
        for _ in range(trials_per_drug):
            c = int(rng.choice([x for x in range(n_cond) if x % k == i % k])) if rng.random() < 0.9 else int(rng.integers(0, n_cond))
            nct = f"NCT{n_tr:08d}"; n_tr += 1
            date = f"{int(rng.integers(2000, 2024))}-{int(rng.integers(1, 13)):02d}-01"
            org = f"org{(i % k) * 3 + int(rng.integers(0, 3))}"
            conn.execute("INSERT INTO trial(nct_id,title,overall_status,start_date,phase,lead_sponsor,org_key) VALUES (?,?,?,?,?,?,?)",
                         (nct, f"Trial {nct}", "COMPLETED", date, "PHASE3", f"Org", org))
            conn.execute("INSERT INTO trial_drug VALUES (?,?,?,?)", (nct, f"drug{i}", f"Drug{i}", "DRUG"))
            conn.execute("INSERT OR IGNORE INTO trial_condition VALUES (?,?)", (nct, f"cond{c}"))
    for j in range(n_cl):
        conn.execute("INSERT INTO cell_line VALUES (?,?,?,?,?)", (f"cl{j}", j, f"CL{j}", "X", codes[j % k]))
    resp = []
    for i in range(n_drugs):
        for j in rng.choice(n_cl, size=min(n_cl, 40), replace=False):
            sens = int((j % k) == (i % k) and rng.random() < 0.7)
            resp.append((int(j), f"cl{j}", f"CL{j}", i, f"drug{i}", f"Drug{i}", "T", "P", "X", 0.0, 0.5, 0.0, sens))
    conn.executemany("INSERT INTO gdsc_response VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", resp)
    fa = []
    for i in range(n_drugs):
        pool = [x for x in range(n_ae) if x % k == i % k]
        for a in rng.choice(pool, size=min(4, len(pool)), replace=False):
            fa.append((f"drug{i}", f"ae{a}", int(rng.integers(1, 500))))
    conn.executemany("INSERT INTO faers_reaction VALUES (?,?,?)", fa)
    conn.commit()
    stats = db.table_counts(conn)
    conn.close()
    return stats
