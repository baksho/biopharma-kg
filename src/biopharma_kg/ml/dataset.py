"""Build link-prediction tasks from the same relational views that feed the RDF graph.

Leakage controls (each is covered by a unit test):
  * Trial *nodes* are NOT part of the ML graph. If they were, a test edge (drug, condition) could be
    recovered through the 2-hop path drug <- trial -> condition.
  * No inverse relations are added to the training graph (they would duplicate test edges).
  * Temporal task: every time-stamped context edge (sponsor links, literature links) must predate
    the validation cutoff; test/validation edges are the newest first-seen (drug, condition) pairs.
  * Validation/test triples are dropped if their head or tail never occurs in training.
  * Evaluation is *filtered* (other known true tails are removed) and *type-constrained* (only
    entities that occur as tail of the target relation are ranked).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# relation name -> (SQL, head prefix, tail prefix, time kind)
#   time kind: None (static) | 'date' (column `t` is an ISO date) | 'year' (column `t` is a year)
RELATIONS: dict[str, tuple[str, str, str, str | None]] = {
    "investigatedFor": ("SELECT drug_key AS h, condition_key AS t_, first_start AS t FROM v_drug_condition", "drug", "cond", "date"),
    "investigatedBy": ("SELECT drug_key AS h, org_key AS t_, first_start AS t FROM v_drug_org", "drug", "org", "date"),
    "sensitiveIn": ("SELECT drug_key AS h, cell_line_key AS t_, NULL AS t FROM v_drug_sensitive_cell_line", "drug", "cl", None),
    "hasReportedReaction": ("SELECT drug_key AS h, ae_key AS t_, NULL AS t FROM v_drug_ae", "drug", "ae", None),
    "hasTarget": ("SELECT drug_key AS h, target_symbol AS t_, NULL AS t FROM drug_target", "drug", "tgt", None),
    "hasPharmClass": ("SELECT drug_key AS h, class_key AS t_, NULL AS t FROM v_drug_pharm_class", "drug", "pc", None),
    "cellLineOfType": ("SELECT cell_line_key AS h, cancer_type_code AS t_, NULL AS t FROM cell_line WHERE cancer_type_code IS NOT NULL", "cl", "ct", None),
    "relatedToCancerType": (
        "SELECT condition_key AS h, code AS t_, NULL AS t FROM condition_cancer_type WHERE condition_key IN "
        "(SELECT condition_key FROM v_drug_condition UNION SELECT object_key FROM extracted_relation "
        "WHERE object_kind='condition' AND object_key IS NOT NULL)", "cond", "ct", None),
    "broaderType": ("SELECT code AS h, parent_code AS t_, NULL AS t FROM cancer_type WHERE parent_code IS NOT NULL", "ct", "ct", None),
}
# literature-extracted edges carry the publication year as their time
LITERATURE: dict[str, tuple[str, str, str]] = {
    "extractedTreats": ("treats", "drug", "cond"),
    "extractedAdverseEffect": ("adverse_effect", "drug", "ae"),
    "extractedInhibits": ("inhibits", "drug", "tgt"),
}
_LIT_SQL = (
    "SELECT r.drug_key AS h, r.object_key AS t_, MIN(a.pub_year) AS t FROM extracted_relation r "
    "JOIN pubmed_article a ON a.pmid = r.pmid WHERE r.predicate = ? AND r.object_key IS NOT NULL "
    "GROUP BY r.drug_key, r.object_key"
)


def load_edges(conn: sqlite3.Connection, literature: bool = True) -> dict[str, pd.DataFrame]:
    """Return ``{relation: DataFrame[head, tail, time]}`` with type-prefixed entity ids."""
    out: dict[str, pd.DataFrame] = {}
    for rel, (sql, hp, tp, kind) in RELATIONS.items():
        df = pd.read_sql_query(sql, conn)
        df = df.dropna(subset=["h", "t_"])
        df["head"] = hp + ":" + df["h"].astype(str)
        df["tail"] = tp + ":" + df["t_"].astype(str)
        df["time"] = df["t"] if kind == "date" else None
        out[rel] = df[["head", "tail", "time"]].drop_duplicates(subset=["head", "tail"]).reset_index(drop=True)
    if literature:
        for rel, (pred, hp, tp) in LITERATURE.items():
            df = pd.read_sql_query(_LIT_SQL, conn, params=(pred,))
            df["head"] = hp + ":" + df["h"].astype(str)
            df["tail"] = tp + ":" + df["t_"].astype(str)
            # a year-only time is stored as the last day of that year => usable only for cutoffs after it
            df["time"] = df["t"].map(lambda y: None if pd.isna(y) else f"{int(y)}-12-31")
            out[rel] = df[["head", "tail", "time"]].reset_index(drop=True)
    return {k: v for k, v in out.items() if len(v)}


@dataclass
class Task:
    name: str
    target: str
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    train: np.ndarray                      # (n, 3) all relations (context + target-train)
    valid: np.ndarray                      # (m, 3) target relation only
    test: np.ndarray                       # (k, 3) target relation only
    candidate_mask: np.ndarray             # bool (E,): entities allowed as tails when ranking
    known_tails: dict[int, set[int]]       # head id -> all true tails of the target relation (any split)
    info: dict = field(default_factory=dict)

    @property
    def target_rel_id(self) -> int:
        return self.relation_to_id[self.target]

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)


class TaskTooSmall(RuntimeError):
    pass


def _encode(edges: dict[str, pd.DataFrame]) -> tuple[dict[str, int], dict[str, int]]:
    ents: set[str] = set()
    for df in edges.values():
        ents.update(df["head"])
        ents.update(df["tail"])
    return {e: i for i, e in enumerate(sorted(ents))}, {r: i for i, r in enumerate(sorted(edges))}


def build_task(
    name: str,
    target: str,
    edges: dict[str, pd.DataFrame],
    seed: int = 0,
    temporal: bool = False,
    temporal_test_fraction: float = 0.2,
    valid_fraction: float = 0.1,
    random_test_fraction: float = 0.1,
    min_test: int = 20,
    min_train: int = 50,
) -> Task:
    if target not in edges:
        raise TaskTooSmall(f"relation {target!r} has no edges (is the source loaded?)")
    e2i, r2i = _encode(edges)
    tgt = edges[target].copy()
    info: dict = {"target_edges_total": len(tgt), "split": "temporal" if temporal else "random"}

    # ---- split the target relation --------------------------------------------------------------
    cutoff: str | None = None
    if temporal:
        dated = tgt.dropna(subset=["time"]).sort_values("time", kind="stable")
        undated = tgt[tgt["time"].isna()]
        if len(dated) < (min_test + min_train):
            raise TaskTooSmall(f"only {len(dated)} dated {target} edges; need >= {min_test + min_train} for a temporal split")
        n = len(dated)
        test_start = dated["time"].iloc[int(np.floor(n * (1 - temporal_test_fraction)))]
        val_start = dated["time"].iloc[int(np.floor(n * (1 - temporal_test_fraction - valid_fraction)))]
        if not (val_start < test_start):
            raise TaskTooSmall("dates too concentrated for a temporal split (validation and test windows coincide)")
        test = dated[dated["time"] >= test_start]
        valid = dated[(dated["time"] >= val_start) & (dated["time"] < test_start)]
        train_t = pd.concat([dated[dated["time"] < val_start], undated])
        cutoff = val_start
        info.update(cutoff_valid_start=val_start, cutoff_test_start=test_start, undated_edges_put_in_train=len(undated))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(tgt))
        n_test = max(1, int(round(len(tgt) * random_test_fraction)))
        n_val = max(1, int(round(len(tgt) * valid_fraction)))
        test = tgt.iloc[perm[:n_test]]
        valid = tgt.iloc[perm[n_test : n_test + n_val]]
        train_t = tgt.iloc[perm[n_test + n_val :]]

    # ---- context relations (time-aware for the temporal task) ---------------------------------------
    parts = [(r2i[target], train_t)]
    dropped_future = 0
    for rel, df in edges.items():
        if rel == target:
            continue
        if cutoff is not None and df["time"].notna().any():
            keep = df["time"].isna() | (df["time"] < cutoff)
            dropped_future += int((~keep).sum())
            df = df[keep]
        parts.append((r2i[rel], df))
    info["context_edges_dropped_as_future"] = dropped_future

    def to_arr(rid: int, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.stack([df["head"].map(e2i).to_numpy(), np.full(len(df), rid), df["tail"].map(e2i).to_numpy()], axis=1).astype(np.int64)

    train = np.concatenate([to_arr(rid, df) for rid, df in parts], axis=0)
    seen = set(train[:, 0].tolist()) | set(train[:, 2].tolist())

    def covered(df: pd.DataFrame) -> pd.DataFrame:
        m = df["head"].map(e2i).isin(seen) & df["tail"].map(e2i).isin(seen)
        return df[m]

    valid_c, test_c = covered(valid), covered(test)
    info.update(valid_dropped_unseen_entity=len(valid) - len(valid_c), test_dropped_unseen_entity=len(test) - len(test_c))
    valid_a, test_a = to_arr(r2i[target], valid_c), to_arr(r2i[target], test_c)
    n_train_target = int((train[:, 1] == r2i[target]).sum())
    if len(test_a) < min_test or n_train_target < min_train:
        raise TaskTooSmall(f"{name}: {len(test_a)} test and {n_train_target} train {target} edges (need >= {min_test} / {min_train})")

    cand = np.zeros(len(e2i), dtype=bool)
    all_t = to_arr(r2i[target], tgt)
    cand[all_t[:, 2]] = True
    known: dict[int, set[int]] = {}
    for h, _, t in all_t:
        known.setdefault(int(h), set()).add(int(t))
    info.update(num_entities=len(e2i), num_relations=len(r2i), train_triples=len(train), train_target_triples=n_train_target,
                valid_triples=len(valid_a), test_triples=len(test_a), candidate_tails=int(cand.sum()))
    return Task(name, target, e2i, r2i, train, valid_a, test_a, cand, known, info)


def shortcut_report(task: Task) -> dict:
    """How many test pairs are directly linked by *another* relation in the training graph?
    (e.g. a literature 'extractedTreats' edge between the same drug and condition.) These are
    legitimate context features when dated before the cutoff, but readers should know about them."""
    r = task.target_rel_id
    ctx = task.train[task.train[:, 1] != r]
    pairs: dict[tuple[int, int], set[int]] = {}
    for h, rel, t in ctx:
        pairs.setdefault((int(h), int(t)), set()).add(int(rel))
    inv = {v: k for k, v in task.relation_to_id.items()}
    counts: dict[str, int] = {}
    n = 0
    for h, _, t in task.test:
        rels = pairs.get((int(h), int(t)))
        if rels:
            n += 1
            for x in rels:
                counts[inv[x]] = counts.get(inv[x], 0) + 1
    return {"test_pairs_with_other_direct_edge": n, "test_pairs": len(task.test), "by_relation": counts}


def assert_no_leakage(task: Task) -> None:
    """Hard checks used by tests and by the runner before training."""
    r = task.target_rel_id
    train_t = {(int(h), int(t)) for h, rr, t in task.train if rr == r}
    for name, arr in (("valid", task.valid), ("test", task.test)):
        both = {(int(h), int(t)) for h, _, t in arr} & train_t
        if both:
            raise AssertionError(f"{len(both)} {name} pairs also occur in the training graph")
    vt = {(int(h), int(t)) for h, _, t in task.valid}
    tt = {(int(h), int(t)) for h, _, t in task.test}
    if vt & tt:
        raise AssertionError("validation and test share pairs")
