"""Graph-ML tests. The synthetic graph below exists ONLY to test mechanics (splits, leakage guards, evaluator
correctness, that training learns planted structure). No result from it is ever reported as a project result."""
import pytest
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pykeen")

import numpy as np
import pandas as pd
import pytest
import torch

from biopharma_kg.ml.dataset import TaskTooSmall, assert_no_leakage, build_task, shortcut_report
from biopharma_kg.ml.evaluate import evaluate, metrics_from_ranks, ranks_for
from biopharma_kg.ml.models import popularity_baseline, train_pykeen, train_rgcn

K, ND, NC = 4, 60, 40   # clusters, drugs, conditions


def synthetic_edges(seed=0, p=0.6, with_dates=True):
    rng = np.random.default_rng(seed)
    rows, ctx_org, ctx_ct = [], [], []
    for d in range(ND):
        k = d % K
        for c in range(NC):
            if c % K == k and rng.random() < p:
                year = int(rng.integers(2000, 2024))
                rows.append((f"drug:d{d}", f"cond:c{c}", f"{year}-{int(rng.integers(1, 13)):02d}-15" if with_dates else None))
        ctx_org.append((f"drug:d{d}", f"org:o{k}", None))
    for c in range(NC):
        ctx_ct.append((f"cond:c{c}", f"ct:t{c % K}", None))
    mk = lambda r: pd.DataFrame(r, columns=["head", "tail", "time"])  # noqa: E731
    return {"investigatedFor": mk(rows), "investigatedBy": mk(ctx_org), "relatedToCancerType": mk(ctx_ct)}


# ------------------------------------------------------------------ splits and leakage
def test_random_split_is_disjoint_and_covered():
    t = build_task("t", "investigatedFor", synthetic_edges(), seed=1, temporal=False, min_test=5, min_train=20)
    assert_no_leakage(t)
    assert len(t.test) >= 5 and len(t.valid) >= 1
    tr = {(h, tt) for h, r, tt in t.train if r == t.target_rel_id}
    assert not ({(h, tt) for h, _, tt in t.test} & tr)
    seen = set(t.train[:, 0]) | set(t.train[:, 2])
    assert set(t.test[:, 0]) <= seen and set(t.test[:, 2]) <= seen          # no unseen entities in test
    assert all(t.candidate_mask[x] for x in t.test[:, 2])                    # true tails are always candidates
    assert not t.candidate_mask[t.entity_to_id["org:o0"]]                    # type constraint excludes non-conditions


def test_temporal_split_respects_time():
    edges = synthetic_edges()
    t = build_task("t", "investigatedFor", edges, temporal=True, min_test=5, min_train=20)
    assert_no_leakage(t)
    i2e = {v: k for k, v in t.entity_to_id.items()}
    tm = {(h, tt): d for h, tt, d in edges["investigatedFor"][["head", "tail", "time"]].itertuples(index=False)}
    pair = lambda row: tm[(i2e[row[0]], i2e[row[2]])]  # noqa: E731
    test_times = [pair(r) for r in t.test]
    valid_times = [pair(r) for r in t.valid]
    train_times = [pair(r) for r in t.train if r[1] == t.target_rel_id]
    assert min(test_times) >= t.info["cutoff_test_start"] and max(valid_times) < t.info["cutoff_test_start"]
    assert min(valid_times) >= t.info["cutoff_valid_start"] and max(train_times) < t.info["cutoff_valid_start"]


def test_temporal_context_edges_after_cutoff_are_dropped():
    edges = synthetic_edges()
    # a dated context edge in the future of the cutoff must not reach the training graph
    edges["investigatedBy"] = pd.concat([edges["investigatedBy"], pd.DataFrame(
        [("drug:d0", "org:o_future", "2030-01-01"), ("drug:d0", "org:o_past", "1999-01-01")], columns=["head", "tail", "time"])])
    t = build_task("t", "investigatedFor", edges, temporal=True, min_test=5, min_train=20)
    assert t.info["context_edges_dropped_as_future"] >= 1
    assert "org:o_future" not in t.entity_to_id or t.entity_to_id["org:o_future"] not in set(t.train[:, 2])
    assert t.entity_to_id["org:o_past"] in set(t.train[:, 2])


def test_leakage_guard_fires():
    t = build_task("t", "investigatedFor", synthetic_edges(), seed=0, min_test=5, min_train=20)
    t.valid = np.concatenate([t.valid, t.train[t.train[:, 1] == t.target_rel_id][:1]])
    with pytest.raises(AssertionError):
        assert_no_leakage(t)


def test_too_small_and_missing_relation():
    with pytest.raises(TaskTooSmall):
        build_task("t", "investigatedFor", synthetic_edges(), seed=0, min_test=500)
    with pytest.raises(TaskTooSmall):
        build_task("t", "nonexistent", synthetic_edges())
    with pytest.raises(TaskTooSmall):
        build_task("t", "investigatedFor", synthetic_edges(with_dates=False), temporal=True)


def test_shortcut_report_counts_direct_context_edges():
    edges = synthetic_edges()
    t0 = build_task("t", "investigatedFor", edges, seed=0, min_test=5, min_train=20)
    # add a literature-like edge that duplicates every target pair -> every test pair is a shortcut
    lit = edges["investigatedFor"][["head", "tail"]].assign(time=None)
    t1 = build_task("t", "investigatedFor", {**edges, "extractedTreats": lit}, seed=0, min_test=5, min_train=20)
    assert shortcut_report(t0)["test_pairs_with_other_direct_edge"] == 0
    rep = shortcut_report(t1)
    assert rep["test_pairs_with_other_direct_edge"] == rep["test_pairs"] > 0 and "extractedTreats" in rep["by_relation"]


# ------------------------------------------------------------------ evaluator correctness
def brute_force_ranks(task, triples, scores_by_head):
    out = []
    for h, r, t in triples:
        s = scores_by_head[int(h)].copy()
        for e in range(len(s)):
            if not task.candidate_mask[e] or (e != t and e in task.known_tails.get(int(h), set())):
                s[e] = -np.inf
        better = sum(1 for e in range(len(s)) if s[e] > s[t])
        ties = sum(1 for e in range(len(s)) if e != t and s[e] == s[t])
        out.append(1 + better + 0.5 * ties)
    return np.array(out, dtype=float)


def test_evaluator_matches_brute_force_including_ties():
    t = build_task("t", "investigatedFor", synthetic_edges(), seed=2, min_test=5, min_train=20)
    rng = np.random.default_rng(0)
    table = rng.integers(0, 5, size=(t.num_entities, t.num_entities)).astype(np.float32)  # integer scores -> many ties
    fn = lambda hr: torch.as_tensor(table[hr[:, 0].numpy()])  # noqa: E731
    got = ranks_for(t, t.test, fn, batch_size=7)
    want = brute_force_ranks(t, t.test, {h: table[h] for h in range(t.num_entities)})
    np.testing.assert_allclose(got, want)


def test_metrics_from_ranks():
    m = metrics_from_ranks(np.array([1, 2, 4, 20]))
    assert m["hits@1"] == 0.25 and m["hits@3"] == 0.5 and m["hits@10"] == 0.75 and m["n"] == 4
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.25 + 0.05) / 4)
    assert np.isnan(metrics_from_ranks(np.zeros(0))["mrr"])


def test_evaluator_agrees_with_pykeen_rank_based_evaluator():
    """Independent cross-check against PyKEEN's filtered tail evaluation (type constraint disabled so they are comparable)."""
    from pykeen.evaluation import RankBasedEvaluator
    from pykeen.models import TransE
    from pykeen.triples import TriplesFactory
    t = build_task("t", "investigatedFor", synthetic_edges(), seed=3, min_test=5, min_train=20)
    assert t.info["test_dropped_unseen_entity"] == 0 and t.info["valid_dropped_unseen_entity"] == 0
    t.candidate_mask[:] = True
    tf = TriplesFactory(mapped_triples=torch.as_tensor(t.train), entity_to_id=t.entity_to_id, relation_to_id=t.relation_to_id)
    torch.manual_seed(0)
    model = TransE(triples_factory=tf, embedding_dim=8).eval()
    ours = metrics_from_ranks(ranks_for(t, t.test, lambda hr: model.score_t(hr_batch=hr), batch_size=16))
    res = RankBasedEvaluator(filtered=True).evaluate(
        model, torch.as_tensor(t.test), batch_size=32, use_tqdm=False, targets=("tail",),
        additional_filter_triples=[torch.as_tensor(t.train), torch.as_tensor(t.valid)])
    flat = res.to_flat_dict()
    key = "tail.realistic.inverse_harmonic_mean_rank"
    assert key in flat, sorted(flat)[:40]
    assert ours["mrr"] == pytest.approx(flat[key], rel=1e-4), (ours, key, flat[key])


# ------------------------------------------------------------------ training learns planted structure
@pytest.fixture(scope="module")
def planted_task():
    return build_task("planted", "investigatedFor", synthetic_edges(seed=5), seed=0, temporal=False, min_test=5, min_train=20)


def test_popularity_baseline_is_head_independent(planted_task):
    tr = popularity_baseline(planted_task)
    s = tr.score_fn(torch.tensor([[0, 0], [1, 0]]))
    assert torch.equal(s[0], s[1])


def test_rgcn_loss_decreases_and_beats_popularity(planted_task):
    short = train_rgcn(planted_task, dim=16, epochs=2, seed=0)
    longer = train_rgcn(planted_task, dim=16, epochs=150, seed=0)
    assert longer.final_loss < short.final_loss
    m = evaluate(planted_task, longer.score_fn)
    base = evaluate(planted_task, popularity_baseline(planted_task).score_fn)
    print("\nPLANTED rgcn", m, "\nPLANTED popularity", base)
    assert m["mrr"] > base["mrr"] + 0.05


@pytest.mark.parametrize("model", ["TransE", "RotatE", "ComplEx"])
def test_pykeen_models_train_and_beat_popularity(planted_task, model):
    tr = train_pykeen(planted_task, model, dim=16, epochs=60, batch_size=64, seed=0)
    m = evaluate(planted_task, tr.score_fn)
    base = evaluate(planted_task, popularity_baseline(planted_task).score_fn)
    print(f"\nPLANTED {model}", m["mrr"], "popularity", base["mrr"])
    assert np.isfinite(m["mrr"]) and m["n"] == len(planted_task.test)
    assert m["mrr"] > base["mrr"] + 0.03, (model, m, base)
