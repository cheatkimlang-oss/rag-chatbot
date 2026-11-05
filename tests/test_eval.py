from eval.metrics import ndcg_at_k, reciprocal_rank, recall_at_k
from eval.run_eval import ranked_at, score, sweep


def test_metrics():
    assert recall_at_k(["a", "b"], {"b"}, 5) == 1.0 and recall_at_k(["a"], {"z"}, 5) == 0.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert ndcg_at_k(["x", "a"], {"a"}) < ndcg_at_k(["a", "x"], {"a"}) == 1.0


def _rec(kind, gate, cards, expected=(), slice_=None):
    return {"query_id": "q", "query": "q", "slice": slice_ or kind, "kind": kind, "gate": gate,
            "expected": list(expected), "expected_senior_id": None, "senior_id": None,
            "cards": [{"project_id": p, "dense": d, "lexical": lx} for p, d, lx in cards],
            "latency_ms": 1}


def test_gate_replay_matches_service_rules():
    rec = _rec("topic", 0.6, [("a", 0.6, False), ("b", 0.2, True), ("c", 0.3, False)])
    assert ranked_at(rec, 0.5) == ["a", "b"]           # c: below gate and no keyword match
    assert ranked_at(rec, 0.7) == []                    # abstains
    assert ranked_at(_rec("name", None, [("p", None, True)]), 0.99) == ["p"]   # names never gated


def test_sweep_picks_smallest_threshold_meeting_both_targets():
    records = [_rec("topic", 0.70, [("a", 0.70, True)], {"a"}),
               _rec("topic", 0.60, [("b", 0.60, True)], {"b"}),
               _rec("negative", 0.55, [("a", 0.55, False)]),
               _rec("negative", 0.40, [("b", 0.40, False)])]
    table, chosen = sweep(records)
    assert chosen == 0.575          # first t with 0.55 < t <= 0.60
    assert score(records, chosen)["negative"]["abstention"] == 1.0
