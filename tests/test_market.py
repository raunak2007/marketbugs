"""Mode S, credit and pre-freeze protocol fixes (2026-11-10): normalization, label consistency, information
boundaries, reward telescoping, reproducible trajectories, gold-not-in-pool handling, single traceback,
no_trace retrieval boundary, token-budget truncation."""
import hashlib, math
import numpy as np
import pytest
from src.agents.views import inference_view
from src.agents.prompts import AGENTS, probability_agent_prompt, candidate_block
from src.agents.specialists import label_maps
from src.agents.run_values import permuted_view
from src.agents.pipeline import condition_query
from src.market.sequential import run_market, market_prompt, market_text, to_cid_probs, trajectory, ORDERS
from src.market import pools
from src.analysis.credit import market_rewards, value_vector, credit
from src.llm import tokens
from tests.fixtures import make_instance


class FakeSrv:
    """Deterministic stand-in for the Gemma server: logits are a hash of the prompt."""
    def __init__(self):
        self.prompts = []

    def label_logits(self, prompts, ns):
        out = []
        for p, n in zip(prompts, ns):
            self.prompts.append(p)
            seed = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            raw = np.random.RandomState(seed).randn(n) * 2
            pr = np.exp(raw - raw.max()); pr /= pr.sum()
            out.append({"raw": raw.tolist(), "probs": pr.tolist(), "input_tokens": len(p) // 4, "latency_ms": 1.0,
                        "token_mapping_hash": "fake", "model_revision": "fake@0"})
        return out


def _setup(i=0, perm=0):
    inst, cands = make_instance(i)
    view = permuted_view(inference_view(inst, cands, "test_informed"), perm)
    notes = {a: f"RANKED: [A]\\n{a} thinks [A]" for a in AGENTS}
    cids = sorted(c["cid"] for c in view["pool"])
    p0 = list(pools.normalize(np.arange(1, len(cids) + 1)))
    return inst, view, notes, cids, p0


def test_market_rows_normalized_and_complete():
    inst, view, notes, cids, p0 = _setup()
    rows, tel = run_market(view, notes, p0, FakeSrv(), inst["id"], "test_informed")
    assert len(rows) == 10 and len(tel) == 10  # 5 steps x 2 orders
    for r in rows:
        assert r["mode"] == "S" and abs(sum(r["probs"]) - 1) < 1e-9 and min(r["probs"]) >= 0
        assert r["top1"] == cids[int(np.argmax(r["probs"]))]
    finals = [r for r in rows if len(r["note_order"]) == 5]
    assert {tuple(r["note_order"]) for r in finals} == {ORDERS["canonical"], ORDERS["reversed"]}
    assert all(r["subset_mask"] == 31 for r in finals)


@pytest.mark.parametrize("perm", [0, 1, 2])
def test_candidate_label_consistency(perm):
    inst, view, notes, cids, p0 = _setup(3, perm)
    lab2cid, cid2lab = label_maps(view)
    probs = {c: (k + 1) / 210 for k, c in enumerate(cids)}
    shown = market_text(view, probs).splitlines()
    top = max(probs, key=probs.get)
    assert shown[0].startswith(f"[{cid2lab[top]}] ")  # highest-probability cid shown under its own display label
    out = {"probs": [probs[c["cid"]] for c in view["pool"]]}  # server returns display order
    assert to_cid_probs(view, out) == probs                    # and maps back to the same cids
    blk = candidate_block(view)
    for lab, c in zip(lab2cid, view["pool"]):
        assert f"[{lab}] {c['path']} :: {c['qualname']}" in blk


def test_market_prompt_information_boundary():
    """Prompts depend only on (view, note, market): changing gold / evaluator fields changes nothing, and no
    outcome or reward vocabulary appears."""
    inst, view, notes, cids, p0 = _setup(5)
    s1, s2 = FakeSrv(), FakeSrv()
    run_market(view, notes, p0, s1, inst["id"], "test_informed")
    inst2 = dict(inst, gold_functions=["something/else.py::other"], mutation_operator="X")
    _, cands = make_instance(5)
    view2 = inference_view(inst2, cands, "test_informed")
    run_market(view2, notes, p0, s2, inst["id"], "test_informed")
    assert s1.prompts == s2.prompts
    for p in s1.prompts:
        low = p.lower()
        assert not any(w in low for w in ("reward", "gold", "resolved", "correct answer", "log score"))


def test_reward_telescoping_and_gold_missing():
    inst, view, notes, cids, p0 = _setup(7)
    rows, _ = run_market(view, notes, p0, FakeSrv(), inst["id"], "test_informed")
    for name in ORDERS:
        traj = trajectory(rows, inst["id"], name, {"probs": p0})
        assert len(traj) == 6
        r = market_rewards(traj, cids, {cids[4]})
        lg = [math.log(t[4]) for t in traj]
        assert abs(sum(r) - (lg[-1] - lg[0])) < 1e-9
    assert market_rewards(traj, cids, set()) is None  # retrieval miss: excluded, never inf


def test_market_trajectories_reproducible():
    inst, view, notes, cids, p0 = _setup(9)
    a, _ = run_market(view, notes, p0, FakeSrv(), inst["id"], "test_informed")
    b, _ = run_market(view, notes, p0, FakeSrv(), inst["id"], "test_informed")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "inference_cost_usd"} for r in rows]  # wall-clock
    assert strip(a) == strip(b)


def test_credit_excludes_gold_missing_and_is_efficient():
    rng = np.random.RandomState(0)
    cids = [chr(65 + i) for i in range(20)]
    table = {m: list(pools.normalize(rng.rand(20))) for m in range(32)}
    c = credit(table, cids, {"C"})
    assert abs(sum(c["shapley"]) - c["v_full"]) < 1e-6 and all(math.isfinite(x) for x in c["shapley"])
    assert value_vector(table, cids, set()) is None and credit(table, cids, set()) is None


def test_traceback_appears_once_in_probability_prompt():
    inst, cands = make_instance(11)
    v = inference_view(inst, cands, "test_informed")
    p = probability_agent_prompt(v, {a: "n" for a in AGENTS}, 31)[0]
    assert p.count("in caller_11") == 1
    v2 = inference_view(inst, cands, "no_trace")
    assert "in caller_11" not in probability_agent_prompt(v2, {a: "n" for a in AGENTS}, 31)[0]


def test_no_trace_condition_query_has_no_trace():
    inst, _ = make_instance(12)
    q, tr = condition_query(inst, "no_trace")
    assert tr is None and inst["trace"] not in q and "caller_12" not in q
    q2, tr2 = condition_query(inst, "test_informed")
    assert tr2 == inst["trace"]


def test_code_truncated_to_token_budget():
    long_code = "def f():\n" + "    x = x + 1\n" * 2000
    out = tokens.truncate_code(long_code)
    assert out.endswith(tokens.TRUNC_MARK)
    assert tokens.n_tokens(out[: -len(tokens.TRUNC_MARK)]) <= tokens.CODE_TOKEN_BUDGET + 2
    assert tokens.truncate_code("def g():\n    return 1") == "def g():\n    return 1"


def test_pools_normalized_and_hedge_weights():
    S = np.stack([pools.normalize(r) for r in np.random.RandomState(1).rand(5, 20)])
    for p in (pools.p_vote(S), pools.p_avg(S), pools.log_pool(S, np.ones(5))):
        assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
    w = pools.hedge_weights(np.log(np.random.RandomState(2).rand(10, 5)), 1.0)
    assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
