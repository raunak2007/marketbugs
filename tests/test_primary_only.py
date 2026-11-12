"""--primary-only computes a subset of the full C-mode table: every computed row must equal the full-table row, and
the primary methods (P_hedge, B_full5) plus context methods must come out identical. Deterministic fakes, no GPU."""
import hashlib, json
from pathlib import Path

import numpy as np
import pytest

from src.agents.pilot import load_repo
from src.agents.pipeline import run_instance
from src.agents.run_test import PRIMARY_MASKS
from src.analysis.methods import method_dists
from tests.test_market import FakeSrv

ROOT = Path(__file__).resolve().parents[1]


class Srv(FakeSrv):
    def generate(self, prompts, **kw):
        return [{"text": "Suspect: " + hashlib.sha256(p.encode()).hexdigest()[:12], "input_tokens": len(p) // 4,
                 "output_tokens": 5, "latency_ms": 1.0, "model_revision": "fake@0"} for p in prompts]


def emb(texts):
    return np.array([np.random.RandomState(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).randn(16)
                     for t in texts])


@pytest.mark.skipif(not (ROOT / "data/marketbugs/repos/cachetools").exists(), reason="dev repo not checked out")
def test_primary_only_rows_equal_full_table_rows():
    inst = json.loads(open(ROOT / "data/marketbugs/dev/cachetools.jsonl").readline())
    repo = load_repo("cachetools", emb)
    full = run_instance(inst, "test_informed", repo, emb, Srv(), extras=False, perms=(0, 1), mode_s=False)
    prim = run_instance(inst, "test_informed", repo, emb, Srv(), extras=False, perms=(0, 1), mode_s=False,
                        masks=PRIMARY_MASKS)
    assert prim["notes"] == full["notes"] and prim["candidates"] == full["candidates"]
    key = lambda r: (r["mode"], r["subset_mask"], r["order_perm"], tuple(r["note_order"]))
    fk = {key(r): r for r in full["values"]}
    assert {r["subset_mask"] for r in prim["values"]} == set(PRIMARY_MASKS)
    assert len(prim["values"]) == 2 * (len(PRIMARY_MASKS) + 4)  # 2 perms x (7 masks + 4 extra rotations)
    for r in prim["values"]:
        f = fk[key(r)]
        assert r["prompt_hash"] == f["prompt_hash"] and r["probs"] == f["probs"]
    n = full["candidates"]["pool_size"]
    w = [0.38, 0.11, 0.15, 0.12, 0.24]
    Df, _ = method_dists(full["values"], n, (0, 1), hedge_w=w, with_s=False, what="full")
    Dp, _ = method_dists(prim["values"], n, (0, 1), hedge_w=w, with_s=False, what="prim")
    for m in ("p_empty", "p_full", "B_full5", "P_avg", "P_hedge_eq", "P_hedge"):
        np.testing.assert_array_equal(Dp[m], Df[m])


def test_mode_s_requires_mask_zero():
    with pytest.raises(ValueError):
        inst = json.loads(open(ROOT / "data/marketbugs/dev/cachetools.jsonl").readline())
        run_instance(inst, "test_informed", load_repo("cachetools", emb), emb, Srv(), extras=False, perms=(0,),
                     mode_s=True, masks=(1, 31))
