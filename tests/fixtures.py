"""Synthetic instances with planted gold strings for leak testing."""
import random


def make_instance(i, source="marketbugs"):
    gold = f"pkg.mod_{i}.secret_gold_fn_{i}"
    inst = {"id": f"mb-fake-{i:04d}", "source": source, "repo": "fake", "repo_license": "MIT", "base_commit": "deadbeef",
            "split": "dev", "issue_text": f"test_thing_{i} fails", "trace": f"File mod_{i}.py, in caller_{i}",
            "conditions": {"test_informed": f"FAILED test_thing_{i}: assert 3 == 4\nFile mod_{i}.py, in caller_{i}",
                           "no_trace": f"FAILED test_thing_{i}: assert 3 == 4"},
            "failing_tests": [f"tests/test_x.py::test_thing_{i}"], "gold_functions": [gold],
            "mutation_operator": f"MUTOP_LEAK_{i}", "mutation_span": [10 + i, 4, 10 + i, 9], "stratum": "near",
            "generation_seed": 4242 + i, "validation_status": "ok", "test_environment": "py3.11",
            "patch": f"PATCH_LEAK_{i}", "test_patch": f"TESTPATCH_LEAK_{i}", "FAIL_TO_PASS": [f"F2P_LEAK_{i}"]}
    if source == "swebench_lite":
        inst.update(conditions=None, failing_tests=None, trace=None, stratum="na")
    # Pool deliberately shows the gold function's code but under a neutral qualname; the gold
    # qualname string itself must not appear (pool qualnames are the real ones in production; the
    # leak test allow-lists exact spans from permitted fields).
    pool = [{"cid": chr(65 + k), "path": f"pkg/m{k}.py", "qualname": f"cand_{i}_{k}", "code": f"def cand_{i}_{k}():\n    return {k}",
             "retrieval_source": "bm25", "retrieval_score": 1.0 / (k + 1)} for k in range(20)]
    order = [c["cid"] for c in pool]; random.Random(i).shuffle(order)
    cands = {"id": inst["id"], "pool_size": 20, "gold_in_pool": True, "pool": pool, "candidate_order": order, "order_seed": i}
    return inst, cands
