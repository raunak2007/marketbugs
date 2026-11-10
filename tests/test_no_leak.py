"""PLAN 4.4: built prompts must not contain evaluator-only strings, except exact spans that are
legitimately present in permitted input fields."""
import json, pytest
from src.agents.views import inference_view, INSTANCE_ALLOWED
from src.agents.prompts import probability_agent_prompt, AGENTS
from tests.fixtures import make_instance

EVAL_FIELDS = ("gold_functions", "mutation_operator", "mutation_span", "patch", "test_patch", "FAIL_TO_PASS", "generation_seed")


def forbidden_strings(inst):
    out = []
    for f in EVAL_FIELDS:
        val = inst.get(f)
        if val is None:
            continue
        items = val if isinstance(val, list) else [val]
        if f == "mutation_span":
            items = [json.dumps(val)]
        for it in items:
            s = str(it)
            out.append(s)
            if f == "gold_functions":
                out.append(s.rsplit(".", 1)[-1])  # bare function name too
    return [s for s in out if len(s) >= 4]


def permitted_text(inst):
    parts = [str(inst.get(k) or "") for k in INSTANCE_ALLOWED]
    parts += list((inst.get("conditions") or {}).values())
    return "\n".join(parts)


def all_prompts(inst, cands):
    conds = ["test_informed", "no_trace"] if inst["source"] == "marketbugs" else [None]
    for c in conds:
        view = inference_view(inst, cands, c)
        notes = {a: f"note from {a}" for a in AGENTS}
        for mask in range(32):
            yield probability_agent_prompt(view, notes, mask)[0]
    # VERIFY: extend with specialist / forecaster / market builders as they are added.


@pytest.mark.parametrize("source", ["marketbugs", "swebench_lite"])
def test_no_leak_50_instances(source):
    for i in range(50):
        inst, cands = make_instance(i, source)
        allowed = permitted_text(inst)
        for p in all_prompts(inst, cands):
            for s in forbidden_strings(inst):
                if s in p and s not in allowed:
                    raise AssertionError(f"leak of {s!r} in prompt for {inst['id']}")


def test_detector_catches_planted_leak():
    inst, cands = make_instance(0)
    cands["pool"][0]["code"] += "\n# " + inst["gold_functions"][0]
    view = inference_view(inst, cands)
    p = probability_agent_prompt(view, {}, 0)[0]
    assert any(s in p and s not in permitted_text(inst) for s in forbidden_strings(inst))


def test_swebench_view_drops_tests_and_trace():
    inst, cands = make_instance(1, "swebench_lite")
    inst["failing_tests"] = ["SHOULD_NOT_APPEAR"]
    v = inference_view(inst, cands)
    assert v["failing_tests"] is None and v["trace"] is None


def test_view_whitelist_excludes_eval_keys():
    inst, cands = make_instance(2)
    v = inference_view(inst, cands)
    assert not any(k.startswith(("gold_", "mutation_")) for k in v)
    assert all(set(c) <= {"cid", "path", "qualname", "code", "retrieval_source", "retrieval_score"} for c in v["pool"])
