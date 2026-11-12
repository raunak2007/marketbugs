import json
from pathlib import Path

from src.agents.views import inference_view
from src.llm import tokens

ROOT = Path(__file__).resolve().parents[1]


def test_under_budget_is_unchanged():
    s = "Failing test: tests/test_x.py::test_y\nAssertionError: 1 != 2"
    assert tokens.truncate_issue(s) == s
    assert tokens.truncate_issue(None) is None


def test_over_budget_keeps_head_and_tail():
    s = "HEAD-MARK " + "x, " * 20000 + " TAIL: AssertionError: boom"
    out = tokens.truncate_issue(s, budget=512, head=128)
    assert out.startswith("HEAD-MARK") and out.endswith("AssertionError: boom")
    assert "tokens truncated" in out
    assert tokens.n_tokens(out) <= 512 + 20


def test_dev_cap_changes_only_the_two_outliers():
    changed = []
    for p in sorted((ROOT / "data/marketbugs/dev").glob("*.jsonl")):
        if p.name == "eval_only.jsonl":
            continue
        for l in open(p):
            r = json.loads(l)
            for cond, raw in r["conditions"].items():
                v = inference_view(r, {"pool": [], "candidate_order": []}, cond)
                if v["issue_text"] != raw:
                    changed.append(r["id"])
                    assert tokens.n_tokens(v["issue_text"]) <= tokens.ISSUE_TOKEN_BUDGET + 20
    assert sorted(set(changed)) == ["mb-cachetools-0034", "mb-sortedcontainers-0038"]
