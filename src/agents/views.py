"""Inference views. The ONLY path from instance/candidate rows to prompt builders.

Whitelist, not blacklist: anything not listed here is invisible to inference code (PLAN 0.3, 4.4).
"""
from src.llm.tokens import truncate_issue

INSTANCE_ALLOWED = ("id", "source", "repo", "issue_text", "trace", "failing_tests")
CANDIDATE_ALLOWED = ("cid", "path", "qualname", "code", "retrieval_source", "retrieval_score")
FORBIDDEN_PREFIXES = ("gold_", "mutation_")
FORBIDDEN_KEYS = ("gold_in_pool", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "generation_seed")


def inference_view(instance, candidates, condition=None):
    v = {k: instance.get(k) for k in INSTANCE_ALLOWED}
    if instance["source"] == "marketbugs":
        cond = condition or "test_informed"
        v["issue_text"] = instance["conditions"][cond]
        if cond == "no_trace":
            v["trace"] = None
    else:  # SWE-bench: issue-only (problem_statement); never failing tests
        v["trace"], v["failing_tests"] = None, None
    v["issue_text"] = truncate_issue(v["issue_text"])  # configs/models.yaml prompting.issue_token_budget
    by_cid = {c["cid"]: c for c in candidates["pool"]}
    v["pool"] = [{k: by_cid[cid][k] for k in CANDIDATE_ALLOWED} for cid in candidates["candidate_order"]]
    return v
