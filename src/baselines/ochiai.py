"""B3: Ochiai spectrum-based fault localization (PLAN 7). MarketBugs only. RUNS EACH REPO'S TEST SUITE (local CPU).

python -m src.baselines.ochiai dev [repo ...]

For each instance: apply the mutant (the buggy program -- what a developer would have), run the FULL suite once
under coverage with a per-test context (pytest plugin `src/baselines/pytest_plugin/ochiai_spectra.py`, alone on PYTHONPATH), then score every non-test
function: ochiai = ef / sqrt(F * (ef + ep)), F = #failing tests, ef/ep = failing/passing tests covering any body
line. Ties broken by a seeded shuffle (seed = sha(instance id)). Repo file is always restored.
Outputs artifacts/baselines/b3_ochiai_<split>_<repo>.jsonl with the top-100 ranking (rankings only) and, in a
separate evaluator file, gold ranks.
"""
import json, math, os, random, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path
import yaml
from src.cands.functions import index_repo
from src.gen.mutate import Mutation, apply
from src.llm.hashing import sha

ROOT = Path(__file__).resolve().parents[2]
MB = ROOT / "data/marketbugs"
OUT = ROOT / "artifacts/baselines"


def spectra(d, extra_args, timeout):
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src/baselines/pytest_plugin"), OCHIAI_OUT=td, OCHIAI_SOURCE=str(d))
        cmd = [str(d / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
               "-p", "ochiai_spectra", "--tb=no", *extra_args]
        try:
            subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        p = Path(td) / "spectra.json"
        return json.loads(p.read_text()) if p.exists() else None


def rank(fns, spec, seed):
    outcome = spec["outcomes"]  # nodeid -> "passed"/"failed"
    F = sum(v == "failed" for v in outcome.values())
    cov = defaultdict(set)  # fn key -> set(nodeid)
    by_path = defaultdict(list)
    for f in fns:
        by_path[f.path].append(f)
    for rel, lines in spec["lines"].items():
        for f in by_path.get(rel, ()):
            for ln in range(f.lineno + 1, f.end_lineno + 1):
                cov[f.key] |= set(lines.get(str(ln), ()))
    scored = []
    for f in fns:
        ts = cov.get(f.key, set())
        ef = sum(outcome.get(t) == "failed" for t in ts); ep = sum(outcome.get(t) == "passed" for t in ts)
        s = ef / math.sqrt(F * (ef + ep)) if F and (ef + ep) else 0.0
        scored.append((s, f.key))
    rng = random.Random(seed); rng.shuffle(scored)
    scored.sort(key=lambda x: -x[0])  # stable: shuffle order breaks ties
    return F, scored


def main(split, names):
    cfgs = yaml.safe_load((ROOT / "configs/repos.yaml").read_text())["repos"]
    patches = {json.loads(l)["id"]: json.loads(l) for l in open(MB / split / "eval_only.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    names = names or [n for n, c in cfgs.items() if c["split"] == split]
    for name in names:
        cfg, d = cfgs[name], MB / "repos" / name
        fns = index_repo(d)
        rows = [json.loads(l) for l in open(MB / split / f"{name}.jsonl")]
        ranks, evals = [], []
        for r in rows:
            pt = patches[r["id"]]; p = d / pt["path"]; orig = p.read_text(encoding="utf-8")
            try:
                p.write_text(apply(orig, Mutation("", tuple(r["mutation_span"]), pt["old"], pt["new"])), encoding="utf-8")
                spec = spectra(d, cfg.get("pytest_args", []), max(300, cfg.get("timeout", 180) * 3))
            finally:
                p.write_text(orig, encoding="utf-8")
            if spec is None:
                ranks.append({"id": r["id"], "status": "spectra_failed"}); evals.append({"id": r["id"], "gold_rank": None}); continue
            F, scored = rank(fns, spec, int(sha(r["id"]), 16) % 10**9)
            keys = [k for _, k in scored]
            gold = set(r["gold_functions"])
            gr = next((i + 1 for i, k in enumerate(keys) if k in gold), None)
            ranks.append({"id": r["id"], "status": "ok", "n_failed_tests": F, "ranking": scored[:100]})
            evals.append({"id": r["id"], "gold_rank": gr, "gold_score": next((s for s, k in scored if k in gold), None)})
            print(r["id"], "F=", F, "gold_rank=", gr, flush=True)
        (OUT / f"b3_ochiai_{split}_{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ranks))
        (OUT / f"b3_ochiai_{split}_{name}.eval.jsonl").write_text("".join(json.dumps(x) + "\n" for x in evals))
        ok = [e["gold_rank"] for e in evals if e["gold_rank"] is not None]
        n = len(evals)
        print(name, {"n": n, "spectra_ok": sum(x["status"] == "ok" for x in ranks),
                     "top1": round(sum(g == 1 for g in ok) / n, 3), "top5": round(sum(g <= 5 for g in ok) / n, 3),
                     "mrr": round(sum(1 / g for g in ok) / n, 3)}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
