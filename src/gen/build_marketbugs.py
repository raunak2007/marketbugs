"""MarketBugs builder (PLAN 5.1). RUNS EACH REPO'S TEST SUITE -> run locally (CPU), not on Modal.

python -m src.gen.build_marketbugs <repo_name> [--cap N]
Requires configs/repos.yaml (approved by human) and data/marketbugs/repos/<name>/.venv (from src.gen.eval_repos).
Validity filters: original suite passes; mutant fails >=1 test; failing set identical over 3 runs
(runs 2-3 re-run only the failing tests: VERIFY deviation noted in BLOCKERS); exactly one function changed;
gold >= 3 statements; dedupe by (function, operator, span).
"""
import ast, json, random, re, subprocess, sys, hashlib
from pathlib import Path
import yaml
from src.gen.mutate import candidates_in_function, apply, is_valid_mutant
from src.cands.functions import index_repo, trace_frames, enclosing
from src.cands.retrieval import Graph

ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT / "data/marketbugs/repos"
FAILED_RE = re.compile(r"^FAILED (\S+?)(?: - (.*))?$", re.M)
MIN_STATEMENTS = 3


def pytest(d, args, timeout=180):
    cmd = [str(d / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", "--tb=native", "-rf", *args]
    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", ""


def failing(out):
    return sorted({m.group(1) for m in FAILED_RE.finditer(out)})


def first_failure_block(out, test_id):
    """The native traceback and assertion message for one failing test from -q --tb=native output."""
    name = test_id.split("::")[-1]
    blocks = re.split(r"\n_{3,} (.+?) _{3,}\n", out)
    for i in range(1, len(blocks) - 1, 2):
        if name in blocks[i]:
            body = blocks[i + 1].split("\n= ")[0].strip()
            tb_lines = [l for l in body.splitlines()]
            msg = next((l for l in reversed(tb_lines) if re.match(r"^\w*(Error|Exception)\b", l)), tb_lines[-1] if tb_lines else "")
            return body, msg
    return "", ""


def stratum(fns, graph, trace, gold_key, repo_root):
    frame_fns = []
    for p, ln, _ in trace_frames(trace, repo_root):
        f = enclosing(fns, p, ln)
        if f:
            frame_fns.append(f.key)
    if gold_key in frame_fns:
        return "direct"
    if any(gold_key in graph.neighbors(k) for k in frame_fns):
        return "near"
    return "far"


def build(name, cfg, cap, seed=123):
    d = REPOS / name
    code, out = pytest(d, [], timeout=300)
    if code != 0:
        raise SystemExit(f"{name}: original suite does not pass ({code})")
    fns = index_repo(d)
    graph = Graph(fns)
    rng = random.Random(seed)
    pool = []
    for f in fns:
        if f.n_statements < MIN_STATEMENTS:
            continue
        src_path = d / f.path
        src = src_path.read_text(encoding="utf-8")
        node = next((n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.lineno == f.lineno), None)
        if node is None:
            continue
        for m in candidates_in_function(src, node, rng):
            pool.append((f, m))
    rng.shuffle(pool)
    out_rows, seen, tried, log = [], set(), 0, {"no_fail": 0, "flaky": 0, "timeout": 0, "invalid": 0, "multi_fn": 0}
    for f, m in pool:
        if len(out_rows) >= cap:
            break
        key = (f.key, m.operator, m.span)
        if key in seen:
            continue
        seen.add(key); tried += 1
        p = d / f.path
        orig = p.read_text(encoding="utf-8")
        new = apply(orig, m)
        if not is_valid_mutant(orig, new):
            log["invalid"] += 1; continue
        if enclosing(fns, f.path, m.span[0]).key != f.key or enclosing(fns, f.path, m.span[2]).key != f.key:
            log["multi_fn"] += 1; continue
        try:
            p.write_text(new, encoding="utf-8")
            c1, o1 = pytest(d, ["-x"], timeout=cfg.get("timeout", 180))
            if c1 == "timeout":
                log["timeout"] += 1; continue
            fail1 = failing(o1)
            if not fail1:
                log["no_fail"] += 1; continue
            reps = [failing(pytest(d, fail1)[1]) for _ in range(2)]
            if any(r != fail1 for r in reps):
                log["flaky"] += 1; continue
            _, o_full = pytest(d, fail1)
        finally:
            p.write_text(orig, encoding="utf-8")
        t0 = fail1[0]
        tb, msg = first_failure_block(o_full, t0)
        test_informed = f"Failing test: {t0}\n{msg}\n\n{tb}"
        no_trace = f"Failing test: {t0}\n{msg}"
        iid = f"mb-{name}-{len(out_rows):04d}"
        out_rows.append({
            "id": iid, "source": "marketbugs", "repo": name, "repo_license": cfg["license"], "base_commit": cfg["commit"],
            "split": cfg["split"], "issue_text": test_informed, "trace": tb,
            "conditions": {"test_informed": test_informed, "no_trace": no_trace}, "failing_tests": fail1,
            "gold_functions": [f.key], "mutation_operator": m.operator, "mutation_span": list(m.span),
            "stratum": stratum(fns, graph, tb, f.key, d), "generation_seed": seed, "validation_status": "ok",
            "test_environment": "py3.11 + " + hashlib.sha256((d / ".venv/pyvenv.cfg").read_bytes()).hexdigest()[:12]})
        # evaluator-only patch, kept apart from instances
        with open(ROOT / f"data/marketbugs/{cfg['split']}/eval_only.jsonl", "a") as ef:
            ef.write(json.dumps({"id": iid, "path": f.path, "old": m.old, "new": m.new}) + "\n")
        print(iid, m.operator, f.qualname, out_rows[-1]["stratum"], flush=True)
    log.update(tried=tried, valid=len(out_rows))
    return out_rows, log


def main():
    name = sys.argv[1]
    cfg = yaml.safe_load((ROOT / "configs/repos.yaml").read_text())["repos"][name]
    cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else (40 if cfg["split"] == "dev" else 30)
    rows, log = build(name, cfg, cap)
    outp = ROOT / f"data/marketbugs/{cfg['split']}/{name}.jsonl"
    outp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ROOT / f"artifacts/gen_log_{name}.json").write_text(json.dumps(log, indent=1))
    print(name, log)


if __name__ == "__main__":
    main()
