"""SWE-bench Lite: stratified sample + reference-patch function localization labels (PLAN 5.2).

Label name: "reference-patch function localization" (not root cause).
Inference input: problem_statement only. patch/test_patch/FAIL_TO_PASS are evaluator-only and are
kept in a separate file (data/swebench_lite/eval_only.jsonl) that inference code never opens.
"""
import json, random, subprocess, shutil, sys, tempfile, collections
from pathlib import Path
import pyarrow.parquet as pq
from unidiff import PatchSet
from src.cands.functions import index_repo, enclosing

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/swebench_lite/raw"
OUT = ROOT / "data/swebench_lite"
MIRRORS = ROOT / "data/swebench_lite/repos"
DATASET_REV = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
SAMPLE_SEED, N_SAMPLE = 20261110, 100


def stratified_sample(rows, n, seed):
    """Proportional allocation by repo (largest remainder, >=1 per repo), seeded draw within repo."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["repo"]].append(r)
    tot = len(rows)
    quota = {k: max(1, n * len(v) // tot) for k, v in by.items()}
    rema = sorted(by, key=lambda k: -(n * len(by[k]) / tot - n * len(by[k]) // tot))
    i = 0
    while sum(quota.values()) < n:
        quota[rema[i % len(rema)]] += 1; i += 1
    while sum(quota.values()) > n:
        k = max(quota, key=lambda k: quota[k]); quota[k] -= 1
    rng = random.Random(seed)
    out = []
    for k in sorted(by):
        ids = sorted(by[k], key=lambda r: r["instance_id"])
        out += rng.sample(ids, quota[k])
    return sorted(out, key=lambda r: r["instance_id"]), quota


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, shell=True, check=True, capture_output=True, text=True).stdout


def checkout(repo, commit):
    """Blobless mirror + git archive of base_commit into a temp dir. No code is executed."""
    m = MIRRORS / repo.replace("/", "__")
    if not m.exists():
        MIRRORS.mkdir(parents=True, exist_ok=True)
        sh(f"git clone -q --bare --filter=blob:none https://github.com/{repo}.git {m}")
    d = Path(tempfile.mkdtemp(prefix="swe_"))
    sh(f"git --git-dir={m} archive {commit} | tar -x -C {d}")
    return d


def gold_functions(patch_text, fns):
    """Map hunks to enclosing functions using PRE-patch line ranges.
    Returns (keys, files, reason) where reason flags no-function / parse failures."""
    try:
        ps = PatchSet(patch_text)
    except Exception as e:
        return [], [], f"parse_failure:{type(e).__name__}"
    keys, files = [], []
    for pf in ps:
        if not pf.source_file.endswith(".py") and not pf.path.endswith(".py"):
            files.append(pf.path); continue
        files.append(pf.path)
        for h in pf:
            lines = [l.source_line_no for l in h if l.is_removed]
            if not lines:  # pure insertion: anchor on the context line before the first added line
                prev = None
                for l in h:
                    if l.is_added:
                        break
                    if l.source_line_no:
                        prev = l.source_line_no
                lines = [prev or h.source_start]
            for ln in lines:
                f = enclosing(fns, pf.path, ln)
                if f and f.key not in keys:
                    keys.append(f.key)
    return keys, files, None if keys else "no_function"


def main(limit=None):
    rows = pq.read_table(RAW / "test.parquet").to_pylist()
    sample, quota = stratified_sample(rows, N_SAMPLE, SAMPLE_SEED)
    (OUT / "sample.json").write_text(json.dumps({"seed": SAMPLE_SEED, "dataset_rev": DATASET_REV, "quota": quota,
                                                  "ids": [r["instance_id"] for r in sample]}, indent=1))
    log = collections.Counter()
    inst_f, eval_f = open(OUT / "instances.jsonl", "w"), open(OUT / "eval_only.jsonl", "w")
    for r in sample[:limit]:
        d = None
        try:
            d = checkout(r["repo"], r["base_commit"])
            fns = index_repo(d)
            keys, files, why = gold_functions(r["patch"], fns)
        except subprocess.CalledProcessError as e:
            keys, files, why = [], [], "checkout_failure"
        finally:
            if d:
                shutil.rmtree(d, ignore_errors=True)
        bucket = why or ("multi_function" if len(keys) > 1 else "single_function")
        log[bucket] += 1
        inst_f.write(json.dumps({
            "id": "swe-" + r["instance_id"], "source": "swebench_lite", "repo": r["repo"], "repo_license": "see upstream",
            "base_commit": r["base_commit"], "split": "test", "issue_text": r["problem_statement"], "trace": None,
            "conditions": None, "failing_tests": None, "gold_functions": keys, "mutation_operator": None,
            "mutation_span": None, "stratum": "na", "generation_seed": SAMPLE_SEED,
            "validation_status": "ok" if keys else why, "test_environment": None}) + "\n")
        eval_f.write(json.dumps({"id": "swe-" + r["instance_id"], "gold_files": files, "bucket": bucket,
                                 "patch": r["patch"], "test_patch": r["test_patch"], "FAIL_TO_PASS": r["FAIL_TO_PASS"]}) + "\n")
        print(r["instance_id"], bucket, len(keys), flush=True)
    (OUT / "build_log.json").write_text(json.dumps(log, indent=1))
    print(dict(log))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
