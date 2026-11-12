"""Two-stage repository-cluster bootstrap for paired Top-1 differences (protocol v1.0 section 4).

python -m src.analysis.bootstrap_test --input artifacts/validation/heldout_v1/per_instance.csv \
    --seed 20261110 --replicates 10000 --output artifacts/validation/heldout_v1/confidence_intervals.json

Per replicate: (1) sample the repositories with replacement; (2) for each sampled repository COPY, resample its
paired instance records with replacement (same count as the repo has); (3) the statistic is the mean paired
difference over all sampled records. 95% percentile CI = 2.5 / 97.5 percentiles. A paired instance-level bootstrap
CI is reported alongside for diagnostics only; it never replaces the cluster CI. A CI spanning zero is
"inconclusive", never "equivalent". Only the primary contrast is confirmatory.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

CONTRASTS = {"primary": ("P_hedge", "B_full5", "confirmatory (preregistered)"),
             "real_vs_no_notes": ("p_full", "p_empty", "exploratory"),
             "real_vs_placebo": ("p_full", "placebo_full", "exploratory")}


def cluster_bootstrap(groups, replicates, seed):
    """groups: list of 1-D arrays (one per repository). Returns the replicate means (length `replicates`)."""
    rng = np.random.default_rng(seed)
    G = len(groups)
    out = np.empty(replicates)
    for b in range(replicates):
        picks = rng.integers(0, G, G)
        tot, cnt = 0.0, 0
        for g in picks:
            x = groups[g]
            s = x[rng.integers(0, len(x), len(x))]
            tot += s.sum(); cnt += len(s)
        out[b] = tot / cnt
    return out


def instance_bootstrap(x, replicates, seed):
    rng = np.random.default_rng(seed + 1)
    return x[rng.integers(0, len(x), (replicates, len(x)))].mean(1)


def verdict(lo, hi, label):
    if lo > 0:
        v = "positive (CI excludes 0)"
    elif hi < 0:
        v = "negative (CI excludes 0)"
    else:
        v = "inconclusive (CI spans 0)"
    return v if label.startswith("confirmatory") else v + "; exploratory"


def run(rows, seed, replicates):
    res = {}
    for nm, (a, b, label) in CONTRASTS.items():
        col = f"diff.{nm}"
        if not rows or col not in rows[0] or rows[0][col] in ("", None):
            res[nm] = {"omitted": f"column {col} not present"}; continue
        by = defaultdict(list)
        for r in rows:
            by[r["repo"]].append(float(r[col]))
        repos = sorted(by)
        groups = [np.array(by[r]) for r in repos]
        allx = np.concatenate(groups)
        cb = cluster_bootstrap(groups, replicates, seed)
        ib = instance_bootstrap(allx, replicates, seed)
        lo, hi = (float(x) for x in np.percentile(cb, [2.5, 97.5]))
        res[nm] = {"contrast": f"{a} - {b} (Top-1)", "label": label, "observed": float(allx.mean()), "n": int(len(allx)),
                   "n_repos": len(repos), "cluster_ci95": [lo, hi], "verdict": verdict(lo, hi, label),
                   "instance_ci95_diagnostic_only": [float(x) for x in np.percentile(ib, [2.5, 97.5])],
                   "per_repo": {r: {"n": len(by[r]), "diff": float(np.mean(by[r]))} for r in repos},
                   "wins": int((allx > 0).sum()), "losses": int((allx < 0).sum()), "ties": int((allx == 0).sum())}
    return {"seed": seed, "replicates": replicates, "method": "two-stage repository-cluster percentile bootstrap",
            "contrasts": res}


def _md(res):
    L = ["## Paired Top-1 differences with 95% repository-cluster bootstrap CIs", "",
         f"{res['replicates']} replicates, seed {res['seed']}. Instance-level CI is diagnostic only.", "",
         "| contrast | status | observed | cluster 95% CI | instance CI (diag.) | verdict |", "|---|---|---|---|---|---|"]
    for nm, v in res["contrasts"].items():
        if "omitted" in v:
            L.append(f"| {nm} | omitted | {v['omitted']} | | | |"); continue
        c, i = v["cluster_ci95"], v["instance_ci95_diagnostic_only"]
        L.append(f"| {v['contrast']} | {v['label']} | {v['observed']:+.4f} | [{c[0]:+.4f}, {c[1]:+.4f}] | "
                 f"[{i[0]:+.4f}, {i[1]:+.4f}] | {v['verdict']} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.input)))
    res = run(rows, a.seed, a.replicates)
    Path(a.output).write_text(json.dumps(res, indent=1) + "\n")
    md = Path(a.input).parent / "RESULTS.md"
    if md.exists():
        t = md.read_text()
        s, e = "<!-- BOOTSTRAP:BEGIN -->", "<!-- BOOTSTRAP:END -->"
        if s in t and e in t:
            md.write_text(t[:t.index(s) + len(s)] + "\n" + _md(res) + "\n" + t[t.index(e):])
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
