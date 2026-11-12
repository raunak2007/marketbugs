"""G2 freeze: fit Hedge eta/weights and per-method temperatures on the COMPLETE dev run; write
artifacts/frozen_params.json and configs/FROZEN.txt (protocol v1.0 section 4). EVALUATOR SIDE, DEV ONLY.

python -m src.analysis.freeze_dev --tags g2_cachetools g2_intervaltree g2_sortedcontainers \
    --protocol avg2 --output artifacts/frozen_params.json [--cond test_informed]

Stops (exit != 0, nothing written) if: dev coverage is not exactly the 120 dev instances; any id is duplicated
across tags; any instance lacks a row for any required (mask, note order, permutation); rows mix model revisions or
token mappings; any row is schema-invalid; or src/configs/schemas have uncommitted changes.
eta: existing 2-fold dev CV (src.market.pools.fit_eta). Weights: Hedge on dev gold-in-pool instances only.
Temperatures: one per method by dev NLL (gold-in-pool only). Test data is never read (only id/repo/split for the
split-manifest hash).
"""
import argparse, json, math, platform, sys, time
from importlib import metadata
from pathlib import Path
import numpy as np
import yaml
from src import frozen
from src.analysis.methods import AGENTS, PROTOCOL_PERMS, method_dists, MethodDataError
from src.analysis.score import gold_cids, score_row
from src.market import pools
from src.market.placebo import PLACEBO_SEED
from src.schemas.validate import validate_row

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SEED = 20261110


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def dev_instances(root=ROOT):
    out = {}
    for p in sorted((root / "data/marketbugs/dev").glob("*.jsonl")):
        if p.name != "eval_only.jsonl":
            for l in open(p):
                r = json.loads(l)
                if r["split"] != "dev":
                    raise SystemExit(f"non-dev row in dev file: {r['id']}")
                out[r["id"]] = r
    return out


def _versions():
    v = {"python": platform.python_version()}
    for pkg in ("numpy", "jsonschema", "torch", "transformers", "sentence-transformers", "modal"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return v


def collect(tags, cond, protocol, root=ROOT):
    """Load + check the dev run. Returns (cands, values_by_id, problems)."""
    perms = PROTOCOL_PERMS[protocol]
    inst = dev_instances(root)
    cands, vals, problems = {}, {}, []
    for t in tags:
        d = root / "artifacts/runs" / t / cond
        if not (d / "candidates_inf.jsonl").exists():
            problems.append(f"{d}: no candidates_inf.jsonl"); continue
        for c in jl(d / "candidates_inf.jsonl"):
            if c["id"] in cands:
                problems.append(f"duplicate instance {c['id']} (tag {t})")
            cands[c["id"]] = c
        for v in jl(d / "values_inf.jsonl"):
            vals.setdefault(v["id"], []).append(v)
    missing, extra = sorted(set(inst) - set(cands)), sorted(set(cands) - set(inst))
    if missing:
        problems.append(f"dev run incomplete: {len(missing)} of {len(inst)} dev instances missing, e.g. {missing[:3]}")
    if extra:
        problems.append(f"non-dev ids in run: {extra[:3]}")
    revs, maps = set(), set()
    for i, c in cands.items():
        if i not in inst:
            continue
        g = set(gold_cids(inst[i], c))
        for v in vals.get(i, []):
            revs.add(v["model_revision"]); maps.add(v["token_mapping_hash"])
            errs = validate_row("value_row", score_row(dict(v), g))
            if errs:
                problems.append(f"{i}: invalid value row {errs[:1]}")
        if (cr := validate_row("candidates", dict(c, gold_in_pool=bool(g)))):
            problems.append(f"{i}: invalid candidates row {cr[:1]}")
        try:  # every required (mask, order) x protocol permutation present => not a single-permutation run
            method_dists(vals.get(i, []), c["pool_size"], perms, with_s=False, what=i)
        except MethodDataError as e:
            problems.append(f"{i}: {e} (mixed/incomplete protocol?)")
    if len(revs) > 1 or len(maps) > 1:
        problems.append(f"mixed model revisions {revs} / token mappings {maps}")
    return inst, cands, vals, revs, maps, problems


def fit(inst, cands, vals, protocol):
    perms = PROTOCOL_PERMS[protocol]
    ids = sorted(cands)
    S_all, D_all, gidx = {}, {}, {}
    for i in ids:
        D, S = method_dists(vals[i], cands[i]["pool_size"], perms, with_s=True, what=i)
        D_all[i], S_all[i] = D, S
        cids = sorted(c["cid"] for c in cands[i]["pool"])
        g = gold_cids(inst[i], cands[i])
        if g:
            gidx[i] = cids.index(sorted(g)[0])
    gip = [i for i in ids if i in gidx]
    eta, eta_cv = pools.fit_eta([S_all[i] for i in gip], [gidx[i] for i in gip])
    w = pools.hedge_weights([[math.log(max(S_all[i][b][gidx[i]], 1e-12)) for b in range(5)] for i in gip], eta)
    for i in ids:
        D_all[i]["P_hedge"] = pools.log_pool(S_all[i], w)
    temps = {}
    methods = sorted({m for D in D_all.values() for m in D})
    for m in methods:
        have = [i for i in gip if m in D_all[i]]
        if len(have) == len(gip):
            temps[m] = pools.fit_temperature([D_all[i][m] for i in have], [gidx[i] for i in have])[0]
    temps["placebo_full"] = temps["p_full"]  # same prompt family; placebo is never fit (no dev placebo arm)
    return {"eta": eta, "eta_cv_nll": {str(k): v for k, v in eta_cv.items()},
            "hedge_weights": {a: float(x) for a, x in zip(AGENTS, w)}, "temperatures": temps,
            "n_dev": len(ids), "n_dev_gold_in_pool": len(gip)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--protocol", default="avg2", choices=sorted(PROTOCOL_PERMS))
    ap.add_argument("--cond", default="test_informed")
    ap.add_argument("--output", default=str(frozen.FROZEN_JSON))
    ap.add_argument("--frozen-txt", default=str(frozen.FROZEN_TXT))
    ap.add_argument("--allow-dirty", action="store_true", help="tests only: skip the clean-tree requirement")
    a = ap.parse_args(argv)
    if a.cond != "test_informed":
        sys.exit("primary condition is test_informed")
    if not a.allow_dirty and not frozen.tree_clean():
        sys.exit("refusing to freeze: uncommitted changes under src/configs/schemas (commit code first)")
    inst, cands, vals, revs, maps, problems = collect(a.tags, a.cond, a.protocol)
    if problems:
        print("FREEZE REFUSED:", *problems[:20], sep="\n  ")
        sys.exit(f"{len(problems)} problem(s); nothing written")
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["probability_agent"]
    rev = f"{cfg['hf_id']}@{cfg['revision']}"
    if revs != {rev}:
        sys.exit(f"dev rows model revision {revs} != configs/models.yaml {rev}")
    f = fit(inst, cands, vals, a.protocol)
    ch = frozen.config_hashes()
    params = {"protocol_version": "1.0", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "code_commit": frozen.head_sha(), "condition": a.cond, "protocol": a.protocol,
              "candidate_perms": list(PROTOCOL_PERMS[a.protocol]), "dev_tags": a.tags,
              "model": {"hf_id": cfg["hf_id"], "revision": cfg["revision"], "revision_full": rev,
                        "dtype": cfg.get("dtype", "bfloat16"), "token_mapping_hash": next(iter(maps))},
              "generation": {"temperature": 0.0, "seed": 0, "max_new_tokens": 150},
              "config_hashes": ch, "config_hashes_sha256": frozen.sha256_bytes(json.dumps(ch, sort_keys=True).encode()),
              "split_manifest_sha256": frozen.split_manifest_hash(), "placebo_seed": PLACEBO_SEED,
              "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": 10000, "package_versions": _versions(), **f}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=1, sort_keys=True) + "\n")
    Path(a.frozen_txt).write_text(frozen.render_frozen_txt(params, frozen.sha256_file(out)))
    print(f"frozen: eta={f['eta']} weights={f['hedge_weights']} n_dev={f['n_dev']} -> {out}, {a.frozen_txt}")
    print("next: commit configs/FROZEN.txt + artifacts/frozen_params.json BEFORE any test inference")


if __name__ == "__main__":
    main()
