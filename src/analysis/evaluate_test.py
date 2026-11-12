"""Held-out evaluator (protocol v1.0 section 4). EVALUATOR SIDE: separate process, imports nothing from src.agents /
src.llm / src.cands. Reads inference-only artifacts after ALL inference finished; attaches gold only here.

python -m src.analysis.evaluate_test --tag heldout_v1 --frozen artifacts/frozen_params.json --require-n 210 --audit-only
python -m src.analysis.evaluate_test --tag heldout_v1 --frozen artifacts/frozen_params.json --require-n 210 \
    --output-dir artifacts/validation/heldout_v1 [--allow-partial]

--audit-only never reads gold fields. Hard failures: frozen files inconsistent or different from the run manifest;
run not complete (unless --allow-partial, which labels every output PARTIAL); missing / duplicated / unexpected ids;
any required method without exactly one distribution per instance after aggregation; non-finite / wrong-length
distributions; model revision or token mapping different from the frozen one; schema-invalid rows; placebo
manifest inconsistent with the seeded derangement. Frozen Hedge weights / eta / temperatures are applied as loaded;
nothing is fit here.
"""
import argparse, csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from src import frozen
from src.analysis.methods import (AGENTS, CORE, FULL, PROTOCOL_PERMS, MethodDataError, method_dists, placebo_dist,
                                  index_rows, disagreement, subset)
from src.analysis.score import gold_cids, metrics, ece
from src.market import pools
from src.market.placebo import assign_donors, assignment_hash
from src.schemas.validate import validate_row

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {"primary": ("P_hedge", "B_full5"), "real_vs_no_notes": ("p_full", "p_empty"),
         "real_vs_placebo": ("p_full", "placebo_full")}
PAIR_LABEL = {"primary": "confirmatory (preregistered)", "real_vs_no_notes": "exploratory",
              "real_vs_placebo": "exploratory"}
CHECK_MASKS = (0, 1, 2, 4, 8, 16, 31)
NOTE_MAX_NEW_TOKENS = 150
METRICS = ("top1", "top3", "top5", "rr")


class AuditError(RuntimeError):
    pass


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def test_ids(input_dir):
    """(id, repo, stratum) only -- no gold fields are kept."""
    out = []
    for p in sorted(Path(input_dir).glob("*.jsonl")):
        if p.name == "eval_only.jsonl":
            continue
        for l in open(p):
            r = json.loads(l)
            if r["split"] != "test":
                raise AuditError(f"non-test row {r['id']} in {p}")
            out.append((r["id"], r["repo"], r["stratum"]))
    return out


def test_gold(input_dir):
    out = {}
    for p in sorted(Path(input_dir).glob("*.jsonl")):
        if p.name != "eval_only.jsonl":
            for l in open(p):
                r = json.loads(l); out[r["id"]] = r
    return out


def audit(d, params, kv, expected, require_n, allow_partial=False):
    """Returns dict with per-instance method distributions (sorted-cid order) and audit facts. Gold-free."""
    perms = PROTOCOL_PERMS[params["protocol"]]
    P = []
    man = json.loads((d / "run_manifest.json").read_text()) if (d / "run_manifest.json").exists() else None
    if man is None:
        raise AuditError(f"{d}/run_manifest.json missing")
    if man["frozen_params_sha256"] != kv["frozen_params_sha256"]:
        raise AuditError("run was produced under different frozen params than the current FROZEN.txt")
    if man["protocol"] != params["protocol"] or man["placebo_seed"] != params["placebo_seed"]:
        raise AuditError("run manifest protocol/seed differ from frozen")
    status = json.loads((d / "run_status.json").read_text()) if (d / "run_status.json").exists() else {}
    if not status.get("complete"):
        (P if not allow_partial else []).append(f"run not complete: {status.get('stopped')!r}, real "
                                                f"{status.get('real_complete')}, placebo {status.get('placebo_complete')}")
    exp_ids = [i for i, _, _ in expected]
    repo_of = {i: r for i, r, _ in expected}
    if len(exp_ids) != require_n:
        P.append(f"expected {require_n} test instances in input, found {len(exp_ids)}")
    cl = jl(d / "candidates_inf.jsonl")
    ids = [c["id"] for c in cl]
    if len(ids) != len(set(ids)):
        P.append("duplicate ids in candidates_inf.jsonl")
    cands = {c["id"]: c for c in cl}
    unexpected = sorted(set(cands) - set(exp_ids))
    missing = [i for i in exp_ids if i not in cands]
    if unexpected:
        P.append(f"unexpected ids {unexpected[:3]}")
    if missing and not allow_partial:
        P.append(f"{len(missing)} missing instances, e.g. {missing[:3]}")
    vals, plc, pman, notes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for v in jl(d / "values_inf.jsonl"):
        vals[v["id"]].append(v)
    for v in jl(d / "placebo_values_inf.jsonl"):
        plc[v["id"]].append(v)
    for m in jl(d / "placebo_manifest.jsonl"):
        pman[m["id"]].append(m)
    for n in jl(d / "notes.jsonl"):
        notes[n["id"]].append(n)
    rev, tmh = params["model"]["revision_full"], params["model"]["token_mapping_hash"]
    w = [params["hedge_weights"][a] for a in AGENTS]
    assign = assign_donors([(i, r) for i, r, _ in expected], params["placebo_seed"])
    pa = d / "placebo_assignment.json"
    if pa.exists() and json.loads(pa.read_text())["sha256"] != assignment_hash(assign):
        P.append("stored placebo assignment differs from the seeded derangement")
    want_s, want_plc = man.get("mode_s", True), "placebo" in man.get("arms", [])
    D, ok_ids = {}, []
    for i in [x for x in exp_ids if x in cands]:
        c = cands[i]; n = c["pool_size"]
        if n != 20:
            P.append(f"{i}: pool_size {n} != K=20")
        if (e := validate_row("candidates", dict(c, gold_in_pool=False))):  # placeholder: audit never reads gold
            P.append(f"{i}: candidates schema {e[:1]}")
        for v in vals[i]:
            if v["model_revision"] != rev or v["token_mapping_hash"] != tmh:
                P.append(f"{i}: value row model revision/token mapping != frozen"); break
            if (e := validate_row("value_row", dict(v, gold_logscore=-1.0))):  # placeholder only for the schema check
                P.append(f"{i}: value_row schema {e[:1]}"); break
        if len(notes[i]) != 5 or sorted(x["agent"] for x in notes[i]) != sorted(AGENTS):
            P.append(f"{i}: expected exactly 5 specialist notes")
        for nrow in notes[i]:
            if (e := validate_row("note", nrow)):
                P.append(f"{i}: note schema {e[:1]}"); break
        try:
            Di, _ = method_dists(vals[i], n, perms, hedge_w=w, with_s=want_s, what=i)
            if want_s and "S_mean" not in Di:
                raise MethodDataError(f"{i}: Mode S rows incomplete")
            if want_plc:
                if i in plc:
                    for v in plc[i]:
                        if v["model_revision"] != rev or v["token_mapping_hash"] != tmh:
                            raise MethodDataError(f"{i}: placebo row model revision/token mapping != frozen")
                        if (e := validate_row("placebo_row", v)):
                            raise MethodDataError(f"{i}: placebo_row schema {e[:1]}")
                        if v["donor_id"] != assign[i]:
                            raise MethodDataError(f"{i}: placebo donor {v['donor_id']} != assigned {assign[i]}")
                    Di["placebo_full"] = placebo_dist(plc[i], n, perms, i)
                    ms = pman[i]
                    if sorted(m["specialist"] for m in ms) != sorted(AGENTS):
                        raise MethodDataError(f"{i}: placebo manifest needs one row per specialist")
                    for m in ms:
                        if validate_row("placebo_manifest", m):
                            raise MethodDataError(f"{i}: placebo manifest schema")
                        if m["donor_id"] == i or m["donor_id"] != assign[i] or repo_of.get(m["donor_id"]) != repo_of[i]:
                            raise MethodDataError(f"{i}: bad donor {m['donor_id']}")
                elif not allow_partial:
                    raise MethodDataError(f"{i}: no placebo rows")
            D[i] = Di; ok_ids.append(i)
        except MethodDataError as e:
            P.append(str(e))
    if P:
        raise AuditError(f"{len(P)} audit problem(s):\n  " + "\n  ".join(P[:30]))
    req = list(CORE) + (["placebo_full"] if want_plc else []) + (["S_canonical", "S_reversed", "S_mean"] if want_s else [])
    complete_ids = [i for i in ok_ids if all(m in D[i] for m in req)]
    return {"D": D, "ids": complete_ids, "cands": cands, "vals": vals, "plc": plc, "notes": notes, "assign": assign,
            "manifest": man, "status": status, "missing": missing, "required_methods": req, "perms": perms,
            "partial": bool(missing) or not status.get("complete")}


def _mean(x):
    x = [v for v in x if v is not None]
    return round(float(np.mean(x)), 4) if x else None


def evaluate(A, params, inst, expected):
    ids, D, cands, perms = A["ids"], A["D"], A["cands"], A["perms"]
    temps = params["temperatures"]
    repo_of = {i: r for i, r, _ in expected}
    strat_of = {i: s for i, _, s in expected}
    per, gip = [], []
    for i in ids:
        cids = sorted(c["cid"] for c in cands[i]["pool"])
        g = gold_cids(inst[i], cands[i])
        row = {"id": i, "repo": repo_of[i], "stratum": strat_of[i], "gold_in_pool": int(bool(g)),
               "pool_size": len(cids)}
        if g:
            gip.append(i)
        for m in A["required_methods"]:
            p = D[i][m]
            mt = metrics(p, cids, g)  # retrieval miss => top-k/rr = 0 (unconditional)
            mtT = metrics(pools.temper(p, temps[m]), cids, g) if m in temps else None
            for k in METRICS:
                row[f"{m}.{k}"] = mt[k]
            row[f"{m}.conf"] = mt["conf"]
            row[f"{m}.nll_T"] = mtT["nll"] if mtT else None          # None for retrieval misses
            row[f"{m}.brier_T"] = mtT["brier"] if mtT else None
        for nm, (x, y) in PAIRS.items():
            if f"{x}.top1" in row and f"{y}.top1" in row:
                row[f"diff.{nm}"] = row[f"{x}.top1"] - row[f"{y}.top1"]
        per.append(row)
    by_id = {r["id"]: r for r in per}

    def summ(rows, m):
        out = {k: _mean([r[f"{m}.{k}"] for r in rows]) for k in METRICS}
        out["n"] = len(rows)
        return out
    methods = {}
    for m in A["required_methods"]:
        g_rows = [r for r in per if r["gold_in_pool"]]
        nll = [r[f"{m}.nll_T"] for r in g_rows]
        methods[m] = {"all": summ(per, m), "gold_in_pool": summ(g_rows, m),
                      "nll_tempered_gold_in_pool": _mean(nll), "nll_n": len([x for x in nll if x is not None]),
                      "nll_excluded_retrieval_miss": len(per) - len(g_rows),
                      "brier_tempered_all": _mean([r[f"{m}.brier_T"] for r in per]),
                      "brier_tempered_gold_in_pool": _mean([r[f"{m}.brier_T"] for r in g_rows]),
                      "ece_all": round(float(ece([r[f"{m}.conf"] for r in per], [r[f"{m}.top1"] for r in per])), 4),
                      "temperature": temps.get(m)}
    paired = {}
    for nm, (x, y) in PAIRS.items():
        if x not in methods or y not in methods:
            paired[nm] = {"omitted": f"{x} or {y} not available"}; continue
        dd = [r[f"diff.{nm}"] for r in per]
        paired[nm] = {"label": PAIR_LABEL[nm], "a": x, "b": y, "observed_diff_top1": _mean(dd), "n": len(dd),
                      "wins": sum(v > 0 for v in dd), "losses": sum(v < 0 for v in dd), "ties": sum(v == 0 for v in dd),
                      "per_repo": {r: {"n": len(v), "diff": _mean(v)} for r, v in sorted(_group(per, "repo", f"diff.{nm}").items())}}
    per_repo = {r: {"n": len(rows), "gold_in_pool": sum(x["gold_in_pool"] for x in rows),
                    **{f"{m}.top1": _mean([x[f"{m}.top1"] for x in rows]) for m in A["required_methods"]}}
                for r, rows in sorted(_rows_by(per, "repo").items())}
    by_stratum = {s: {"n": len(rows), **{f"{m}.top1": _mean([x[f"{m}.top1"] for x in rows])
                                          for m in ("p_empty", "p_full", "B_full5", "P_hedge", "placebo_full") if m in methods}}
                  for s, rows in sorted(_rows_by(per, "stratum").items())}
    # ordering instability (not accuracy): candidate permutations 0 vs 1, and Mode S canonical vs reversed
    order = {}
    if len(perms) >= 2:
        for m in CHECK_MASKS:
            dis = [disagreement(index_rows(A["vals"][i], cands[i]["pool_size"], i), m, subset(m)) for i in ids]
            dis = [x for x in dis if x is not None]
            order[f"cand_perm_top1_disagree.mask{m}"] = {"frac": _mean(dis), "n": len(dis)}
        pd = [int(np.argmax(_p(A["plc"][i], 0)) != np.argmax(_p(A["plc"][i], 1))) for i in ids if A["plc"].get(i)]
        order["cand_perm_top1_disagree.placebo_full"] = {"frac": _mean(pd), "n": len(pd)}
    if "S_canonical" in methods:
        sd = [int(np.argmax(D[i]["S_canonical"]) != np.argmax(D[i]["S_reversed"])) for i in ids]
        order["S_final_top1_canonical_vs_reversed"] = {"frac": _mean(sd), "n": len(sd)}
    # note / prompt length balance between arms
    bal = {}
    if "placebo_full" in methods:
        nt = {i: {n["agent"]: n.get("output_tokens") for n in A["notes"][i]} for i in A["notes"]}
        diffs, trunc_r, trunc_p = [], [], []
        for i in ids:
            dn = A["assign"][i]
            own, don = sum(nt[i].values()), sum(nt[dn].values())
            diffs.append(don - own)
            trunc_r += [int(v >= NOTE_MAX_NEW_TOKENS) for v in nt[i].values()]
            trunc_p += [int(v >= NOTE_MAX_NEW_TOKENS) for v in nt[dn].values()]
        tel = {t["prompt_hash"]: t["input_tokens"] for t in jl(_d(A) / "telemetry.jsonl") if t["kind"] == "C"}
        real_in = [tel.get(v["prompt_hash"]) for i in ids for v in A["vals"][i]
                   if v["mode"] == "C" and v["subset_mask"] == FULL and tuple(v["note_order"]) == AGENTS and v["order_perm"] in perms]
        plc_in = [v["input_tokens"] for i in ids for v in A["plc"][i]]
        bal = {"note_output_tokens_sum_mean_diff_donor_minus_own": _mean(diffs),
               "note_output_tokens_sum_mean_abs_diff": _mean([abs(x) for x in diffs]),
               "note_truncation_rate_real": _mean(trunc_r), "note_truncation_rate_placebo": _mean(trunc_p),
               "prompt_input_tokens_mean_real_full": _mean(real_in), "prompt_input_tokens_mean_placebo_full": _mean(plc_in),
               "note": "notes are capped at 150 new tokens at generation; 'truncation' = note hit that cap"}
    cost = _costs(A, perms)
    return per, {"methods": methods, "paired": paired, "per_repo": per_repo, "by_stratum": by_stratum,
                 "ordering_instability": order, "arm_balance": bal, "cost": cost,
                 "n_instances": len(ids), "n_gold_in_pool": len(gip), "n_retrieval_miss": len(ids) - len(gip)}


def _d(A):
    return A["_dir"]


def _p(rows, perm):
    return next(np.array(r["probs"]) for r in rows if r["order_perm"] == perm)


def _rows_by(per, key):
    g = defaultdict(list)
    for r in per:
        g[r[key]].append(r)
    return g


def _group(per, key, col):
    g = defaultdict(list)
    for r in per:
        g[r[key]].append(r[col])
    return g


def _costs(A, perms):
    tel = jl(_d(A) / "telemetry.jsonl")
    n = max(len(A["ids"]), 1)
    usd, calls = defaultdict(float), defaultdict(int)
    for t in tel:
        k = t["kind"].split(":")[0]
        usd[k] += t["est_usd"]; calls[k] += 1
    note = (usd["note"] + usd["note_retry"]) / n
    c = usd["C"] / max(calls["C"], 1)
    pl = usd["placebo"] / max(calls["placebo"], 1)
    s = usd["S"] / max(calls["S"], 1)
    k = len(perms)
    calls_per = {"p_empty": k, "p_full": k, "B_full5": 5 * k, "P_hedge": 5 * k, "placebo_full": k,
                 "S_one_order": 5 * k}
    return {"note_generation_usd_per_instance": round(note, 5), "note_calls_per_instance": round((calls["note"] + calls["note_retry"]) / n, 3),
            "label_calls_per_instance_by_method": calls_per,
            "usd_per_instance_by_method_incl_amortized_notes": {
                "p_empty": round(k * c, 5), "p_full": round(note + k * c, 5), "B_full5": round(note + 5 * k * c, 5),
                "P_hedge": round(note + 5 * k * c, 5),
                "placebo_full": round(k * pl, 5), "S_one_order": round(note + 5 * k * s, 5)},
            "placebo_note": "placebo reuses donor notes already generated for the real arm (no new note cost)",
            "total_est_usd_logged": round(sum(usd.values()), 4)}


def provenance(params, kv, A):
    drift = frozen.git("diff", "--name-only", params["code_commit"], "HEAD", "--", *frozen.CODE_PATHS).stdout.split()
    return {"evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "head": frozen.head_sha(),
            "frozen_code_commit": params["code_commit"], "frozen_params_sha256": kv["frozen_params_sha256"],
            "code_changed_since_freeze": drift, "model": params["model"], "protocol": params["protocol"],
            "seeds": {"placebo": params["placebo_seed"], "bootstrap": params["bootstrap_seed"], "generation": 0},
            "hedge": {"eta": params["eta"], "weights": params["hedge_weights"]},
            "run_manifest": A["manifest"], "run_status": {k: v for k, v in A["status"].items() if not k.startswith("incomplete")},
            "package_versions_at_freeze": params.get("package_versions")}


def write_outputs(out, per, S, prov, A):
    out.mkdir(parents=True, exist_ok=True)
    cols = list(per[0].keys()) if per else []
    with open(out / "per_instance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(per)
    pr = S["per_repo"]
    rcols = ["repo"] + (list(next(iter(pr.values())).keys()) if pr else [])
    with open(out / "per_repo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rcols); w.writeheader(); w.writerows([{"repo": r, **v} for r, v in pr.items()])
    summary = {"partial": A["partial"], "missing_instances": A["missing"], "provenance": prov, **S}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (out / "RESULTS.md").write_text(results_md(summary))


def results_md(s):
    L = []
    if s["partial"]:
        L += ["# PARTIAL held-out analysis: NOT a 210-instance result, not confirmatory", "",
              f"Missing instances: {len(s['missing_instances'])}. Stopped: {s['provenance']['run_status'].get('stopped')!r}.", ""]
    else:
        L += ["# MarketBugs held-out results (protocol v1.0)", ""]
    p = s["provenance"]
    L += [f"n = {s['n_instances']} instances; gold in the 20-candidate pool for {s['n_gold_in_pool']} "
          f"({s['n_retrieval_miss']} retrieval misses scored as failures in all unconditional metrics).", "",
          "## Top-1 / Top-3 / Top-5 / MRR", "", "| method | all: top1 | top3 | top5 | MRR | gold-in-pool: top1 | NLL (T, gold-in-pool, n) |",
          "|---|---|---|---|---|---|---|"]
    for m, v in s["methods"].items():
        a, g = v["all"], v["gold_in_pool"]
        L.append(f"| {m} | {a['top1']} | {a['top3']} | {a['top5']} | {a['rr']} | {g['top1']} | {v['nll_tempered_gold_in_pool']} ({v['nll_n']}) |")
    L += ["", "NLL is over gold-in-pool instances only (retrieval misses have no finite log score; count excluded "
          "is stated per method in summary.json). ECE over 210 cases / 7 repositories is a noisy estimate.", "",
          "## Paired Top-1 differences", "", "| contrast | status | observed diff | W / L / T | n |", "|---|---|---|---|---|"]
    for nm, v in s["paired"].items():
        if "omitted" in v:
            L.append(f"| {nm} | omitted | {v['omitted']} | | |"); continue
        L.append(f"| {v['a']} − {v['b']} | {v['label']} | {v['observed_diff_top1']} | {v['wins']} / {v['losses']} / {v['ties']} | {v['n']} |")
    L += ["", "<!-- BOOTSTRAP:BEGIN -->", "Confidence intervals: not yet computed (run src.analysis.bootstrap_test). "
          "No conclusion is drawn without the repository-cluster CI.", "<!-- BOOTSTRAP:END -->", "",
          "## Ordering instability (not accuracy)", ""]
    for k, v in s["ordering_instability"].items():
        L.append(f"- {k}: {v['frac']} (n = {v['n']})")
    if s["arm_balance"]:
        L += ["", "## Real vs placebo arm balance", ""] + [f"- {k}: {v}" for k, v in s["arm_balance"].items()]
    L += ["", "## Cost", "", "```", json.dumps(s["cost"], indent=1), "```", "", "## Provenance", "", "```",
          json.dumps({k: p[k] for k in ("head", "frozen_code_commit", "frozen_params_sha256", "code_changed_since_freeze",
                                        "model", "protocol", "seeds", "hedge", "evaluated_utc")}, indent=1), "```", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", default="test_informed")
    ap.add_argument("--frozen", default=str(frozen.FROZEN_JSON))
    ap.add_argument("--frozen-txt", default=str(frozen.FROZEN_TXT))
    ap.add_argument("--input-dir", default=str(ROOT / "data/marketbugs/test"))
    ap.add_argument("--require-n", type=int, default=210)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--runs-dir", default=str(ROOT / "artifacts/runs"))
    a = ap.parse_args(argv)
    params, kv = frozen.load_frozen(txt=a.frozen_txt, js=a.frozen)
    d = Path(a.runs_dir) / a.tag / a.cond
    expected = test_ids(a.input_dir)
    A = audit(d, params, kv, expected, a.require_n, a.allow_partial)
    A["_dir"] = d
    print(f"audit OK: {len(A['ids'])} complete instances, methods {A['required_methods']}"
          + (" (PARTIAL)" if A["partial"] else ""))
    if a.audit_only:
        return A
    inst = test_gold(a.input_dir)  # gold enters here, after the audit, in this process only
    per, S = evaluate(A, params, inst, expected)
    out = Path(a.output_dir or ROOT / "artifacts/validation" / a.tag)
    write_outputs(out, per, S, provenance(params, kv, A), A)
    print(f"wrote {out}/per_instance.csv, per_repo.csv, summary.json, RESULTS.md")
    return per, S


if __name__ == "__main__":
    try:
        main()
    except (AuditError, frozen.FreezeError) as e:
        sys.exit(f"STOP: {e}")
