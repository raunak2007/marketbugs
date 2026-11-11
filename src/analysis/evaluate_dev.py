"""EVALUATOR SIDE, DEV ONLY (pre-G2). Attach gold to a run, validate, and report G1 checks + method comparison.

python -m src.analysis.evaluate_dev <tag> [--cond test_informed]

Methods (all on the same pool / notes):
  B4 p_empty, B5 p_full (canonical order), B_full5 (mean of 5 cyclic note orders), singletons,
  P_vote, P_avg, P_hedge_eq (equal-weight log pool), P_hedge (dev-fit eta by 2-fold CV; *learned supervised pool*),
  S_canonical / S_reversed (final market) and S_mean (mean of the two finals).
Unconditional metrics count retrieval misses as misses; NLL/ECE-by-gold and all credit/rewards use gold-in-pool
instances only, with exclusion counts reported. NOTE: P_hedge eta here is fit and scored on the same dev data
(in-sample); the frozen eta for test comes from the G2 fit, not from this table.
"""
import json, math, sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from src.analysis.score import gold_cids, metrics, ece, score_row
from src.analysis.credit import credit, market_rewards, CREDIT_LABEL
from src.agents.prompts import AGENTS
from src.market import pools
from src.market.sequential import ORDERS
from src.schemas.validate import validate_row

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ROT = [tuple(AGENTS[i:] + AGENTS[:i]) for i in range(5)]
CHECK_MASKS = (0, 1, 2, 4, 8, 16, 31)


def jl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def load_instances():
    out = {}
    for p in (ROOT / "data/marketbugs/dev").glob("*.jsonl"):
        if p.name != "eval_only.jsonl":
            for l in open(p):
                r = json.loads(l); out[r["id"]] = r
    return out


def attach_and_validate(d, inst):
    """Writes candidates.jsonl / values.jsonl (with gold) next to the inference files; returns invalid count."""
    cands = {c["id"]: c for c in jl(d / "candidates_inf.jsonl")}
    bad = 0
    out_c, out_v = [], []
    for i, c in cands.items():
        g = gold_cids(inst[i], c)
        cc = dict(c, gold_in_pool=bool(g)); out_c.append(cc)
        bad += bool(validate_row("candidates", cc))
    gc = {i: set(gold_cids(inst[i], c)) for i, c in cands.items()}
    for v in jl(d / "values_inf.jsonl"):
        vv = score_row(dict(v), gc[v["id"]]); out_v.append(vv)
        bad += bool(validate_row("value_row", vv))
    for n in jl(d / "notes.jsonl"):
        bad += bool(validate_row("note", n))
    (d / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_c))
    (d / "values.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_v))
    return cands, gc, out_v, bad


def main(tag, cond):
    d = ART / "runs" / tag / cond
    inst = load_instances()
    cands, gc, vals, n_invalid = attach_and_validate(d, inst)
    notes = defaultdict(dict)
    for n in jl(d / "notes.jsonl"):
        notes[n["id"]][n["agent"]] = n
    by = defaultdict(list)
    for v in vals:
        by[v["id"]].append(v)
    ids = [i for i in cands if len([v for v in by[i] if v["mode"] == "C" and v["order_perm"] == 0]) >= 36]
    CID = lambda i: sorted(c["cid"] for c in cands[i]["pool"])

    def get(i, mode, mask, perm=0, order=None):
        for v in by[i]:
            if v["mode"] == mode and v["subset_mask"] == mask and v["order_perm"] == perm and (
                    order is None or tuple(v["note_order"]) == tuple(order)):
                return np.array(v["probs"])
        return None

    # per-instance method distributions
    D, singles_all = defaultdict(dict), {}
    for i in ids:
        S = np.stack([get(i, "C", 1 << b, order=[a]) for b, a in enumerate(AGENTS)]); singles_all[i] = S
        D["p_empty"][i] = get(i, "C", 0, order=[])
        D["p_full"][i] = get(i, "C", 31, order=AGENTS)
        D["B_full5"][i] = pools.normalize(np.mean([get(i, "C", 31, order=r) for r in ROT], 0))
        for b, a in enumerate(AGENTS):
            D[f"single.{a}"][i] = S[b]
        D["P_vote"][i], D["P_avg"][i], D["P_hedge_eq"][i] = pools.p_vote(S), pools.p_avg(S), pools.log_pool(S, np.ones(5))
        for nm, seq in ORDERS.items():
            D[f"S_{nm}"][i] = get(i, "S", 31, order=seq)
        if D["S_canonical"][i] is not None and D["S_reversed"][i] is not None:
            D["S_mean"][i] = pools.normalize((D["S_canonical"][i] + D["S_reversed"][i]) / 2)
    gip = [i for i in ids if gc[i]]
    gidx = {i: CID(i).index(sorted(gc[i])[0]) for i in gip}
    eta, eta_cv = (None, {})
    if len(gip) >= 4:
        eta, eta_cv = pools.fit_eta([singles_all[i] for i in gip], [gidx[i] for i in gip])
        w = pools.hedge_weights([[math.log(max(singles_all[i][b][gidx[i]], 1e-12)) for b in range(5)] for i in gip], eta)
        for i in ids:
            D["P_hedge"][i] = pools.log_pool(singles_all[i], w)

    rep = {"tag": tag, "condition": cond, "n_instances": len(ids), "n_gold_in_pool": len(gip),
           "n_retrieval_miss": len(ids) - len(gip), "schema_invalid_rows": n_invalid,
           "by_repo": dict(sorted({r: sum(inst[i]["repo"] == r for i in ids) for r in {inst[i]["repo"] for i in ids}}.items())),
           "random_top1": round(float(np.mean([1 / cands[i]["pool_size"] if gc[i] else 0 for i in ids])), 4),
           "hedge_eta": eta, "hedge_eta_cv_nll": eta_cv, "methods": {}, "per_repo_top1": defaultdict(dict)}
    for m, dist in D.items():
        rows = {i: metrics(p, CID(i), gc[i]) for i, p in dist.items() if p is not None}
        if not rows:
            continue
        T, _ = pools.fit_temperature([dist[i] for i in gip if i in rows], [gidx[i] for i in gip if i in rows]) if gip else (1.0, None)
        cal = [metrics(pools.temper(dist[i], T), CID(i), gc[i]) for i in rows]
        agg = {k: round(float(np.mean([r[k] for r in rows.values()])), 4) for k in ("top1", "top3", "top5", "rr")}
        nll = [r["nll"] for r in rows.values() if r["nll"] is not None]
        agg.update(n=len(rows), nll_gold_in_pool=round(float(np.mean(nll)), 4) if nll else None, temperature=T,
                   nll_tempered=round(float(np.mean([c["nll"] for c in cal if c["nll"] is not None])), 4) if nll else None,
                   ece=round(float(ece([r["conf"] for r in rows.values()], [r["top1"] for r in rows.values()])), 4))
        rep["methods"][m] = agg
        for r in sorted({inst[i]["repo"] for i in rows}):
            ii = [i for i in rows if inst[i]["repo"] == r]
            rep["per_repo_top1"][r][m] = round(float(np.mean([rows[i]["top1"] for i in ii])), 3)

    # PLAN 4.3 order sensitivity over 3 candidate permutations
    changed, tot = 0, 0
    for i in ids:
        for m in CHECK_MASKS:
            tops = [None if (p := get(i, "C", m, perm, order=[a for b, a in enumerate(AGENTS) if m >> b & 1])) is None else CID(i)[int(np.argmax(p))] for perm in (0, 1, 2)]
            if None not in tops:
                tot += 1; changed += len(set(tops)) > 1
    rep["order_sensitivity"] = {"frac_top1_changed": round(changed / tot, 4) if tot else None, "n": tot,
                                "threshold": 0.15, "exceeds": (changed / tot > 0.15) if tot else None}
    # Mode S order sensitivity + rewards (gold-in-pool only)
    s_ids = [i for i in ids if D["S_canonical"].get(i) is not None and D["S_reversed"].get(i) is not None]
    rep["S_order_sensitivity"] = {"frac_final_top1_differs": round(float(np.mean(
        [np.argmax(D["S_canonical"][i]) != np.argmax(D["S_reversed"][i]) for i in s_ids])), 4) if s_ids else None, "n": len(s_ids)}
    rew = {nm: defaultdict(list) for nm in ORDERS}; tele_err = 0.0
    for i in s_ids:
        if not gc[i]:
            continue
        for nm, seq in ORDERS.items():
            traj = [D["p_empty"][i]] + [get(i, "S", sum(1 << AGENTS.index(a) for a in seq[:k]), order=seq[:k]) for k in range(1, 6)]
            r = market_rewards([list(t) for t in traj], CID(i), gc[i])
            lg = [math.log(max(sum(t[CID(i).index(c)] for c in gc[i]), 1e-300)) for t in (traj[0], traj[-1])]
            tele_err = max(tele_err, abs(sum(r) - (lg[1] - lg[0])))
            for a, x in zip(seq, r):
                rew[nm][a].append(x)
    rep["S_rewards_mean"] = {nm: {a: round(float(np.mean(v)), 4) for a, v in rr.items()} for nm, rr in rew.items()}
    rep["S_rewards_excluded_gold_miss"] = len([i for i in s_ids if not gc[i]])
    rep["S_reward_telescoping_max_err"] = tele_err
    # exact Shapley (C mode), gold-in-pool only
    sh = defaultdict(list)
    for i in gip:
        tbl = {m: list(get(i, "C", m, order=[a for b, a in enumerate(AGENTS) if m >> b & 1])) for m in range(32)}
        c = credit(tbl, CID(i), gc[i])
        for a, x in zip(AGENTS, c["shapley"]):
            sh[a].append(x)
    rep["shapley_mean"] = {a: round(float(np.mean(v)), 4) for a, v in sh.items()}
    rep["shapley_label"] = CREDIT_LABEL
    rep["shapley_excluded_gold_miss"] = len(ids) - len(gip)
    # K=10 vs K=20 (report only)
    k10 = jl(d / "k10.jsonl"); k10r = defaultdict(list)
    for r in k10:
        g = set(inst[r["id"]]["gold_functions"]); keys = r["candidate_keys"]
        top = int(np.argmax(r["probs"]))
        k10r[r["subset_mask"]].append(int(keys[top] in g)); k10r[f"gip_{r['subset_mask']}"].append(int(bool(set(keys) & g)))
    rep["k10"] = {str(k): round(float(np.mean(v)), 4) for k, v in k10r.items()}
    # notes: validity + copy check (G1 #5)
    parsed, copy = [], []
    for i in ids:
        for a in AGENTS:
            n = notes[i].get(a, {}); parsed.append(bool(n.get("ranked_suspects")))
            if n.get("text") and inst[i]["trace"]:
                sm = SequenceMatcher(None, n["text"], inst[i]["trace"], autojunk=False)
                copy.append(sm.find_longest_match(0, len(n["text"]), 0, len(inst[i]["trace"])).size / len(n["text"]))
    rep["notes"] = {"parsed_frac": round(float(np.mean(parsed)), 4), "copy_frac_mean": round(float(np.mean(copy)), 4) if copy else None,
                    "copy_frac_max": round(float(max(copy)), 4) if copy else None}
    # cost / tokens / latency / calls (telemetry)
    tel = jl(d / "telemetry.jsonl"); kinds = defaultdict(lambda: defaultdict(float))
    for t in tel:
        k = t["kind"].split(":")[0]
        for f in ("input_tokens", "output_tokens", "latency_ms", "est_usd"):
            kinds[k][f] += t[f]
        kinds[k]["calls"] += 1
    n_ic = max(len(ids), 1)
    rep["telemetry_per_instance"] = {k: {f: round(v / n_ic, 5) for f, v in dd.items()} for k, dd in kinds.items()}
    note_usd = (kinds["note"]["est_usd"] + kinds["note_retry"]["est_usd"]) / n_ic
    c_call = kinds["C"]["est_usd"] / max(kinds["C"]["calls"], 1)
    s_call = kinds["S"]["est_usd"] / max(kinds["S"]["calls"], 1)
    rep["method_cost_usd_per_instance"] = {  # A10G-only estimate; notes counted for every method that uses them
        "p_empty": round(c_call, 5), "p_full": round(note_usd + c_call, 5), "B_full5": round(note_usd + 5 * c_call, 5),
        "P_hedge": round(note_usd + 5 * c_call, 5), "S_one_order": round(note_usd + c_call + 5 * s_call, 5),
        "C_table_32": round(note_usd + 32 * c_call, 5)}
    rep["est_usd_per_instance_all_calls"] = round(sum(dd["est_usd"] for dd in kinds.values()) / n_ic, 5)
    rep["per_repo_top1"] = dict(rep["per_repo_top1"])
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[a.index("--cond") + 1] if "--cond" in a else "test_informed")
