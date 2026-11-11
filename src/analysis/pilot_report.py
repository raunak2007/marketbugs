"""SUPERSEDED by src.analysis.evaluate_dev (old-format pre-fix pilot files only). G1 pilot report (DEV ONLY). EVALUATOR SIDE: reads gold via src.analysis.score.

python -m src.analysis.pilot_report <tag> [<tag> ...]     # tags: artifacts/pilot_<tag>_{candidates,notes,values_C}.jsonl

Reports, over all instances in the given tags:
- p_empty (mask 0) and p_full (mask 31) Top-1/Top-5/NLL vs random-in-pool (G1 #1)
- each singleton (one note) and specialist note top-1 accuracy (the note's own first-ranked suspect)
- exploratory aggregates of the 5 singleton posteriors: P_vote, P_avg, P_hedge(equal-weight log pool)
- note-copy check (G1 #5): longest common substring of note vs trace, as a fraction of note length
- cost/latency per instance and extrapolation from artifacts/costs.jsonl
"""
import json, math, sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from src.analysis.score import gold_cids, metrics, random_top1
from src.agents.prompts import AGENTS

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
CIDS = "ABCDEFGHIJKLMNOPQRST"


def load(tag, kind):
    p = ART / f"pilot_{tag}_{kind}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def instances():
    out = {}
    for split in ("dev",):
        for p in (ROOT / "data/marketbugs" / split).glob("*.jsonl"):
            if p.name != "eval_only.jsonl":
                for l in open(p):
                    r = json.loads(l); out[r["id"]] = r
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def main(tags):
    inst = instances()
    cands, notes, vals = {}, defaultdict(dict), defaultdict(dict)
    for t in tags:
        for c in load(t, "candidates"):
            cands[c["id"]] = c
        for n in load(t, "notes"):
            notes[n["id"]][n["agent"]] = n
        for v in load(t, "values_C"):
            vals[v["id"]][v["subset_mask"]] = v
    ids = [i for i in cands if len(vals[i]) == 32]
    if not ids:
        sys.exit("no complete instances")
    M = defaultdict(list)
    copy_fr = []
    for i in ids:
        c = cands[i]; n = c["pool_size"]; cids = list(CIDS[:n])
        g = gold_cids(inst[i], c)
        M["gold_in_pool"].append(bool(g)); M["random_top1"].append(random_top1(n, bool(g)))
        for name, m in (("p_empty", 0), ("p_full", 31)):
            for k, x in metrics(vals[i][m]["probs"], cids, g).items():
                M[f"{name}.{k}"].append(x)
        singles = []
        for b, a in enumerate(AGENTS):
            p = np.array(vals[i][1 << b]["probs"]); singles.append(p)
            M[f"single.{a}.top1"].append(metrics(p, cids, g)["top1"])
            rs = notes[i].get(a, {}).get("ranked_suspects") or []
            M[f"note.{a}.top1"].append(int(bool(rs) and rs[0] in g))
            M[f"note.{a}.parsed"].append(int(bool(rs)))
            txt = notes[i].get(a, {}).get("text", "")
            if txt:
                sm = SequenceMatcher(None, txt, inst[i]["trace"], autojunk=False)
                copy_fr.append(sm.find_longest_match(0, len(txt), 0, len(inst[i]["trace"])).size / len(txt))
        S = np.stack(singles)
        votes = np.bincount(S.argmax(1), minlength=n).astype(float); votes /= votes.sum()
        lp = np.log(np.clip(S, 1e-12, 1)).mean(0); hedge = np.exp(lp - lp.max()); hedge /= hedge.sum()
        for name, p in (("P_vote", votes), ("P_avg", S.mean(0)), ("P_hedge_eq", hedge)):
            for k, x in metrics(p, cids, g).items():
                M[f"{name}.{k}"].append(x)
    rep = {"n": len(ids), "by_repo": dict(sorted({r: sum(inst[i]["repo"] == r for i in ids)
                                                   for r in {inst[i]["repo"] for i in ids}}.items()))}
    for k, v in M.items():
        rep[k] = mean([float(x) if isinstance(x, bool) else x for x in v])
    rep["note_copy_frac.mean"] = mean(copy_fr); rep["note_copy_frac.max"] = round(max(copy_fr), 3) if copy_fr else None
    costs = [json.loads(l) for l in open(ART / "costs.jsonl")]
    runs = [c for c in costs if c.get("job", "").startswith("pilot") and c.get("n", 0) > 0]
    tot_n = sum(c["n"] for c in runs); tot_c = sum(c["conservative_usd"] for c in runs)
    rep["cost.per_instance_conservative_usd"] = round(tot_c / tot_n, 4) if tot_n else None
    rep["cost.all_pilot_attempts_conservative_usd"] = round(sum(c.get("conservative_usd", c.get("est_usd", 0)) for c in costs
                                                               if c.get("job", "").startswith("pilot")), 3)
    (ART / "pilot_report.json").write_text(json.dumps(rep, indent=1))
    keys = ["n", "by_repo", "gold_in_pool", "random_top1", "p_empty.top1", "p_empty.top5", "p_empty.nll", "p_empty.conf",
            "p_full.top1", "p_full.top5", "p_full.nll", "p_full.conf", "P_vote.top1", "P_avg.top1", "P_hedge_eq.top1",
            "P_hedge_eq.nll"] + [f"single.{a}.top1" for a in AGENTS] + [f"note.{a}.top1" for a in AGENTS] + \
           [f"note.{a}.parsed" for a in AGENTS] + ["note_copy_frac.mean", "note_copy_frac.max",
                                                  "cost.per_instance_conservative_usd", "cost.all_pilot_attempts_conservative_usd"]
    for k in keys:
        print(f"{k:42} {rep.get(k)}")


if __name__ == "__main__":
    main(sys.argv[1:])
