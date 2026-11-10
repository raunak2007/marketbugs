"""Recompute MarketBugs strata per PLAN 5.1 with test-code frames included.

Bug fixed: strata were computed on an index that excluded test files, so traces whose only in-repo frames are
the failing test (assertion-style failures) could never be 'near' and fell to 'far'. PLAN: near = gold within one
static call-hop of a *traceback-frame function*; the failing test function is one.

python -m src.gen.restratify            # dry run: print old -> new counts
python -m src.gen.restratify --write    # rewrite stratum in place; old->new audit in artifacts/restratify_audit.json
(schemas are frozen after G0, so no audit field is added to rows)
"""
import json, sys
from collections import Counter
from pathlib import Path
from src.cands.functions import index_repo, trace_frames, enclosing
from src.cands.retrieval import Graph

ROOT = Path(__file__).resolve().parents[2]
MB = ROOT / "data/marketbugs"


def stratum(fns, graph, trace, gold, d):
    frames = []
    for p, ln, _ in trace_frames(trace, d):
        f = enclosing(fns, p, ln)
        if f:
            frames.append(f.key)
    if gold & set(frames):
        return "direct"
    if any(gold & graph.neighbors(k) for k in frames):
        return "near"
    return "far"


def main(write):
    tot = Counter()
    ap = ROOT / "artifacts/restratify_audit.json"
    audit = json.loads(ap.read_text()) if ap.exists() else {}
    for path in sorted(MB.glob("*/*.jsonl")):
        if path.name == "eval_only.jsonl":
            continue
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            continue
        d = MB / "repos" / rows[0]["repo"]
        fns = index_repo(d, include_tests=True); g = Graph(fns)
        moves = Counter()
        for r in rows:
            old = audit.get(r["id"], {}).get("v1", r["stratum"])
            new = stratum(fns, g, r["trace"], set(r["gold_functions"]), d)
            moves[(old, new)] += 1; tot[new] += 1
            audit[r["id"]] = {"v1": old, "v2": new}; r["stratum"] = new
        print(f"{path.parent.name}/{path.stem:17}", dict(moves))
        if write:
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if write:
        ap.write_text(json.dumps(audit, indent=0))
    print("new totals:", dict(tot), "(written)" if write else "(dry run)")


if __name__ == "__main__":
    main("--write" in sys.argv)
