"""Pre-pilot sanity on the DEV split only (CPU, no Modal): can the gold function reach the pool?

python -m src.analysis.dev_retrieval_check
Reports per dev repo: strata, B1 BM25 end-to-end gold rank (Top-1/5/20, MRR), trace-frame recall,
graph(frames) top-40 recall, and a BM25+graph k=20 pool recall LOWER BOUND (the pilot pool adds embeddings).
For the test split it prints only instance and stratum counts -- no outcome is computed on test.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from src.cands.functions import index_repo
from src.cands.retrieval import BM25, Graph, frame_functions
from src.cands.pool import build_pool
from src.llm.hashing import sha

ROOT = Path(__file__).resolve().parents[2]
MB = ROOT / "data/marketbugs"


def rank_of(gold, ranked):
    keys = [f.key for f, _ in ranked]
    return next((i + 1 for i, k in enumerate(keys) if k in gold), None)


def main():
    out = {}
    for path in sorted((MB / "dev").glob("*.jsonl")):
        if path.name == "eval_only.jsonl":
            continue
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            continue
        repo = rows[0]["repo"]; d = MB / "repos" / repo
        fns = index_repo(d); bm, gr = BM25(fns), Graph(fns)
        m = defaultdict(list); strata = Counter()
        for r in rows:
            gold = set(r["gold_functions"]); q = r["conditions"]["test_informed"]
            strata[r["stratum"]] += 1
            rk = rank_of(gold, bm.rank(q))
            m["bm25_top1"].append(rk == 1); m["bm25_top5"].append(bool(rk and rk <= 5))
            m["bm25_top20"].append(bool(rk and rk <= 20)); m["bm25_mrr"].append(1 / rk if rk else 0.0)
            frames = frame_functions(fns, r["trace"], d)
            m["in_frames"].append(any(f.key in gold for f in frames))
            g40 = gr.rank(frames, 40)
            m["graph40"].append(rank_of(gold, g40) is not None)
            pool = build_pool(r["id"], {"bm25": bm.rank(q, 40), "graph": g40}, 20, order_seed=int(sha(r["id"]), 16) % 10**6)
            keys = {f"{c['path']}::{c['qualname']}" for c in pool["pool"]}
            m["pool20_lb"].append(bool(keys & gold)); m["pool_size"].append(pool["pool_size"])
        out[repo] = {"n": len(rows), "n_functions": len(fns), "strata": dict(strata),
                     **{k: round(sum(v) / len(v), 3) for k, v in m.items()}}
    test = {}
    for path in sorted((MB / "test").glob("*.jsonl")):
        if path.name != "eval_only.jsonl":
            rows = [json.loads(l) for l in open(path)]
            test[path.stem] = {"n": len(rows), "strata": dict(Counter(r["stratum"] for r in rows))}
    res = {"dev": out, "test_counts_only": test}
    (ROOT / "artifacts/dev_retrieval_check.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'repo':17}{'n':>4}{'fns':>5}  {'B1@1':>5}{'B1@5':>6}{'B1@20':>6}{'MRR':>6}{'frame':>6}{'grph40':>7}{'pool_lb':>8}  strata"
    print(hdr)
    for k, v in out.items():
        print(f"{k:17}{v['n']:>4}{v['n_functions']:>5}  {v['bm25_top1']:>5}{v['bm25_top5']:>6}{v['bm25_top20']:>6}"
              f"{v['bm25_mrr']:>6}{v['in_frames']:>6}{v['graph40']:>7}{v['pool20_lb']:>8}  {v['strata']}")
    print("test (counts only):", json.dumps(test))


if __name__ == "__main__":
    main()
