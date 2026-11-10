"""Pool builder (PLAN 4.2): interleave BM25 / embedding / graph by rank, dedupe, truncate to K.
Never inserts gold. Seeded shuffle for display order."""
import random
from src.agents.prompts import LABELS


def build_pool(inst_id, ranked_lists, k=20, order_seed=0):
    """ranked_lists: {"bm25": [(Fn, score)], "emb": [...], "graph": [...]} -- each already ranked."""
    srcs = [s for s in ("graph", "bm25", "emb") if ranked_lists.get(s)]
    chosen, seen, depth = [], set(), 0
    while len(chosen) < k and any(depth < len(ranked_lists[s]) for s in srcs):
        for s in srcs:
            lst = ranked_lists[s]
            if depth < len(lst):
                f, sc = lst[depth]
                if f.key not in seen:
                    seen.add(f.key); chosen.append((f, s, sc))
                    if len(chosen) == k:
                        break
        depth += 1
    pool = [{"cid": LABELS[i], "path": f.path, "qualname": f.qualname, "code": f.code,
             "retrieval_source": s, "retrieval_score": float(sc)} for i, (f, s, sc) in enumerate(chosen)]
    order = [c["cid"] for c in pool]
    random.Random(order_seed).shuffle(order)  # display order does not encode retrieval rank
    return {"id": inst_id, "pool_size": len(pool), "pool": pool, "candidate_order": order, "order_seed": order_seed}


def attach_gold_flag(cands, gold_functions):
    """EVALUATOR side only. gold_functions entries are 'path::qualname'."""
    keys = {f"{c['path']}::{c['qualname']}" for c in cands["pool"]}
    cands["gold_in_pool"] = any(g in keys for g in gold_functions)
    return cands
