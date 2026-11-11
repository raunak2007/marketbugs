"""Mode S: sequential market (PLAN 6, E4; exploratory). INFERENCE SIDE: never sees gold, outcomes or rewards.

Initial market = p_empty (C-mode mask 0). Agents move one at a time in a fixed order (canonical and reversed).
Agent i sees the issue, the candidate block, the current market's top-5 probabilities and its own note; the
Gemma probability-extraction interface returns the updated distribution, which becomes the market.
Rewards r_i = log p_after(G) - log p_before(G) are computed afterwards by src.analysis.market_score (evaluator).

Not incentive-compatible and not trained: agents are a frozen model prompted with a market state.
"""
import time
from src.agents.prompts import AGENTS, LABELS, template, candidate_block
from src.agents.specialists import label_maps
from src.llm.hashing import sha
from src.agents.run_values import GPU_USD_PER_S

ORDERS = {"canonical": tuple(AGENTS), "reversed": tuple(reversed(AGENTS))}
TOP_K_SHOWN = 5


def market_text(view, probs_by_cid, k=TOP_K_SHOWN):
    """Top-k by probability, ties by display label (deterministic), shown with display labels."""
    _, cid2lab = label_maps(view)
    items = sorted(probs_by_cid.items(), key=lambda kv: (-kv[1], cid2lab[kv[0]]))[:k]
    return "\n".join(f"[{cid2lab[c]}] {p:.2f}" for c, p in items)


def market_prompt(view, agent, rendered_note, probs_by_cid):
    return template("market").format(
        role=agent, issue=view["issue_text"], candidates=candidate_block(view),
        market=market_text(view, probs_by_cid), note=rendered_note,
        last_label=LABELS[len(view["pool"]) - 1])


def to_cid_probs(view, out):
    """Server output is in display order A..; return {cid: prob}."""
    return {c["cid"]: p for c, p in zip(view["pool"], out["probs"])}


def avg_logprob(dists):
    """Order-sensitivity protocol (ANALYSIS_PLAN, adopted at G1): mean of log-probs over fixed permutations,
    renormalized. dists: list of {cid: prob}."""
    import math
    cids = sorted(dists[0])
    lp = {c: sum(math.log(max(d[c], 1e-300)) for d in dists) / len(dists) for c in cids}
    m = max(lp.values())
    z = sum(math.exp(v - m) for v in lp.values())
    return {c: math.exp(lp[c] - m) / z for c in cids}


def run_market(view, rendered_notes, p_empty_cid_sorted, srv, inst_id, cond, orders=ORDERS, views=None, notes_by_view=None):
    """p_empty_cid_sorted: probs in sorted-cid order (for the averaged protocol: the perm-averaged p_empty).
    views: candidate-display permutations [(perm_id, view)], default [(0, view)]. With several views, each step
    queries every view and the market moves to the log-prob average (the next agent sees that averaged market).
    Returns (value_rows with mode 'S', one per (order, perm, step), telemetry). One batched call per step."""
    views = views or [(0, view)]
    notes_by_view = notes_by_view or {pid: rendered_notes for pid, _ in views}
    cids = sorted(c["cid"] for c in view["pool"])
    p0 = dict(zip(cids, p_empty_cid_sorted))
    state = {name: dict(p0) for name in orders}
    rows, tel = [], []
    for k in range(len(AGENTS)):
        jobs = [(nm, pid, v) for nm in orders for pid, v in views]
        prompts = [market_prompt(v, orders[nm][k], notes_by_view[pid][orders[nm][k]], state[nm]) for nm, pid, v in jobs]
        t0 = time.time()
        outs = srv.label_logits(prompts, [len(view["pool"])] * len(prompts))
        wall = time.time() - t0
        new_by = {}
        for (nm, pid, v), p, o in zip(jobs, prompts, outs):
            new = to_cid_probs(v, o)
            new_by.setdefault(nm, []).append(new)
            done = orders[nm][:k + 1]
            probs = [new[c] for c in cids]
            rows.append({"id": inst_id, "mode": "S", "subset_mask": sum(1 << AGENTS.index(a) for a in done),
                         "order_perm": pid, "note_order": list(done),
                         "raw_candidate_logits": [dict(zip([c["cid"] for c in v["pool"]], o["raw"]))[c] for c in cids],
                         "probs": probs, "top1": cids[max(range(len(probs)), key=probs.__getitem__)],
                         "token_mapping_hash": o["token_mapping_hash"], "prompt_hash": sha(p),
                         "model_revision": o["model_revision"], "inference_cost_usd": wall / len(prompts) * GPU_USD_PER_S})
            tel.append({"id": inst_id, "condition": cond, "kind": f"S:{nm}", "step": k + 1, "agent": orders[nm][k],
                        "order_perm": pid, "prompt_hash": sha(p), "input_tokens": o["input_tokens"], "output_tokens": 0,
                        "latency_ms": o["latency_ms"], "wall_share_s": wall / len(prompts),
                        "est_usd": wall / len(prompts) * GPU_USD_PER_S})
        for nm, ds in new_by.items():
            state[nm] = ds[0] if len(ds) == 1 else avg_logprob(ds)
    return rows, tel


def trajectory(rows, inst_id, order_name, p_empty_row):
    """Evaluator/analysis helper: [p0, p1, ..., p5] (sorted-cid lists) for one instance and order."""
    seq = tuple(ORDERS[order_name])
    steps = sorted((r for r in rows if r["id"] == inst_id and r["mode"] == "S" and r["order_perm"] == 0
                    and tuple(r["note_order"]) == seq[:len(r["note_order"])]), key=lambda r: len(r["note_order"]))
    return [p_empty_row["probs"]] + [r["probs"] for r in steps]
