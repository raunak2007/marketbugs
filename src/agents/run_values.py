"""C-mode value-table runner (PLAN 8) + F-mode forecaster. Inference side: never reads gold.
Gold scores are attached afterwards by the evaluator-side scorer."""
import json, time
from src.agents.prompts import probability_agent_prompt, subset_agents, AGENTS, LABELS, template_hash
from src.agents.specialists import render_note, forecaster_prompt
from src.llm.hashing import sha

# VERIFY: Modal A10G list price; replace with measured per-call cost at the pilot.
GPU_USD_PER_S = 1.10 / 3600


def permuted_view(view, perm_seed):
    """Display order permutation for order-sensitivity checks (perm 0 = stored order)."""
    if perm_seed == 0:
        return view
    import random
    v = dict(view); pool = list(view["pool"]); random.Random(10_000 + perm_seed).shuffle(pool); v["pool"] = pool
    return v


def value_rows_for_instance(view, notes, server, order_perm=0, masks=range(32), note_order_override=None):
    """notes: {agent: canonical_note_text}. Returns list of value_row dicts minus gold fields."""
    v = permuted_view(view, order_perm)
    rendered = {a: render_note(t, v) for a, t in notes.items()}
    prompts, metas = [], []
    for m in masks:
        p, present = probability_agent_prompt(v, rendered, m, order=note_order_override)
        prompts.append(p); metas.append((m, present))
    t0 = time.time()
    outs = server.label_logits(prompts, [len(v["pool"])] * len(prompts))
    wall = time.time() - t0
    rows = []
    disp_cids = [c["cid"] for c in v["pool"]]
    for (m, present), p, o in zip(metas, prompts, outs):
        # map display-order outputs back to cid order (A..T in pool order)
        by_cid = dict(zip(disp_cids, zip(o["raw"], o["probs"])))
        cids = sorted(by_cid)
        raw = [by_cid[c][0] for c in cids]; probs = [by_cid[c][1] for c in cids]
        rows.append({"id": view["id"], "mode": "C", "subset_mask": m, "order_perm": order_perm, "note_order": present,
                     "raw_candidate_logits": raw, "probs": probs, "top1": cids[max(range(len(probs)), key=probs.__getitem__)],
                     "token_mapping_hash": o["token_mapping_hash"], "prompt_hash": sha(p),
                     "template_hash": template_hash("probability_agent"), "model_revision": o["model_revision"],
                     "input_tokens": o["input_tokens"], "latency_ms": o["latency_ms"],
                     "inference_cost_usd": wall * GPU_USD_PER_S / len(prompts)})
    return rows
