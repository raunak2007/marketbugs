"""Per-instance inference pipeline (INFERENCE SIDE: never reads gold; gold is attached later by src.analysis).

One call of `run_instance` produces, for one instance and one condition:
  candidates   pool (schema candidates.v2 minus the evaluator-only gold_in_pool flag)
  notes        5 specialist notes (schema note.v2 fields only)
  values       C-mode 32-subset table + B_full5 note-order rotations + order-sensitivity candidate permutations
               + Mode S trajectories (value_row.v2 fields only; gold_logscore attached by the evaluator)
  k10          PLAN 4.3 K=10 control rows (masks 0 and 31) -- report only, never used by a method
  telemetry    every model call: kind, prompt hash, tokens, latency, estimated cost (schemas are frozen, so
               this lives in a sidecar instead of extra row fields)

Condition boundary (fix 2026-11-10): retrieval, graph frames, semantic scores and evidence are all computed from
inference_view(inst, ., cond). Under `no_trace` the view has no trace, so nothing trace-derived reaches the pool.
"""
import time
import numpy as np
from src.agents.views import inference_view
from src.agents.evidence import build_evidence
from src.agents.specialists import specialist_prompt, parse_note, render_note
from src.agents.prompts import AGENTS, probability_agent_prompt, template_hash
from src.agents.run_values import permuted_view, GPU_USD_PER_S
from src.cands.retrieval import frame_functions
from src.cands.pool import build_pool
from src.market.sequential import run_market, avg_logprob
from src.llm.hashing import sha

ORDER_CHECK_MASKS = (0, 1, 2, 4, 8, 16, 31)   # PLAN 4.3: p_empty, singletons, p_full
ORDER_PERMS = (1, 2)                           # + stored order 0 = 3 candidate-order permutations
ROTATIONS = [AGENTS[i:] + AGENTS[:i] for i in range(1, 5)]  # B_full5 = rotation 0 (canonical mask 31) + these 4
FULL = 31


def condition_query(inst, cond):
    """The only text retrieval may see: the inference view's issue text and (test_informed only) its trace."""
    v = inference_view(inst, {"pool": [], "candidate_order": []}, cond)
    return v["issue_text"], v.get("trace")


def build_candidates(inst, cond, repo, emb, k=20):
    fns, bm, em, gr, repo_dir = repo
    q, trace = condition_query(inst, cond)
    frames = frame_functions(fns, trace, repo_dir) if trace else []
    lists = {"bm25": bm.rank(q, 40), "emb": em.rank(q, 40), "graph": gr.rank(frames, 40) if frames else []}
    return build_pool(inst["id"], lists, k, order_seed=int(sha(inst["id"] + "|" + cond), 16) % 10**6), q


def _tel(kind, inst_id, cond, prompts, outs, wall, gen=False):
    n = max(len(prompts), 1)
    return [{"id": inst_id, "condition": cond, "kind": kind, "prompt_hash": sha(p),
             "input_tokens": o["input_tokens"], "output_tokens": o.get("output_tokens", 0),
             "latency_ms": o["latency_ms"], "wall_share_s": wall / n, "est_usd": wall / n * GPU_USD_PER_S}
            for p, o in zip(prompts, outs)]


def _rows(view_id, mode, mask, perm, order, prompts_outs, disp_view, tel_cost):
    """value_row.v2 fields only, probs mapped from display order back to cid order."""
    out = []
    disp_cids = [c["cid"] for c in disp_view["pool"]]
    for (p, o), present in zip(prompts_outs, order):
        by_cid = dict(zip(disp_cids, zip(o["raw"], o["probs"])))
        cids = sorted(by_cid)
        probs = [by_cid[c][1] for c in cids]
        out.append({"id": view_id, "mode": mode, "subset_mask": mask if isinstance(mask, int) else mask.pop(0),
                    "order_perm": perm, "note_order": present, "raw_candidate_logits": [by_cid[c][0] for c in cids],
                    "probs": probs, "top1": cids[int(np.argmax(probs))], "token_mapping_hash": o["token_mapping_hash"],
                    "prompt_hash": sha(p), "model_revision": o["model_revision"], "inference_cost_usd": tel_cost})
    return out


def label_batch(srv, prompts, n):
    t0 = time.time()
    outs = srv.label_logits(prompts, [n] * len(prompts))
    return outs, time.time() - t0


def run_instance(inst, cond, repo, emb, srv, extras=True, perms=(0,), notes_override=None, cands_override=None,
                 mode_s=True, masks=None):
    """perms: candidate permutations every method is computed on (protocol avg2 = (0, 1)).
    masks: C-mode subsets to compute (None = all 32). B_full5 rotations are always computed. Rows for the computed
    masks are built exactly as in the full table (same prompts, same order within the batch); Mode S needs mask 0.
    notes_override: {agent: canonical note} from an earlier run of the same instance/condition/pool -- skips
    regeneration (temperature-0 notes are deterministic; reuse keeps notes identical and saves cost).
    cands_override: the stored pool those notes were written for (required with notes_override)."""
    fns, bm, em, gr, repo_dir = repo
    if notes_override is not None and cands_override is None:
        raise ValueError("reusing notes requires the pool they were written for")
    if cands_override is not None:
        cands, q = cands_override, condition_query(inst, cond)[0]
    else:
        cands, q = build_candidates(inst, cond, repo, emb)
    view = inference_view(inst, cands, cond)
    n = len(view["pool"])
    sem = dict(zip([c["cid"] for c in view["pool"]],
                   (np.array(emb([c["code"][:2000] for c in view["pool"]])) @ np.array(emb([q[:2000]])[0])).tolist()))
    ev = build_evidence(view, fns, gr, sem, repo_dir)
    tel, notes_rows, notes = [], [], {}

    # --- specialist notes (5 generations, one retry each if unparseable)
    prompts = [specialist_prompt(a, view, ev) for a in AGENTS]
    if notes_override is not None:
        notes = {a: notes_override[a]["text"] or "(no usable note)" for a in AGENTS}
        notes_rows = [notes_override[a] for a in AGENTS]
        raw = {a: "(reused from earlier run)" for a in AGENTS}
        prompts = []
    t0 = time.time(); gens = srv.generate(prompts, max_new_tokens=150, temperature=0.0, seed=0) if prompts else []; w = time.time() - t0
    tel += _tel("note", inst["id"], cond, prompts, gens, w, gen=True)
    raw = raw if notes_override is not None else {}
    for a, p, g in zip(AGENTS, prompts, gens):
        canon, ranked = parse_note(g["text"], view)
        if canon is None:
            p2 = p + "\n\nRemember: first line must be 'RANKED: X,Y,...'"
            t0 = time.time(); g = srv.generate([p2], max_new_tokens=150)[0]; w = time.time() - t0
            tel += _tel("note_retry", inst["id"], cond, [p2], [g], w, gen=True)
            canon, ranked = parse_note(g["text"], view)
        notes[a] = canon or "(no usable note)"; raw[a] = g["text"]
        notes_rows.append({"id": inst["id"], "agent": a, "copy_idx": 0, "text": canon or "", "ranked_suspects": ranked,
                           "prompt_hash": sha(p), "model_revision": g["model_revision"], "temperature": 0.0, "seed": 0,
                           "input_tokens": g["input_tokens"], "output_tokens": g["output_tokens"],
                           "latency_ms": g["latency_ms"]})

    rendered = {a: render_note(t, view) for a, t in notes.items()}
    values = []

    # --- C-mode 32-subset table (+ B_full5 rotations) for every protocol permutation, one batch per instance
    # (protocol "avg2", adopted at G1: every method averages log-probs over candidate permutations 0 and 1)
    pv_list = [(perm, permuted_view(view, perm)) for perm in perms]
    rn_by = {perm: {a: render_note(t_, pv) for a, t_ in notes.items()} for perm, pv in pv_list}
    ps, pres, meta = [], [], []
    for perm, pv in pv_list:
        for m in (range(32) if masks is None else masks):
            p, present = probability_agent_prompt(pv, rn_by[perm], m); ps.append(p); pres.append(present); meta.append((perm, m, pv))
        for rot in ROTATIONS:
            p, present = probability_agent_prompt(pv, rn_by[perm], FULL, order=list(rot)); ps.append(p); pres.append(present); meta.append((perm, FULL, pv))
    outs, w = label_batch(srv, ps, n)
    tel += _tel("C", inst["id"], cond, ps, outs, w)
    for (perm, m, pv), p, o, present in zip(meta, ps, outs, pres):
        values += _rows(inst["id"], "C", m, perm, [present], [(p, o)], pv, w / len(ps) * GPU_USD_PER_S)
    if mode_s and masks is not None and 0 not in masks:
        raise ValueError("Mode S starts from p_empty: masks must include 0")
    empties = [{c: pr for c, pr in zip(sorted(x["cid"] for x in view["pool"]), r["probs"])}
               for r in values if r["mode"] == "C" and r["subset_mask"] == 0]
    p_empty = None
    if empties:
        p_empty_d = empties[0] if len(empties) == 1 else avg_logprob(empties)
        p_empty = [p_empty_d[c] for c in sorted(p_empty_d)]

    k10 = []
    if extras:
        # --- PLAN 4.3 order sensitivity: 2 extra candidate display permutations on the check masks
        ps, pres, meta = [], [], []
        for perm in [x for x in ORDER_PERMS if x not in perms]:
            pv = permuted_view(view, perm); rn = {a: render_note(t_, pv) for a, t_ in notes.items()}
            for m in ORDER_CHECK_MASKS:
                p, present = probability_agent_prompt(pv, rn, m); ps.append(p); pres.append(present); meta.append((perm, m, pv))
        outs, w = label_batch(srv, ps, n)
        tel += _tel("order_perm", inst["id"], cond, ps, outs, w)
        for (perm, m, pv), p, o, present in zip(meta, ps, outs, pres):
            values += _rows(inst["id"], "C", m, perm, [present], [(p, o)], pv, w / len(ps) * GPU_USD_PER_S)
        # --- PLAN 4.3 K=10 vs K=20 (report only)
        c10 = build_pool(inst["id"], {"bm25": bm.rank(q, 40), "emb": em.rank(q, 40),
                                      "graph": gr.rank(frame_functions(fns, view["trace"], repo_dir), 40) if view.get("trace") else []},
                         10, order_seed=int(sha(inst["id"] + "|" + cond), 16) % 10**6)
        v10 = inference_view(inst, c10, cond); rn = {a: render_note(t_, v10) for a, t_ in notes.items()}
        ps10 = [probability_agent_prompt(v10, rn, m)[0] for m in (0, FULL)]
        outs, w = label_batch(srv, ps10, len(v10["pool"]))
        tel += _tel("k10", inst["id"], cond, ps10, outs, w)
        k10 = [{"id": inst["id"], "subset_mask": m, "candidate_keys": [f"{c['path']}::{c['qualname']}" for c in c10["pool"]],
                "probs": _rows(inst["id"], "C", m, 0, [[]], [(p, o)], v10, 0)[0]["probs"]}
               for m, p, o in zip((0, FULL), ps10, outs)]

    # --- Mode S: sequential market, canonical and reversed orders, starting from p_empty
    if mode_s:
        s_rows, s_tel = run_market(view, rendered, p_empty, srv, inst["id"], cond, views=pv_list, notes_by_view=rn_by)
        values += s_rows; tel += s_tel
    return {"candidates": cands, "notes": notes_rows, "values": values, "k10": k10, "telemetry": tel, "raw_notes": raw}
