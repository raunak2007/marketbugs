"""Gold-free method aggregation shared by src.analysis.freeze_dev and src.analysis.evaluate_test.

Input: inference-only value rows of ONE instance (no gold fields read). Output: one distribution per method, in
sorted-cid order. Imports nothing from src.agents / src.llm (the evaluator must not import inference code).

Aggregation order (protocol v1.0 section 2):
  1. candidate-order averaging FIRST: for every (mode, mask, note order), mean of log-probs over the protocol's
     candidate permutations, then renormalize (`perm_avg`);
  2. note-order averaging SECOND: B_full5 = arithmetic mean of the five cyclic full-note-order rotations, each
     already permutation-averaged, then renormalize.
Missing rows, duplicate keys, wrong lengths or non-finite values raise MethodDataError (hard failure).
"""
import math
import numpy as np
from src.market import pools

AGENTS = ("trace", "graph", "spec", "static", "semantic")  # must equal src.agents.prompts.AGENTS (tested)
FULL = 31
ROT = [tuple(AGENTS[i:] + AGENTS[:i]) for i in range(5)]
S_ORDERS = {"canonical": AGENTS, "reversed": tuple(reversed(AGENTS))}
PROTOCOL_PERMS = {"single": (0,), "avg2": (0, 1)}
CORE = ("p_empty", "p_full", "B_full5", "P_hedge") + tuple(f"single.{a}" for a in AGENTS)


class MethodDataError(ValueError):
    pass


def subset(mask):
    return tuple(a for i, a in enumerate(AGENTS) if mask >> i & 1)


def check_dist(p, n, what):
    p = np.asarray(p, float)
    if p.ndim != 1 or len(p) != n:
        raise MethodDataError(f"{what}: expected {n} probabilities, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or abs(p.sum() - 1) > 1e-4:
        raise MethodDataError(f"{what}: not a finite probability distribution (sum={p.sum()})")
    return p


def index_rows(rows, n, what="", mode_field="mode"):
    """{(mode, mask, perm, note_order): probs}; duplicates are a hard error."""
    idx = {}
    for r in rows:
        k = (r.get(mode_field, "P"), r["subset_mask"], r["order_perm"], tuple(r["note_order"]))
        if k in idx:
            raise MethodDataError(f"{what}: duplicate row {k}")
        idx[k] = check_dist(r["probs"], n, f"{what} {k}")
    return idx


def perm_avg(dists):
    """Mean of log-probabilities over candidate permutations, renormalized."""
    if len(dists) == 1:
        return np.asarray(dists[0], float)
    lp = np.mean([np.log(np.clip(np.asarray(d, float), 1e-300, 1)) for d in dists], 0)
    return pools.normalize(np.exp(lp - lp.max()))


def get(idx, mode, mask, order, perms, what=""):
    ps = []
    for q in perms:
        k = (mode, mask, q, tuple(order))
        if k not in idx:
            raise MethodDataError(f"{what}: missing row {k}")
        ps.append(idx[k])
    return perm_avg(ps)


def b_full5(idx, perms, what=""):
    return pools.normalize(np.mean([get(idx, "C", FULL, r, perms, what) for r in ROT], 0))


def method_dists(rows, n, perms, hedge_w=None, with_s=True, what=""):
    """rows: value rows of one instance. hedge_w: frozen Hedge weights in AGENTS order (None -> no P_hedge)."""
    idx = index_rows(rows, n, what)
    D = {"p_empty": get(idx, "C", 0, (), perms, what), "p_full": get(idx, "C", FULL, AGENTS, perms, what),
         "B_full5": b_full5(idx, perms, what)}
    S = np.stack([get(idx, "C", 1 << b, (a,), perms, what) for b, a in enumerate(AGENTS)])
    for b, a in enumerate(AGENTS):
        D[f"single.{a}"] = S[b]
    D["P_avg"], D["P_hedge_eq"] = pools.p_avg(S), pools.log_pool(S, np.ones(5))
    if hedge_w is not None:
        D["P_hedge"] = pools.log_pool(S, np.asarray(hedge_w, float))
    if with_s:
        for nm, seq in S_ORDERS.items():
            try:
                D[f"S_{nm}"] = get(idx, "S", FULL, seq, perms, what)
            except MethodDataError:
                D.pop("S_canonical", None)
                break
        if "S_canonical" in D and "S_reversed" in D:
            D["S_mean"] = pools.normalize((D["S_canonical"] + D["S_reversed"]) / 2)
    for k, p in D.items():
        check_dist(p, n, f"{what} method {k}")
    return D, S


def placebo_dist(prows, n, perms, what=""):
    idx = index_rows(prows, n, what, mode_field="arm")
    p = get(idx, "placebo_full", FULL, AGENTS, perms, what)
    if len(idx) != len(perms):
        raise MethodDataError(f"{what}: expected {len(perms)} placebo rows, got {len(idx)}")
    return check_dist(p, n, f"{what} placebo_full")


def disagreement(idx, mask, order, a=0, b=1):
    """Top-1 disagreement between two candidate permutations for one (mask, order); None if either missing."""
    ka, kb = ("C", mask, a, tuple(order)), ("C", mask, b, tuple(order))
    if ka not in idx or kb not in idx:
        return None
    return int(np.argmax(idx[ka]) != np.argmax(idx[kb]))


def logscore(p, gold_idx):
    m = float(sum(p[g] for g in gold_idx))
    return math.log(m) if m > 0 else float("-inf")
