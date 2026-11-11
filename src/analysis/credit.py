"""EVALUATOR SIDE. Evidence-value credit (E2) and Mode S market rewards (E4).

Gold-not-in-pool handling (fix 2026-11-10): log p(G) is -inf for every subset when gold is a retrieval miss, so
v(S) and r_i would be undefined (inf - inf). Those instances are EXCLUDED from Shapley / pivotal / marginal credit
and from market rewards, and reported as a count; they still count as misses in unconditional localization metrics.
Label: credit = "value of an evidence source given the fixed probability agent".
"""
import math
from src.market.shapley import shapley, pivotal, marginal, N

CREDIT_LABEL = "value of an evidence source given the fixed probability agent"


def gold_mass(probs_cid_sorted, cids_sorted, gcids):
    return sum(p for c, p in zip(cids_sorted, probs_cid_sorted) if c in gcids)


def value_vector(table, cids_sorted, gcids):
    """table: {mask: probs (sorted-cid order)} for the 32 canonical C rows. v(S) = log p_S(G) - log p_empty(G).
    Returns None if gold is not in the pool (excluded) -- never a vector containing inf/nan."""
    if not gcids:
        return None
    lg = [math.log(max(gold_mass(table[m], cids_sorted, gcids), 1e-300)) for m in range(1 << N)]
    return [x - lg[0] for x in lg]


def credit(table, cids_sorted, gcids, order=tuple(range(N))):
    v = value_vector(table, cids_sorted, gcids)
    if v is None:
        return None
    phi = shapley(v)
    assert abs(sum(phi) - v[(1 << N) - 1]) < 1e-6, "Shapley efficiency violated"
    return {"v_full": v[-1], "shapley": phi, "pivotal": pivotal(v), "marginal": marginal(v, order)}


def market_rewards(traj, cids_sorted, gcids):
    """traj: [p0, p1, ..., p5] (sorted-cid lists). r_k = log p_k(G) - log p_{k-1}(G). None if gold not in pool.
    Telescoping: sum(r) == log p5(G) - log p0(G)."""
    if not gcids:
        return None
    lg = [math.log(max(gold_mass(p, cids_sorted, gcids), 1e-300)) for p in traj]
    return [lg[k] - lg[k - 1] for k in range(1, len(lg))]
