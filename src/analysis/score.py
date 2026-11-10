"""EVALUATOR SIDE. The only module allowed to read gold_* fields. Never imported by src/agents or src/llm."""
import math
import numpy as np


def gold_cids(instance, cands):
    gold = set(instance["gold_functions"])
    return [c["cid"] for c in cands["pool"] if f"{c['path']}::{c['qualname']}" in gold]


def score_row(row, gcids):
    cids = sorted(c for c in "ABCDEFGHIJKLMNOPQRST"[: len(row["probs"])])
    mass = sum(p for c, p in zip(cids, row["probs"]) if c in gcids)
    row["gold_logscore"] = math.log(mass) if mass > 0 else float("-inf")
    return row


def metrics(probs, cids, gcids):
    """Top-1/3/5, reciprocal rank, NLL, Brier for one distribution. Gold not in pool => miss, NLL=None."""
    probs = np.asarray(probs)
    order = [cids[i] for i in np.argsort(-probs, kind="stable")]
    ranks = [order.index(g) + 1 for g in gcids if g in order]
    r = min(ranks) if ranks else None
    y = np.array([c in gcids for c in cids], float)
    mass = float(probs[y.astype(bool)].sum()) if y.any() else 0.0
    return {"top1": int(r == 1) if r else 0, "top3": int(bool(r) and r <= 3), "top5": int(bool(r) and r <= 5),
            "rr": 1.0 / r if r else 0.0, "nll": -math.log(mass) if mass > 0 else None,
            "brier": float(((probs - y) ** 2).sum()), "conf": float(probs.max()), "gold_in_pool": bool(y.any())}


def ece(confs, corrects, bins=10):
    confs, corrects = np.asarray(confs), np.asarray(corrects, float)
    e, n = 0.0, len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (confs > lo) & (confs <= hi) if b else (confs >= lo) & (confs <= hi)
        if m.any():
            e += m.sum() / n * abs(confs[m].mean() - corrects[m].mean())
    return e


def random_top1(n_pool, gold_in_pool):
    return (1.0 / n_pool) if gold_in_pool else 0.0
