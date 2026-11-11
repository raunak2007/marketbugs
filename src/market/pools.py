"""Pooling of the five C-mode singleton posteriors + dev fitting (PLAN 7; G2).

Labels (PLAN naming discipline): P_hedge with dev-fit weights is a *learned supervised pool*, not a market.
Fitting functions take gold log-scores as plain numbers supplied by the evaluator; nothing here reads gold fields.
"""
import numpy as np

ETAS = (0.1, 0.3, 1.0, 3.0)
TEMPS = tuple(np.round(np.exp(np.linspace(np.log(0.25), np.log(4.0), 25)), 4))
EPS = 1e-12


def normalize(p):
    p = np.clip(np.asarray(p, float), 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1.0 / len(p))


def temper(p, T):
    lp = np.log(np.clip(np.asarray(p, float), EPS, 1)) / T
    return normalize(np.exp(lp - lp.max()))


def p_vote(S):
    S = np.asarray(S); v = np.bincount(S.argmax(1), minlength=S.shape[1]).astype(float)
    return v / v.sum()


def p_avg(S):
    return normalize(np.asarray(S).mean(0))


def log_pool(S, w):
    w = np.asarray(w, float); w = w / w.sum()
    lp = (w[:, None] * np.log(np.clip(np.asarray(S), EPS, 1))).sum(0)
    return normalize(np.exp(lp - lp.max()))


def hedge_weights(logscores, eta):
    """logscores: (n_instances, 5) log p_i(G) over the dev stream in its seeded order. w_i ~ exp(eta * G_i)."""
    G = np.asarray(logscores, float).sum(0)
    z = eta * (G - G.max())
    w = np.exp(z)
    return w / w.sum()


def fit_eta(S_list, gold_idx, seed=20261110):
    """2-fold CV on dev (gold-in-pool instances only). S_list: list of (5, n_cand) singleton posteriors.
    Returns (best_eta, {eta: mean held-out NLL})."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(S_list)); folds = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    ls = np.array([[np.log(max(S[i][g], EPS)) for i in range(S.shape[0])] for S, g in zip(map(np.asarray, S_list), gold_idx)])
    res = {}
    for eta in ETAS:
        nll = []
        for a, b in ((0, 1), (1, 0)):
            w = hedge_weights(ls[folds[a]], eta)
            nll += [-np.log(max(log_pool(S_list[j], w)[gold_idx[j]], EPS)) for j in folds[b]]
        res[eta] = float(np.mean(nll))
    best = min(ETAS, key=lambda e: (res[e], e))
    return best, res


def fit_temperature(P_list, gold_idx):
    """One scalar T per method by dev NLL (gold-in-pool only). Returns (T, nll_at_T)."""
    best = None
    for T in TEMPS:
        nll = float(np.mean([-np.log(max(temper(p, T)[g], EPS)) for p, g in zip(P_list, gold_idx)]))
        if best is None or nll < best[1] - 1e-12:
            best = (float(T), nll)
    return best
