"""Exact Shapley / marginal / pivotal credit over the 5-player evidence game (PLAN 8).
Interpretation: value of an evidence source given the fixed probability agent."""
from math import factorial

N = 5


def shapley(v):
    """v: sequence of 32 values indexed by subset mask, v[0] = 0 by construction."""
    phi = [0.0] * N
    for i in range(N):
        for S in range(1 << N):
            if S >> i & 1:
                continue
            s = bin(S).count("1")
            w = factorial(s) * factorial(N - s - 1) / factorial(N)
            phi[i] += w * (v[S | 1 << i] - v[S])
    return phi


def pivotal(v):
    full = (1 << N) - 1
    return [v[full] - v[full & ~(1 << i)] for i in range(N)]


def marginal(v, order):
    out, S = [0.0] * N, 0
    for i in order:
        out[i] = v[S | 1 << i] - v[S]
        S |= 1 << i
    return out
