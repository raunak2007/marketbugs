import random
from src.market.shapley import shapley, pivotal, marginal


def test_efficiency_random_games():
    rng = random.Random(0)
    for _ in range(200):
        v = [0.0] + [rng.uniform(-3, 3) for _ in range(31)]
        assert abs(sum(shapley(v)) - v[31]) < 1e-6


def test_additive_game():
    w = [0.5, -1.0, 2.0, 0.0, 0.3]
    v = [sum(w[i] for i in range(5) if S >> i & 1) for S in range(32)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(shapley(v), w))
    assert all(abs(a - b) < 1e-12 for a, b in zip(pivotal(v), w))
    assert all(abs(a - b) < 1e-12 for a, b in zip(marginal(v, [4, 3, 2, 1, 0]), w))


def test_negative_credit_not_clipped():
    v = [0.0] * 32
    for S in range(32):
        if S & 1:
            v[S] = -1.0
    assert shapley(v)[0] == -1.0
