import ast, random
from src.gen.mutate import candidates_in_function, apply, is_valid_mutant

SRC = '''import os

def target(xs, n, flag):
    """Doc é."""
    total = 0
    for x in xs:
        if x <= n and flag:
            total += x * 2
        log(x, n)
    return total + 1

def other(a):
    return a
'''


def test_all_operators_produce_minimal_parseable_edits():
    fn = [n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef)][0]
    muts = candidates_in_function(SRC, fn, random.Random(0))
    ops = {m.operator for m in muts}
    assert {"cmp_flip", "arith_swap", "bool_swap", "off_by_one", "arg_swap", "cond_removal", "return_change", "wrong_var", "stmt_deletion"} <= ops
    for m in muts:
        new = apply(SRC, m)
        assert is_valid_mutant(SRC, new), m
        # minimal edit: only lines within the span changed, nothing else reformatted
        a, b = SRC.split("\n"), new.split("\n")
        assert len(a) == len(b)
        changed = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert changed and all(m.span[0] <= c <= m.span[2] for c in changed)
        # never touches the other function
        assert "def other(a):\n    return a" in new


def test_cmp_flip_exact():
    fn = [n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef)][0]
    m = next(m for m in candidates_in_function(SRC, fn, random.Random(0)) if m.operator == "cmp_flip")
    assert "if x < n and flag:" in apply(SRC, m)
