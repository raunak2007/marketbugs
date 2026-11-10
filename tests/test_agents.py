from src.agents.views import inference_view
from src.agents.specialists import parse_note, render_note, specialist_prompt, label_maps
from src.agents.evidence import build_evidence, static_flags
from src.agents.run_values import permuted_view
from src.cands.pool import build_pool
from src.cands.functions import Fn
from tests.fixtures import make_instance


def _view(i=0):
    inst, c = make_instance(i)
    return inference_view(inst, c)


def test_parse_and_render_roundtrip_across_permutations():
    v = _view()
    lab2cid, cid2lab = label_maps(v)
    canon, ranked = parse_note("RANKED: C, A,F\n[C] compares with <= \n[A] also odd\nextra\nmore", v)
    assert ranked == [lab2cid["C"], lab2cid["A"], lab2cid["F"]]
    assert render_note(canon, v).startswith("RANKED: [C],[A],[F]")
    v2 = permuted_view(v, 3)
    _, cid2lab2 = label_maps(v2)
    assert render_note(canon, v2).startswith(f"RANKED: [{cid2lab2[ranked[0]]}]")


def test_unparseable_note():
    assert parse_note("I think it's the second one", _view()) == (None, [])


def test_all_specialist_prompts_build():
    v = _view(); ev = build_evidence(v, semantic_scores={c["cid"]: 0.1 for c in v["pool"]})
    for a in ("trace", "graph", "spec", "static", "semantic"):
        assert "RANKED:" in specialist_prompt(a, v, ev)


def test_static_flags():
    fl = static_flags("def f(xs, n, unused):\n    tmp = 3\n    if n <= len(xs):\n        return xs[n - 1]\n        print(1)\n    return None")
    s = " | ".join(fl)
    assert "boundary" in s and "+/-1" in s and "unreachable" in s and "tmp" in s and "unused" in s


def _fn(i):
    return Fn(f"p/{i}.py", f"f{i}", 1, 2, f"def f{i}(): pass", "", 1)


def test_pool_interleave_dedupe_truncate_seeded():
    fs = [_fn(i) for i in range(40)]
    lists = {"bm25": [(f, 1.0) for f in fs[:25]], "emb": [(f, 1.0) for f in fs[10:35]], "graph": [(fs[30], 2.0)]}
    p = build_pool("x", lists, 20, 7)
    keys = [c["qualname"] for c in p["pool"]]
    assert len(keys) == 20 == len(set(keys)) and keys[0] == "f30"
    assert p["candidate_order"] == build_pool("x", lists, 20, 7)["candidate_order"]
    assert "gold_in_pool" not in p
