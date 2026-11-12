"""Protocol v1.0 section 5: required checks before any held-out run. Synthetic data only; no model, no test split."""
import copy, csv, json, math, re
from pathlib import Path
import numpy as np
import pytest
from src import frozen
from src.agents.views import inference_view
from src.agents.prompts import AGENTS as P_AGENTS, probability_agent_prompt
from src.agents.specialists import specialist_prompt, render_note
from src.agents.evidence import build_evidence
from src.agents.run_values import permuted_view
from src.agents.placebo import placebo_prompts
from src.agents import run_test
from src.market.sequential import market_prompt
from src.market.placebo import assign_donors, derangement, assignment_hash
from src.market import pools
from src.analysis import methods as M
from src.analysis import evaluate_test as E
from src.analysis import bootstrap_test as B
from src.analysis.score import metrics
from tests.fixtures import make_instance
from tests.test_no_leak import forbidden_strings, permitted_text

ROOT = Path(__file__).resolve().parents[1]
PERMS = (0, 1)
REV, TMH = "fake/model@0", "fakehash"
ROT = [tuple(P_AGENTS[i:] + P_AGENTS[:i]) for i in range(5)]


def mk_test_instance(i, repo):
    inst, cands = make_instance(i)
    inst = dict(inst, id=f"mb-{repo}-{i:04d}", repo=repo, split="test")
    cands = dict(cands, id=inst["id"])
    return inst, cands


def canon_notes(i):
    return {a: f"RANKED: [{{{chr(65 + (i + k) % 20)}}}],[{{B}}]\n{a} suspects [{{{chr(65 + (i + k) % 20)}}}] (inst {i})"
            for k, a in enumerate(P_AGENTS)}


# ------------------------------------------------------------------ 1. leakage
def _all_arm_prompts(inst, cands, donor_canon):
    view = inference_view(inst, cands, "test_informed")
    ev = build_evidence(view)
    out = [specialist_prompt(a, view, ev) for a in P_AGENTS]
    for perm in PERMS:
        pv = permuted_view(view, perm)
        rn = {a: render_note(t, pv) for a, t in canon_notes(0).items()}
        out += [probability_agent_prompt(pv, rn, m)[0] for m in range(32)]
        out += [probability_agent_prompt(pv, rn, 31, order=list(r))[0] for r in ROT[1:]]
        out += [market_prompt(pv, a, rn[a], {c["cid"]: 0.05 for c in pv["pool"]}) for a in P_AGENTS]
    out += [j[2] for j in placebo_prompts(view, donor_canon, PERMS)]
    return out


@pytest.mark.parametrize("i", range(10))
def test_1_test_inference_cannot_see_evaluator_fields(i):
    inst, cands = mk_test_instance(i, "r1")
    inst.update(gold_in_pool=True, FAIL_TO_PASS=[f"F2P_LEAK_{i}"], patch=f"PATCH_LEAK_{i}")
    v = inference_view(inst, cands, "test_informed")
    assert not set(v) & {"gold_functions", "mutation_operator", "mutation_span", "gold_in_pool", "patch",
                         "FAIL_TO_PASS", "test_patch"}
    allowed = permitted_text(inst)
    for p in _all_arm_prompts(inst, cands, canon_notes(i + 1)):
        for s in forbidden_strings(inst) + [f"F2P_LEAK_{i}", f"PATCH_LEAK_{i}"]:
            assert not (s in p and s not in allowed), f"leak {s!r}"


def test_1_new_inference_modules_never_read_gold():
    bad = re.compile(r"""["'](gold_functions|gold_in_pool|mutation_\w+|test_patch|FAIL_TO_PASS)["']|^\s*(from|import)\s+src\.analysis""")
    for f in ("src/agents/run_test.py", "src/agents/placebo.py", "src/market/placebo.py", "src/frozen.py"):
        hits = [l for l in (ROOT / f).read_text().splitlines() if bad.search(l) and not l.lstrip().startswith("#")]
        assert not hits, (f, hits)


# ------------------------------------------------------------------ 2. gold-invariance of every prompt
def test_2_changing_gold_leaves_all_prompts_byte_identical():
    for i in range(5):
        inst, cands = mk_test_instance(i, "r1")
        a = _all_arm_prompts(inst, cands, canon_notes(i + 1))
        inst2 = dict(inst, gold_functions=[f"{cands['pool'][3]['path']}::{cands['pool'][3]['qualname']}"],
                     mutation_operator="OTHER", mutation_span=[1, 2, 3, 4], stratum="far")
        b = _all_arm_prompts(inst2, cands, canon_notes(i + 1))
        assert a == b


# ------------------------------------------------------------------ 3. donors
def test_3_derangement_properties():
    ids = [(f"mb-{r}-{k:04d}", r) for r in ("attrs", "toolz", "tomli") for k in range(30)]
    a1, a2 = assign_donors(ids, 20261111), assign_donors(list(reversed(ids)), 20261111)
    assert a1 == a2  # deterministic, input-order independent
    repo = dict(ids)
    for rec, don in a1.items():
        assert rec != don and repo[rec] == repo[don]
    assert sorted(a1.values()) == sorted(a1)  # a bijection within each repo
    assert assign_donors(ids, 1) != a1
    assert assignment_hash(a1) == assignment_hash(dict(reversed(list(a1.items()))))
    with pytest.raises(ValueError):
        derangement(["only"], 1, "x")


def test_3_assignment_ignores_gold():
    rows = [mk_test_instance(k, "r1")[0] for k in range(8)]
    a = assign_donors([(r["id"], r["repo"]) for r in rows])
    for r in rows:
        r["gold_functions"] = ["changed::fn"]; r["mutation_operator"] = "X"
    assert assign_donors([(r["id"], r["repo"]) for r in rows]) == a  # signature takes ids + repos only


# ------------------------------------------------------------------ 4. genuine vs placebo differ only in notes
def test_4_placebo_keeps_everything_but_notes():
    inst, cands = mk_test_instance(3, "r1")
    view = inference_view(inst, cands, "test_informed")
    real = placebo_prompts(view, canon_notes(3), PERMS)    # real arm = same builder with own notes
    plc = placebo_prompts(view, canon_notes(9), PERMS)
    for (pr, vr, p1, o1, _), (pp, vp, p2, o2, _) in zip(real, plc):
        assert pr == pp and [c["cid"] for c in vr["pool"]] == [c["cid"] for c in vp["pool"]]
        assert vr["pool"] == vp["pool"] and o1 == o2 == list(P_AGENTS)
        h1, t1 = p1.split("\n\nSpecialist notes:\n"); h2, t2 = p2.split("\n\nSpecialist notes:\n")
        assert h1 == h2  # issue, trace, candidate block (code + labels), in the same display order
        assert t1.splitlines()[-1] == t2.splitlines()[-1]  # same question / label range
        assert p1 != p2
    # and the real-arm builder matches pipeline's mask-31 canonical prompt exactly
    pv = permuted_view(view, 1)
    rn = {a: render_note(t, pv) for a, t in canon_notes(3).items()}
    assert probability_agent_prompt(pv, rn, 31)[0] == real[1][2]


# ------------------------------------------------------------------ 5. aggregation order
def test_5_candidate_then_note_order_averaging():
    rng = np.random.RandomState(0)
    rows, raw = [], {}
    for perm in PERMS:
        for r in ROT:
            p = rng.dirichlet(np.ones(20) * 0.3); raw[(perm, r)] = p
            rows.append({"mode": "C", "subset_mask": 31, "order_perm": perm, "note_order": list(r), "probs": list(p)})
    idx = M.index_rows(rows, 20)
    got = M.b_full5(idx, PERMS)
    geo = lambda a, b: pools.normalize(np.exp((np.log(a) + np.log(b)) / 2))
    want = pools.normalize(np.mean([geo(raw[(0, r)], raw[(1, r)]) for r in ROT], 0))
    assert np.allclose(got, want, atol=1e-12)
    swapped_arith_perm = pools.normalize(np.mean([(raw[(0, r)] + raw[(1, r)]) / 2 for r in ROT], 0))
    swapped_order = geo(pools.normalize(np.mean([raw[(0, r)] for r in ROT], 0)),
                        pools.normalize(np.mean([raw[(1, r)] for r in ROT], 0)))
    assert not np.allclose(got, swapped_arith_perm, atol=1e-6) and not np.allclose(got, swapped_order, atol=1e-6)
    # perm averaging is the renormalized geometric mean
    a, b = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    assert np.allclose(M.perm_avg([a, b]), pools.normalize(np.sqrt(a * b)))


def test_5_method_agents_match_prompt_agents():
    assert M.AGENTS == P_AGENTS


# ------------------------------------------------------------------ synthetic complete run for 6-10
def _probs(rng):
    p = rng.dirichlet(np.ones(20) * 0.5)
    return [float(x) for x in p]


def synth_run(tmp, repos=("r1", "r2", "r3"), n=4, miss=("r1", 0)):
    d = tmp / "runs" / "t" / "test_informed"; d.mkdir(parents=True)
    rng = np.random.RandomState(1)
    rows = []
    for r in repos:
        for k in range(n):
            inst, cands = mk_test_instance(k, r)
            g = cands["pool"][(k * 7) % 20]
            inst["gold_functions"] = [] if (r, k) == miss else [f"{g['path']}::{g['qualname']}"]
            rows.append((inst, cands))
    expected = [(i["id"], i["repo"], i["stratum"]) for i, _ in rows]
    assign = assign_donors([(i, r) for i, r, _ in expected], 20261111)
    files = {k: [] for k in ("candidates_inf", "values_inf", "notes", "telemetry", "placebo_values_inf", "placebo_manifest")}
    base = lambda i: {"raw_candidate_logits": [0.0] * 20, "token_mapping_hash": TMH, "prompt_hash": "h" * 16,
                      "model_revision": REV, "inference_cost_usd": 0.0, "id": i}
    for inst, cands in rows:
        i = inst["id"]
        files["candidates_inf"].append(cands)
        for k, a in enumerate(P_AGENTS):
            files["notes"].append({"id": i, "agent": a, "copy_idx": 0, "text": canon_notes(k)[a], "ranked_suspects": ["A"],
                                   "prompt_hash": "n" * 16, "model_revision": REV, "temperature": 0.0, "seed": 0,
                                   "input_tokens": 100, "output_tokens": 100 + 10 * k, "latency_ms": 1.0})
            files["telemetry"].append({"id": i, "kind": "note", "prompt_hash": "n", "input_tokens": 100, "output_tokens": 100,
                                       "latency_ms": 1.0, "est_usd": 0.001})
        for perm in PERMS:
            for m in range(32):
                p = _probs(rng)
                files["values_inf"].append({**base(i), "mode": "C", "subset_mask": m, "order_perm": perm,
                                            "note_order": list(M.subset(m)), "probs": p, "top1": chr(65 + int(np.argmax(p))),
                                            "prompt_hash": f"{i}-{perm}-{m}"[-16:]})
                files["telemetry"].append({"id": i, "kind": "C", "prompt_hash": f"{i}-{perm}-{m}"[-16:], "input_tokens": 500,
                                           "output_tokens": 0, "latency_ms": 1.0, "est_usd": 0.0001})
            for r in ROT[1:]:
                p = _probs(rng)
                files["values_inf"].append({**base(i), "mode": "C", "subset_mask": 31, "order_perm": perm,
                                            "note_order": list(r), "probs": p, "top1": chr(65 + int(np.argmax(p)))})
            for seq in (P_AGENTS, tuple(reversed(P_AGENTS))):
                p = _probs(rng)
                files["values_inf"].append({**base(i), "mode": "S", "subset_mask": 31, "order_perm": perm,
                                            "note_order": list(seq), "probs": p, "top1": chr(65 + int(np.argmax(p)))})
            p = _probs(rng)
            files["placebo_values_inf"].append({**base(i), "arm": "placebo_full", "subset_mask": 31, "order_perm": perm,
                                                "note_order": list(P_AGENTS), "donor_id": assign[i], "probs": p,
                                                "top1": chr(65 + int(np.argmax(p))), "input_tokens": 520})
            files["telemetry"].append({"id": i, "kind": "placebo", "prompt_hash": "p", "input_tokens": 520,
                                       "output_tokens": 0, "latency_ms": 1.0, "est_usd": 0.0001})
        for a in P_AGENTS:
            files["placebo_manifest"].append({"id": i, "repo": inst["repo"], "specialist": a, "donor_id": assign[i],
                                              "seed": 20261111, "source_note_sha256": "0" * 64,
                                              "donor_note_output_tokens": 100, "rendered_prompt_sha256": {"0": "1" * 64}})
    for k, v in files.items():
        (d / f"{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
    params = {"protocol": "avg2", "placebo_seed": 20261111, "condition": "test_informed", "code_commit": "x",
              "model": {"revision_full": REV, "token_mapping_hash": TMH, "dtype": "bfloat16"},
              "hedge_weights": {a: w for a, w in zip(P_AGENTS, (0.4, 0.2, 0.2, 0.1, 0.1))}, "eta": 0.1,
              "temperatures": {m: 1.5 for m in list(M.CORE) + ["placebo_full", "S_canonical", "S_reversed", "S_mean", "P_avg", "P_hedge_eq"]},
              "bootstrap_seed": 20261110}
    kv = {"frozen_params_sha256": "f" * 64}
    (d / "run_manifest.json").write_text(json.dumps({"frozen_params_sha256": kv["frozen_params_sha256"], "protocol": "avg2",
                                                     "placebo_seed": 20261111, "arms": ["placebo", "real"], "mode_s": True}))
    (d / "run_status.json").write_text(json.dumps({"complete": True, "stopped": None}))
    (d / "placebo_assignment.json").write_text(json.dumps({"sha256": assignment_hash(assign)}))
    inst = {i["id"]: i for i, _ in rows}
    return d, params, kv, expected, inst


def _audit(d, params, kv, expected, **kw):
    A = E.audit(d, params, kv, expected, len(expected), **kw); A["_dir"] = d
    return A


# ------------------------------------------------------------------ 6. frozen params only
def test_6_evaluator_never_fits_and_hedge_ignores_test_gold(tmp_path):
    src = (ROOT / "src/analysis/evaluate_test.py").read_text()
    assert not re.search(r"fit_eta|fit_temperature|hedge_weights\(|freeze_dev", src)
    d, params, kv, expected, inst = synth_run(tmp_path)
    A = _audit(d, params, kv, expected)
    _, S1 = E.evaluate(A, params, inst, expected)
    inst2 = copy.deepcopy(inst)
    for r in inst2.values():
        r["gold_functions"] = ["pkg/m0.py::cand_0_0"]
    A2 = _audit(d, params, kv, expected)
    for i in A["ids"]:
        assert np.array_equal(A["D"][i]["P_hedge"], A2["D"][i]["P_hedge"])
        S = np.stack([A["D"][i][f"single.{a}"] for a in P_AGENTS])
        assert np.allclose(A["D"][i]["P_hedge"], pools.log_pool(S, [params["hedge_weights"][a] for a in P_AGENTS]))
    assert S1["methods"]["P_hedge"]["temperature"] == 1.5


def test_6_freeze_dev_reads_dev_split_only(tmp_path):
    from src.analysis import freeze_dev
    (tmp_path / "data/marketbugs/dev").mkdir(parents=True)
    (tmp_path / "data/marketbugs/dev/x.jsonl").write_text(json.dumps({"id": "a", "split": "test"}) + "\n")
    with pytest.raises(SystemExit):
        freeze_dev.dev_instances(tmp_path)


# ------------------------------------------------------------------ 7. hard failures
def _mutate(d, fname, fn):
    p = d / fname
    rows = [json.loads(l) for l in open(p)]
    (d / fname).write_text("".join(json.dumps(r) + "\n" for r in fn(rows)))


BREAKS = {
    "missing_instance": ("candidates_inf.jsonl", lambda rs: rs[1:]),
    "duplicate_instance": ("candidates_inf.jsonl", lambda rs: rs + rs[:1]),
    "missing_permutation": ("values_inf.jsonl", lambda rs: [r for r in rs if not (r["order_perm"] == 1 and r["subset_mask"] == 0 and r["mode"] == "C")]),
    "missing_rotation": ("values_inf.jsonl", lambda rs: [r for r in rs if not (r["mode"] == "C" and tuple(r["note_order"]) == ROT[2])]),
    "nonfinite": ("values_inf.jsonl", lambda rs: [dict(r, probs=[float("nan")] * 20) if k == 3 else r for k, r in enumerate(rs)]),
    "wrong_length": ("values_inf.jsonl", lambda rs: [dict(r, probs=r["probs"][:19]) if k == 3 else r for k, r in enumerate(rs)]),
    "model_revision": ("values_inf.jsonl", lambda rs: [dict(r, model_revision="other@1") if k == 5 else r for k, r in enumerate(rs)]),
    "schema_invalid": ("values_inf.jsonl", lambda rs: [dict(r, extra_field=1) if k == 7 else r for k, r in enumerate(rs)]),
    "duplicate_row": ("values_inf.jsonl", lambda rs: rs + rs[:1]),
    "placebo_missing_perm": ("placebo_values_inf.jsonl", lambda rs: rs[1:]),
    "placebo_wrong_donor": ("placebo_values_inf.jsonl", lambda rs: [dict(r, donor_id=r["id"]) if k == 0 else r for k, r in enumerate(rs)]),
    "manifest_self_donor": ("placebo_manifest.jsonl", lambda rs: [dict(r, donor_id=r["id"]) if k == 0 else r for k, r in enumerate(rs)]),
    "placebo_revision": ("placebo_values_inf.jsonl", lambda rs: [dict(r, model_revision="x@2") if k == 2 else r for k, r in enumerate(rs)]),
}


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_7_audit_hard_fails(tmp_path, name):
    d, params, kv, expected, inst = synth_run(tmp_path)
    _audit(d, params, kv, expected)  # clean run passes
    f, fn = BREAKS[name]
    _mutate(d, f, fn)
    with pytest.raises(E.AuditError):
        _audit(d, params, kv, expected)


def test_7_incomplete_status_and_frozen_mismatch_fail(tmp_path):
    d, params, kv, expected, inst = synth_run(tmp_path)
    (d / "run_status.json").write_text(json.dumps({"complete": False, "stopped": "budget guard"}))
    with pytest.raises(E.AuditError):
        _audit(d, params, kv, expected)
    A = _audit(d, params, kv, expected, allow_partial=True)
    assert A["partial"]
    with pytest.raises(E.AuditError):
        _audit(d, params, {"frozen_params_sha256": "0" * 64}, expected)


# ------------------------------------------------------------------ 8. retrieval misses score zero
def test_8_gold_missing_from_pool_scores_zero(tmp_path):
    m = metrics(np.full(20, 0.05), [chr(65 + k) for k in range(20)], [])
    assert (m["top1"], m["top3"], m["top5"], m["rr"], m["nll"]) == (0, 0, 0, 0.0, None)
    d, params, kv, expected, inst = synth_run(tmp_path)
    per, S = E.evaluate(_audit(d, params, kv, expected), params, inst, expected)
    miss = [r for r in per if not r["gold_in_pool"]]
    assert len(miss) == 1 and S["n_retrieval_miss"] == 1 and S["n_instances"] == len(expected)
    for m_ in ("p_empty", "p_full", "B_full5", "P_hedge", "placebo_full"):
        assert all(miss[0][f"{m_}.{k}"] == 0 for k in ("top1", "top3", "top5", "rr"))
        assert miss[0][f"{m_}.nll_T"] is None
        assert S["methods"][m_]["all"]["n"] == len(expected) and S["methods"][m_]["nll_excluded_retrieval_miss"] == 1
    out = tmp_path / "val"
    E.write_outputs(out, per, S, {"run_status": {}, **{k: None for k in ("head", "frozen_code_commit", "frozen_params_sha256",
                    "code_changed_since_freeze", "model", "protocol", "seeds", "hedge", "evaluated_utc")}}, {"partial": False, "missing": []})
    assert (out / "per_instance.csv").exists() and "diff.primary" in (out / "per_instance.csv").read_text().splitlines()[0]


# ------------------------------------------------------------------ 9. bootstrap
def test_9_bootstrap_two_stage_and_fixed():
    # repo-constant differences: within-repo resampling adds no variance, so every replicate mean is a mean of
    # resampled REPO values -> proves repos are resampled at stage 1
    groups = [np.full(30, 1.0), np.full(30, 0.0), np.full(30, 0.0)]
    reps = B.cluster_bootstrap(groups, 2000, 7)
    assert set(np.round(reps * 3).astype(int)) <= {0, 1, 2, 3}
    assert len(set(np.round(reps, 6))) == 4
    # one repo with mixed values: variance comes only from stage-2 within-repo resampling
    reps2 = B.cluster_bootstrap([np.array([0.0, 1.0] * 15)], 2000, 7)
    assert reps2.std() > 0 and abs(reps2.mean() - 0.5) < 0.02
    rows = []
    for r, v in zip("abcdefg", (1, 0, 1, 0, -1, 1, 0)):
        for k in range(30):
            rows.append({"repo": r, "diff.primary": str(v if k % 3 else 0)})
    res1 = B.run(rows, 20261110, 1000)
    res2 = B.run(rows, 20261110, 1000)
    assert res1 == res2
    c = res1["contrasts"]["primary"]
    assert c["n"] == 210 and c["n_repos"] == 7 and abs(c["observed"] - 2 / 7 * 2 / 3) < 1e-12
    assert c["cluster_ci95"][0] <= c["observed"] <= c["cluster_ci95"][1]
    # cluster CI is wider than the instance CI when differences cluster by repo
    ci, ii = c["cluster_ci95"], c["instance_ci95_diagnostic_only"]
    assert ci[1] - ci[0] > ii[1] - ii[0]
    assert "omitted" in res1["contrasts"]["real_vs_placebo"]
    assert B.verdict(-0.1, 0.2, "confirmatory").startswith("inconclusive")
    assert B.verdict(0.01, 0.2, "confirmatory").startswith("positive")


# ------------------------------------------------------------------ 10. resumption + immutable provenance
def test_10_reconcile_drops_partial_rows_and_rejects_duplicates(tmp_path):
    d, params, kv, expected, inst = synth_run(tmp_path)
    good = d / "candidates_inf.jsonl"
    lines = good.read_text().splitlines()
    last_id = json.loads(lines[-1])["id"]
    good.write_text("\n".join(lines[:-1]) + "\n")  # last instance crashed before its done marker
    pv = [json.loads(l) for l in open(d / "placebo_values_inf.jsonl")]
    first = pv[0]["id"]
    (d / "placebo_values_inf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pv[1:]))  # half-written placebo
    done, pdone = run_test.reconcile(d, PERMS)
    assert last_id not in done and first not in pdone
    for f in ("values_inf", "notes", "telemetry", "placebo_values_inf", "placebo_manifest"):
        rows = [json.loads(l) for l in open(d / f"{f}.jsonl")]
        assert not any(r["id"] == last_id for r in rows), f
    assert not any(r["id"] == first for r in map(json.loads, open(d / "placebo_manifest.jsonl")))
    assert not any(r["id"] == first and r["kind"] == "placebo" for r in map(json.loads, open(d / "telemetry.jsonl")))
    before = {f: (d / f).read_text() for f in ("values_inf.jsonl", "telemetry.jsonl")}
    run_test.reconcile(d, PERMS)  # idempotent
    assert before == {f: (d / f).read_text() for f in before}
    good.write_text(good.read_text() + good.read_text().splitlines()[0] + "\n")
    with pytest.raises(RuntimeError):
        run_test.reconcile(d, PERMS)


def test_10_run_manifest_is_immutable(tmp_path):
    class A:
        protocol, arms, seed, backend, no_mode_s, extras = "avg2", {"real", "placebo"}, 20261111, "local", False, False
    params = {"code_commit": "c", "model": {"dtype": "bfloat16"}}
    assign = {"a": "b", "b": "a"}
    m = run_test.run_manifest(tmp_path, params, {"frozen_params_sha256": "1" * 64}, A, assign, ["a", "b"])
    assert run_test.run_manifest(tmp_path, params, {"frozen_params_sha256": "1" * 64}, A, assign, ["a", "b"]) == m
    with pytest.raises(frozen.FreezeError):
        run_test.run_manifest(tmp_path, params, {"frozen_params_sha256": "2" * 64}, A, assign, ["a", "b"])
    A.backend = "modal"
    with pytest.raises(frozen.FreezeError):
        run_test.run_manifest(tmp_path, params, {"frozen_params_sha256": "1" * 64}, A, assign, ["a", "b"])


# ------------------------------------------------------------------ freeze gate
def test_freeze_gate_rejects_empty_or_tampered(tmp_path):
    txt, js = tmp_path / "FROZEN.txt", tmp_path / "frozen_params.json"
    txt.write_text("")
    with pytest.raises(frozen.FreezeError):
        frozen.load_frozen(tmp_path, txt, js)
    params = {"code_commit": "abc", "config_hashes_sha256": "c" * 64, "model": {"revision_full": REV, "dtype": "bfloat16"},
              "protocol": "avg2", "eta": 0.1, "hedge_weights": {a: 0.2 for a in P_AGENTS}, "temperatures": {"p_full": 2.0},
              "split_manifest_sha256": "s" * 64, "placebo_seed": 20261111, "bootstrap_seed": 20261110,
              "condition": "test_informed"}
    js.write_text(json.dumps(params))
    txt.write_text(frozen.render_frozen_txt(params, frozen.sha256_file(js)))
    assert frozen.load_frozen(tmp_path, txt, js)[0]["eta"] == 0.1
    js.write_text(json.dumps(dict(params, eta=0.3)))  # weights/eta edited after the freeze
    with pytest.raises(frozen.FreezeError):
        frozen.load_frozen(tmp_path, txt, js)
    js.write_text(json.dumps(params))
    txt.write_text(txt.read_text().replace("eta: 0.1", "eta: 0.3"))
    with pytest.raises(frozen.FreezeError):
        frozen.load_frozen(tmp_path, txt, js)


def test_repo_freeze_file_is_currently_unpopulated_so_test_runs_are_blocked():
    if (ROOT / "configs/FROZEN.txt").read_text().strip():
        pytest.skip("FROZEN.txt populated (post-G2)")
    with pytest.raises(frozen.FreezeError):
        frozen.verify_frozen()


def test_cli_vs_frozen_mismatch_refused():
    class A:
        protocol, seed, cond, dtype = "single", 20261111, "test_informed", None
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["probability_agent"]
    params = {"protocol": "avg2", "placebo_seed": 20261111, "condition": "test_informed",
              "model": {"dtype": "bfloat16", "revision_full": f"{cfg['hf_id']}@{cfg['revision']}"}}
    with pytest.raises(frozen.FreezeError):
        run_test.check_cli_vs_frozen(params, A)
    A.protocol, A.dtype = "avg2", "float16"
    with pytest.raises(frozen.FreezeError):
        run_test.check_cli_vs_frozen(params, A)
    A.dtype = None
    run_test.check_cli_vs_frozen(params, A)


def test_evaluator_imports_no_inference_code():
    for f in ("src/analysis/evaluate_test.py", "src/analysis/methods.py", "src/analysis/bootstrap_test.py",
              "src/market/placebo.py", "src/market/pools.py", "src/frozen.py"):
        src = (ROOT / f).read_text()
        assert not re.search(r"^\s*(from|import)\s+src\.(agents|llm|cands)", src, re.M), f


def test_placebo_instance_makes_exactly_two_calls_and_valid_rows():
    from tests.test_market import FakeSrv
    from src.schemas.validate import validate_row
    inst, cands = mk_test_instance(2, "r1")
    donor = {a: {"text": t, "output_tokens": 120} for a, t in canon_notes(5).items()}
    srv = FakeSrv()
    rows, man, tel = run_test.placebo_instance(inst, cands, donor, "mb-r1-0005", "test_informed", srv, PERMS)
    assert len(srv.prompts) == 2 and len(rows) == 2 and len(tel) == 2 and len(man) == 5
    for r in rows:
        assert not validate_row("placebo_row", r) and r["donor_id"] == "mb-r1-0005"
        assert sorted(r["probs"]) and abs(sum(r["probs"]) - 1) < 1e-9
    for m in man:
        m["seed"] = 20261111
        assert not validate_row("placebo_manifest", m)
    view = inference_view(inst, cands, "test_informed")
    assert srv.prompts == [j[2] for j in placebo_prompts(view, {a: d["text"] for a, d in donor.items()}, PERMS)]
