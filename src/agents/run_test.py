"""Held-out MarketBugs runner (protocol v1.0 section 4). INFERENCE SIDE: never reads gold.

python -m src.agents.run_test --input-dir data/marketbugs/test --tag heldout_v1 --cond test_informed \
    --protocol avg2 --arms real,placebo --seed 20261111 --backend modal --max-total-usd 45 --resume
    [--prior-usd X] [--dtype bfloat16] [--max-hours H] [--extras] [--no-mode-s]

Gates (hard stop before any model call): configs/FROZEN.txt + artifacts/frozen_params.json present, consistent and
committed; configs/prompts/schemas/split unchanged since the freeze; code unchanged since the frozen commit; CLI
protocol / seed / condition / dtype equal the frozen values; on resume, the run manifest's frozen hash, backend,
arms and placebo assignment equal the current ones.

Arms: real = pipeline.run_instance (C-mode 32-subset table + B_full5 rotations on the protocol permutations, Mode S
unless --no-mode-s); placebo = exactly len(perms) label-logit calls per instance (mask 31, canonical note order) with
the five notes of the instance's donor (src.market.placebo derangement). Pools, notes and subset tables are never
regenerated for the placebo arm.

Output: artifacts/runs/<tag>/<cond>/{candidates_inf,notes,values_inf,k10,telemetry,raw_notes}.jsonl (real),
placebo_values_inf.jsonl + placebo_manifest.jsonl (new sidecar schemas, v2 rows untouched), placebo_assignment.json,
run_manifest.json, run_status.json. Writes append per instance; on start `reconcile` drops rows of unfinished
instances (atomic rewrite), so resuming never duplicates or orphans rows. A partial run is NOT a 210-instance result.
"""
import argparse, json, os, sys, threading, time, _thread
from pathlib import Path
import yaml
from src import frozen
from src.agents.pipeline import run_instance, label_batch
from src.agents.views import inference_view
from src.agents.placebo import placebo_prompts
from src.agents.run_values import GPU_USD_PER_S
from src.agents.prompts import AGENTS
from src.market.placebo import assign_donors, assignment_hash, sha256

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
PROTOCOL_PERMS = {"single": (0,), "avg2": (0, 1)}
REAL_KINDS = ("notes", "values_inf", "k10", "telemetry", "raw_notes")
RESERVE_USD = 1.0
STALL_S = 15 * 60


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def _atomic_write(p, rows):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def load_test_rows(input_dir, root=ROOT):
    """Test instances in file order (never reordered by outcome). Refuses non-test rows / non-test repos."""
    repos = yaml.safe_load((root / "configs/repos.yaml").read_text())["repos"]
    test_repos = {r for r, v in repos.items() if v["split"] == "test"}
    rows = []
    for p in sorted(Path(input_dir).glob("*.jsonl")):
        if p.name == "eval_only.jsonl":
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("split") != "test" or r["repo"] not in test_repos or r["source"] != "marketbugs":
                sys.exit(f"refusing non-test row {r['id']} ({r.get('split')}, {r['repo']})")
            rows.append(r)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate instance ids in test input")
    return rows


def reconcile(d, perms):
    """Drop rows of unfinished instances; hard-fail on duplicates of finished ones. Returns (real_done, plc_done)."""
    cands = jl(d / "candidates_inf.jsonl")
    ids = [c["id"] for c in cands]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{d}: duplicate finished instances in candidates_inf.jsonl")
    done = set(ids)
    pv = jl(d / "placebo_values_inf.jsonl")
    by = {}
    for r in pv:
        by.setdefault(r["id"], []).append(r["order_perm"])
    for i, ps in by.items():
        if len(ps) != len(set(ps)):
            raise RuntimeError(f"{d}: duplicate placebo rows for {i}")
    pdone = {i for i, ps in by.items() if sorted(ps) == sorted(perms) and i in done}
    dropped = {}
    plan = {k: (lambda r: r["id"] in done) for k in REAL_KINDS if k != "telemetry"}
    plan["telemetry"] = lambda r: (r["id"] in pdone) if r["kind"] == "placebo" else (r["id"] in done)
    plan["placebo_values_inf"] = lambda r: r["id"] in pdone
    plan["placebo_manifest"] = lambda r: r["id"] in pdone
    for k, keep in plan.items():
        p = d / f"{k}.jsonl"
        rows = jl(p)
        kept = [r for r in rows if keep(r)]
        if len(kept) != len(rows):
            _atomic_write(p, kept); dropped[k] = len(rows) - len(kept)
    if dropped:
        print(f"reconcile: dropped partial rows of unfinished instances {dropped}", flush=True)
    return done, pdone


def prior_spend(costs_path, prior_usd=0.0):
    """Conservative paid spend so far: max(sum of logged conservative_usd for paid backends, provider-billed)."""
    s = sum(r.get("conservative_usd") or 0.0 for r in jl(costs_path) if r.get("backend", "modal") != "local")
    return max(s, prior_usd or 0.0)


def check_cli_vs_frozen(params, a):
    errs = []
    if a.protocol != params["protocol"]:
        errs.append(f"--protocol {a.protocol} != frozen {params['protocol']}")
    if a.seed != params["placebo_seed"]:
        errs.append(f"--seed {a.seed} != frozen placebo_seed {params['placebo_seed']}")
    if a.cond != params["condition"]:
        errs.append(f"--cond {a.cond} != frozen {params['condition']}")
    if (a.dtype or params["model"]["dtype"]) != params["model"]["dtype"]:
        errs.append(f"--dtype {a.dtype} != frozen {params['model']['dtype']} (precision may not change silently)")
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["probability_agent"]
    if f"{cfg['hf_id']}@{cfg['revision']}" != params["model"]["revision_full"]:
        errs.append("configs/models.yaml model revision != frozen")
    if errs:
        raise frozen.FreezeError("; ".join(errs))


def run_manifest(d, params, kv, a, assign, ids):
    m = {"frozen_params_sha256": kv["frozen_params_sha256"], "frozen_code_commit": params["code_commit"],
         "protocol": a.protocol, "arms": sorted(a.arms), "placebo_seed": a.seed, "backend": a.backend,
         "dtype": params["model"]["dtype"], "mode_s": not a.no_mode_s, "extras": a.extras,
         "placebo_assignment_sha256": assignment_hash(assign), "input_ids_sha256": sha256(json.dumps(ids))}
    p = d / "run_manifest.json"
    if p.exists():
        old = json.loads(p.read_text())
        diff = {k: (old.get(k), v) for k, v in m.items() if old.get(k) != v}
        if diff:
            raise frozen.FreezeError(f"run manifest mismatch on resume (frozen provenance is immutable): {diff}")
        return old
    if any((d / f).exists() for f in ("candidates_inf.jsonl", "placebo_values_inf.jsonl")):
        raise frozen.FreezeError(f"{d} has inference rows but no run_manifest.json")
    m.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), head_at_start=frozen.head_sha())
    p.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    (d / "placebo_assignment.json").write_text(json.dumps(
        {"seed": a.seed, "sha256": assignment_hash(assign), "donor_of": dict(sorted(assign.items()))}, indent=1) + "\n")
    return m


def placebo_instance(inst, cands, donor_notes, donor_id, cond, srv, perms):
    """Exactly len(perms) label-logit calls. Returns (placebo rows, manifest rows, telemetry)."""
    view = inference_view(inst, cands, cond)
    canon = {a: donor_notes[a]["text"] for a in AGENTS}
    jobs = placebo_prompts(view, canon, perms)
    t0 = time.time()
    outs, w = label_batch(srv, [j[2] for j in jobs], len(view["pool"]))
    rows, tel = [], []
    cids = sorted(c["cid"] for c in view["pool"])
    for (perm, pv, p, present, rn), o in zip(jobs, outs):
        by_cid = dict(zip([c["cid"] for c in pv["pool"]], zip(o["raw"], o["probs"])))
        probs = [by_cid[c][1] for c in cids]
        rows.append({"id": inst["id"], "arm": "placebo_full", "subset_mask": 31, "order_perm": perm,
                     "note_order": present, "donor_id": donor_id, "raw_candidate_logits": [by_cid[c][0] for c in cids],
                     "probs": probs, "top1": cids[max(range(len(probs)), key=probs.__getitem__)],
                     "token_mapping_hash": o["token_mapping_hash"], "prompt_hash": sha256(p),
                     "model_revision": o["model_revision"], "input_tokens": o["input_tokens"],
                     "inference_cost_usd": w / len(jobs) * GPU_USD_PER_S})
        tel.append({"id": inst["id"], "condition": cond, "kind": "placebo", "order_perm": perm,
                    "prompt_hash": sha256(p)[:16], "input_tokens": o["input_tokens"], "output_tokens": 0,
                    "latency_ms": o["latency_ms"], "wall_share_s": w / len(jobs), "est_usd": w / len(jobs) * GPU_USD_PER_S})
    man = [{"id": inst["id"], "repo": inst["repo"], "specialist": a, "donor_id": donor_id,
            "seed": None, "source_note_sha256": sha256(donor_notes[a]["text"]),
            "donor_note_output_tokens": donor_notes[a].get("output_tokens"),
            "rendered_prompt_sha256": {str(j[0]): sha256(j[2]) for j in jobs}} for a in AGENTS]
    return rows, man, tel


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cond", default="test_informed")
    ap.add_argument("--protocol", default="avg2", choices=sorted(PROTOCOL_PERMS))
    ap.add_argument("--arms", default="real,placebo")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--backend", default="modal", choices=("modal", "local"))
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-total-usd", type=float, required=True)
    ap.add_argument("--prior-usd", type=float, default=0.0, help="provider-billed spend so far, if above the log")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extras", action="store_true", help="PLAN 4.3 extra perms + K=10 control (off by default)")
    ap.add_argument("--no-mode-s", action="store_true", help="cut E4 Mode S (only if the frozen plan cuts it)")
    a = ap.parse_args(argv)
    a.arms = set(a.arms.split(","))
    if not a.arms <= {"real", "placebo"}:
        sys.exit(f"unknown arms {a.arms}")
    params, kv = frozen.verify_frozen()
    check_cli_vs_frozen(params, a)
    for k, v in params["model"].items():
        print(f"frozen model {k}: {v}")
    perms = PROTOCOL_PERMS[a.protocol]
    rows = load_test_rows(a.input_dir)
    by_id = {r["id"]: r for r in rows}
    assign = assign_donors([(r["id"], r["repo"]) for r in rows], a.seed)
    d = ART / "runs" / a.tag / a.cond
    if d.exists() and any(d.iterdir()) and not a.resume:
        sys.exit(f"{d} exists; pass --resume to continue it")
    d.mkdir(parents=True, exist_ok=True)
    run_manifest(d, params, kv, a, assign, [r["id"] for r in rows])
    done, pdone = reconcile(d, perms)

    from src.agents.pilot import make_backend, precompute_embeddings, load_repo, write_instance
    srv, emb, rates, _ = make_backend(a.backend, params["model"]["dtype"])
    prior = prior_spend(ART / "costs.jsonl", a.prior_usd)
    print(f"budget: prior paid spend ${prior:.2f} of ${a.max_total_usd:.2f} total ({a.backend})", flush=True)
    t_start, n_real, n_plc, stopped = time.time(), 0, 0, None
    state = {"t": time.time(), "fired": False, "end": False}

    def _watch():
        while not state["end"]:
            if time.time() - state["t"] > STALL_S:
                state["fired"] = True; print(f"STOP watchdog: no progress in {STALL_S}s", flush=True)
                _thread.interrupt_main(); return
            time.sleep(5)
    threading.Thread(target=_watch, daemon=True).start()
    t4_s = [0.0]
    spend = lambda: (time.time() - t_start) * rates[0] * 1.15 + t4_s[0] * rates[1] * 1.15

    def guard(units):
        per = spend() / units if units else 0.0
        if prior + spend() + 1.5 * per + RESERVE_USD > a.max_total_usd:
            return f"budget guard: prior ${prior:.2f} + this run ${spend():.2f} + next ~${per:.2f} + reserve > ${a.max_total_usd}"
        el = time.time() - t_start
        if a.max_hours and units and el + 1.5 * el / units > a.max_hours * 3600:
            return f"time guard: {el / 3600:.2f}h elapsed, limit {a.max_hours}h"
        return None
    try:
        todo = [r for r in rows if r["id"] not in done] if "real" in a.arms else []
        if todo:
            precompute_embeddings(todo, [a.cond], emb)
            t4_s[0] = (time.time() - t_start) + 120
        repo_cache = {}
        for inst in todo:
            state["t"] = time.time()
            if (stopped := guard(n_real)):
                break
            if inst["repo"] not in repo_cache:
                repo_cache[inst["repo"]] = load_repo(inst["repo"], emb)
            out = run_instance(inst, a.cond, repo_cache[inst["repo"]], emb, srv, extras=a.extras, perms=perms,
                               mode_s=not a.no_mode_s)
            write_instance(d, inst, out)
            done.add(inst["id"]); n_real += 1
            print(f"{inst['id']} real done ({len(done)}/{len(rows)}), ~${prior + spend():.2f} total", flush=True)
        if "placebo" in a.arms and not stopped:
            cands = {c["id"]: c for c in jl(d / "candidates_inf.jsonl")}
            notes = {}
            for n in jl(d / "notes.jsonl"):
                notes.setdefault(n["id"], {})[n["agent"]] = n
            for inst in rows:
                i, dn = inst["id"], assign[inst["id"]]
                if i in pdone or i not in done or dn not in done:
                    continue
                state["t"] = time.time()
                if (stopped := guard(n_real + n_plc)):
                    break
                pr, man, tel = placebo_instance(by_id[i], cands[i], notes[dn], dn, a.cond, srv, perms)
                for m in man:
                    m["seed"] = a.seed
                for k, data in (("placebo_manifest", man), ("telemetry", tel), ("placebo_values_inf", pr)):
                    with open(d / f"{k}.jsonl", "a") as f:  # placebo_values_inf last = done marker
                        f.write("".join(json.dumps(r) + "\n" for r in data))
                pdone.add(i); n_plc += 1
    except KeyboardInterrupt:
        stopped = stopped or ("watchdog stall" if state["fired"] else "interrupted")
    finally:
        state["end"] = True
        exp = [r["id"] for r in rows]
        inc_real = [i for i in exp if i not in done] if "real" in a.arms else []
        inc_plc = [i for i in exp if i not in pdone] if "placebo" in a.arms else []
        status = {"tag": a.tag, "expected": len(exp), "real_complete": len(done), "placebo_complete": len(pdone),
                  "incomplete_real": inc_real, "incomplete_placebo": inc_plc,
                  "complete": not inc_real and not inc_plc and {"real", "placebo"} <= a.arms,
                  "stopped": stopped, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _atomic_write_json(d / "run_status.json", status)
        wall = time.time() - t_start
        with open(ART / "costs.jsonl", "a") as f:
            f.write(json.dumps({"job": f"run_test {a.tag}", "backend": a.backend, "conds": [a.cond],
                                "protocol": a.protocol, "n": n_real, "n_placebo": n_plc, "wall_s": wall,
                                "est_usd": wall * rates[0], "conservative_usd": spend(), "stopped": stopped}) + "\n")
        print(("PARTIAL: " if not status["complete"] else "") + json.dumps({k: v for k, v in status.items()
              if not k.startswith("incomplete")}), flush=True)


def _atomic_write_json(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, p)


if __name__ == "__main__":
    main()
