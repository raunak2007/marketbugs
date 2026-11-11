"""Dev runs (G1 pilot, G2 fitting data) via src.agents.pipeline. INFERENCE SIDE.

python -m src.agents.pilot <instances.jsonl> --tag <name> [--n 30] [--cond test_informed|no_trace|both]
                           [--max-usd X] [--no-extras]

Writes artifacts/runs/<tag>/<cond>/{candidates_inf,notes,values_inf,k10,telemetry,raw_notes}.jsonl, appending one
instance at a time (crash-safe; rerunning the same tag resumes, skipping finished ids). Gold is attached later by
src.analysis.attach_gold. Refuses non-dev instances (no test outcomes before the G2 freeze).
"""
import json, sys, time, threading, _thread
from pathlib import Path
import numpy as np
import modal
from src.agents.pipeline import run_instance
from src.agents.run_values import GPU_USD_PER_S
from src.cands.functions import index_repo
from src.cands.retrieval import BM25, Embedding, Graph
from src.llm.hashing import sha

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"

# Hard budget guard (human cap 2026-11-10). Conservative: bill BOTH GPUs (A10G + T4 embedder) for the whole wall
# time, +15% for Modal CPU/memory, and keep a reserve for one hung call / scaledown.
T4_USD_PER_S = 0.59 / 3600
BURN_USD_PER_S = (GPU_USD_PER_S + T4_USD_PER_S) * 1.15
RESERVE_USD = 2.0
STALL_S = 15 * 60
KINDS = ("candidates_inf", "notes", "values_inf", "k10", "telemetry", "raw_notes")


class _Srv:
    def __init__(self):
        G = modal.Cls.from_name("emfl-gemma", "Gemma")()
        self.label_logits = lambda p, n: G.label_logits.remote(p, n)
        self.generate = lambda p, **kw: G.generate.remote(p, **kw)


def _done_ids(d):
    p = d / "candidates_inf.jsonl"
    return {json.loads(l)["id"] for l in open(p)} if p.exists() else set()


def run(path, tag, n, conds, emb, srv, max_usd, extras=True):
    rows = [json.loads(l) for l in open(path)][:n]
    bad = [r["id"] for r in rows if r.get("split") not in ("dev", None) and r["source"] == "marketbugs"]
    if bad:
        sys.exit(f"refusing non-dev instances before G2 freeze: {bad[:3]}...")
    repo_cache, t_start, done = {}, time.time(), 0
    state = {"t": time.time(), "fired": False, "end": False}
    stopped = None

    def _watch():
        while not state["end"]:
            if time.time() - state["t"] > STALL_S:
                state["fired"] = True; print(f"STOP watchdog: no progress in {STALL_S}s", flush=True)
                _thread.interrupt_main(); return
            time.sleep(5)
    threading.Thread(target=_watch, daemon=True).start()
    todo = [(inst, c) for c in conds for inst in rows]
    try:
        for inst, cond in todo:
            d = ART / "runs" / tag / cond; d.mkdir(parents=True, exist_ok=True)
            if inst["id"] in _done_ids(d):
                continue
            state["t"] = time.time()
            spent = (time.time() - t_start) * BURN_USD_PER_S
            per = spent / done if done else 0.0
            if spent + per * 1.5 + RESERVE_USD > max_usd:
                stopped = f"budget guard: spent~${spent:.2f}, next~${per:.2f}, reserve ${RESERVE_USD}, cap ${max_usd}"
                print("STOP", stopped, flush=True); break
            repo_dir = ROOT / "data/marketbugs/repos" / inst["repo"]
            if inst["repo"] not in repo_cache:
                fns = index_repo(repo_dir)
                repo_cache[inst["repo"]] = (fns, BM25(fns), Embedding(fns, emb), Graph(fns), repo_dir)
            out = run_instance(inst, cond, repo_cache[inst["repo"]], emb, srv, extras=extras)
            out["candidates_inf"], out["values_inf"] = out.pop("candidates"), out.pop("values")
            out["raw_notes"] = [{"id": inst["id"], "agent": a, "raw_text": t} for a, t in out["raw_notes"].items()]
            for k in KINDS:  # candidates_inf last-written => marks the instance done only if all others landed
                data = out[k] if isinstance(out[k], list) else [out[k]]
                if k != "candidates_inf":
                    with open(d / f"{k}.jsonl", "a") as f:
                        f.write("".join(json.dumps(r) + "\n" for r in data))
            with open(d / "candidates_inf.jsonl", "a") as f:
                f.write(json.dumps(out["candidates_inf"]) + "\n")
            done += 1
            tel = out["telemetry"]
            print(f"{inst['id']} [{cond}] done: {len(tel)} calls, {sum(t['input_tokens'] for t in tel)} in-tok, "
                  f"~${(time.time() - t_start) * BURN_USD_PER_S:.2f} so far (conservative)", flush=True)
    except KeyboardInterrupt:
        stopped = stopped or ("watchdog stall" if state["fired"] else "interrupted")
    finally:
        state["end"] = True
        wall = time.time() - t_start
        with open(ART / "costs.jsonl", "a") as f:
            f.write(json.dumps({"job": f"run {tag}", "conds": conds, "n": done, "wall_s": wall,
                                "est_usd": wall * GPU_USD_PER_S, "conservative_usd": wall * BURN_USD_PER_S,
                                "stopped": stopped}) + "\n")
        print(f"run {tag}: {done} instance-conditions, wall {wall:.0f}s, conservative ${wall * BURN_USD_PER_S:.2f}")


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    E = modal.Cls.from_name("emfl-gemma", "Embedder")()
    cache = {}

    def emb(texts):
        miss = [t for t in texts if sha(t) not in cache]
        for i in range(0, len(miss), 512):
            for t, v in zip(miss[i:i + 512], E.embed.remote(miss[i:i + 512])):
                cache[sha(t)] = v
        return np.array([cache[sha(t)] for t in texts])
    c = _arg("--cond", "test_informed")
    run(sys.argv[1], _arg("--tag", Path(sys.argv[1]).stem), int(_arg("--n", 30)),
        ["test_informed", "no_trace"] if c == "both" else [c], emb, _Srv(), float(_arg("--max-usd", 10.79)),
        extras="--no-extras" not in sys.argv)
