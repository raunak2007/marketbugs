"""Dev runs (G1 pilot, G2 fitting data) via src.agents.pipeline. INFERENCE SIDE.

python -m src.agents.pilot <instances.jsonl> --tag <name> [--n 30] [--cond test_informed|no_trace|both]
                           [--max-usd X] [--no-extras] [--protocol single|avg2] [--reuse-notes-from <tag>]
                           [--skip K] [--backend modal|local] [--max-hours H] [--dtype bfloat16|float16]

--backend local loads Gemma + the embedder in-process (Kaggle/any free GPU): cost rates are 0, no $ guard, and
--max-hours stops cleanly between instances before a session limit kills the process mid-write.

Writes artifacts/runs/<tag>/<cond>/{candidates_inf,notes,values_inf,k10,telemetry,raw_notes}.jsonl, appending one
instance at a time (crash-safe; rerunning the same tag resumes, skipping finished ids). Gold is attached later by
src.analysis.attach_gold. Refuses non-dev instances (no test outcomes before the G2 freeze).
"""
import json, sys, time, threading, _thread
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
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
XL_USD_PER_S = 2.00 / 3600  # L40S, rounded up; billed per XL call wall + cold start + 120 s scaledown
BURN_USD_PER_S = (GPU_USD_PER_S + T4_USD_PER_S) * 1.15
RESERVE_USD = 1.0  # lowered from 2.0: conservative estimate matched Modal billing ($3.39 est vs $3.35 billed)
STALL_S = 15 * 60
KINDS = ("candidates_inf", "notes", "values_inf", "k10", "telemetry", "raw_notes")


XL_GPU = "L40S"  # 48 GB: notes prompts of ~17k tokens OOM on A10G in the sliding-window mask build (G2)
XL_LOG = ART / "xl_fallback.jsonl"


class _Srv:
    """All value calls (label_logits) run on the configured A10G. generate() is greedy and per-prompt independent
    (hf_backend loops over prompts), so a batch that OOMs is re-run prompt by prompt on A10G and only the prompts
    that still OOM go to an XL_GPU container; every such prompt is logged to artifacts/xl_fallback.jsonl."""

    def __init__(self):
        import modal
        C = modal.Cls.from_name("emfl-gemma", "Gemma")
        G = C()
        self._C, self._xl, self.xl_s = C, None, 0.0
        self.label_logits = lambda p, n: G.label_logits.remote(p, n)
        self._gen = lambda p, **kw: G.generate.remote(p, **kw)

    @staticmethod
    def _is_oom(e):
        return "OutOfMemoryError" in f"{type(e).__name__}: {e}"

    def generate(self, prompts, **kw):
        try:
            return self._gen(prompts, **kw)
        except Exception as e:
            if not self._is_oom(e):
                raise
        out = []
        for p in prompts:
            try:
                out += self._gen([p], **kw)
            except Exception as e:
                if not self._is_oom(e):
                    raise
                if self._xl is None:
                    self._xl = self._C.with_options(gpu=XL_GPU)()
                t0 = time.time()
                r = self._xl.generate.remote([p], **kw)
                self.xl_s += (time.time() - t0) + 120  # + scaledown window (cold start is inside the call wall)
                with open(XL_LOG, "a") as f:
                    f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "prompt_hash": sha(p),
                                        "gpu": XL_GPU, "input_tokens": r[0]["input_tokens"]}) + "\n")
                print(f"XL fallback: {r[0]['input_tokens']}-token notes prompt ran on {XL_GPU}", flush=True)
                out += r
        return out


def precompute_embeddings(rows, conds, emb):
    """Embed everything a run needs up front (repo function texts, candidate code, condition queries) so the T4
    embedder is used for minutes, not the whole run. Same texts and model as the per-instance calls."""
    from src.cands.retrieval import fn_text
    from src.agents.pipeline import condition_query
    for repo in sorted({r["repo"] for r in rows}):
        repo_dir = ROOT / "data/marketbugs/repos" / repo
        fns = index_repo(repo_dir)
        texts = [fn_text(f)[:2000] for f in fns] + [f.code[:2000] for f in fns]
        texts += [condition_query(r, c)[0][:2000] for r in rows if r["repo"] == repo for c in conds]
        emb(list(dict.fromkeys(texts)))
        print(f"embedded {repo}: {len(set(texts))} texts", flush=True)


def load_repo(name, emb):
    repo_dir = ROOT / "data/marketbugs/repos" / name
    fns = index_repo(repo_dir)
    return fns, BM25(fns), Embedding(fns, emb), Graph(fns), repo_dir


def write_instance(d, inst, out):
    """Append one finished instance. candidates_inf is written last: it marks the instance done only if all other
    files landed (src.agents.run_test.reconcile drops partial rows of unfinished instances on resume)."""
    out["candidates_inf"], out["values_inf"] = out.pop("candidates"), out.pop("values")
    out["raw_notes"] = [{"id": inst["id"], "agent": a, "raw_text": t} for a, t in out["raw_notes"].items()]
    for k in KINDS:
        data = out[k] if isinstance(out[k], list) else [out[k]]
        if k != "candidates_inf":
            with open(d / f"{k}.jsonl", "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in data))
    with open(d / "candidates_inf.jsonl", "a") as f:
        f.write(json.dumps(out["candidates_inf"]) + "\n")


def _done_ids(d):
    p = d / "candidates_inf.jsonl"
    return {json.loads(l)["id"] for l in open(p)} if p.exists() else set()


PROTOCOLS = {"single": (0,), "avg2": (0, 1)}  # avg2 adopted at G1 (order sensitivity 25.7% > 15%)


def run(path, tag, n, conds, emb, srv, max_usd, extras=True, protocol="single", reuse_from=None, skip=0,
        rates=(GPU_USD_PER_S, T4_USD_PER_S), max_hours=None, backend="modal"):
    rows = [json.loads(l) for l in open(path)][skip:skip + n]
    perms = PROTOCOLS[protocol]
    reuse, reuse_c = {}, {}
    for c in conds:
        cp = ART / "runs" / (reuse_from or "") / c / "candidates_inf.jsonl"
        if reuse_from and cp.exists():
            for l in open(cp):
                r = json.loads(l); reuse_c[(c, r["id"])] = r
        rp = ART / "runs" / (reuse_from or "") / c / "notes.jsonl"
        if reuse_from and rp.exists():
            for l in open(rp):
                r = json.loads(l); reuse.setdefault((c, r["id"]), {})[r["agent"]] = r
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
    precompute_embeddings(rows, conds, emb)
    t4_s = (time.time() - t_start) + 120  # embedder active during precompute + its 120 s scaledown window
    spend = lambda: ((time.time() - t_start) * rates[0] + t4_s * rates[1] + getattr(srv, "xl_s", 0.0) * XL_USD_PER_S) * 1.15
    try:
        for inst, cond in todo:
            d = ART / "runs" / tag / cond; d.mkdir(parents=True, exist_ok=True)
            if inst["id"] in _done_ids(d):
                continue
            state["t"] = time.time()
            spent = spend()
            per = spent / done if done else 0.0
            if spent + per * 1.5 + RESERVE_USD > max_usd:
                stopped = f"budget guard: spent~${spent:.2f}, next~${per:.2f}, reserve ${RESERVE_USD}, cap ${max_usd}"
                print("STOP", stopped, flush=True); break
            if max_hours and done and (time.time() - t_start) + 1.5 * (time.time() - t_start) / done > max_hours * 3600:
                stopped = f"time guard: {(time.time() - t_start) / 3600:.2f}h elapsed, limit {max_hours}h"
                print("STOP", stopped, flush=True); break
            if inst["repo"] not in repo_cache:
                repo_cache[inst["repo"]] = load_repo(inst["repo"], emb)
            nov = reuse.get((cond, inst["id"]))
            ok = bool(nov) and len(nov) == 5 and (cond, inst["id"]) in reuse_c
            out = run_instance(inst, cond, repo_cache[inst["repo"]], emb, srv, extras=extras, perms=perms,
                               notes_override=nov if ok else None, cands_override=reuse_c[(cond, inst["id"])] if ok else None)
            write_instance(d, inst, out)
            done += 1
            tel = out["telemetry"]
            print(f"{inst['id']} [{cond}] done: {len(tel)} calls, {sum(t['input_tokens'] for t in tel)} in-tok, "
                  f"~${spend():.2f} so far ({backend}), {(time.time() - t_start) / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        stopped = stopped or ("watchdog stall" if state["fired"] else "interrupted")
    finally:
        state["end"] = True
        wall = time.time() - t_start
        with open(ART / "costs.jsonl", "a") as f:
            f.write(json.dumps({"job": f"run {tag}", "ts": datetime.now(timezone.utc).isoformat(), "backend": backend, "conds": conds, "protocol": protocol, "n": done, "wall_s": wall,
                                "est_usd": wall * rates[0],
                                "conservative_usd": max(spend() if "spend" in dir() else 0.0, wall * (rates[0] + rates[1]) * 1.15),
                                "stopped": stopped}) + "\n")
        print(f"run {tag}: {done} instance-conditions, wall {wall:.0f}s, conservative ${wall * BURN_USD_PER_S:.2f}")


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _local_backend(dtype, device_map=None):
    """In-process Gemma + embedder (same pinned models/revisions as the Modal images)."""
    import yaml
    from src.llm.hf_backend import HFGemma, EMB_MODEL, EMB_REV
    from sentence_transformers import SentenceTransformer
    pa = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["probability_agent"]
    g = HFGemma(pa["hf_id"], pa["revision"], dtype=dtype or pa.get("dtype", "bfloat16"), device_map=device_map)
    print("backend info:", json.dumps({k: v for k, v in g.info().items() if k != "mapping"}), flush=True)
    m = SentenceTransformer(EMB_MODEL, revision=EMB_REV, device="cuda")
    embed = lambda texts: m.encode(texts, batch_size=64, normalize_embeddings=True).tolist()
    return g, embed


def make_backend(backend, dtype=None, max_usd=10.79, device_map=None, workspace=None, workspace_cap_usd=None):
    """-> (srv, emb, rates, max_usd). emb caches by text hash so precomputed embeddings are reused.
    Modal runs must name the expected workspace + its monthly cap (src.llm.modal_guard); max_usd is lowered to what
    is left under that cap."""
    if backend == "local":
        srv, embed_fn = _local_backend(dtype, device_map)
        rates, max_usd = (0.0, 0.0), float("inf")
    elif backend == "modal":
        import modal
        from src.llm.modal_guard import remaining_budget
        max_usd = min(max_usd, remaining_budget(workspace, workspace_cap_usd))
        E = modal.Cls.from_name("emfl-gemma", "Embedder")()
        srv, embed_fn = _Srv(), E.embed.remote
        rates = (GPU_USD_PER_S, T4_USD_PER_S)
    else:
        raise ValueError(backend)
    cache = {}

    def emb(texts):
        miss = [t for t in texts if sha(t) not in cache]
        for i in range(0, len(miss), 512):
            for t, v in zip(miss[i:i + 512], embed_fn(miss[i:i + 512])):
                cache[sha(t)] = v
        return np.array([cache[sha(t)] for t in texts])
    return srv, emb, rates, max_usd


if __name__ == "__main__":
    backend = _arg("--backend", "modal")
    ws, wcap = _arg("--workspace", None), _arg("--workspace-cap-usd", None)
    srv, emb, rates, max_usd = make_backend(backend, _arg("--dtype", None), float(_arg("--max-usd", 10.79)),
                                           _arg("--device-map", None), ws, float(wcap) if wcap else None)
    if backend == "modal":
        backend = f"modal:{ws}"  # ledger key for src.llm.modal_guard
    c = _arg("--cond", "test_informed")
    mh = _arg("--max-hours", None)
    run(sys.argv[1], _arg("--tag", Path(sys.argv[1]).stem), int(_arg("--n", 30)),
        ["test_informed", "no_trace"] if c == "both" else [c], emb, srv, max_usd,
        extras="--no-extras" not in sys.argv, protocol=_arg("--protocol", "single"),
        reuse_from=_arg("--reuse-notes-from", None), skip=int(_arg("--skip", 0)),
        rates=rates, max_hours=float(mh) if mh else None, backend=backend)
