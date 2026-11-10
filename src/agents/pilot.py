"""G1 pilot (PLAN 11 Phase 1): pools -> notes -> 32-subset C tables on N dev instances, then score
(evaluator side, after inference) and extrapolate cost.

python -m src.agents.pilot data/marketbugs/dev/<repo>.jsonl [--n 30]
"""
import json, sys, time, random
from pathlib import Path
import numpy as np
import modal
from src.agents.views import inference_view
from src.agents.evidence import build_evidence
from src.agents.specialists import specialist_prompt, parse_note
from src.agents.prompts import AGENTS, template_hash
from src.agents.run_values import value_rows_for_instance, GPU_USD_PER_S
from src.cands.functions import index_repo
from src.cands.retrieval import BM25, Embedding, Graph, frame_functions
from src.cands.pool import build_pool
from src.llm.hashing import sha

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


class _Srv:
    def __init__(self):
        G = modal.Cls.from_name("emfl-gemma", "Gemma")()
        self.g = G
        self.label_logits = lambda p, n: G.label_logits.remote(p, n)
        self.generate = lambda p, **kw: G.generate.remote(p, **kw)


def run(path, n, emb):
    rows = [json.loads(l) for l in open(path)][:n]
    srv = _Srv()
    repo_cache = {}
    out_c, out_n, out_v = [], [], []
    t_start = time.time()
    for inst in rows:
        repo_dir = ROOT / "data/marketbugs/repos" / inst["repo"]
        if inst["repo"] not in repo_cache:
            fns = index_repo(repo_dir)
            repo_cache[inst["repo"]] = (fns, BM25(fns), Embedding(fns, emb), Graph(fns))
        fns, bm, em, gr = repo_cache[inst["repo"]]
        q = inst["conditions"]["test_informed"]  # inference input only
        frames = frame_functions(fns, inst["trace"], repo_dir)
        cands = build_pool(inst["id"], {"bm25": bm.rank(q, 40), "emb": em.rank(q, 40), "graph": gr.rank(frames, 40)},
                           20, order_seed=int(sha(inst["id"]), 16) % 10**6)
        view = inference_view(inst, cands, "test_informed")
        sem = dict(zip([c["cid"] for c in view["pool"]],
                       (np.array(emb([c["code"][:2000] for c in view["pool"]])) @ np.array(emb([q[:2000]])[0])).tolist()))
        ev = build_evidence(view, fns, gr, sem, repo_dir)
        prompts = [specialist_prompt(a, view, ev) for a in AGENTS]
        gens = srv.generate(prompts, max_new_tokens=150, temperature=0.0, seed=0)
        notes = {}
        for a, p, g in zip(AGENTS, prompts, gens):
            canon, ranked = parse_note(g["text"], view)
            if canon is None:  # retry once at temperature 0 (deterministic; logged), then empty
                g = srv.generate([p + "\n\nRemember: first line must be 'RANKED: X,Y,...'"], max_new_tokens=150)[0]
                canon, ranked = parse_note(g["text"], view)
            notes[a] = canon or "(no usable note)"
            out_n.append({"id": inst["id"], "agent": a, "copy_idx": 0, "text": canon or "", "raw_text": g["text"],
                          "ranked_suspects": ranked, "prompt_hash": sha(p), "template_hash": template_hash("specialist"),
                          "model_revision": g["model_revision"], "temperature": 0.0, "seed": 0,
                          "input_tokens": g["input_tokens"], "output_tokens": g["output_tokens"], "latency_ms": g["latency_ms"]})
        vr = value_rows_for_instance(view, notes, srv)
        out_c.append(cands); out_v += vr
        print(inst["id"], "done", flush=True)
    wall = time.time() - t_start
    tag = Path(path).stem
    for name, data in (("candidates", out_c), ("notes", out_n), ("values_C", out_v)):
        (ART / f"pilot_{tag}_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    cost = wall * GPU_USD_PER_S
    with open(ART / "costs.jsonl", "a") as f:
        f.write(json.dumps({"job": f"pilot {tag}", "n": len(rows), "wall_s": wall, "est_usd": cost}) + "\n")
    print(f"pilot {tag}: {len(rows)} instances, wall {wall:.0f}s, est ${cost:.2f}")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 30
    E = modal.Cls.from_name("emfl-gemma", "Embedder")()
    cache = {}

    def emb(texts):
        miss = [t for t in texts if sha(t) not in cache]
        for i in range(0, len(miss), 512):
            for t, v in zip(miss[i:i + 512], E.embed.remote(miss[i:i + 512])):
                cache[sha(t)] = v
        return np.array([cache[sha(t)] for t in texts])
    run(sys.argv[1], n, emb)
