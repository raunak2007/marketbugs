"""Modal HF-forward-pass server for label-logit extraction and short generation (logic in src.llm.hf_backend).

Run:  modal run src/llm/modal_server.py::smoke
Uses an HF forward pass (preferred by PLAN 4.1); no vLLM, so no top-k logprob dependence.
VERIFY: Gemma 4 support in the pinned transformers version (checked in smoke()).
"""
import modal, yaml, pathlib

_LOCAL_CFG = pathlib.Path(__file__).resolve().parent.parent.parent / "configs/models.yaml"
_CFG_PATH = _LOCAL_CFG if _LOCAL_CFG.exists() else pathlib.Path("/root/configs/models.yaml")
CFG = yaml.safe_load(_CFG_PATH.read_text())
PA = CFG["probability_agent"]

image = (modal.Image.debian_slim(python_version="3.11")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # fragmentation OOM in generate (G2)
         .pip_install("torch==2.8.0", "transformers==5.22.2", "accelerate==1.15.0", "numpy", "pyyaml")
         .add_local_file(str(_LOCAL_CFG), "/root/configs/models.yaml")
         .add_local_python_source("src"))
app = modal.App("emfl-gemma", image=image)
hf_cache = modal.Volume.from_name("emfl-hf-cache", create_if_missing=True)


@app.cls(gpu=CFG["modal"]["gpu"], secrets=[modal.Secret.from_name("huggingface")],
         volumes={"/cache": hf_cache}, scaledown_window=120, timeout=3600)
class Gemma:
    model_id: str = modal.parameter(default=PA["hf_id"])
    revision: str = modal.parameter(default=PA["revision"])

    @modal.enter()
    def load(self):
        from src.llm.hf_backend import HFGemma
        self.b = HFGemma(self.model_id, self.revision, cache_dir="/cache", dtype=PA.get("dtype", "bfloat16"))

    @modal.method()
    def label_logits(self, prompts: list[str], n_candidates: list[int], batch_size: int = 8,
                     max_padded_tokens: int = 24000):
        return self.b.label_logits(prompts, n_candidates, batch_size, max_padded_tokens)

    @modal.method()
    def generate(self, prompts: list[str], max_new_tokens: int = 150, temperature: float = 0.0, seed: int = 0):
        return self.b.generate(prompts, max_new_tokens, temperature, seed)

    @modal.method()
    def info(self):
        return self.b.info()


@app.local_entrypoint()
def smoke():
    """G0 check: 20-way vectors, sums to 1, repeatable across two runs."""
    import json, numpy as np
    g = Gemma()
    print(json.dumps(g.info.remote(), indent=1))
    cands = "\n".join(f"{chr(65+i)}. def f{i}(x): return x + {i}" for i in range(20))
    prompts = [f"Which function returns x + {k}?\n{cands}\nReply with the letter only." for k in (3, 11, 17)]
    a = g.label_logits.remote(prompts, [20] * 3)
    b = g.label_logits.remote(prompts, [20] * 3)
    for k, ra, rb in zip((3, 11, 17), a, b):
        p = np.array(ra["probs"])
        print(k, "top1", chr(65 + int(p.argmax())), "p=%.3f" % p.max(), "sum=%.6f" % p.sum(),
              "maxdiff_rerun=%.2e" % np.abs(p - np.array(rb["probs"])).max())


from src.llm.hf_backend import EMB_MODEL, EMB_REV  # noqa: E402
# The module reads configs/models.yaml at import, so every image must ship it (Embedder crash-looped without it).
emb_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("sentence-transformers==6.3.0", "torch==2.8.0", "pyyaml")
             .add_local_file(str(_LOCAL_CFG), "/root/configs/models.yaml")
             .add_local_python_source("src"))


@app.cls(gpu="T4", image=emb_image, volumes={"/cache": hf_cache}, scaledown_window=120)
class Embedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(EMB_MODEL, revision=EMB_REV, cache_folder="/cache")

    @modal.method()
    def embed(self, texts: list[str]):
        return self.m.encode(texts, batch_size=64, normalize_embeddings=True).tolist()
