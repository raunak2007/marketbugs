"""Modal HF-forward-pass server for label-logit extraction and short generation.

Run:  modal run src/llm/modal_server.py::smoke
Uses an HF forward pass (preferred by PLAN 4.1); no vLLM, so no top-k logprob dependence.
VERIFY: Gemma 4 support in the pinned transformers version (checked in smoke()).
"""
import modal, yaml, pathlib

CFG = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2] / "configs/models.yaml").read_text())
PA = CFG["probability_agent"]

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers>=5.0", "accelerate", "numpy", "pyyaml")
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
        import torch, transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src.llm.extraction import resolve_labels, token_mapping_hash
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, cache_dir="/cache")
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, cache_dir="/cache",
            torch_dtype=torch.bfloat16, device_map="cuda").eval()
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.mapping = resolve_labels(enc, 20)
        self.mhash = token_mapping_hash(self.mapping, self.model_id, self.revision)
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def _render(self, user_text):
        # Chat template, then open the model turn and end on the answer cue so the next token is the label.
        from src.llm.extraction import ANSWER_CUE
        msgs = [{"role": "user", "content": user_text}]
        prefix = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prefix + ANSWER_CUE

    @modal.method()
    def label_logits(self, prompts: list[str], n_candidates: list[int], batch_size: int = 8):
        import torch, time
        from src.llm.extraction import extract
        out = []
        for i in range(0, len(prompts), batch_size):
            chunk = [self._render(p) for p in prompts[i:i + batch_size]]
            t0 = time.time()
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float().cpu().numpy()
            dt = (time.time() - t0) * 1000 / len(chunk)
            for j, row in enumerate(logits):
                raw, probs = extract(row, self.mapping, n_candidates[i + j])
                out.append({"raw": raw.tolist(), "probs": probs.tolist(),
                            "input_tokens": int(enc["attention_mask"][j].sum()), "latency_ms": dt,
                            "token_mapping_hash": self.mhash, "model_revision": f"{self.model_id}@{self.revision}"})
        return out

    @modal.method()
    def generate(self, prompts: list[str], max_new_tokens: int = 150, temperature: float = 0.0, seed: int = 0):
        import torch, time
        torch.manual_seed(seed)
        out = []
        for p in prompts:
            text = self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            t0 = time.time()
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens)
            if temperature > 0:
                kw["temperature"] = temperature
            with torch.no_grad():
                g = self.model.generate(**enc, **kw)
            new = g[0, enc["input_ids"].shape[1]:]
            out.append({"text": self.tok.decode(new, skip_special_tokens=True), "input_tokens": int(enc["input_ids"].shape[1]),
                        "output_tokens": int(len(new)), "latency_ms": (time.time() - t0) * 1000,
                        "model_revision": f"{self.model_id}@{self.revision}"})
        return out

    @modal.method()
    def info(self):
        return {"mapping": self.mapping, "token_mapping_hash": self.mhash, **self.versions}


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
