"""Backend-neutral HF forward-pass model (label-logit extraction + short generation).

Shared by the Modal server (src.llm.modal_server) and the Kaggle runner (src.llm.kaggle_runner) so both use the same
chat-template rendering, answer cue, batching rule and label extraction. Only the device/dtype/cache dir differ.
"""

EMB_MODEL, EMB_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned 2026-11-10 via HF API


class HFGemma:
    def __init__(self, model_id, revision, cache_dir=None, dtype="bfloat16", device="cuda"):
        import torch, transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src.llm.extraction import resolve_labels, token_mapping_hash
        self.torch, self.device, self.model_id, self.revision = torch, device, model_id, revision
        self.dtype_name = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir,
            dtype=getattr(torch, dtype), device_map=device).eval()
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        mapping = resolve_labels(enc, 20)
        self.mhash = token_mapping_hash(mapping, model_id, revision)
        self.mapping = {k: ([int(x) for x in v] if k == "token_ids" else v) for k, v in mapping.items()}
        self.versions = {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                         "dtype": dtype, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else device}

    def _render(self, user_text):
        # Chat template, then open the model turn and end on the answer cue so the next token is the label.
        from src.llm.extraction import ANSWER_CUE
        msgs = [{"role": "user", "content": user_text}]
        prefix = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prefix + ANSWER_CUE

    def label_logits(self, prompts, n_candidates, batch_size=8, max_padded_tokens=24000):
        """Batches are capped by size AND by padded tokens (batch x longest prompt): a fixed batch of 8 OOMed on
        A10G once prompts passed ~4k tokens. Chunking is a deterministic function of the inputs. If a chunk still
        OOMs it is re-run one prompt at a time. VERIFY: bf16 logits can differ in low bits across batch shapes."""
        import time
        torch = self.torch
        from src.llm.extraction import extract
        rendered = [self._render(p) for p in prompts]
        lens = [len(self.tok(r, add_special_tokens=False)["input_ids"]) for r in rendered]
        chunks, cur = [], []
        for i in range(len(rendered)):
            if cur and (len(cur) >= batch_size or (len(cur) + 1) * max(lens[k] for k in cur + [i]) > max_padded_tokens):
                chunks.append(cur); cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)

        def fwd(idx):
            enc = self.tok([rendered[k] for k in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            with torch.no_grad():
                # last position only: full-seq x 262k-vocab logits OOM on A10G
                lg = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float().cpu().numpy()
            return lg, enc["attention_mask"].sum(1).tolist()

        res = {}
        for idx in chunks:
            t0 = time.time()
            try:
                parts = [(idx, *fwd(idx))]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                parts = [([k], *fwd([k])) for k in idx]
            dt = (time.time() - t0) * 1000 / len(idx)
            for sub, lg, ntok in parts:
                for j, k in enumerate(sub):
                    raw, probs = extract(lg[j], self.mapping, n_candidates[k])
                    res[k] = {"raw": raw.tolist(), "probs": probs.tolist(), "input_tokens": int(ntok[j]),
                              "latency_ms": dt, "token_mapping_hash": self.mhash,
                              "model_revision": f"{self.model_id}@{self.revision}"}
        return [res[k] for k in range(len(prompts))]

    def generate(self, prompts, max_new_tokens=150, temperature=0.0, seed=0):
        import time
        torch = self.torch
        torch.manual_seed(seed)
        out = []
        for p in prompts:
            text = self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
            t0 = time.time()
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens)
            if temperature > 0:
                kw["temperature"] = temperature
            with torch.no_grad():
                g = self.model.generate(**enc, **kw)
            new = g[0, enc["input_ids"].shape[1]:]
            out.append({"text": self.tok.decode(new, skip_special_tokens=True),
                        "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": int(len(new)),
                        "latency_ms": (time.time() - t0) * 1000,
                        "model_revision": f"{self.model_id}@{self.revision}"})
        return out

    def info(self):
        import json
        return json.loads(json.dumps({"mapping": self.mapping, "token_mapping_hash": self.mhash,
                                      "model_class": type(self.model).__name__, **self.versions}))
