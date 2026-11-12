"""Kaggle backend smoke test (free T4 x2, NO internet). Pushed with `kaggle kernels push -p scripts/kaggle/smoke --accelerator NvidiaTeslaT4`.

Offline setup (datasets emfl-dev-bundle + emfl-offline-deps, model google/gemma-4 transformers gemma-4-e4b-it v1):
  - pinned wheels installed from the offline-deps dataset (pip --no-index); Kaggle's own torch is used and recorded.
  - Gemma: config/tokenizer/chat template come from the pinned HF revision (offline-deps); model.safetensors comes from
    Kaggle Models and is linked into the pinned snapshot ONLY if its sha256 equals the HF LFS hash for that revision.
    Any mismatch aborts the run before a single forward pass.
  - HF_HUB_OFFLINE=1, so nothing can silently resolve to another revision.

1. BF16 on 3 dev instances, reusing the Modal pilot's notes/candidates, so every value call is directly
   comparable to artifacts/runs/dev120 (compare locally with src.analysis.backend_compare).
2. One fresh dev instance under G2 settings (avg2, extras on) for end-to-end timing.
Dev split only. Outputs land in /kaggle/working/runs.
"""
import glob, hashlib, json, os, shutil, subprocess, sys, time

sh = lambda c: (print("+", c, flush=True), subprocess.run(c, shell=True, check=True))


def one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert len(hits) == 1, (pattern, hits, os.listdir("/kaggle/input"))
    return hits[0]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# Fail fast on a CPU session (pushing without `--accelerator NvidiaTeslaT4` gets the CPU image, torch+cpu).
sh("nvidia-smi -L")
DEPS = os.path.dirname(one("/kaggle/input/**/emfl-offline-deps*/manifest.json"))
man = json.load(open(f"{DEPS}/manifest.json"))
bad = [f for f, h in man["files"].items() if sha256(f"{DEPS}/{f}") != h]
assert not bad, f"offline-deps files corrupted: {bad}"
sh(f"pip install -q --no-index --find-links {DEPS}/wheels 'transformers==5.22.2' 'sentence-transformers==6.3.0' "
   f"'accelerate==1.15.0' rank-bm25 jsonschema pyyaml")

HUB = "/tmp/hf/hub"
shutil.copytree(f"{DEPS}/hf/hub", HUB, dirs_exist_ok=True)
g = man["gemma"]
snap = f"{HUB}/models--{g['hf_id'].replace('/', '--')}/snapshots/{g['revision']}"
w = one(f"/kaggle/input/**/gemma-4-e4b-it/**/{g['weights_file']}")
t0 = time.time()
got = sha256(w)
check = {"weights_path": w, "expected_sha256": g["weights_sha256"], "got_sha256": got,
         "size": os.path.getsize(w), "hash_s": round(time.time() - t0, 1), "match": got == g["weights_sha256"]}
print("WEIGHTS", json.dumps(check), flush=True)
os.makedirs("/kaggle/working/runs", exist_ok=True)
json.dump(check, open("/kaggle/working/weights_check.json", "w"))
if not check["match"]:
    sys.exit("ABORT: Kaggle Models weights != pinned HF revision " + g["revision"])
os.symlink(w, f"{snap}/{g['weights_file']}")
os.environ.update(HF_HUB_CACHE=HUB, HF_HOME="/tmp/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1")

src = one("/kaggle/input/**/emfl-dev-bundle*/src")
P = "/tmp/project"
shutil.copytree(os.path.dirname(src), P, dirs_exist_ok=True)
os.chdir(P)

import torch, transformers
env = {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "bf16_supported": torch.cuda.is_bf16_supported()}
print("ENV", json.dumps(env), flush=True)
json.dump(env, open("/kaggle/working/env.json", "w"))
sh(f"{sys.executable} -m pip freeze > /kaggle/working/pip_freeze.txt")

inst = "data/marketbugs/dev/cachetools.jsonl"
# BF16 only (protocol v1.0 fixes the dtype). kg_bf16: Modal-comparable value calls on reused pilot notes.
# kg_g2timing: one fresh dev instance under the actual G2 settings (avg2, extras on) for runtime per instance.
jobs = [("kg_bf16", "--protocol single --reuse-notes-from dev120 --no-extras --n 3"),
        ("kg_g2timing", "--protocol avg2 --n 1 --skip 5")]
for tag, extra in jobs:
    r = subprocess.run(f"{sys.executable} -m src.agents.pilot {inst} --tag {tag} --backend local --dtype bfloat16 "
                       f"--device-map auto --max-hours 3 {extra}", shell=True)
    print(f"JOB {tag} exit {r.returncode}", flush=True)
    if os.path.isdir(f"artifacts/runs/{tag}"):
        shutil.copytree(f"artifacts/runs/{tag}", f"/kaggle/working/runs/{tag}", dirs_exist_ok=True)
if os.path.exists("artifacts/costs.jsonl"):
    shutil.copy("artifacts/costs.jsonl", "/kaggle/working/costs_kaggle.jsonl")
