"""Kaggle backend smoke test (free T4 x2). Pushed with `kaggle kernels push -p scripts/kaggle/smoke`.

1. bf16 vs fp16 on 3 dev instances, reusing the Modal pilot's notes/candidates, so every value call is directly
   comparable to artifacts/runs/dev120 (compare locally with src.analysis.backend_compare).
2. One fresh instance (notes generated, extras on) for end-to-end timing.
Dev split only. Outputs land in /kaggle/working/runs.
"""
import glob, json, os, shutil, subprocess, sys

sh = lambda c: (print("+", c, flush=True), subprocess.run(c, shell=True, check=True))
sh("pip install -q 'transformers==5.22.2' sentence-transformers rank-bm25 jsonschema")
src = glob.glob("/kaggle/input/**/emfl-dev-bundle*/src", recursive=True) or glob.glob("/kaggle/input/*/src")
assert src, os.listdir("/kaggle/input")
P = "/tmp/project"
shutil.copytree(os.path.dirname(src[0]), P, dirs_exist_ok=True)
os.chdir(P)

import torch
env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in
       range(torch.cuda.device_count())], "bf16_supported": torch.cuda.is_bf16_supported()}
print("ENV", json.dumps(env), flush=True)
os.makedirs("/kaggle/working/runs", exist_ok=True)
json.dump(env, open("/kaggle/working/env.json", "w"))

inst = "data/marketbugs/dev/cachetools.jsonl"
jobs = [("kg_bf16", "bfloat16", "--reuse-notes-from dev120 --no-extras --n 3"),
        ("kg_fp16", "float16", "--reuse-notes-from dev120 --no-extras --n 3"),
        ("kg_fresh", "bfloat16", "--n 1 --skip 5")]
for tag, dt, extra in jobs:
    r = subprocess.run(f"{sys.executable} -m src.agents.pilot {inst} --tag {tag} --backend local --dtype {dt} "
                       f"--protocol single --max-hours 3 {extra}", shell=True)
    print(f"JOB {tag} exit {r.returncode}", flush=True)
    if os.path.isdir(f"artifacts/runs/{tag}"):
        shutil.copytree(f"artifacts/runs/{tag}", f"/kaggle/working/runs/{tag}", dirs_exist_ok=True)
if os.path.exists("artifacts/costs.jsonl"):
    shutil.copy("artifacts/costs.jsonl", "/kaggle/working/costs_kaggle.jsonl")
