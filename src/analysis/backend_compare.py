"""Compare a Kaggle-backend run against the Modal pilot (dev only) on identical value calls.

python -m src.analysis.backend_compare <ref_dir> <run_dir> [<run_dir> ...]
  e.g. artifacts/runs/dev120/test_informed artifacts/kaggle/smoke/runs/kg_bf16/test_informed
Calls are matched on (id, mode, subset_mask, order_perm, note_order) and must share prompt_hash (same prompt text).
"""
import json, sys
from pathlib import Path
import numpy as np


def _rows(d):
    key = lambda r: (r["id"], r["mode"], r["subset_mask"], r["order_perm"], tuple(r["note_order"]))
    return {key(r): r for r in map(json.loads, open(Path(d) / "values_inf.jsonl"))}


def compare(ref_dir, run_dir):
    ref, run = _rows(ref_dir), _rows(run_dir)
    keys = [k for k in run if k in ref]
    same_prompt = sum(ref[k]["prompt_hash"] == run[k]["prompt_hash"] for k in keys)
    keys = [k for k in keys if ref[k]["prompt_hash"] == run[k]["prompt_hash"]]
    if not keys:
        return {"run": str(run_dir), "matched": 0}
    a = np.array([ref[k]["probs"] + [0] * (20 - len(ref[k]["probs"])) for k in keys])
    b = np.array([run[k]["probs"] + [0] * (20 - len(run[k]["probs"])) for k in keys])
    eps = 1e-12
    kl = (a * (np.log(a + eps) - np.log(b + eps))).sum(1)
    tel = [json.loads(l) for l in open(Path(run_dir) / "telemetry.jsonl")]
    lat = lambda kind: float(np.mean([t["latency_ms"] for t in tel if t["kind"] == kind])) if any(
        t["kind"] == kind for t in tel) else None
    return {"run": str(run_dir), "run_calls": len(run), "matched": len(keys), "prompt_hash_equal": same_prompt,
            "top1_agree": float((a.argmax(1) == b.argmax(1)).mean()),
            "max_abs_prob_diff": float(np.abs(a - b).max()), "mean_abs_prob_diff": float(np.abs(a - b).mean()),
            "mean_kl_ref_run": float(kl.mean()), "max_kl_ref_run": float(kl.max()),
            "nan_rows": int(np.isnan(b).any(1).sum()),
            "mean_latency_ms": {k: lat(k) for k in sorted({t["kind"] for t in tel})}}


if __name__ == "__main__":
    for d in sys.argv[2:]:
        print(json.dumps(compare(sys.argv[1], d), indent=1))
