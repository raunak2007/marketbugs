"""Stage the private Kaggle dataset `emfl-offline-deps` so notebooks run with NO internet.

python scripts/kaggle/build_offline.py <out_dir> <kaggle_username>

Contents:
  wheels/    pinned pure-Python deps for Kaggle's Python 3.13 / x86_64 (torch + CUDA libs excluded: Kaggle's own are used
             and recorded per run). Installed with `pip install --no-index --find-links`.
  hf/hub/    HF-cache-layout snapshots at the PINNED revisions: every Gemma file EXCEPT model.safetensors (Kaggle's copy
             of the tokenizer/chat template differs from our revision, so those come from HF), plus the bge embedder.
  manifest.json  sha256 of every staged file + the expected sha256 of Gemma's model.safetensors (from the HF LFS
             pointer at the pinned revision). The notebook links Kaggle Models' weights in only if the hash matches.
"""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.llm.hf_backend import EMB_MODEL, EMB_REV  # noqa: E402

REQS = ["transformers==5.22.2", "sentence-transformers==6.3.0", "accelerate==1.15.0", "rank-bm25", "jsonschema",
        "pyyaml"]
DROP = ("torch", "triton", "nvidia-", "cuda-", "setuptools")  # provided by the Kaggle image
WEIGHTS = "model.safetensors"
EMB_FILES = ("1_Pooling/config.json", "config.json", "config_sentence_transformers.json", "model.safetensors",
             "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.txt")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage_snapshot(out, repo, rev, files):
    snap = out / "hf/hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    for f in files:
        src = hf_hub_download(repo, f, revision=rev)
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, snap / f)  # real file, not a symlink: Kaggle datasets don't keep links


def main(out, user):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "wheels").mkdir(parents=True)
    pa = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["probability_agent"]
    api = HfApi()
    info = api.model_info(pa["hf_id"], revision=pa["revision"], files_metadata=True)
    assert info.sha == pa["revision"], (info.sha, pa["revision"])
    sib = {s.rfilename: s for s in info.siblings}
    gemma_files = [f for f in sib if f not in (WEIGHTS, ".gitattributes", "README.md")]
    stage_snapshot(out, pa["hf_id"], pa["revision"], gemma_files)
    stage_snapshot(out, EMB_MODEL, EMB_REV, EMB_FILES)

    lock = out / "requirements.lock"
    (out / "requirements.in").write_text("\n".join(REQS) + "\n")
    subprocess.run(["uv", "pip", "compile", str(out / "requirements.in"), "-o", str(lock), "--python-version", "3.13",
                    "--python-platform", "x86_64-manylinux_2_28", "--no-header", "--no-annotate", "-q"], check=True)
    pins = [l.strip() for l in lock.read_text().splitlines() if l.strip() and not l.startswith("#")]
    keep = [p for p in pins if not p.lower().startswith(DROP)]
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "-d",
                    str(out / "wheels"), "--python-version", "3.13", "--implementation", "cp", "--abi", "cp313",
                    "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux2014_x86_64",
                    "--platform", "manylinux_2_17_x86_64", *keep], check=True)
    (out / "requirements.kaggle.txt").write_text("\n".join(keep) + "\n")

    manifest = {"gemma": {"hf_id": pa["hf_id"], "revision": pa["revision"], "weights_file": WEIGHTS,
                          "weights_sha256": sib[WEIGHTS].lfs.sha256, "weights_size": sib[WEIGHTS].size},
                "embedder": {"hf_id": EMB_MODEL, "revision": EMB_REV},
                "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    meta = {"title": "emfl-offline-deps", "id": f"{user}/emfl-offline-deps", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"staged {len(manifest['files'])} files, {size / 1e6:.1f} MB -> {out}; dropped "
          f"{[p for p in pins if p not in keep]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
