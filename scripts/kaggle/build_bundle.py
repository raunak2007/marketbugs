"""Stage the private Kaggle dataset: code + configs + DEV instances/repos only (+ pilot notes for backend comparison).

python scripts/kaggle/build_bundle.py <out_dir> <kaggle_username>
Never includes test-split instances, eval_only.jsonl (gold), or test repos: no test outcomes before the G2 freeze.
"""
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV_REPOS = ("cachetools", "intervaltree", "sortedcontainers")
SKIP = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache", ".venv", "*.egg-info")


def main(out, user):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data/marketbugs/dev").mkdir(parents=True)
    shutil.copytree(ROOT / "src", out / "src", ignore=SKIP)
    shutil.copytree(ROOT / "configs", out / "configs", ignore=SKIP)
    shutil.copytree(ROOT / "schemas", out / "schemas", ignore=SKIP)
    for r in DEV_REPOS:
        rows = [json.loads(l) for l in open(ROOT / f"data/marketbugs/dev/{r}.jsonl")]
        assert all(x["split"] == "dev" for x in rows), r
        shutil.copy(ROOT / f"data/marketbugs/dev/{r}.jsonl", out / "data/marketbugs/dev")
        shutil.copytree(ROOT / "data/marketbugs/repos" / r, out / "data/marketbugs/repos" / r, ignore=SKIP)
    src = ROOT / "artifacts/runs/dev120/test_informed"
    dst = out / "artifacts/runs/dev120/test_informed"
    dst.mkdir(parents=True)
    for f in ("candidates_inf.jsonl", "notes.jsonl"):  # notes reuse => value calls directly comparable to Modal
        shutil.copy(src / f, dst / f)
    bad = [p for p in out.rglob("*") if "eval_only" in p.name or str(p.relative_to(out)).startswith("data/marketbugs/test")]
    assert not bad, bad
    meta = {"title": "emfl-dev-bundle", "id": f"{user}/emfl-dev-bundle", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    n = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"staged {n} files, {size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
