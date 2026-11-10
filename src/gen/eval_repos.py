"""Evaluate candidate MarketBugs repos: license, pure-python, suite time x3, determinism."""
import json, subprocess, sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT / "data/marketbugs/repos"
CANDS = {
 "toolz": "https://github.com/pytoolz/toolz", "sortedcontainers": "https://github.com/grantjenks/python-sortedcontainers",
 "more-itertools": "https://github.com/more-itertools/more-itertools", "boltons": "https://github.com/mahmoud/boltons",
 "cachetools": "https://github.com/tkem/cachetools", "dateutil": "https://github.com/dateutil/dateutil",
 "tomli": "https://github.com/hukkin/tomli", "attrs": "https://github.com/python-attrs/attrs",
 "funcy": "https://github.com/Suor/funcy", "pyparsing": "https://github.com/pyparsing/pyparsing",
 "bidict": "https://github.com/jab/bidict", "parse": "https://github.com/r1chardj0n3s/parse",
 "humanize": "https://github.com/python-humanize/humanize", "glom": "https://github.com/mahmoud/glom",
 "isodate": "https://github.com/gweis/isodate", "intervaltree": "https://github.com/chaimleib/intervaltree",
}
def sh(cmd, cwd=None, timeout=900):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout[-3000:] + r.stderr[-2000:]
def main(names):
    REPOS.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        d = REPOS / n; res = {"url": CANDS[n]}
        try:
            if not d.exists():
                sh(f"git clone -q --depth 1 {CANDS[n]} {d}")
            res["commit"] = sh("git rev-parse HEAD", d)[1].strip()
            lic = next((p for p in d.iterdir() if p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))), None)
            res["license_head"] = lic.read_text(errors="ignore")[:200].replace("\n", " ") if lic else None
            res["c_files"] = len([p for p in d.rglob("*.c") if ".venv" not in p.parts]) + len([p for p in d.rglob("*.pyx")])
            res["py_files"] = len([p for p in d.rglob("*.py") if ".venv" not in p.parts])
            code, log = sh("uv venv -q -p 3.11 .venv && uv pip install -q -p .venv -e . pytest pytest-timeout hypothesis coverage "
                           + " ".join(f"-r {r}" for r in ["requirements-test.txt", "test-requirements.txt", "requirements/test.txt", "requirements-dev.txt"] if (d / r).exists()), d)
            res["install_ok"] = code == 0
            if code: res["install_log"] = log[-800:]
            times, summaries = [], []
            for _ in range(3):
                t = time.time(); code, log = sh(".venv/bin/python -m pytest -q -x -p no:cacheprovider --timeout=120", d, timeout=400)
                times.append(round(time.time() - t, 1)); summaries.append((code, log.strip().splitlines()[-1] if log.strip() else ""))
            res["suite_s"], res["suite"] = times, summaries
            res["deterministic"] = len({s for s in summaries}) == 1 and summaries[0][0] == 0
        except Exception as e:
            res["error"] = repr(e)
        out[n] = res; print(n, json.dumps({k: v for k, v in res.items() if k not in ("install_log",)})[:400], flush=True)
    (ROOT / "artifacts/repo_eval.json").write_text(json.dumps(out, indent=1))
if __name__ == "__main__":
    main(sys.argv[1:] or list(CANDS))
