"""Re-apply build_marketbugs.sanitize to already-built instances (text fields only; gold/strata untouched).
python -m src.gen.resanitize [--write]"""
import json, sys
from pathlib import Path
from src.gen.build_marketbugs import sanitize, ROOT

MB = ROOT / "data/marketbugs"
TEXT = ("issue_text", "trace")

def main(write):
    changed = 0
    for path in sorted(MB.glob("*/*.jsonl")):
        if path.name == "eval_only.jsonl":
            continue
        rows = [json.loads(l) for l in open(path)]
        d = MB / "repos" / rows[0]["repo"]
        for r in rows:
            before = json.dumps(r)
            for k in TEXT:
                r[k] = sanitize(r[k], d)
            r["conditions"] = {k: sanitize(v, d) for k, v in r["conditions"].items()}
            changed += json.dumps(r) != before
        if write:
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("rows changed:", changed, "(written)" if write else "(dry run)")

if __name__ == "__main__":
    main("--write" in sys.argv)
