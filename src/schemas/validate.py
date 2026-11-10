"""Schema v2 validators (frozen at G0). Usage: python -m src.schemas.validate <kind> <file.jsonl> [...]"""
import json, sys, math
from pathlib import Path
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
KINDS = ("instance", "candidates", "note", "value_row")
_V = {k: Draft202012Validator(json.loads((SCHEMA_DIR / f"{k}.v2.json").read_text())) for k in KINDS}


def semantic_errors(kind, row):
    """Checks JSON Schema cannot express."""
    errs = []
    if kind == "candidates":
        cids = [c["cid"] for c in row["pool"]]
        if len(cids) != row["pool_size"]:
            errs.append("pool_size != len(pool)")
        if len(set(cids)) != len(cids):
            errs.append("duplicate cid")
        if sorted(row["candidate_order"]) != sorted(cids):
            errs.append("candidate_order is not a permutation of pool cids")
    if kind == "value_row":
        p = row["probs"]
        if len(p) != len(row["raw_candidate_logits"]):
            errs.append("probs/logits length mismatch")
        if not all(math.isfinite(x) and x >= 0 for x in p) or abs(sum(p) - 1) > 1e-4:
            errs.append("probs not a finite distribution")
        if row["gold_logscore"] > 1e-9:
            errs.append("gold_logscore > 0")
    if kind == "instance" and row["source"] == "marketbugs":
        if not row.get("conditions") or not row.get("failing_tests"):
            errs.append("marketbugs instance needs conditions and failing_tests")
    return errs


def validate_row(kind, row):
    errs = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in _V[kind].iter_errors(row)]
    return errs or semantic_errors(kind, row)


def validate_file(kind, path):
    bad = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                errs = validate_row(kind, json.loads(line))
                if errs:
                    bad.append((i, errs))
    return bad


def main(argv):
    kind, files = argv[0], argv[1:]
    n_bad = 0
    for p in files:
        for i, errs in validate_file(kind, p):
            n_bad += 1
            print(f"{p}:{i}: {'; '.join(errs)}")
    print(f"{kind}: {len(files)} file(s), {n_bad} invalid row(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
