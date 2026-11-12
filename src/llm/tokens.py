"""Tokenizer-aware truncation (PLAN 4.2: candidate code truncated to a fixed TOKEN budget, same for every method).

Uses the pinned Gemma tokenizer.json from the local HF cache (artifacts/tok/, revision from configs/models.yaml).
Prompts are built locally, so the server never re-truncates. No silent character fallback: if the tokenizer file is
missing, prompt building fails loudly (tests may inject a tokenizer with set_tokenizer).
"""
from functools import lru_cache
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
CODE_TOKEN_BUDGET = int(CFG["prompting"]["code_token_budget"])
TRUNC_MARK = "\n# ...truncated"
ISSUE_TOKEN_BUDGET = int(CFG["prompting"]["issue_token_budget"])
ISSUE_HEAD_TOKENS = int(CFG["prompting"]["issue_head_tokens"])
_injected = None


def set_tokenizer(tok):
    """tok: object with encode(str)->ids and decode(ids)->str, or None to restore the pinned tokenizer."""
    global _injected
    _injected = tok


@lru_cache(maxsize=1)
def _pinned():
    from tokenizers import Tokenizer
    pa = CFG["probability_agent"]
    org, name = pa["hf_id"].split("/")
    p = ROOT / "artifacts/tok" / f"models--{org}--{name}" / "snapshots" / pa["revision"] / "tokenizer.json"
    if not p.exists():
        raise FileNotFoundError(f"pinned tokenizer not cached: {p} (download it before building prompts)")
    t = Tokenizer.from_file(str(p))

    class _T:
        def encode(self, s):
            return t.encode(s, add_special_tokens=False).ids

        def decode(self, ids):
            return t.decode(ids, skip_special_tokens=False)
    return _T()


def tokenizer():
    return _injected or _pinned()


def n_tokens(s):
    return len(tokenizer().encode(s))


def truncate_code(code, budget=CODE_TOKEN_BUDGET):
    t = tokenizer()
    ids = t.encode(code)
    if len(ids) <= budget:
        return code
    return t.decode(ids[:budget]) + TRUNC_MARK


def truncate_issue(text, budget=ISSUE_TOKEN_BUDGET, head=ISSUE_HEAD_TOKENS):
    """Issue/failing-test text over `budget` tokens -> first `head` + last `budget - head` tokens with a marker that
    states how many tokens were cut. Text within budget is returned unchanged (byte-identical)."""
    if text is None:
        return None
    t = tokenizer()
    ids = t.encode(text)
    if len(ids) <= budget:
        return text
    tail = budget - head
    return (t.decode(ids[:head]) + f"\n[... {len(ids) - budget} tokens truncated ...]\n" + t.decode(ids[-tail:]))
