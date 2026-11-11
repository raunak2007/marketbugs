import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Prompt builders truncate code with the pinned Gemma tokenizer (src.llm.tokens). If it is not cached (fresh CI),
# inject a whitespace tokenizer so unit tests still run; the real budget is exercised when the cache exists.
from src.llm import tokens as _tokens
try:
    _tokens._pinned()
except FileNotFoundError:
    class _WS:
        def encode(self, s): return s.split(" ")
        def decode(self, ids): return " ".join(ids)
    _tokens.set_tokenizer(_WS())
