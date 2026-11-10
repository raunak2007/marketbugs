"""Probability extraction protocol (PLAN 4.1).

Gather logits of the K label tokens at the final prompt position and softmax over those K only.
Never uses top-k logprob lists. Model-agnostic: works on a logits vector + tokenizer encode fn.
"""
from __future__ import annotations
import hashlib, json
import numpy as np

LETTERS = [chr(ord("A") + i) for i in range(20)]
NUMERALS = [str(i) for i in range(1, 21)]
ANSWER_CUE = "Answer:"


class LabelError(RuntimeError):
    pass


def resolve_labels(encode, k=20):
    """Pick a single-token label scheme. `encode(str) -> list[int]` without special tokens.

    Prompt ends with ANSWER_CUE, so the natural continuation is ' X'. We require every label to be
    exactly one token under the SAME prefix convention, checked both standalone and in context
    (ANSWER_CUE + label must tokenize as encode(ANSWER_CUE) + [label_id]).
    """
    cue = encode(ANSWER_CUE)
    for scheme_name, scheme in (("letters", LETTERS[:k]), ("numerals", NUMERALS[:k])):
        for prefix in (" ", ""):
            ids = []
            for lab in scheme:
                standalone = encode(prefix + lab)
                in_ctx = encode(ANSWER_CUE + prefix + lab)
                if len(standalone) != 1 or in_ctx != cue + standalone:
                    break
                ids.append(standalone[0])
            if len(ids) == k and len(set(ids)) == k:
                return {"scheme": scheme_name, "prefix": prefix, "labels": list(scheme), "token_ids": ids}
    raise LabelError("no single-token label scheme found")


def token_mapping_hash(mapping, tokenizer_id, tokenizer_revision):
    blob = json.dumps({"tok": tokenizer_id, "rev": tokenizer_revision, "prefix": mapping["prefix"],
                       "labels": mapping["labels"], "ids": mapping["token_ids"]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def label_logits(final_logits, token_ids):
    """Gather label-token logits from the full-vocab logits at the final position."""
    final_logits = np.asarray(final_logits, dtype=np.float64)
    if final_logits.ndim != 1:
        raise ValueError("expected 1-D final-position logits")
    if max(token_ids) >= final_logits.shape[0]:
        raise ValueError("label token id outside vocab")
    out = final_logits[np.asarray(token_ids)]
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite label logit")  # no zero-filling, ever
    return out


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def extract(final_logits, mapping, n_candidates=None):
    """Return (raw_label_logits, probs) over the first n_candidates labels."""
    ids = mapping["token_ids"][: n_candidates or len(mapping["token_ids"])]
    raw = label_logits(final_logits, ids)
    return raw, softmax(raw)


def gold_logscore(probs, labels, gold_labels):
    """s(p) = log(sum of p over G). -inf if gold not in pool (retrieval miss handled by caller)."""
    idx = [labels.index(g) for g in gold_labels if g in labels]
    if not idx:
        return float("-inf")
    return float(np.log(np.sum(np.asarray(probs)[idx])))
