import glob, numpy as np, pytest
from src.llm.extraction import resolve_labels, extract, token_mapping_hash, LabelError, gold_logscore, label_logits


class FakeTok:
    """Word-level fake: ' X' and 'X' map to distinct single ids; 'Answer:' is its own token."""
    def __init__(self, split_letters=()):
        self.v, self.split = {}, set(split_letters)

    def _id(self, s):
        return self.v.setdefault(s, len(self.v) + 5)

    def encode(self, s):
        out, i = [], 0
        if s.startswith("Answer:"):
            out.append(self._id("Answer:")); s = s[7:]
        if not s:
            return out
        lab = s.strip()
        if lab in self.split:
            return out + [self._id(s[:-1] or "_"), self._id(lab[-1])]
        return out + [self._id(s)]


def test_letters_single_token_space_prefix():
    m = resolve_labels(FakeTok().encode)
    assert m["scheme"] == "letters" and m["prefix"] == " " and len(set(m["token_ids"])) == 20


def test_fallback_to_numerals_when_letter_splits():
    t = FakeTok()
    real = t.encode
    t.encode = lambda s: real(s) + ([999] if s.strip() == "Q" else [])
    m = resolve_labels(t.encode)
    assert m["scheme"] == "numerals"


def test_no_scheme_raises():
    with pytest.raises(LabelError):
        resolve_labels(lambda s: [1, 2])


def _fake_logits(seed, vocab=1000):
    return np.random.default_rng(seed).normal(size=vocab) * 5


def test_probs_sum_finite_repeatable():
    m = resolve_labels(FakeTok().encode)
    lg = _fake_logits(0)
    raw1, p1 = extract(lg, m)
    raw2, p2 = extract(lg.copy(), m)
    assert p1.shape == (20,) and np.all(np.isfinite(p1)) and abs(p1.sum() - 1) < 1e-12
    assert np.array_equal(p1, p2)
    assert np.allclose(raw1, lg[m["token_ids"]])  # gather from full vocab, not top-k


def test_extreme_logits_stay_finite():
    m = resolve_labels(FakeTok().encode)
    lg = np.zeros(1000); lg[m["token_ids"][3]] = 1e4
    _, p = extract(lg, m)
    assert np.all(np.isfinite(p)) and p.argmax() == 3


def test_no_zero_filling_nan_raises():
    m = resolve_labels(FakeTok().encode)
    lg = _fake_logits(1); lg[m["token_ids"][5]] = np.nan
    with pytest.raises(ValueError):
        extract(lg, m)


def test_out_of_vocab_raises():
    with pytest.raises(ValueError):
        label_logits(np.zeros(10), [3, 50])


def test_smaller_pool_renormalizes_over_present_labels_only():
    m = resolve_labels(FakeTok().encode)
    _, p = extract(_fake_logits(2), m, n_candidates=12)
    assert p.shape == (12,) and abs(p.sum() - 1) < 1e-12


def test_label_mapping_round_trip_and_hash_stable():
    m = resolve_labels(FakeTok().encode)
    order = list("QDAMTBCEFGHIJKLNOPRS")
    labels = m["labels"]
    cid_of_label = dict(zip(labels, order))
    label_of_cid = {c: l for l, c in cid_of_label.items()}
    assert all(cid_of_label[label_of_cid[c]] == c for c in order)
    h1 = token_mapping_hash(m, "tok", "rev"); h2 = token_mapping_hash(resolve_labels(FakeTok().encode), "tok", "rev")
    assert h1 == h2 and h1 != token_mapping_hash(m, "tok", "rev2")


def test_gold_logscore():
    p = np.full(20, 0.05)
    labels = [chr(65 + i) for i in range(20)]
    assert abs(gold_logscore(p, labels, ["A", "B"]) - np.log(0.1)) < 1e-12
    assert gold_logscore(p, labels, ["Z"]) == float("-inf")


TOK = glob.glob("artifacts/tok/models--google--gemma-4-E4B-it/snapshots/*/tokenizer.json", recursive=True)


@pytest.mark.skipif(not TOK, reason="Gemma tokenizer not cached")
def test_real_gemma4_tokenizer_labels():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(TOK[0])
    m = resolve_labels(lambda s: t.encode(s, add_special_tokens=False).ids)
    assert m["scheme"] == "letters" and m["prefix"] == " "
