from src.schemas.validate import validate_row
from tests.fixtures import make_instance


def _inst():
    inst, c = make_instance(0)
    for k in ("patch", "test_patch", "FAIL_TO_PASS"):
        inst.pop(k)
    return inst, c


def test_valid_instance_and_candidates():
    inst, c = _inst()
    assert validate_row("instance", inst) == []
    assert validate_row("candidates", c) == []


def test_extra_field_rejected():
    inst, _ = _inst(); inst["oops"] = 1
    assert validate_row("instance", inst)


def test_bad_order_rejected():
    _, c = _inst(); c["candidate_order"][0] = c["candidate_order"][1]
    assert validate_row("candidates", c)


def test_value_row_probs_checked():
    row = {"id": "x", "mode": "C", "subset_mask": 3, "order_perm": 0, "note_order": ["trace", "graph"],
           "raw_candidate_logits": [0.0, 1.0], "probs": [0.3, 0.3], "gold_logscore": -1.0, "top1": "B",
           "token_mapping_hash": "h", "prompt_hash": "h", "model_revision": "m", "inference_cost_usd": 0.0}
    assert validate_row("value_row", row)
    row["probs"] = [0.4, 0.6]
    assert validate_row("value_row", row) == []
