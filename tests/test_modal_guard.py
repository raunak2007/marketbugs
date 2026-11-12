import json, sys, types
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.llm import modal_guard


def _fake_modal(monkeypatch, name, metered):
    class WS:
        def __init__(self):
            self.name = name
            self.billing = types.SimpleNamespace(summary=lambda: types.SimpleNamespace(metered_cost=Decimal(metered)))

        def hydrate(self):
            return self
    m = types.SimpleNamespace(Workspace=types.SimpleNamespace(from_context=WS))
    monkeypatch.setitem(sys.modules, "modal", m)


def _ledger(tmp_path, monkeypatch, rows):
    p = tmp_path / "costs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(modal_guard, "LEDGER", p)


def test_refuses_without_workspace_or_cap():
    with pytest.raises(SystemExit):
        modal_guard.remaining_budget(None, 25)
    with pytest.raises(SystemExit):
        modal_guard.remaining_budget("ws", None)


def test_refuses_wrong_workspace(monkeypatch, tmp_path):
    _ledger(tmp_path, monkeypatch, [])
    _fake_modal(monkeypatch, "aarushagg12", "0")
    with pytest.raises(SystemExit, match="expected 'ratchakraborty'"):
        modal_guard.remaining_budget("ratchakraborty", 25)


def test_uses_larger_of_metered_and_ledger(monkeypatch, tmp_path):
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    _ledger(tmp_path, monkeypatch, [
        {"backend": "modal:ratchakraborty", "ts": f"{month}-03T00:00:00+00:00", "conservative_usd": 7.0},
        {"backend": "modal:ratchakraborty", "ts": "1999-01-01T00:00:00+00:00", "conservative_usd": 100.0},
        {"backend": "modal:rmondal", "ts": f"{month}-03T00:00:00+00:00", "conservative_usd": 100.0},
        {"backend": "modal", "conservative_usd": 100.0}])
    _fake_modal(monkeypatch, "ratchakraborty", "2.50")  # billing lags: ledger (7.0) wins
    assert modal_guard.remaining_budget("ratchakraborty", 25) == pytest.approx(18.0)
    _fake_modal(monkeypatch, "ratchakraborty", "10")
    assert modal_guard.remaining_budget("ratchakraborty", 25) == pytest.approx(15.0)


def test_refuses_when_cap_exhausted(monkeypatch, tmp_path):
    _ledger(tmp_path, monkeypatch, [])
    _fake_modal(monkeypatch, "ratchakraborty", "25.01")
    with pytest.raises(SystemExit, match="monthly cap"):
        modal_guard.remaining_budget("ratchakraborty", 25)
