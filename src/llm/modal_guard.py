"""Workspace spend guard for Modal runs (paid runs need an explicit cap, enforced in code).

Every Modal run must name the workspace it expects and a monthly USD cap for that workspace. Before the run:
  1. the active Modal profile's workspace must equal the expected one (no billing the wrong account);
  2. already = max(Modal's metered cost this month, our own ledger of conservative run costs this month in that
     workspace), since Modal's billing can lag behind recent runs;
  3. the run's own budget becomes cap - already; the pilot's per-instance guard then stops before crossing it.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "artifacts/costs.jsonl"


def ledger_usd(workspace, now=None):
    now = now or datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    tot = 0.0
    if LEDGER.exists():
        for l in open(LEDGER):
            r = json.loads(l)
            if r.get("backend") == f"modal:{workspace}" and str(r.get("ts", "")).startswith(month):
                tot += float(r.get("conservative_usd", 0.0))
    return tot


def remaining_budget(expect_workspace, cap_usd):
    """-> USD this run may spend. Exits (no GPU started) on a workspace mismatch or an exhausted cap."""
    if not expect_workspace or cap_usd is None:
        sys.exit("Modal run refused: pass --workspace <name> and --workspace-cap-usd <monthly cap>")
    import modal
    ws = modal.Workspace.from_context()
    ws.hydrate()
    if ws.name != expect_workspace:
        sys.exit(f"Modal run refused: active workspace is {ws.name!r}, expected {expect_workspace!r} "
                 f"(use MODAL_PROFILE=<profile>)")
    metered = float(ws.billing.summary().metered_cost)
    already = max(metered, ledger_usd(expect_workspace))
    left = float(cap_usd) - already
    print(f"workspace guard: {ws.name}, metered this month ${metered:.2f}, ledger ${ledger_usd(expect_workspace):.2f}, "
          f"cap ${cap_usd:.2f}, this run may spend ${left:.2f}", flush=True)
    if left <= 0:
        sys.exit(f"Modal run refused: workspace {ws.name} at ${already:.2f} of its ${cap_usd:.2f} monthly cap")
    return left
