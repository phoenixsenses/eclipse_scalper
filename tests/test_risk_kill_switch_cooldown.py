from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.risk.guards import check_kill_switch
from src.microphys.risk.policy import RiskPolicy


def test_kill_switch_drawdown_and_cooldown() -> None:
    st = {"kill_until_ts_ms": 0}
    pol = RiskPolicy(drawdown_kill_pct=0.02, kill_cooldown_minutes=1)
    active, reason = check_kill_switch(st, {"drawdown_pct": 0.05}, pol, 1_000)
    assert active is True
    assert reason == "RISK_KILL_DRAWDOWN"
    assert int(st["kill_until_ts_ms"]) > 1_000
    active2, reason2 = check_kill_switch(st, {"drawdown_pct": 0.0}, pol, 1_500)
    assert active2 is True
    assert reason2 == "RISK_KILL_COOLDOWN_ACTIVE"

