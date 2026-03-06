from __future__ import annotations

from typing import Any, Dict, Tuple

from src.microphys.risk.policy import RiskPolicy


def check_kill_switch(state: Dict[str, Any], mtm: Dict[str, Any], policy: RiskPolicy, now_ts_ms: int) -> Tuple[bool, str]:
    kill_until = int(state.get("kill_until_ts_ms", 0) or 0)
    if kill_until > int(now_ts_ms):
        return True, "RISK_KILL_COOLDOWN_ACTIVE"
    dd = float(mtm.get("drawdown_pct", 0.0) or 0.0)
    if dd >= float(policy.drawdown_kill_pct):
        state["kill_until_ts_ms"] = int(now_ts_ms) + int(policy.kill_cooldown_minutes) * 60 * 1000
        return True, "RISK_KILL_DRAWDOWN"
    return False, "OK"

