from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.risk.policy import RiskPolicy
from src.microphys.risk.sizer import compute_risk_decision


def test_risk_sizer_deterministic() -> None:
    policy = RiskPolicy()
    kwargs = dict(
        ts_ms=1,
        symbol="ETHUSDT",
        desired_side="buy",
        signal_row={"ensemble_score": 0.42, "spread_z": 0.1, "rv_z": 0.2, "liq_rate_z": 0.0, "expected_fill_rate": 0.9},
        gating_row={"confidence_score": 0.7},
        live_status={"missing_bars_pct_1h": 0.0, "regime_shift": 0.0},
        policy=policy,
        mtm={"equity": 10000.0, "gross_notional": 0.0, "by_symbol": {}},
    )
    a = compute_risk_decision(**kwargs)
    b = compute_risk_decision(**kwargs)
    assert a == b
    assert a.action == "TRADE"

