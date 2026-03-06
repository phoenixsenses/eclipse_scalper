from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.risk.policy import RiskPolicy
from src.microphys.risk.sizer import compute_risk_decision


def test_risk_exposure_cap_skips_when_full() -> None:
    policy = RiskPolicy(max_gross_exposure=0.1, max_position_per_symbol=0.1, min_trade_notional=10.0, max_trade_notional=1000.0)
    dec = compute_risk_decision(
        ts_ms=1,
        symbol="ETHUSDT",
        desired_side="buy",
        signal_row={"ensemble_score": 1.0, "spread_z": 0.0, "rv_z": 0.0, "liq_rate_z": 0.0},
        gating_row={"confidence_score": 1.0},
        live_status={"missing_bars_pct_1h": 0.0, "regime_shift": 0.0},
        policy=policy,
        mtm={"equity": 10000.0, "gross_notional": 1000.0, "by_symbol": {"ETHUSDT": {"notional": 1000.0}}},
    )
    assert dec.action == "SKIP"
    assert dec.reason == "RISK_CAP_EXPOSURE"

