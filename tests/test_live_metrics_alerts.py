from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.live.alerts import evaluate_alerts
from src.microphys.live.config import LiveSettings
from src.microphys.live.metrics import compute_live_metrics


def test_compute_live_metrics_basic() -> None:
    physics = pd.DataFrame(
        {
            "ts_ms": [1_700_000_000_000 + i * 100 for i in range(100)],
            "spread": [0.001] * 100,
            "F_ofi_z": [(-1 + 2 * (i / 99)) for i in range(100)],
        }
    )
    trades = pd.DataFrame(
        {
            "entry_ts_utc": [f"2024-03-01T00:00:{i:02d}Z" for i in range(10)],
            "pnl_net": [0.001] * 10,
            "pnl_gross": [0.002] * 10,
        }
    )
    m = compute_live_metrics(
        physics_recent=physics,
        live_trades=trades,
        baseline={"spread_median": 0.001, "ofi_ref": [0.0] * 10, "regime_ref": [0, 1] * 5},
        db_last_event_ts=1_700_000_000.0,
        interval_ms=100,
    )
    assert "signal_rate_per_hour" in m
    assert "spread_median" in m


def test_alert_rules_fire() -> None:
    cfg = LiveSettings()
    status = {
        "data_freshness_sec": 999.0,
        "missing_bars_pct_1h": 50.0,
        "spread_jump_frac": 0.8,
        "ofi_shift": 2.0,
        "regime_shift": 1.0,
        "signal_rate_per_hour": 0.0,
    }
    alerts = evaluate_alerts(status, cfg)
    codes = {a["code"] for a in alerts}
    assert "data_stale" in codes
    assert "missing_bars" in codes
    assert "spread_jump" in codes
