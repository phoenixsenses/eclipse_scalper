from __future__ import annotations

from pathlib import Path

try:
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips


def test_exec_sim_horizon_before_fallback_reduces_skips() -> None:
    events = [
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 1, "payload": {"price": 100.0}},
        {"ts_utc": "2026-03-01T00:00:01Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 2, "payload": {"price": 100.2}},
        {"ts_utc": "2026-03-01T00:00:02Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 3, "payload": {"price": 100.4}},
    ]
    decisions = [
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "decision_id": "d1", "params": {}},
        {"ts_utc": "2026-03-01T00:00:01Z", "symbol": "ETHUSDT", "decision_id": "d2", "params": {}},
    ]
    cfg = ExecSimConfig(fee_bps=0.0, qty=1.0, horizon_sec=5, horizon_or_before_enabled=True)
    fills, skipped = simulate_fills_with_skips(decisions, events, cfg)
    assert len(fills) == 2
    assert len(skipped) == 0
    assert all(str(f.get("horizon_price_source")) == "before" for f in fills)
    assert all("pnl" in f for f in fills)

    fills2, skipped2 = simulate_fills_with_skips(decisions, events, cfg)
    assert fills == fills2
    assert skipped == skipped2

