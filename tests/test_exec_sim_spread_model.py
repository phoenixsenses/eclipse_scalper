from __future__ import annotations

from pathlib import Path

try:
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips


def test_exec_sim_spread_model_buy_math() -> None:
    events = [
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 1, "payload": {"price": 100.0}},
        {"ts_utc": "2026-03-01T00:00:10Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 2, "payload": {"price": 101.0}},
    ]
    decisions = [{"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "decision_id": "d1", "params": {"side": "buy"}}]
    cfg = ExecSimConfig(
        fee_bps=0.0,
        spread_bps=10.0,
        use_spread_model=True,
        qty=1.0,
        horizon_sec=10,
        side_rule="from_params",
    )
    fills, skipped = simulate_fills_with_skips(decisions, events, cfg)
    assert not skipped
    f = fills[0]
    assert abs(float(f["fill_px_raw"]) - 100.0) < 1e-9
    assert abs(float(f["horizon_px_raw"]) - 101.0) < 1e-9
    assert abs(float(f["fill_px"]) - 100.05) < 1e-9
    assert abs(float(f["horizon_px"]) - 100.9495) < 1e-9
    assert abs(float(f["pnl_gross"]) - (100.9495 - 100.05)) < 1e-9
    assert abs(float(f["pnl"]) - float(f["pnl_gross"])) < 1e-9

