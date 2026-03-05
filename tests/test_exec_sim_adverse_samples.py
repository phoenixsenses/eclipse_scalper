from __future__ import annotations

from pathlib import Path

try:
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips


def test_exec_sim_adverse_samples_and_value() -> None:
    events = [
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 1, "payload": {"price": 100.0}},
        {"ts_utc": "2026-03-01T00:00:01Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 2, "payload": {"price": 99.8}},
        {"ts_utc": "2026-03-01T00:00:02Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 3, "payload": {"price": 99.6}},
        {"ts_utc": "2026-03-01T00:00:03Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 4, "payload": {"price": 100.1}},
        {"ts_utc": "2026-03-01T00:00:04Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 5, "payload": {"price": 100.2}},
    ]
    decisions = [{"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "decision_id": "d1", "params": {"side": "buy"}}]
    cfg = ExecSimConfig(fee_bps=0.0, qty=1.0, horizon_sec=4, side_rule="from_params")
    fills, skipped = simulate_fills_with_skips(decisions, events, cfg)
    assert not skipped
    assert len(fills) == 1
    f = fills[0]
    assert int(f["adverse_samples"]) == 5
    assert abs(float(f["adverse_move"]) - (99.6 - 100.0)) < 1e-9
    fills2, skipped2 = simulate_fills_with_skips(decisions, events, cfg)
    assert fills == fills2
    assert skipped == skipped2

