from __future__ import annotations

from pathlib import Path

try:
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills


def test_exec_sim_determinism() -> None:
    events = [
        {
            "event_index": 1,
            "ts_utc": "2026-03-01T00:00:00Z",
            "symbol": "ETHUSDT",
            "source_table": "agg_trades",
            "rowid": 1,
            "payload": {"price": 100.0},
        },
        {
            "event_index": 2,
            "ts_utc": "2026-03-01T00:00:01Z",
            "symbol": "ETHUSDT",
            "source_table": "agg_trades",
            "rowid": 2,
            "payload": {"price": 100.5},
        },
        {
            "event_index": 3,
            "ts_utc": "2026-03-01T00:00:03Z",
            "symbol": "ETHUSDT",
            "source_table": "agg_trades",
            "rowid": 3,
            "payload": {"price": 101.0},
        },
    ]
    decisions = [
        {
            "ts_utc": "2026-03-01T00:00:00Z",
            "symbol": "ETHUSDT",
            "action": "signal",
            "decision_id": "d1",
            "params": {},
        },
        {
            "ts_utc": "2026-03-01T00:00:01Z",
            "symbol": "ETHUSDT",
            "action": "signal",
            "decision_id": "d2",
            "params": {},
        },
    ]
    cfg = ExecSimConfig(fee_bps=1.0, qty=1.0, horizon_sec=2)
    f1 = simulate_fills(decisions, events, cfg)
    f2 = simulate_fills(decisions, events, cfg)
    assert f1 == f2
    assert len(f1) == 2
    assert all(x["status"] == "filled" for x in f1)

