from __future__ import annotations

import json
from pathlib import Path

try:
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips


def test_exec_sim_skipped_reasons_and_determinism() -> None:
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
            "payload": {"price": 101.0},
        },
    ]
    decisions = [
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "decision_id": "ok", "params": {}},
        {"ts_utc": "2026-03-01T00:00:01Z", "symbol": "ETHUSDT", "decision_id": "nohorizon", "params": {}},
        {"ts_utc": "2026-03-01T00:00:03Z", "symbol": "ETHUSDT", "decision_id": "late", "params": {}},
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "BTCUSDT", "decision_id": "nosym", "params": {}},
    ]
    cfg = ExecSimConfig(fee_bps=1.0, qty=1.0, horizon_sec=1)

    f1, s1 = simulate_fills_with_skips(decisions, events, cfg)
    f2, s2 = simulate_fills_with_skips(decisions, events, cfg)
    assert f1 == f2
    assert s1 == s2
    assert len(f1) == 2
    assert all(f["status"] == "filled" for f in f1)
    reasons = sorted(x["reason"] for x in s1)
    assert reasons == ["no_event_at_or_after_ts", "no_event_at_or_after_ts"]

    b1 = "\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in f1) + "\n"
    b2 = "\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in f2) + "\n"
    assert b1.encode("utf-8") == b2.encode("utf-8")
