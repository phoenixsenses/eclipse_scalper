from __future__ import annotations

import json
from pathlib import Path

try:
    from tools.strategies.micro_edge_pocket import MicroEdgePocketStrategy
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.strategies.micro_edge_pocket import MicroEdgePocketStrategy


def _event(ts_ms: int, intensity: float, imbalance: float, spread: float) -> dict:
    return {
        "ts_utc": "2026-03-01T00:00:00Z",
        "symbol": "ETHUSDT",
        "source_table": "agg_trades",
        "event_index": 1,
        "payload": {
            "ts_ms": ts_ms,
            "trade_intensity": intensity,
            "imbalance": imbalance,
            "spread": spread,
            "price": 100.0,
        },
    }


def test_micro_edge_pocket_thresholds_cooldown_and_determinism() -> None:
    cfg = {
        "rule": "micro_edge_v3_passive_alpha",
        "side": "buy",
        "symbol_whitelist": ["ETHUSDT"],
        "event_source_table": "agg_trades",
        "min_trade_count_window": 1,
        "cooldown_ms": 250,
        "filters": {"imbalance_gte": 0.4, "intensity_gte": 2500, "spread_lte": 0.0003},
    }
    s1 = MicroEdgePocketStrategy(**cfg)
    s2 = MicroEdgePocketStrategy(**cfg)
    events = [
        _event(1709251200000, 2000.0, 0.5, 0.0002),  # intensity below => no decision
        _event(1709251200100, 2600.0, 0.5, 0.0002),  # should fire
        _event(1709251200200, 2600.0, 0.5, 0.0002),  # cooldown => no decision
        _event(1709251200400, 2600.0, 0.5, 0.0002),  # cooldown passed => fire
    ]
    d1 = []
    d2 = []
    for ev in events:
        d1.extend(s1.on_event(ev))
        d2.extend(s2.on_event(ev))
    assert len(d1) == 2
    assert len(d2) == 2
    assert d1 == d2
    p0 = d1[0]["params"]
    assert p0["rule"] == "micro_edge_v3_passive_alpha"
    assert p0["side"] == "buy"
    assert "pocket_id" in p0 and len(str(p0["pocket_id"])) > 0
    assert p0["filters"]["imbalance_gte"] == 0.4
