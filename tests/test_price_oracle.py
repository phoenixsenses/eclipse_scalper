from __future__ import annotations

from pathlib import Path

try:
    from execution.sim.price_oracle import PriceOracle
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.sim.price_oracle import PriceOracle


def _events():
    return [
        {"ts_utc": "2026-03-01T00:00:00Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 1, "payload": {"price": 100.0}},
        {"ts_utc": "2026-03-01T00:00:05Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 2, "payload": {"price": 101.0}},
        {"ts_utc": "2026-03-01T00:00:09Z", "symbol": "ETHUSDT", "source_table": "agg_trades", "rowid": 3, "payload": {"price": 102.0}},
    ]


def test_price_oracle_lookup_and_determinism() -> None:
    o1 = PriceOracle.build(_events(), ("price",))
    o2 = PriceOracle.build(_events(), ("price",))
    t = 1772323207.0  # 2026-03-01T00:00:07Z
    a1 = o1.price_at_or_after("ETHUSDT", t)
    b1 = o1.price_at_or_before("ETHUSDT", t)
    a2 = o2.price_at_or_after("ETHUSDT", t)
    b2 = o2.price_at_or_before("ETHUSDT", t)
    assert a1 is not None and a1.ts_utc == "2026-03-01T00:00:09Z"
    assert b1 is not None and b1.ts_utc == "2026-03-01T00:00:05Z"
    assert a2 is not None and a2.ts_utc == a1.ts_utc
    assert b2 is not None and b2.ts_utc == b1.ts_utc
