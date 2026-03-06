from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

try:
    from core.micro_features import MicroFeatureEngine
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.micro_features import MicroFeatureEngine


def _mk_db() -> Path:
    p = Path("data") / f"test_micro_min_{uuid.uuid4().hex}.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_builder_with_minimal_three_tables() -> None:
    db = _mk_db()
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE agg_trades(symbol TEXT, ts INTEGER, price REAL, qty REAL, side TEXT)")
        conn.execute("CREATE TABLE mark_prices(symbol TEXT, ts INTEGER, mark_price REAL)")
        conn.execute("CREATE TABLE liquidations(symbol TEXT, ts INTEGER, side TEXT, qty REAL)")
        now = int(time.time())
        for i in range(40):
            ts = now - (39 - i)
            conn.execute(
                "INSERT INTO agg_trades(symbol, ts, price, qty, side) VALUES(?,?,?,?,?)",
                ("BTCUSDT", ts, 100.0 + (0.01 * i), 1.0, "buy" if i % 2 == 0 else "sell"),
            )
            conn.execute(
                "INSERT INTO mark_prices(symbol, ts, mark_price) VALUES(?,?,?)",
                ("BTCUSDT", ts, 100.0 + (0.01 * i)),
            )
        conn.commit()
    finally:
        conn.close()

    try:
        eng = MicroFeatureEngine(str(db), "BTCUSDT", lookback_sec=120, update_interval_sec=1.0)
        feat = eng._compute_once()
        assert feat is not None
        ready, reason, _detail = eng.get_readiness("BTCUSDT")
        assert ready is True
        assert reason == "ok"
    finally:
        db.unlink(missing_ok=True)

