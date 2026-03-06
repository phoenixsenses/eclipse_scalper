from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

try:
    from tools.state_reconstruct import reconstruct_state_vectors, write_state_vectors_jsonl
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.state_reconstruct import reconstruct_state_vectors, write_state_vectors_jsonl


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL, quantity REAL, is_buyer_maker INTEGER)"
        )
        conn.execute(
            "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, mark_price REAL)"
        )
        conn.execute(
            "CREATE TABLE liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, quantity REAL, side TEXT)"
        )
        rows_t = [
            (1709251200000, "ETHUSDT", 100.0, 1.0, 0),
            (1709251201000, "ETHUSDT", 100.1, 2.0, 1),
            (1709251202000, "ETHUSDT", 100.2, 1.0, 0),
        ]
        rows_m = [
            (1709251200000, "ETHUSDT", 100.0),
            (1709251201000, "ETHUSDT", 100.05),
            (1709251202000, "ETHUSDT", 100.10),
        ]
        rows_l = [
            (1709251201500, "ETHUSDT", 3.0, "sell"),
        ]
        conn.executemany("INSERT INTO agg_trades (ts_ms, symbol, price, quantity, is_buyer_maker) VALUES (?, ?, ?, ?, ?)", rows_t)
        conn.executemany("INSERT INTO mark_prices (ts_ms, symbol, mark_price) VALUES (?, ?, ?)", rows_m)
        conn.executemany("INSERT INTO liquidations (ts_ms, symbol, quantity, side) VALUES (?, ?, ?, ?)", rows_l)
        conn.commit()
    finally:
        conn.close()


def test_state_reconstruct_deterministic() -> None:
    base = Path("eclipse_scalper/localtests/state_reconstruct") / uuid.uuid4().hex
    db = base / "db.sqlite"
    _mk_db(db)

    r1 = reconstruct_state_vectors(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        window_sec=30.0,
    )
    r2 = reconstruct_state_vectors(
        db=db,
        symbols=["ETHUSDT"],
        start_iso="2024-03-01T00:00:00Z",
        end_iso="2024-03-01T00:01:00Z",
        window_sec=30.0,
    )
    assert r1 == r2
    assert len(r1) > 0
    st = r1[-1]["state"]
    for key in ("order_flow_imbalance", "spread_proxy", "liquidity_pressure", "trade_rate", "vol_proxy"):
        assert key in st

    out1 = base / "state_1.jsonl"
    out2 = base / "state_2.jsonl"
    write_state_vectors_jsonl(out1, r1)
    write_state_vectors_jsonl(out2, r2)
    assert out1.read_bytes() == out2.read_bytes()

