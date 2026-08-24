from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.io.sqlite_reader import SQLiteMicroReader, discover_mappings


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE x_trades(id INTEGER PRIMARY KEY, event_ts INTEGER, s TEXT, p REAL, q REAL, is_buyer_maker INTEGER)"
        )
        conn.execute(
            "CREATE TABLE x_mark(id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, mark_price REAL)"
        )
        conn.execute(
            "CREATE TABLE x_liq(id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, side TEXT, quantity REAL, price REAL)"
        )
        conn.execute("INSERT INTO x_trades(event_ts,s,p,q,is_buyer_maker) VALUES(1,'BTCUSDT',101.0,2.0,0)")
        conn.execute("INSERT INTO x_mark(ts_ms,symbol,mark_price) VALUES(1,'BTCUSDT',101.5)")
        conn.execute("INSERT INTO x_liq(ts_ms,symbol,side,quantity,price) VALUES(1,'BTCUSDT','sell',3.0,101.2)")
        conn.commit()
    finally:
        conn.close()


def test_reader_mapping_finds_candidate_tables() -> None:
    base = Path("localtests") / "phase1_reader" / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    db = base / "mini.db"
    _make_db(db)

    maps = discover_mappings(db)
    assert maps["trades"] is not None
    assert maps["book"] is not None
    assert maps["liquidations"] is not None

    r = SQLiteMicroReader(db)
    trades = r.read_trades("btcusdt", 0.0, 2.0)
    marks = r.read_top_of_book("BTCUSDT", 0.0, 2.0)
    liqs = r.read_liquidations("BTCUSDT", 0.0, 2.0)

    assert len(trades) == 1
    assert trades[0].symbol == "BTCUSDT"
    assert abs(trades[0].price - 101.0) < 1e-12
    assert trades[0].side == "buy"

    assert len(marks) == 1
    assert marks[0].symbol == "BTCUSDT"
    assert marks[0].bid_px > 0

    assert len(liqs) == 1
    assert liqs[0].symbol == "BTCUSDT"
