from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

try:
    from tools.ingestion_check import run_ingestion_check
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.ingestion_check import run_ingestion_check


def _mk_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT, price REAL)"
        )
        now_ms = int(time.time() * 1000)
        conn.execute("INSERT INTO agg_trades (ts_ms, symbol, price) VALUES (?, ?, ?)", (now_ms, "ETHUSDT", 1.0))
        conn.commit()
    finally:
        conn.close()


def test_ingestion_check_ok_delta_and_lag() -> None:
    db = Path("eclipse_scalper/localtests/ingestion_check") / f"{uuid.uuid4().hex}.db"
    _mk_db(db)

    def writer():
        time.sleep(0.2)
        conn = sqlite3.connect(str(db))
        try:
            now_ms = int(time.time() * 1000)
            conn.execute("INSERT INTO agg_trades (ts_ms, symbol, price) VALUES (?, ?, ?)", (now_ms, "ETHUSDT", 1.1))
            conn.commit()
        finally:
            conn.close()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    res = run_ingestion_check(db=db, symbols=["ETHUSDT"], window_sec=1, max_lag_sec=60)
    t.join(timeout=2)
    assert res.verdict == "OK"
    assert res.rows_delta > 0
    assert res.lag_sec is not None and res.lag_sec <= 60


def test_ingestion_check_degraded_when_stale() -> None:
    db = Path("eclipse_scalper/localtests/ingestion_check") / f"{uuid.uuid4().hex}.db"
    _mk_db(db)
    # do not write new rows; max_lag too strict
    res = run_ingestion_check(db=db, symbols=["ETHUSDT"], window_sec=1, max_lag_sec=0)
    assert res.verdict == "DEGRADED"
    assert res.reason in ("rows_delta_zero", "lag_exceeded", "lag_missing")

