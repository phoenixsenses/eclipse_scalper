from __future__ import annotations

import sqlite3
import sys
import time
import uuid
from pathlib import Path

try:
    from tools import micro_diag
except ModuleNotFoundError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import micro_diag


def test_micro_diag_smoke(monkeypatch, capsys) -> None:
    db = Path("data") / f"test_micro_diag_{uuid.uuid4().hex}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE agg_trades(symbol TEXT, ts_ms INTEGER, price REAL, quantity REAL, is_buyer_maker INTEGER)")
        conn.execute("CREATE TABLE mark_prices(symbol TEXT, ts_ms INTEGER, mark_price REAL)")
        conn.execute("CREATE TABLE liquidations(symbol TEXT, ts_ms INTEGER, side TEXT, quantity REAL)")
        now_ms = int(time.time() * 1000)
        for i in range(20):
            ts = now_ms - ((20 - i) * 1000)
            conn.execute(
                "INSERT INTO agg_trades(symbol, ts_ms, price, quantity, is_buyer_maker) VALUES(?,?,?,?,?)",
                ("BTCUSDT", ts, 100.0 + i, 1.0, 0),
            )
            conn.execute(
                "INSERT INTO mark_prices(symbol, ts_ms, mark_price) VALUES(?,?,?)",
                ("BTCUSDT", ts, 100.0 + i),
            )
        conn.commit()
    finally:
        conn.close()
    try:
        monkeypatch.setattr(sys, "argv", ["micro_diag", "--db", str(db), "--symbol", "BTCUSDT", "--window-sec", "30"])
        code = micro_diag.main()
        out = capsys.readouterr().out
        assert code in (0, 1)
        assert '"symbol": "BTCUSDT"' in out
        assert '"reason":' in out
    finally:
        db.unlink(missing_ok=True)

