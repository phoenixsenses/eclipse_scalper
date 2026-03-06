from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

try:
    from core.trade_logger import TradeLogger
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.trade_logger import TradeLogger


def test_trade_logger_insert_and_daily_summary() -> None:
    db = Path("data") / f"test_trade_logger_{uuid.uuid4().hex}.db"
    try:
        lg = TradeLogger(db_path=str(db))
        lg.log_trade(
            {
                "trade_id": "t1",
                "entry_time": 1_700_000_000.0,
                "exit_time": 1_700_000_100.0,
                "side": "sell",
                "regime": "UP",
                "entry_price": 100.0,
                "exit_price": 99.9,
                "pnl_bps": 10.0,
                "exit_type": "HORIZON_EXIT",
                "exit_reason": "time",
                "elapsed_sec": 100.0,
            }
        )
        lg.log_trade(
            {
                "trade_id": "t2",
                "entry_time": 1_700_000_200.0,
                "exit_time": 1_700_000_260.0,
                "side": "sell",
                "regime": "DOWN",
                "entry_price": 100.0,
                "exit_price": 100.2,
                "pnl_bps": -20.0,
                "exit_type": "SCRATCH",
                "exit_reason": "scratch_max_adverse",
                "elapsed_sec": 60.0,
            }
        )
        lg.log_risk_event(
            {
                "event_id": "e1",
                "timestamp": 1_700_000_300.0,
                "event_type": "entry_blocked",
                "details": {"reason": "risk"},
                "risk_state": {"x": 1},
            }
        )
        s = lg.update_daily_summary("2023-11-14")
        assert int(s["total_trades"]) == 2
        assert int(s["wins"]) == 1
        assert int(s["losses"]) == 1
        assert int(s["scratches"]) == 1
        assert "max_drawdown_bps" in s
        conn = sqlite3.connect(str(db))
        try:
            jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        assert str(jm).lower() == "wal"
    finally:
        db.unlink(missing_ok=True)
