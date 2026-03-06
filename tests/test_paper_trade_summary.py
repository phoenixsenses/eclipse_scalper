from __future__ import annotations

import json
import uuid
from pathlib import Path

try:
    from core.trade_logger import TradeLogger
    from tools.paper_trade_summary import generate_summary, main as summary_main
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.trade_logger import TradeLogger
    from tools.paper_trade_summary import generate_summary, main as summary_main


def test_generate_summary_and_main(monkeypatch) -> None:
    base = f"test_paper_trade_summary_{uuid.uuid4().hex}"
    db = Path("data") / f"{base}.db"
    out_md = Path("data") / f"{base}.md"
    out_json = Path("data") / f"{base}.json"
    try:
        lg = TradeLogger(db_path=str(db))
        lg.log_trade(
            {
                "trade_id": "a",
                "entry_time": 1_700_000_000.0,
                "exit_time": 1_700_000_100.0,
                "side": "buy",
                "regime": "UP",
                "entry_price": 100.0,
                "exit_price": 100.1,
                "pnl_bps": 10.0,
                "exit_type": "TAKE_PROFIT",
                "exit_reason": "take_profit",
                "elapsed_sec": 100.0,
            }
        )
        lg.log_trade(
            {
                "trade_id": "b",
                "entry_time": 1_700_000_200.0,
                "exit_time": 1_700_000_300.0,
                "side": "sell",
                "regime": "DOWN",
                "entry_price": 100.0,
                "exit_price": 100.2,
                "pnl_bps": -20.0,
                "exit_type": "SCRATCH",
                "exit_reason": "scratch_max_adverse",
                "elapsed_sec": 100.0,
            }
        )
        s = generate_summary(str(db))
        assert int(s["total_trades"]) == 2
        assert "daily" in s and isinstance(s["daily"], list)
        monkeypatch.setattr(
            "sys.argv",
            [
                "paper_trade_summary",
                "--db",
                str(db),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
            ],
        )
        rc = summary_main()
        assert rc == 0
        assert out_md.exists()
        assert out_json.exists()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert int(payload["total_trades"]) == 2
        assert payload["run_summary"]["run_type"] == "paper_trade_summary"
    finally:
        db.unlink(missing_ok=True)
        out_md.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)


def test_generate_summary_empty_db() -> None:
    db = Path("data") / f"test_paper_trade_summary_empty_{uuid.uuid4().hex}.db"
    try:
        TradeLogger(db_path=str(db))  # schema only, no trades
        s = generate_summary(str(db))
        assert int(s["total_trades"]) == 0
        assert float(s["win_rate"]) == 0.0
    finally:
        db.unlink(missing_ok=True)
