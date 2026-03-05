from __future__ import annotations

import uuid
from pathlib import Path

try:
    from core.trade_logger import TradeLogger
    from monitoring.status_snapshot import collect_last_decisions, collect_pnl, render_status_text
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.trade_logger import TradeLogger
    from monitoring.status_snapshot import collect_last_decisions, collect_pnl, render_status_text


def test_collect_pnl_and_render() -> None:
    db = Path("data") / f"test_status_snapshot_{uuid.uuid4().hex}.db"
    try:
        lg = TradeLogger(str(db))
        lg.log_trade(
            {
                "trade_id": "ss1",
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
        p = collect_pnl(str(db))
        assert p["ok"] is True
        assert int(p["total_trades"]) == 1
        txt = render_status_text()
        assert "Eclipse Scalper - Paper Run" in txt
    finally:
        db.unlink(missing_ok=True)


def test_collect_last_decisions() -> None:
    jp = Path("logs") / f"test_status_snapshot_{uuid.uuid4().hex}.jsonl"
    jp.parent.mkdir(parents=True, exist_ok=True)
    try:
        jp.write_text(
            "\n".join(
                [
                    '{"ts":"2026-01-01T00:00:00Z","event":"entry.blocked","symbol":"ETHUSDT","data":{"reason":"risk"}}',
                    '{"ts":"2026-01-01T00:00:01Z","event":"entry.submitted","symbol":"ETHUSDT","data":{"reason":"ok"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows = collect_last_decisions(str(jp), limit=2)
        assert len(rows) == 2
        assert rows[0]["event"] in ("entry.submitted", "entry.blocked")
    finally:
        jp.unlink(missing_ok=True)
