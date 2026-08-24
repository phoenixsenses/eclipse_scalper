from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger("notifications.daily_summary")

from notifications.events import NotificationEvent, NotificationSeverity
from tools.collection_health import analyze_collection_health
from tools.paper_trade_summary import generate_summary


def compose_daily_summary(
    *,
    trades_db: str = "data/paper_trades.db",
    micro_db: str = "data/microstructure.db",
    day_index: int = 0,
    day_total: int = 60,
) -> NotificationEvent:
    s = generate_summary(trades_db, days=1)
    total = generate_summary(trades_db, days=0)
    uptime = 0.0
    gaps_today = 0
    db_rows = 0
    try:
        h = analyze_collection_health(Path(micro_db), symbols=[], gap_threshold_sec=60, alert_threshold_sec=300)
        gaps_today = int(h.get("gap_count", 0) or 0)
        uptime = float(h.get("uptime_pct", 0.0) or 0.0)
        conn = sqlite3.connect(str(micro_db))
        try:
            row = conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()
            db_rows = int((row or [0])[0] or 0)
        finally:
            conn.close()
    except Exception:
        pass
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    wins = int(round(float(s.get("win_rate", 0.0) or 0.0) * int(s.get("total_trades", 0) or 0)))
    losses = int(int(s.get("total_trades", 0) or 0) - wins)
    body = (
        f"DAILY SUMMARY - {d}\n\n"
        f"Trades: {int(s.get('total_trades', 0))} ({wins}W / {losses}L) - {float(s.get('win_rate', 0.0))*100.0:.1f}% win\n"
        f"PnL: {float(s.get('total_pnl_bps', 0.0)):+.2f} bps\n"
        f"Scratches: {float(s.get('scratch_rate', 0.0))*100.0:.1f}%\n"
        f"Max drawdown: -{float(s.get('max_drawdown_bps', 0.0)):.2f} bps\n\n"
        f"Regime: UP {float(s.get('regime_up_pct', 0.0))*100.0:.1f}% | DOWN {float(s.get('regime_down_pct', 0.0))*100.0:.1f}%\n\n"
        f"Cumulative (Day {int(day_index)}/{int(day_total)}):\n"
        f"Total PnL: {float(total.get('total_pnl_bps', 0.0)):+.2f} bps\n"
        f"Total trades: {int(total.get('total_trades', 0))}\n"
        f"Win rate: {float(total.get('win_rate', 0.0))*100.0:.1f}%\n"
        f"Peak drawdown: -{float(total.get('max_drawdown_bps', 0.0)):.2f} bps\n\n"
        f"Data health: {uptime:.2f}% uptime | {gaps_today} gaps today\n"
        f"DB rows: {db_rows}"
    )
    return NotificationEvent(NotificationSeverity.DAILY, "daily_summary", "DAILY SUMMARY", body)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compose/send daily summary notification.")
    p.add_argument("--db", default="data/paper_trades.db")
    p.add_argument("--micro-db", default="data/microstructure.db")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    evt = compose_daily_summary(trades_db=str(args.db), micro_db=str(args.micro_db))
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("ECLIPSE_TG_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("ECLIPSE_TG_CHAT_ID")
    if not token or not chat:
        _log.info("No Telegram credentials — summary:\n%s", evt.render())
        return 0
    from notifications.telegram import Notifier
    import asyncio

    async def _run() -> int:
        n = Notifier(token=token, chat_id=chat)
        await n.speak(evt.render(), priority="normal", silent=False)
        _log.info("daily_summary_sent")
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
