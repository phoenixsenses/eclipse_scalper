from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notifications.events import NotificationEvent, NotificationSeverity


def _safe_count_rows(db_path: str) -> int:
    try:
        p = Path(str(db_path))
        if not p.exists():
            return 0
        conn = sqlite3.connect(str(p))
        try:
            row = conn.execute("SELECT COUNT(*) FROM mark_prices").fetchone()
            return int((row or [0])[0] or 0)
        finally:
            conn.close()
    except Exception:
        return 0


def build_startup_event(*, symbols: str, horizon_sec: int, scratch_enabled: bool, watchdog_active: bool) -> NotificationEvent:
    body = (
        "ECLIPSE SCALPER STARTED - Paper Mode\n"
        f"Config: sell_UP + buy_UP | h={int(horizon_sec)}s | {symbols}\n"
        "Risk: max_pos=1 | daily_limit=50bps | drawdown=100bps\n"
        f"Scratch: {'ENABLED' if scratch_enabled else 'DISABLED'}\n"
        f"Watchdog: {'ACTIVE' if watchdog_active else 'DISABLED'}"
    )
    return NotificationEvent(
        NotificationSeverity.SUCCESS,
        "startup",
        "ECLIPSE STARTED",
        body,
        raw_payload={"symbols": symbols, "horizon_sec": int(horizon_sec)},
    )


def build_heartbeat_event(bot: Any, started_ts: float) -> NotificationEvent:
    now = time.time()
    up_sec = max(0.0, now - float(started_ts or now))
    trades = int(getattr(getattr(bot, "state", None), "total_trades", 0) or 0)
    wins = int(getattr(getattr(bot, "state", None), "total_wins", 0) or 0)
    win_rate = (wins / trades) if trades > 0 else 0.0
    pnl = float(getattr(getattr(bot, "state", None), "daily_pnl_bps", 0.0) or 0.0)
    reg = str(getattr(bot, "_notify_last_regime", "UNKNOWN") or "UNKNOWN")
    rret = float(getattr(bot, "_notify_last_rolling_return", 0.0) or 0.0)
    db_rows = _safe_count_rows(os.getenv("MICRO_DB_PATH", "data/microstructure.db"))
    body = (
        f"HEARTBEAT - {datetime.now(timezone.utc).strftime('%H:%M')} UTC\n"
        f"Status: Running | Uptime: {int(up_sec//3600)}h {int((up_sec%3600)//60)}m\n"
        f"Trades today: {trades} | PnL: {pnl:+.2f} bps | {win_rate*100.0:.1f}% win\n"
        f"Regime: {reg} | Return: {rret*100.0:+.3f}%\n"
        f"Data rows(mark_prices): {db_rows}"
    )
    return NotificationEvent(
        NotificationSeverity.INFO,
        "heartbeat",
        "HEARTBEAT",
        body,
        raw_payload={"trades": trades, "daily_pnl_bps": pnl},
        silent=True,
    )


def build_crash_event(error_text: str, uptime_sec: float) -> NotificationEvent:
    body = (
        "ECLIPSE SCALPER CRASHED\n"
        f"Error: {error_text}\n"
        f"Uptime was: {int(float(uptime_sec or 0.0)//3600)}h {int((float(uptime_sec or 0.0)%3600)//60)}m"
    )
    return NotificationEvent(
        NotificationSeverity.CRITICAL,
        "crash",
        "CRASH",
        body,
        raw_payload={"error_text": str(error_text)},
    )


def build_data_stale_event(*, stale_sec: int, last_row_utc: str, symbols: str) -> NotificationEvent:
    body = (
        f"DATA STALE: No new rows for {int(stale_sec)}s\n"
        f"Last row: {last_row_utc}\n"
        f"Symbols affected: {symbols}"
    )
    return NotificationEvent(
        NotificationSeverity.CRITICAL,
        "data_stale",
        "DATA STALE",
        body,
        raw_payload={"stale_sec": int(stale_sec), "symbols": symbols},
    )

