from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_date(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def _json(v: Any) -> str:
    return json.dumps(v if v is not None else {}, ensure_ascii=True, separators=(",", ":"))


@dataclass
class TradeLogger:
    db_path: str = "data/paper_trades.db"

    def __post_init__(self) -> None:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    entry_time REAL,
                    exit_time REAL,
                    side TEXT,
                    regime TEXT,
                    regime_age_sec REAL,
                    rolling_return REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_bps REAL,
                    exit_type TEXT,
                    exit_reason TEXT,
                    peak_favorable_bps REAL,
                    max_adverse_bps REAL,
                    elapsed_sec REAL,
                    pocket_filters TEXT,
                    scratch_config TEXT,
                    horizon_price REAL
                );
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    details TEXT,
                    risk_state TEXT
                );
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    scratches INTEGER,
                    total_pnl_bps REAL,
                    max_drawdown_bps REAL,
                    regime_up_pct REAL,
                    regime_down_pct REAL,
                    signals_blocked INTEGER,
                    avg_hold_sec REAL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        td = dict(trade_data or {})
        trade_id = str(td.get("trade_id") or "")
        if not trade_id:
            raise ValueError("trade_id is required")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO trades (
                    trade_id, entry_time, exit_time, side, regime, regime_age_sec, rolling_return,
                    entry_price, exit_price, pnl_bps, exit_type, exit_reason,
                    peak_favorable_bps, max_adverse_bps, elapsed_sec, pocket_filters, scratch_config, horizon_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    float(td.get("entry_time") or 0.0),
                    float(td.get("exit_time") or 0.0),
                    str(td.get("side") or ""),
                    str(td.get("regime") or ""),
                    float(td.get("regime_age_sec") or 0.0),
                    float(td.get("rolling_return") or 0.0),
                    float(td.get("entry_price") or 0.0),
                    float(td.get("exit_price") or 0.0),
                    float(td.get("pnl_bps") or 0.0),
                    str(td.get("exit_type") or ""),
                    str(td.get("exit_reason") or ""),
                    float(td.get("peak_favorable_bps") or 0.0),
                    float(td.get("max_adverse_bps") or 0.0),
                    float(td.get("elapsed_sec") or 0.0),
                    _json(td.get("pocket_filters")),
                    _json(td.get("scratch_config")),
                    float(td.get("horizon_price") or 0.0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def log_risk_event(self, event_data: Dict[str, Any]) -> None:
        ed = dict(event_data or {})
        event_id = str(ed.get("event_id") or "")
        if not event_id:
            raise ValueError("event_id is required")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO risk_events (
                    event_id, timestamp, event_type, details, risk_state
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    float(ed.get("timestamp") or 0.0),
                    str(ed.get("event_type") or ""),
                    _json(ed.get("details")),
                    _json(ed.get("risk_state")),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def update_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        conn = self._connect()
        try:
            if not date:
                row = conn.execute("SELECT MAX(exit_time) FROM trades").fetchone()
                ts = float(row[0] or 0.0)
                if ts <= 0:
                    return {"date": "", "total_trades": 0}
                date = _utc_date(ts)
            start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            end = start + 86400.0
            rows = conn.execute(
                """
                SELECT pnl_bps, regime, elapsed_sec, exit_reason
                FROM trades
                WHERE exit_time >= ? AND exit_time < ?
                """,
                (start, end),
            ).fetchall()
            total = len(rows)
            wins = sum(1 for r in rows if float(r[0] or 0.0) > 0.0)
            losses = sum(1 for r in rows if float(r[0] or 0.0) <= 0.0)
            scratches = sum(1 for r in rows if "scratch" in str(r[3] or "").lower())
            pnl_sum = sum(float(r[0] or 0.0) for r in rows)
            hold_avg = (sum(float(r[2] or 0.0) for r in rows) / total) if total else 0.0
            up_n = sum(1 for r in rows if str(r[1] or "").upper() == "UP")
            down_n = sum(1 for r in rows if str(r[1] or "").upper() == "DOWN")
            regime_up_pct = (up_n / total) if total else 0.0
            regime_down_pct = (down_n / total) if total else 0.0
            eq = 1.0
            peak = 1.0
            max_dd = 0.0
            for r in rows:
                eq *= (1.0 + float(r[0] or 0.0) / 10000.0)
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            max_dd_bps = max_dd * 10000.0
            signals_blocked = conn.execute(
                """
                SELECT COUNT(*) FROM risk_events
                WHERE timestamp >= ? AND timestamp < ? AND event_type='entry_blocked'
                """,
                (start, end),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_summary (
                    date, total_trades, wins, losses, scratches, total_pnl_bps, max_drawdown_bps,
                    regime_up_pct, regime_down_pct, signals_blocked, avg_hold_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date,
                    int(total),
                    int(wins),
                    int(losses),
                    int(scratches),
                    float(pnl_sum),
                    float(max_dd_bps),
                    float(regime_up_pct),
                    float(regime_down_pct),
                    int(signals_blocked or 0),
                    float(hold_avg),
                ),
            )
            conn.commit()
            return {
                "date": date,
                "total_trades": int(total),
                "wins": int(wins),
                "losses": int(losses),
                "scratches": int(scratches),
                "total_pnl_bps": float(pnl_sum),
                "max_drawdown_bps": float(max_dd_bps),
                "regime_up_pct": float(regime_up_pct),
                "regime_down_pct": float(regime_down_pct),
                "signals_blocked": int(signals_blocked or 0),
                "avg_hold_sec": float(hold_avg),
            }
        finally:
            conn.close()

