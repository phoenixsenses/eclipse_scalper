from __future__ import annotations

import argparse
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from notifications.telegram import Notifier
from notifications.health_alerts import build_data_stale_event
from tools.check_data_ready import detect_ts_col, list_tables, normalize_ts_to_seconds, table_columns
from tools.health_state import write_component_health, write_overall_health, utc_now_iso


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("collection_watchdog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = RotatingFileHandler(str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
    return logger


def _pick_table(conn: sqlite3.Connection, requested: str = "") -> Optional[Tuple[str, str]]:
    req = str(requested or "").strip()
    if req:
        cols = table_columns(conn, req)
        ts_col = detect_ts_col(cols)
        return (req, ts_col) if ts_col else None
    tables = list_tables(conn)
    for t in ("mark_prices", "agg_trades", "liquidations"):
        if t in tables:
            cols = table_columns(conn, t)
            ts_col = detect_ts_col(cols)
            if ts_col:
                return t, ts_col
    for t in tables:
        cols = table_columns(conn, t)
        ts_col = detect_ts_col(cols)
        if ts_col:
            return t, ts_col
    return None


def _symbol_col(conn: sqlite3.Connection, table: str) -> Optional[str]:
    cols = table_columns(conn, table)
    lower = [c.lower() for c in cols]
    if "symbol" in lower:
        return cols[lower.index("symbol")]
    return None


def get_latest_timestamp(
    db_path: Path,
    symbols: List[str],
    table: str = "",
) -> Dict[str, Optional[float]]:
    if not db_path.exists():
        raise FileNotFoundError(str(db_path))
    conn = sqlite3.connect(str(db_path))
    try:
        picked = _pick_table(conn, requested=table)
        if not picked:
            raise RuntimeError("no timestamped table found")
        table_name, ts_col = picked
        sym_col = _symbol_col(conn, table_name)
        out: Dict[str, Optional[float]] = {"_table": table_name, "_ts_col": ts_col}
        if sym_col and symbols:
            for sym in symbols:
                row = conn.execute(
                    f"SELECT MAX({ts_col}) FROM {table_name} WHERE {sym_col}=?",
                    (sym,),
                ).fetchone()
                out[sym] = (normalize_ts_to_seconds(float(row[0])) if row and row[0] is not None else None)
        else:
            row = conn.execute(f"SELECT MAX({ts_col}) FROM {table_name}").fetchone()
            out["ALL"] = (normalize_ts_to_seconds(float(row[0])) if row and row[0] is not None else None)
        return out
    finally:
        conn.close()


async def _send_telegram(text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run alert] {text}")
        return
    token = os.getenv("ECLIPSE_TG_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("ECLIPSE_TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[alert skipped] missing telegram token/chat id env")
        return
    notifier = Notifier(token=token, chat_id=chat_id)
    await notifier.speak(text, priority="critical")


def _is_stale(latest_ts_sec: Optional[float], stale_threshold_sec: int, now_sec: Optional[float] = None) -> bool:
    if latest_ts_sec is None:
        return True
    now = float(time.time() if now_sec is None else now_sec)
    return (now - float(latest_ts_sec)) > float(stale_threshold_sec)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Microstructure collection watchdog.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--table", default="", help="Optional table override (default picks mark_prices first).")
    p.add_argument("--check-interval-sec", type=int, default=300)
    p.add_argument("--stale-threshold-sec", type=int, default=300)
    p.add_argument("--log", default="logs/watchdog.log")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


async def run_loop(args: argparse.Namespace) -> int:
    db_path = Path(str(args.db))
    symbols = _parse_symbols(args.symbols)
    logger = _setup_logger(Path(str(args.log)))
    start_msg = f"Eclipse Watchdog started, monitoring {db_path}"
    await _send_telegram(start_msg, bool(args.dry_run))
    print(start_msg)
    logger.info(start_msg)
    try:
        while True:
            await run_once(
                db_path=db_path,
                symbols=symbols,
                table=str(args.table),
                stale_threshold_sec=int(args.stale_threshold_sec),
                logger=logger,
                dry_run=bool(args.dry_run),
            )
            await asyncio.sleep(max(1, int(args.check_interval_sec)))
    except KeyboardInterrupt:
        logger.info("watchdog_stopped")
        print("watchdog_stopped")
        return 0


def main() -> int:
    args = _args()
    return asyncio.run(run_loop(args))


async def run_once(
    *,
    db_path: Path,
    symbols: List[str],
    table: str,
    stale_threshold_sec: int,
    logger: logging.Logger,
    dry_run: bool,
) -> Dict[str, object]:
    now = time.time()
    try:
        latest = get_latest_timestamp(db_path=db_path, symbols=symbols, table=str(table))
        table_name = str(latest.get("_table"))
        ts_col = str(latest.get("_ts_col"))
        stale_symbols: List[str] = []
        for sym in symbols if symbols else ["ALL"]:
            ts = latest.get(sym)
            if _is_stale(ts, int(stale_threshold_sec), now_sec=now):
                stale_symbols.append(sym)
        if stale_symbols:
            parts = []
            max_age = 0
            last_ts = None
            for sym in stale_symbols:
                ts = latest.get(sym)
                age = None if ts is None else (now - float(ts))
                if age is not None:
                    max_age = max(max_age, int(age))
                if ts is not None and (last_ts is None or float(ts) < float(last_ts)):
                    last_ts = float(ts)
                parts.append(f"{sym}:last_ts={ts} age_sec={None if age is None else int(age)}")
            msg = (
                f"COLLECTOR STALE table={table_name} ts_col={ts_col} "
                f"threshold_sec={int(stale_threshold_sec)} details={' | '.join(parts)}"
            )
            logger.warning(msg)
            write_component_health(
                "watchdog",
                {
                    "status": "degraded",
                    "connected": False,
                    "last_progress_ts_utc": (last_row_utc if 'last_row_utc' in locals() else None),
                    "progress_lag_sec": int(max_age),
                    "reconnects_last_5m": 0,
                    "errors_last_5m": 1,
                },
            )
            write_overall_health(
                {
                    "ts_utc": utc_now_iso(),
                    "mode": "paper",
                    "state": "degraded",
                    "reason": "watchdog_stale_data",
                    "components": {
                        "watchdog": {
                            "status": "degraded",
                            "connected": False,
                            "last_progress_ts_utc": (last_row_utc if 'last_row_utc' in locals() else None),
                            "progress_lag_sec": int(max_age),
                            "reconnects_last_5m": 0,
                            "errors_last_5m": 1,
                        }
                    },
                }
            )
            if callable(build_data_stale_event):
                last_row_utc = (
                    time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(last_ts)))
                    if last_ts is not None
                    else "unknown"
                )
                evt = build_data_stale_event(
                    stale_sec=int(max_age),
                    last_row_utc=last_row_utc,
                    symbols=",".join(stale_symbols),
                )
                await _send_telegram(f"{msg}\n{evt.render()}", bool(dry_run))
            else:
                await _send_telegram(msg, bool(dry_run))
            return {"ok": False, "stale_symbols": stale_symbols, "table": table_name}
        logger.info(f"OK table={table_name} ts_col={ts_col} symbols={','.join(symbols)}")
        write_component_health(
            "watchdog",
            {
                "status": "ok",
                "connected": True,
                "last_progress_ts_utc": utc_now_iso(),
                "progress_lag_sec": 0,
                "reconnects_last_5m": 0,
                "errors_last_5m": 0,
            },
        )
        return {"ok": True, "stale_symbols": [], "table": table_name}
    except Exception as exc:
        msg = f"watchdog_check_error err={type(exc).__name__}:{exc}"
        logger.error(msg)
        await _send_telegram(msg, bool(dry_run))
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    sys.exit(main())
