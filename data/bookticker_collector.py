"""Binance Futures bookTicker collector.

Writes top-of-book bid/ask snapshots to data/microstructure.db for paper
execution modeling. Public WebSocket only; no authenticated requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.health_state import write_component_health, utc_now_iso  # noqa: E402
from ami.storage.union_reader import resolve_writer_db_path  # noqa: E402

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise SystemExit("websockets is required: pip install websockets") from exc


LOG = logging.getLogger("bookticker_collector")
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_DB_PATH = REPO_ROOT / "data" / "microstructure.db"
WS_BASE = "wss://fstream.binance.com/stream?streams="
FLUSH_INTERVAL_SEC = 5.0
FLUSH_BATCH_SIZE = 5_000
MAX_PENDING_TICKS = 50_000
IDLE_RECONNECT_SEC = 30.0

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS book_ticker (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  bid_price REAL NOT NULL,
  bid_qty REAL NOT NULL,
  bid_depth_usd REAL,
  ask_price REAL NOT NULL,
  ask_qty REAL NOT NULL,
  mid_price REAL NOT NULL,
  spread_pct REAL NOT NULL,
  book_imbalance REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bt_symbol_ts ON book_ticker(symbol, ts_ms);
CREATE INDEX IF NOT EXISTS idx_bt_ts ON book_ticker(ts_ms);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA_SQL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(book_ticker)").fetchall()}
    if "bid_depth_usd" not in columns:
        conn.execute("ALTER TABLE book_ticker ADD COLUMN bid_depth_usd REAL")
    conn.commit()
    return conn


def _float(raw: Any) -> float:
    value = float(raw)
    if not (value == value) or value < 0:
        raise ValueError(f"invalid numeric value: {raw!r}")
    return value


def parse_bookticker(raw: str) -> tuple[int, str, float, float, float, float, float, float, float, float]:
    payload = json.loads(raw)
    data = payload.get("data", payload)
    symbol = str(data["s"]).upper()
    bid_price = _float(data["b"])
    bid_qty = _float(data["B"])
    ask_price = _float(data["a"])
    ask_qty = _float(data["A"])
    mid_price = (bid_price + ask_price) / 2.0
    if mid_price <= 0:
        raise ValueError("mid_price must be positive")
    qty_total = bid_qty + ask_qty
    spread_pct = (ask_price - bid_price) / mid_price
    book_imbalance = ((bid_qty - ask_qty) / qty_total) if qty_total > 0 else 0.0
    ts_ms = int(data.get("E") or data.get("T") or time.time() * 1000)
    return (
        ts_ms,
        symbol,
        bid_price,
        bid_qty,
        bid_price * bid_qty,
        ask_price,
        ask_qty,
        mid_price,
        spread_pct,
        book_imbalance,
    )


def flush_ticks(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, float, float, float, float, float, float, float, float]],
) -> None:
    if not rows:
        return
    for attempt in range(6):
        try:
            conn.executemany(
                """
                INSERT INTO book_ticker
                (ts_ms, symbol, bid_price, bid_qty, bid_depth_usd, ask_price, ask_qty, mid_price, spread_pct, book_imbalance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            rows.clear()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt >= 5:
                raise
            conn.rollback()
            time.sleep(0.25 * (attempt + 1))


def safe_flush_ticks(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, float, float, float, float, float, float, float, float]],
    *,
    context: str,
) -> bool:
    try:
        flush_ticks(conn, rows)
        return True
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        LOG.warning("bookTicker flush skipped due to database lock context=%s pending=%s", context, len(rows))
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def trim_pending(rows: list[tuple[int, str, float, float, float, float, float, float, float, float]]) -> None:
    if len(rows) <= MAX_PENDING_TICKS:
        return
    drop_count = len(rows) - MAX_PENDING_TICKS
    del rows[:drop_count]
    LOG.warning("bookTicker pending buffer trimmed dropped=%s retained=%s", drop_count, len(rows))


def _ts_ms_to_iso(ts_ms: int | None) -> str | None:
    if not ts_ms:
        return None
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def write_component_status(
    *,
    connected: bool,
    ticks: int,
    last_tick_ts_ms: int | None,
    last_progress_ts_utc: str | None,
    last_error: str = "",
) -> None:
    tick_event_age_sec = None
    if last_tick_ts_ms:
        tick_event_age_sec = max(0.0, time.time() - (float(last_tick_ts_ms) / 1000.0))
    payload = {
        "status": "ok" if connected else "degraded",
        "connected": bool(connected),
        "last_progress_ts_utc": last_progress_ts_utc,
        "backend": "websockets",
        "ticks_seen_today": int(ticks),
        "last_tick_ts_utc": _ts_ms_to_iso(last_tick_ts_ms),
        "last_tick_event_age_sec": tick_event_age_sec,
        "last_error": str(last_error or ""),
        "ts_utc": utc_now_iso(),
    }
    write_component_health("bookticker", payload)


def build_stream_url(symbols: list[str]) -> str:
    streams = "/".join(f"{sym.lower()}@bookTicker" for sym in symbols)
    return WS_BASE + streams


async def collect(db_path: Path, symbols: list[str], heartbeat_interval: float, max_seconds: float | None) -> None:
    conn = init_db(db_path)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop(*_: object) -> None:
        loop.call_soon_threadsafe(stop.set)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, request_stop)
        except ValueError:
            pass

    url = build_stream_url(symbols)
    start = time.monotonic()
    ticks = 0
    last_commit = time.monotonic()
    last_status = 0.0
    last_tick_ts_ms: int | None = None
    last_progress_ts_utc: str | None = None
    pending: list[tuple[int, str, float, float, float, float, float, float, float, float]] = []
    backoff = 1.0
    LOG.info("starting bookTicker collector symbols=%s db=%s", ",".join(symbols), db_path)
    for symbol in symbols:
        LOG.info("BookTicker: %s subscribed", symbol)

    try:
        while not stop.is_set():
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                break
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=4096) as ws:
                    LOG.info("connected %s", url)
                    backoff = 1.0
                    last_frame = time.monotonic()
                    write_component_status(
                        connected=True,
                        ticks=ticks,
                        last_tick_ts_ms=last_tick_ts_ms,
                        last_progress_ts_utc=last_progress_ts_utc,
                    )
                    while not stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            now = time.monotonic()
                            if now - last_status >= 5.0:
                                write_component_status(
                                    connected=True,
                                    ticks=ticks,
                                    last_tick_ts_ms=last_tick_ts_ms,
                                    last_progress_ts_utc=last_progress_ts_utc,
                                )
                                last_status = now
                            if now - last_frame >= IDLE_RECONNECT_SEC:
                                raise TimeoutError(f"bookTicker idle timeout no frames for {now - last_frame:.1f}s")
                            if max_seconds is not None and (now - start) >= max_seconds:
                                stop.set()
                            continue
                        row = parse_bookticker(str(raw))
                        if row[1] not in symbols:
                            continue
                        last_frame = time.monotonic()
                        pending.append(row)
                        ticks += 1
                        last_tick_ts_ms = int(row[0])
                        last_progress_ts_utc = utc_now_iso()
                        now = time.monotonic()
                        if len(pending) >= FLUSH_BATCH_SIZE or now - last_commit >= FLUSH_INTERVAL_SEC:
                            if safe_flush_ticks(conn, pending, context="periodic"):
                                last_commit = now
                            trim_pending(pending)
                        if now - last_status >= heartbeat_interval:
                            safe_flush_ticks(conn, pending, context="heartbeat")
                            trim_pending(pending)
                            write_component_status(
                                connected=True,
                                ticks=ticks,
                                last_tick_ts_ms=last_tick_ts_ms,
                                last_progress_ts_utc=last_progress_ts_utc,
                            )
                            last_status = now
            except Exception as exc:
                safe_flush_ticks(conn, pending, context="exception")
                trim_pending(pending)
                if stop.is_set():
                    break
                write_component_status(
                    connected=False,
                    ticks=ticks,
                    last_tick_ts_ms=last_tick_ts_ms,
                    last_progress_ts_utc=last_progress_ts_utc,
                    last_error=str(exc),
                )
                LOG.warning("bookTicker connection failed: %s; reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.7)
    finally:
        safe_flush_ticks(conn, pending, context="shutdown")
        write_component_status(
            connected=False,
            ticks=ticks,
            last_tick_ts_ms=last_tick_ts_ms,
            last_progress_ts_utc=last_progress_ts_utc,
            last_error="stopped",
        )
        conn.close()
        LOG.info("stopped bookTicker collector ticks=%s", ticks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Binance Futures bookTicker ticks.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    # The default below is the PRE-ROTATION path, which Phase-4 deleted. Opening it
    # would silently create an empty database and write a live feed nobody reads.
    db_path = resolve_writer_db_path(args.db_path)
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    asyncio.run(collect(db_path, symbols, float(args.heartbeat_interval), args.max_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
