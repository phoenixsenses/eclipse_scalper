# data/microstructure_collector.py — Binance Futures Microstructure Data Collector
#
# Collects real-time websocket data from Binance Futures:
#   - Liquidation events (forceOrder)
#   - Aggregated trades (aggTrade) — taker buy/sell aggression
#   - Mark price + funding rate (markPrice@1s)
#
# Stores to SQLite with millisecond timestamps, batched writes.
#
# Usage:
#   python -m data.microstructure_collector --symbols BTCUSDT,ETHUSDT
#   python -m data.microstructure_collector --symbols BTCUSDT --db-path data/micro.db

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import ssl
import sys
import time
import traceback
import getpass
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    websockets = None

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


# ═══════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS liquidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    trade_time_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_liq_ts ON liquidations(ts_ms);
CREATE INDEX IF NOT EXISTS idx_liq_symbol_ts ON liquidations(symbol, ts_ms);

CREATE TABLE IF NOT EXISTS agg_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    is_buyer_maker INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trade_ts ON agg_trades(ts_ms);
CREATE INDEX IF NOT EXISTS idx_trade_symbol_ts ON agg_trades(symbol, ts_ms);

CREATE TABLE IF NOT EXISTS mark_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    mark_price REAL NOT NULL,
    funding_rate REAL,
    next_funding_time_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mark_ts ON mark_prices(ts_ms);
CREATE INDEX IF NOT EXISTS idx_mark_symbol_ts ON mark_prices(symbol, ts_ms);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════════════════════
# EVENT BUFFERS
# ═══════════════════════════════════════════════════════════════════════

class FlushFailureFatal(RuntimeError):
    """Raised when DB flush repeatedly fails and collector should exit."""


class EventBuffer:
    """Thread-safe buffer for batched SQLite writes."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        flush_interval: float = 5.0,
        max_buffer_size: int = 1000,
        flush_fail_threshold: int = 5,
        flush_retry_count: int = 3,
        flush_retry_sleep_sec: float = 0.25,
        on_flush_success=None,
        on_flush_error=None,
    ):
        self.conn = conn
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.flush_fail_threshold = max(1, int(flush_fail_threshold))
        self.flush_retry_count = max(1, int(flush_retry_count))
        self.flush_retry_sleep_sec = max(0.05, float(flush_retry_sleep_sec))
        self.on_flush_success = on_flush_success
        self.on_flush_error = on_flush_error

        self.liquidations: List[Tuple] = []
        self.agg_trades: List[Tuple] = []
        self.mark_prices: List[Tuple] = []

        self.last_flush = time.time()
        self.total_flushed = defaultdict(int)
        self.last_error = ""
        self.consecutive_flush_failures = 0

    def add_liquidation(self, ts_ms: int, symbol: str, side: str, price: float,
                        quantity: float, notional: float, trade_time_ms: int):
        self.liquidations.append((ts_ms, symbol, side, price, quantity, notional, trade_time_ms))
        self._maybe_flush()

    def add_agg_trade(self, ts_ms: int, symbol: str, price: float, quantity: float,
                      notional: float, is_buyer_maker: int):
        self.agg_trades.append((ts_ms, symbol, price, quantity, notional, is_buyer_maker))
        self._maybe_flush()

    def add_mark_price(self, ts_ms: int, symbol: str, mark_price: float,
                       funding_rate: Optional[float], next_funding_time_ms: Optional[int]):
        self.mark_prices.append((ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms))
        self._maybe_flush()

    def _maybe_flush(self):
        total = len(self.liquidations) + len(self.agg_trades) + len(self.mark_prices)
        elapsed = time.time() - self.last_flush
        if total >= self.max_buffer_size or elapsed >= self.flush_interval:
            self.flush()

    def flush(self):
        """Flush all buffered events to SQLite."""
        if not self.liquidations and not self.agg_trades and not self.mark_prices:
            return
        pending_liqs = list(self.liquidations)
        pending_trades = list(self.agg_trades)
        pending_marks = list(self.mark_prices)
        attempt_error: Optional[BaseException] = None
        for attempt in range(1, self.flush_retry_count + 1):
            try:
                if pending_liqs:
                    self.conn.executemany(
                        "INSERT INTO liquidations (ts_ms, symbol, side, price, quantity, notional, trade_time_ms) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        pending_liqs,
                    )
                if pending_trades:
                    self.conn.executemany(
                        "INSERT INTO agg_trades (ts_ms, symbol, price, quantity, notional, is_buyer_maker) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        pending_trades,
                    )
                if pending_marks:
                    self.conn.executemany(
                        "INSERT INTO mark_prices (ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms) "
                        "VALUES (?, ?, ?, ?, ?)",
                        pending_marks,
                    )
                self.conn.commit()
                self.total_flushed["liquidations"] += len(pending_liqs)
                self.total_flushed["agg_trades"] += len(pending_trades)
                self.total_flushed["mark_prices"] += len(pending_marks)
                self.liquidations.clear()
                self.agg_trades.clear()
                self.mark_prices.clear()
                self.last_flush = time.time()
                self.last_error = ""
                self.consecutive_flush_failures = 0
                if self.on_flush_success:
                    try:
                        self.on_flush_success(
                            {
                                "liquidations": len(pending_liqs),
                                "agg_trades": len(pending_trades),
                                "mark_prices": len(pending_marks),
                            }
                        )
                    except Exception:
                        pass
                return
            except Exception as e:
                attempt_error = e
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                locked = "database is locked" in str(e).lower()
                if locked and attempt < self.flush_retry_count:
                    time.sleep(self.flush_retry_sleep_sec * attempt)
                    continue
                break
        self.consecutive_flush_failures += 1
        self.last_error = f"{type(attempt_error).__name__}: {attempt_error}"
        print(
            f"  [ERROR] Flush failed (consecutive={self.consecutive_flush_failures}/{self.flush_fail_threshold}): "
            f"{self.last_error}",
            flush=True,
        )
        if attempt_error is not None:
            tb = "".join(
                traceback.format_exception(type(attempt_error), attempt_error, attempt_error.__traceback__)
            )
            print(tb, flush=True)
        if self.on_flush_error:
            try:
                self.on_flush_error(self.last_error)
            except Exception:
                pass
        if self.consecutive_flush_failures >= self.flush_fail_threshold:
            raise FlushFailureFatal(self.last_error)


# ═══════════════════════════════════════════════════════════════════════
# STREAM PARSER
# ═══════════════════════════════════════════════════════════════════════

class StreamParser:
    """Parse Binance futures websocket messages and route to buffer."""

    def __init__(self, buffer: EventBuffer):
        self.buffer = buffer
        self.counts = defaultdict(int)  # per-stream event counts since last stats
        self.counts_total = defaultdict(int)

    def parse(self, raw: str):
        """Parse a combined stream message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        stream = msg.get("stream", "")
        data = msg.get("data", {})

        if "@forceOrder" in stream:
            self._parse_liquidation(data)
        elif "@aggTrade" in stream:
            self._parse_agg_trade(data)
        elif "@markPrice" in stream:
            self._parse_mark_price(data)

    def _parse_liquidation(self, data: dict):
        """Parse forceOrder event.

        Example:
        {
            "e": "forceOrder",
            "E": 1568014460893,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "o": "LIMIT",
                "f": "IOC",
                "q": "0.014",
                "p": "9910",
                "ap": "9910",
                "X": "FILLED",
                "l": "0.014",
                "z": "0.014",
                "T": 1568014460893
            }
        }
        """
        order = data.get("o", {})
        symbol = order.get("s", "")
        side = order.get("S", "")
        price = float(order.get("p", 0))
        quantity = float(order.get("q", 0))
        event_time = int(data.get("E", 0))
        trade_time = int(order.get("T", event_time))

        if symbol and price > 0 and quantity > 0:
            notional = price * quantity
            self.buffer.add_liquidation(event_time, symbol, side, price, quantity, notional, trade_time)
            self.counts["liquidations"] += 1
            self.counts_total["liquidations"] += 1

    def _parse_agg_trade(self, data: dict):
        """Parse aggTrade event.

        Example:
        {
            "e": "aggTrade",
            "E": 1591261134288,
            "a": 11891,
            "s": "BTCUSDT",
            "p": "9645.01",
            "q": "0.156",
            "f": 25,
            "l": 25,
            "T": 1591261134199,
            "m": false
        }
        """
        symbol = data.get("s", "")
        price = float(data.get("p", 0))
        quantity = float(data.get("q", 0))
        trade_time = int(data.get("T", 0))
        is_buyer_maker = 1 if data.get("m", False) else 0

        if symbol and price > 0 and quantity > 0:
            notional = price * quantity
            self.buffer.add_agg_trade(trade_time, symbol, price, quantity, notional, is_buyer_maker)
            self.counts["agg_trades"] += 1
            self.counts_total["agg_trades"] += 1

    def _parse_mark_price(self, data: dict):
        """Parse markPrice event.

        Example:
        {
            "e": "markPriceUpdate",
            "E": 1562305380000,
            "s": "BTCUSDT",
            "p": "11185.87786614",
            "i": "11185.87786614",
            "P": "11185.87786614",
            "r": "0.00030000",
            "T": 1562306400000
        }
        """
        symbol = data.get("s", "")
        mark_price = float(data.get("p", 0))
        event_time = int(data.get("E", 0))
        funding_rate_str = data.get("r")
        next_funding_time = data.get("T")

        funding_rate = float(funding_rate_str) if funding_rate_str else None
        next_funding_ms = int(next_funding_time) if next_funding_time else None

        if symbol and mark_price > 0:
            self.buffer.add_mark_price(event_time, symbol, mark_price, funding_rate, next_funding_ms)
            self.counts["mark_prices"] += 1
            self.counts_total["mark_prices"] += 1

    def get_and_reset_counts(self) -> Dict[str, int]:
        """Return current counts and reset for next interval."""
        counts = dict(self.counts)
        self.counts.clear()
        return counts


# ═══════════════════════════════════════════════════════════════════════
# WEBSOCKET COLLECTOR
# ═══════════════════════════════════════════════════════════════════════

class MicrostructureCollector:
    """Async websocket collector for Binance Futures microstructure data."""

    BINANCE_WS = "wss://fstream.binance.com/stream"

    def __init__(
        self,
        symbols: List[str],
        db_path: str = "data/microstructure.db",
        flush_interval: float = 5.0,
        stats_interval: float = 60.0,
        flush_fail_threshold: int = 5,
        heartbeat_path: str = "logs/collector_heartbeat.json",
    ):
        self.symbols = [s.lower() for s in symbols]
        self.db_path = db_path
        self.flush_interval = flush_interval
        self.stats_interval = stats_interval

        self.conn: Optional[sqlite3.Connection] = None
        self.buffer: Optional[EventBuffer] = None
        self.parser: Optional[StreamParser] = None

        self._running = False
        self._start_time = 0.0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._backend = "websockets"
        self._ssl_context = ssl.create_default_context()
        self._connected = False
        self._last_message_ts_utc: Optional[str] = None
        self._last_flush_ts_utc: Optional[str] = None
        self._last_error = ""
        self._exit_code = 0
        self.flush_fail_threshold = max(1, int(flush_fail_threshold))
        self.heartbeat_path = Path(heartbeat_path)

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all symbols."""
        streams = []
        for sym in self.symbols:
            streams.append(f"{sym}@forceOrder")
            streams.append(f"{sym}@aggTrade")
            streams.append(f"{sym}@markPrice@1s")
        stream_str = "/".join(streams)
        return f"{self.BINANCE_WS}?streams={stream_str}"

    async def run(self) -> int:
        """Main run loop with auto-reconnect."""
        if not WS_AVAILABLE:
            print("ERROR: websockets library not available. Install with: pip install websockets")
            return 3

        self.conn = init_db(self.db_path)
        self.buffer = EventBuffer(
            self.conn,
            flush_interval=self.flush_interval,
            flush_fail_threshold=self.flush_fail_threshold,
            on_flush_success=self._on_flush_success,
            on_flush_error=self._on_flush_error,
        )
        self.parser = StreamParser(self.buffer)
        self._running = True
        self._start_time = time.time()

        symbols_upper = [s.upper() for s in self.symbols]
        print(f"\n{'=' * 80}")
        print(f"MICROSTRUCTURE DATA COLLECTOR")
        print(f"{'=' * 80}")
        print(f"  Symbols: {symbols_upper}")
        print(f"  Database: {self.db_path}")
        print(f"  Flush interval: {self.flush_interval}s")
        print(f"  Stats interval: {self.stats_interval}s")
        print(f"  Streams: forceOrder, aggTrade, markPrice@1s")
        print(f"{'=' * 80}")
        print(f"  Starting collection... Press Ctrl+C to stop.\n", flush=True)

        # Start stats printer task
        stats_task = asyncio.create_task(self._stats_loop())

        try:
            while self._running:
                try:
                    await self._connect_and_listen()
                except FlushFailureFatal as e:
                    self._last_error = f"fatal_flush:{e}"
                    print(f"  [FATAL] {self._last_error}", flush=True)
                    self._exit_code = 2
                    self._running = False
                    break
                except Exception as e:
                    if not self._running:
                        break
                    print(f"  [WARN] Connection error: {e}", flush=True)
                    self._last_error = f"connection_error:{type(e).__name__}: {e}"
                    self._log_connection_diagnostics(e, self._build_stream_url())
                    if self._is_winerror5(e) and self._backend == "websockets" and AIOHTTP_AVAILABLE:
                        self._backend = "aiohttp"
                        print("  [INFO] Switching websocket backend to aiohttp fallback after WinError 5", flush=True)
                    self._write_heartbeat()
                    print(f"  Reconnecting in {self._reconnect_delay:.0f}s...", flush=True)
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2, self._max_reconnect_delay
                    )
        finally:
            stats_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
            self._shutdown()
        return self._exit_code

    async def _connect_and_listen(self):
        """Connect to websocket and process messages."""
        url = self._build_stream_url()
        print(f"  Connecting to {url[:80]}...", flush=True)

        if self._backend == "aiohttp" and AIOHTTP_AVAILABLE:
            await self._connect_and_listen_aiohttp(url)
        else:
            await self._connect_and_listen_websockets(url)

    async def _connect_and_listen_websockets(
        self,
        url: str,
        max_seconds: Optional[int] = None,
        max_messages: Optional[int] = None,
        parse_messages: bool = True,
    ) -> int:
        start = time.time()
        msg_count = 0
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                ssl=self._ssl_context,
            ) as ws:
                self._connected = True
                self._reconnect_delay = 1.0  # reset on successful connect
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                peer = None
                try:
                    peer = ws.transport.get_extra_info("peername")
                except Exception:
                    peer = None
                print(f"  [{now}] Connected. Receiving data... peer={peer}", flush=True)
                async for message in ws:
                    msg_count += 1
                    self._last_message_ts_utc = datetime.now(timezone.utc).isoformat()
                    if parse_messages and self.parser:
                        self.parser.parse(message)
                    if max_messages is not None and msg_count >= max_messages:
                        break
                    if max_seconds is not None and (time.time() - start) >= max_seconds:
                        break
                    if not self._running and parse_messages:
                        break
        finally:
            self._connected = False
        return msg_count

    async def _connect_and_listen_aiohttp(
        self,
        url: str,
        max_seconds: Optional[int] = None,
        max_messages: Optional[int] = None,
        parse_messages: bool = True,
    ) -> int:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp backend requested but aiohttp is not available")
        start = time.time()
        msg_count = 0
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(url, heartbeat=20, ssl=self._ssl_context) as ws:
                    self._connected = True
                    self._reconnect_delay = 1.0
                    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    peer = None
                    try:
                        conn = getattr(ws, "_response", None)
                        if conn is not None and getattr(conn, "connection", None):
                            tr = conn.connection.transport
                            if tr is not None:
                                peer = tr.get_extra_info("peername")
                    except Exception:
                        peer = None
                    print(f"  [{now}] Connected (aiohttp). Receiving data... peer={peer}", flush=True)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            msg_count += 1
                            self._last_message_ts_utc = datetime.now(timezone.utc).isoformat()
                            if parse_messages and self.parser:
                                self.parser.parse(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            raise RuntimeError(f"aiohttp ws error: {ws.exception()}")
                        if max_messages is not None and msg_count >= max_messages:
                            break
                        if max_seconds is not None and (time.time() - start) >= max_seconds:
                            break
                        if not self._running and parse_messages:
                            break
        finally:
            self._connected = False
        return msg_count

    def _is_winerror5(self, exc: BaseException) -> bool:
        text = f"{exc!r} {exc}"
        if "WinError 5" in text:
            return True
        winerror = getattr(exc, "winerror", None)
        if winerror == 5:
            return True
        cause = getattr(exc, "__cause__", None)
        if cause is not None and cause is not exc:
            return self._is_winerror5(cause)
        return False

    def _log_connection_diagnostics(self, exc: BaseException, url: str):
        host = ""
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            host = ""
        resolved = []
        if host:
            try:
                infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                resolved = sorted({x[4][0] for x in infos if x and len(x) > 4 and x[4]})
            except Exception as dns_exc:
                resolved = [f"DNS_ERROR:{dns_exc}"]
        ssl_details = {
            "verify_mode": str(getattr(self._ssl_context, "verify_mode", "")),
            "check_hostname": bool(getattr(self._ssl_context, "check_hostname", False)),
            "minimum_version": str(getattr(self._ssl_context, "minimum_version", "")),
            "ciphers_count": len(self._ssl_context.get_ciphers() or []),
        }
        print("  [DIAG] connect_failure_context", flush=True)
        print(f"  [DIAG] exc_type={type(exc).__name__}", flush=True)
        print(f"  [DIAG] exc_errno={getattr(exc, 'errno', None)}", flush=True)
        print(f"  [DIAG] exc_repr={repr(exc)}", flush=True)
        print(f"  [DIAG] exc_str={str(exc)}", flush=True)
        print(f"  [DIAG] traceback=\n{traceback.format_exc()}", flush=True)
        print(f"  [DIAG] python_exe={sys.executable}", flush=True)
        print(f"  [DIAG] cwd={os.getcwd()}", flush=True)
        print(f"  [DIAG] user={getpass.getuser()}", flush=True)
        print(f"  [DIAG] url={url}", flush=True)
        print(f"  [DIAG] host={host}", flush=True)
        print(f"  [DIAG] resolved_ips={resolved}", flush=True)
        print(f"  [DIAG] ssl={ssl_details}", flush=True)

    async def run_connect_test(self, max_seconds: int = 15) -> int:
        """Test one websocket connection without DB writes."""
        if not WS_AVAILABLE:
            print("ERROR: websockets library not available. Install with: pip install websockets")
            return 3
        url = self._build_stream_url()
        print(f"[CONNECT_TEST] backend={self._backend} max_seconds={max_seconds} url={url}", flush=True)
        try:
            count = await self._connect_and_listen_websockets(
                url=url,
                max_seconds=max_seconds,
                max_messages=100,
                parse_messages=False,
            )
            print(f"[CONNECT_TEST] PASS backend=websockets messages={count}", flush=True)
            return 0
        except Exception as e:
            self._log_connection_diagnostics(e, url)
            if self._is_winerror5(e) and AIOHTTP_AVAILABLE:
                try:
                    count = await self._connect_and_listen_aiohttp(
                        url=url,
                        max_seconds=max_seconds,
                        max_messages=100,
                        parse_messages=False,
                    )
                    print(f"[CONNECT_TEST] PASS backend=aiohttp messages={count}", flush=True)
                    return 0
                except Exception as e2:
                    self._log_connection_diagnostics(e2, url)
                    print("[CONNECT_TEST] FAIL connect on websockets and aiohttp fallback", flush=True)
                    return 2
            print("[CONNECT_TEST] FAIL connect", flush=True)
            return 2
        except BaseException:
            print("[CONNECT_TEST] UNEXPECTED_ERROR", flush=True)
            print(traceback.format_exc(), flush=True)
            return 3

    async def _stats_loop(self):
        """Print collection statistics periodically."""
        try:
            while self._running:
                await asyncio.sleep(self.stats_interval)
                if not self._running:
                    break
                self._print_stats()
                self._write_heartbeat()
        except asyncio.CancelledError:
            pass

    def _print_stats(self):
        """Print current collection statistics."""
        elapsed = time.time() - self._start_time
        hours = elapsed / 3600
        minutes = elapsed / 60

        counts = self.parser.get_and_reset_counts()
        interval = self.stats_interval

        # Events per second in this interval
        rates = {k: v / interval for k, v in counts.items()}

        # Total counts
        totals = self.parser.counts_total

        # DB file size
        try:
            db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        except OSError:
            db_size_mb = 0

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{now}] Uptime: {hours:.1f}h | "
              f"DB: {db_size_mb:.1f}MB | "
              f"Trades: {rates.get('agg_trades', 0):.0f}/s ({totals.get('agg_trades', 0):,} total) | "
              f"Mark: {rates.get('mark_prices', 0):.0f}/s ({totals.get('mark_prices', 0):,} total) | "
              f"Liqs: {rates.get('liquidations', 0):.1f}/s ({totals.get('liquidations', 0):,} total)",
              flush=True)

    def _shutdown(self):
        """Flush remaining data and close database."""
        print(f"\n  Shutting down...", flush=True)
        if self.buffer:
            try:
                self.buffer.flush()
            except Exception as e:
                self._last_error = f"shutdown_flush_error:{e}"
            totals = self.buffer.total_flushed
            print(f"  Final flush complete.")
            print(f"  Total written: "
                  f"trades={totals.get('agg_trades', 0):,}, "
                  f"mark_prices={totals.get('mark_prices', 0):,}, "
                  f"liquidations={totals.get('liquidations', 0):,}")
        if self.conn:
            self.conn.close()
            print(f"  Database closed: {self.db_path}")
        self._write_heartbeat()

    def stop(self):
        """Signal the collector to stop."""
        self._running = False

    def _on_flush_success(self, writes: Dict[str, int]):
        self._last_flush_ts_utc = datetime.now(timezone.utc).isoformat()
        _ = writes

    def _on_flush_error(self, message: str):
        self._last_error = message
        self._write_heartbeat()

    def _heartbeat_payload(self) -> Dict[str, object]:
        rows = {}
        if self.buffer:
            rows = {
                "agg_trades": int(self.buffer.total_flushed.get("agg_trades", 0)),
                "mark_prices": int(self.buffer.total_flushed.get("mark_prices", 0)),
                "liquidations": int(self.buffer.total_flushed.get("liquidations", 0)),
            }
        return {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "connected": bool(self._connected),
            "last_message_ts_utc": self._last_message_ts_utc,
            "last_flush_ts_utc": self._last_flush_ts_utc,
            "rows_written_since_start": rows,
            "last_error": self._last_error,
            "current_backoff_seconds": float(self._reconnect_delay),
            "backend": self._backend,
            "db_path": self.db_path,
        }

    def _write_heartbeat(self):
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._heartbeat_payload()
            self.heartbeat_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Binance Futures Microstructure Data Collector")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols (default: BTCUSDT,ETHUSDT)")
    parser.add_argument("--db-path", type=str, default="data/microstructure.db",
                        help="SQLite database path (default: data/microstructure.db)")
    parser.add_argument("--flush-interval", type=float, default=5.0,
                        help="Seconds between DB flushes (default: 5)")
    parser.add_argument("--stats-interval", type=float, default=60.0,
                        help="Seconds between stats printout (default: 60)")
    parser.add_argument("--connect-test", action="store_true",
                        help="Run a one-shot websocket connect test (no DB writes).")
    parser.add_argument("--max-seconds", type=int, default=15,
                        help="Max seconds for connect test (default: 15).")
    parser.add_argument("--flush-fail-threshold", type=int, default=5,
                        help="Exit after N consecutive flush failures (default: 5).")
    parser.add_argument("--heartbeat-path", type=str, default="logs/collector_heartbeat.json",
                        help="Heartbeat JSON path (default: logs/collector_heartbeat.json).")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("ERROR: No symbols specified")
        return

    collector = MicrostructureCollector(
        symbols=symbols,
        db_path=args.db_path,
        flush_interval=args.flush_interval,
        stats_interval=args.stats_interval,
        flush_fail_threshold=int(args.flush_fail_threshold),
        heartbeat_path=args.heartbeat_path,
    )

    if args.connect_test:
        code = asyncio.run(collector.run_connect_test(max_seconds=max(1, int(args.max_seconds))))
        raise SystemExit(code)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n  Received stop signal...", flush=True)
        collector.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        code = asyncio.run(collector.run())
        raise SystemExit(code)
    except KeyboardInterrupt:
        pass

    print("  Collector stopped.")


if __name__ == "__main__":
    main()
