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
from ami.storage.union_reader import resolve_writer_db_path
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
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from tools.health_state import write_component_health

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

        if "forceOrder" in stream:
            self._parse_liquidation(data)
        elif "@aggTrade" in stream:
            self._parse_agg_trade(data)
        elif "@markPrice" in stream:
            self._parse_mark_price(data)

    def ingest_rest_agg_trade(self, data: dict) -> bool:
        """Ingest one public /fapi/v1/aggTrades row."""
        before = int(self.counts_total.get("agg_trades", 0))
        self._parse_agg_trade(data)
        return int(self.counts_total.get("agg_trades", 0)) > before

    def ingest_rest_mark_price(self, data: dict) -> bool:
        """Ingest one public /fapi/v1/premiumIndex row as a mark price update."""
        symbol = data.get("symbol", "")
        mark_price = float(data.get("markPrice", 0) or 0)
        event_time = int(data.get("time", 0) or int(time.time() * 1000))
        funding_rate_raw = data.get("lastFundingRate")
        next_funding_raw = data.get("nextFundingTime")
        funding_rate = float(funding_rate_raw) if funding_rate_raw not in (None, "") else None
        next_funding_ms = int(next_funding_raw) if next_funding_raw not in (None, "") else None
        if symbol and mark_price > 0:
            self.buffer.add_mark_price(event_time, symbol, mark_price, funding_rate, next_funding_ms)
            self.counts["mark_prices"] += 1
            self.counts_total["mark_prices"] += 1
            return True
        return False

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

    BINANCE_WS = "wss://fstream.binance.com/market/stream"
    BINANCE_REST = "https://fapi.binance.com"

    def __init__(
        self,
        symbols: List[str],
        db_path: str = "data/microstructure.db",
        flush_interval: float = 5.0,
        stats_interval: float = 60.0,
        flush_fail_threshold: int = 5,
        checkpoint_interval_sec: float = 300.0,
        wal_alert_mb: float = 2048.0,
        heartbeat_path: str = "logs/collector_heartbeat.json",
        stall_timeout_sec: float = 45.0,
        simulate_connection: bool = False,
        simulate_cycle_sec: float = 20.0,
        simulate_down_sec: float = 6.0,
        simulate_max_seconds: float = 0.0,
        rest_fallback_enabled: bool = True,
        rest_poll_interval_sec: float = 5.0,
        liquidation_stream_mode: str = "all_market_arr",
        heartbeat_write_interval_sec: float = 15.0,
        health_root: str = "logs/health",
    ):
        self.symbols = [s.lower() for s in symbols]
        self.db_path = db_path
        self.flush_interval = flush_interval
        self.stats_interval = stats_interval
        self.heartbeat_write_interval_sec = max(1.0, float(heartbeat_write_interval_sec))
        # Overridable so isolated tests (tools/health_cycle_smoke.py) can
        # point a spawned collector at a temp root instead of the real
        # logs/health/ -- otherwise a test subprocess would race the actual
        # live collector for logs/health/collector.json. Production callers
        # never pass this; it defaults to the real path.
        self.health_root = str(health_root)

        self.conn: Optional[sqlite3.Connection] = None
        self.buffer: Optional[EventBuffer] = None
        self.parser: Optional[StreamParser] = None

        self._running = False
        self._start_time = 0.0
        self._reconnect_delay = 1.0
        self._reconnect_base = 1.0
        self._reconnect_factor = 2.0
        self._max_reconnect_delay = 60.0
        self._reconnect_jitter_pct = 0.20
        self._stable_reset_sec = 30.0
        self._connected_since_ts = 0.0
        self._backend = "websockets"
        self._ssl_context = ssl.create_default_context()
        self._connected = False
        self._last_message_ts_utc: Optional[str] = None
        self._last_flush_ts_utc: Optional[str] = None
        self._last_error = ""
        self._exit_code = 0
        self.flush_fail_threshold = max(1, int(flush_fail_threshold))
        self.checkpoint_interval_sec = max(30.0, float(checkpoint_interval_sec))
        self.wal_alert_mb = max(0.0, float(wal_alert_mb))
        self.heartbeat_path = Path(heartbeat_path)
        self.stall_timeout_sec = max(10.0, float(stall_timeout_sec))
        self._reconnect_attempt = 0
        self._recent_reconnect_ts: List[float] = []
        self._reconnect_alert_threshold_5m = max(
            1,
            int(float(os.getenv("COLLECTOR_RECONNECT_ALERT_THRESHOLD_5M", "20") or 20)),
        )
        self._last_checkpoint_ts = 0.0
        self._last_checkpoint_ts_utc: Optional[str] = None
        self.simulate_connection = bool(simulate_connection)
        self.simulate_cycle_sec = max(6.0, float(simulate_cycle_sec))
        self.simulate_down_sec = min(max(1.0, float(simulate_down_sec)), self.simulate_cycle_sec - 1.0)
        self.simulate_max_seconds = max(0.0, float(simulate_max_seconds))
        self.rest_fallback_enabled = bool(rest_fallback_enabled)
        self.rest_poll_interval_sec = max(2.0, float(rest_poll_interval_sec))
        self._rest_fallback_active = False
        self._rest_last_progress_ts_utc: Optional[str] = None
        self._rest_last_error = ""
        self._rest_agg_last_id: Dict[str, int] = {}
        self._last_data_progress_ts_utc: Optional[str] = None
        mode = str(liquidation_stream_mode or "all_market_arr").strip().lower()
        if mode not in {"all_market_arr", "per_symbol"}:
            mode = "all_market_arr"
        self.liquidation_stream_mode = mode

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all symbols."""
        streams = []
        if self.liquidation_stream_mode == "all_market_arr":
            streams.append("!forceOrder@arr")
        for sym in self.symbols:
            if self.liquidation_stream_mode == "per_symbol":
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
        print(f"  Checkpoint interval: {self.checkpoint_interval_sec}s")
        print(f"  Stats interval: {self.stats_interval}s")
        print(f"  Streams: forceOrder({self.liquidation_stream_mode}), aggTrade, markPrice@1s")
        print(f"  REST fallback: {'enabled' if self.rest_fallback_enabled else 'disabled'} interval={self.rest_poll_interval_sec:.1f}s")
        print(f"{'=' * 80}")
        print(f"  Starting collection... Press Ctrl+C to stop.\n", flush=True)

        # Start stats printer task
        stats_task = asyncio.create_task(self._stats_loop())
        rest_task = None
        if self.rest_fallback_enabled and not self.simulate_connection:
            rest_task = asyncio.create_task(self._rest_fallback_loop())

        try:
            if self.simulate_connection:
                await self._run_simulated_connection()
            else:
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
                        self._reconnect_attempt += 1
                        self._recent_reconnect_ts.append(time.time())
                        self._recent_reconnect_ts = [x for x in self._recent_reconnect_ts if (time.time() - x) <= 300.0]
                        reconnects_5m = len(self._recent_reconnect_ts)
                        self._log_connection_diagnostics(e, self._build_stream_url())
                        if self._is_winerror5(e) and self._backend == "websockets" and AIOHTTP_AVAILABLE:
                            self._backend = "aiohttp"
                            print("  [INFO] Switching websocket backend to aiohttp fallback after WinError 5", flush=True)
                        if reconnects_5m >= self._reconnect_alert_threshold_5m:
                            print(
                                f"  [ALERT] reconnect storm: reconnects_last_5m={reconnects_5m} "
                                f"threshold={self._reconnect_alert_threshold_5m}",
                                flush=True,
                            )
                        self._write_heartbeat()
                        sleep_s = self._next_reconnect_sleep()
                        print(
                            f"  [RECONNECT] attempt={self._reconnect_attempt} next_sleep_sec={sleep_s:.2f} backend={self._backend}",
                            flush=True,
                        )
                        await asyncio.sleep(sleep_s)
        finally:
            stats_task.cancel()
            if rest_task:
                rest_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
            if rest_task:
                try:
                    await rest_task
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
                self._connected_since_ts = time.time()
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                peer = None
                try:
                    peer = ws.transport.get_extra_info("peername")
                except Exception:
                    peer = None
                print(f"  [{now}] Connected. Receiving data... peer={peer}", flush=True)
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=self.stall_timeout_sec)
                    except asyncio.TimeoutError:
                        raise RuntimeError(f"stall_timeout_no_messages>{self.stall_timeout_sec:.0f}s")
                    msg_count += 1
                    self._last_message_ts_utc = datetime.now(timezone.utc).isoformat()
                    self._last_data_progress_ts_utc = self._last_message_ts_utc
                    if parse_messages and self.parser:
                        self.parser.parse(message)
                    if max_messages is not None and msg_count >= max_messages:
                        break
                    if max_seconds is not None and (time.time() - start) >= max_seconds:
                        break
                    if not self._running and parse_messages:
                        break
                if (time.time() - self._connected_since_ts) >= self._stable_reset_sec:
                    self._reconnect_delay = self._reconnect_base
                    self._reconnect_attempt = 0
        finally:
            self._connected = False
        return msg_count

    async def _run_simulated_connection(self):
        print(
            f"  [SIM] simulate_connection enabled cycle_sec={self.simulate_cycle_sec:.1f} "
            f"down_sec={self.simulate_down_sec:.1f} max_seconds={self.simulate_max_seconds:.1f}",
            flush=True,
        )
        start = time.time()
        prev_connected = None
        up_sec = max(1.0, self.simulate_cycle_sec - self.simulate_down_sec)
        while self._running:
            elapsed = time.time() - start
            if self.simulate_max_seconds > 0 and elapsed >= self.simulate_max_seconds:
                break
            phase = elapsed % self.simulate_cycle_sec
            connected_now = phase < up_sec
            if prev_connected is None or connected_now != prev_connected:
                if connected_now:
                    self._connected_since_ts = time.time()
                    self._last_error = ""
                    print(f"  [SIM] state=connected elapsed={elapsed:.1f}s", flush=True)
                else:
                    self._reconnect_attempt += 1
                    self._recent_reconnect_ts.append(time.time())
                    self._recent_reconnect_ts = [x for x in self._recent_reconnect_ts if (time.time() - x) <= 300.0]
                    reconnects_5m = len(self._recent_reconnect_ts)
                    self._last_error = "simulated_disconnect"
                    print(f"  [SIM] state=disconnected elapsed={elapsed:.1f}s", flush=True)
                    if reconnects_5m >= self._reconnect_alert_threshold_5m:
                        print(
                            f"  [ALERT] reconnect storm: reconnects_last_5m={reconnects_5m} "
                            f"threshold={self._reconnect_alert_threshold_5m}",
                            flush=True,
                        )
            self._connected = connected_now
            if connected_now:
                self._last_message_ts_utc = datetime.now(timezone.utc).isoformat()
                self._last_data_progress_ts_utc = self._last_message_ts_utc
                if (time.time() - self._connected_since_ts) >= self._stable_reset_sec:
                    self._reconnect_delay = self._reconnect_base
                    self._reconnect_attempt = 0
            self._write_heartbeat()
            prev_connected = connected_now
            await asyncio.sleep(1.0)
        self._connected = False

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
                    self._connected_since_ts = time.time()
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
                    while True:
                        msg = await ws.receive(timeout=self.stall_timeout_sec)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            msg_count += 1
                            self._last_message_ts_utc = datetime.now(timezone.utc).isoformat()
                            self._last_data_progress_ts_utc = self._last_message_ts_utc
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
                    if (time.time() - self._connected_since_ts) >= self._stable_reset_sec:
                        self._reconnect_delay = self._reconnect_base
                        self._reconnect_attempt = 0
        finally:
            self._connected = False
        return msg_count

    def _ws_progress_recent(self) -> bool:
        if not self._last_message_ts_utc:
            return False
        try:
            last = datetime.fromisoformat(self._last_message_ts_utc).timestamp()
        except Exception:
            return False
        return (time.time() - last) <= self.stall_timeout_sec

    async def _rest_fallback_loop(self) -> None:
        if not AIOHTTP_AVAILABLE:
            self._rest_last_error = "aiohttp_unavailable"
            return
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=10, sock_read=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self._running:
                try:
                    if self._ws_progress_recent():
                        self._rest_fallback_active = False
                        await asyncio.sleep(self.rest_poll_interval_sec)
                        continue
                    self._rest_fallback_active = True
                    inserted = 0
                    for sym in self.symbols:
                        inserted += await self._poll_rest_agg_trades(session, sym.upper())
                        inserted += await self._poll_rest_mark_price(session, sym.upper())
                    if inserted > 0:
                        now = datetime.now(timezone.utc).isoformat()
                        self._rest_last_progress_ts_utc = now
                        self._last_data_progress_ts_utc = now
                        self._rest_last_error = ""
                        if self.buffer:
                            self.buffer.flush()
                    self._write_heartbeat()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._rest_last_error = f"{type(exc).__name__}: {exc}"
                    self._last_error = f"rest_fallback_error:{self._rest_last_error}"
                    self._write_heartbeat()
                await asyncio.sleep(self.rest_poll_interval_sec)

    async def _poll_rest_agg_trades(self, session, symbol: str) -> int:
        if not self.parser:
            return 0
        params = {"symbol": symbol, "limit": "100"}
        last_id = self._rest_agg_last_id.get(symbol)
        if last_id is not None:
            params["fromId"] = str(last_id + 1)
        url = f"{self.BINANCE_REST}/fapi/v1/aggTrades"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"aggTrades {symbol} status={resp.status} body={text[:160]}")
            rows = await resp.json()
        inserted = 0
        max_id = last_id
        for row in rows if isinstance(rows, list) else []:
            try:
                agg_id = int(row.get("a"))
            except Exception:
                continue
            if last_id is not None and agg_id <= last_id:
                continue
            payload = {
                "s": symbol,
                "p": row.get("p"),
                "q": row.get("q"),
                "T": row.get("T"),
                "m": row.get("m", False),
            }
            if self.parser.ingest_rest_agg_trade(payload):
                inserted += 1
            max_id = agg_id if max_id is None else max(max_id, agg_id)
        if max_id is not None:
            self._rest_agg_last_id[symbol] = int(max_id)
        return inserted

    async def _poll_rest_mark_price(self, session, symbol: str) -> int:
        if not self.parser:
            return 0
        url = f"{self.BINANCE_REST}/fapi/v1/premiumIndex"
        async with session.get(url, params={"symbol": symbol}) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"premiumIndex {symbol} status={resp.status} body={text[:160]}")
            row = await resp.json()
        return 1 if self.parser.ingest_rest_mark_price(row if isinstance(row, dict) else {}) else 0

    def _next_reconnect_sleep(self) -> float:
        base = max(self._reconnect_base, float(self._reconnect_delay))
        j = max(0.0, min(0.5, float(self._reconnect_jitter_pct)))
        factor = random.uniform(1.0 - j, 1.0 + j)
        sleep_s = max(0.1, base * factor)
        self._reconnect_delay = min(self._max_reconnect_delay, max(self._reconnect_base, base * self._reconnect_factor))
        return sleep_s

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
        """Print collection statistics on `stats_interval`, but refresh the
        heartbeat file on the much shorter `heartbeat_write_interval_sec`
        cadence. Decoupled deliberately: external monitoring reads
        last_message_ts_utc from the heartbeat file, and during genuinely
        healthy operation that file must not go stale for as long as the
        (5-minute-by-default) stats/console cadence -- see
        reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md
        ("Monitoring follow-up"). This does not increase console/log volume:
        _print_stats()/_maybe_checkpoint() still only run every stats_interval.
        """
        try:
            elapsed_since_stats = 0.0
            tick = min(self.stats_interval, self.heartbeat_write_interval_sec)
            while self._running:
                await asyncio.sleep(tick)
                if not self._running:
                    break
                self._write_heartbeat()
                elapsed_since_stats += tick
                if elapsed_since_stats >= self.stats_interval:
                    self._print_stats()
                    self._maybe_checkpoint()
                    elapsed_since_stats = 0.0
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
        wal_size_mb = self._wal_size_mb()

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{now}] Uptime: {hours:.1f}h | "
              f"DB: {db_size_mb:.1f}MB | "
              f"WAL: {wal_size_mb:.1f}MB | "
              f"Trades: {rates.get('agg_trades', 0):.0f}/s ({totals.get('agg_trades', 0):,} total) | "
              f"Mark: {rates.get('mark_prices', 0):.0f}/s ({totals.get('mark_prices', 0):,} total) | "
              f"Liqs: {rates.get('liquidations', 0):.1f}/s ({totals.get('liquidations', 0):,} total)",
              flush=True)
        if self.wal_alert_mb > 0 and wal_size_mb > self.wal_alert_mb:
            print(
                f"  [WARN] WAL size high: {wal_size_mb:.1f}MB threshold={self.wal_alert_mb:.1f}MB",
                flush=True,
            )

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
        try:
            self._maybe_checkpoint()
        except Exception:
            pass
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
        wal_size_mb = self._wal_size_mb()
        return {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "connected": bool(self._connected),
            "last_message_ts_utc": self._last_message_ts_utc,
            "last_data_progress_ts_utc": self._last_data_progress_ts_utc,
            "last_flush_ts_utc": self._last_flush_ts_utc,
            "last_checkpoint_ts_utc": self._last_checkpoint_ts_utc,
            "rows_written_since_start": rows,
            "last_error": self._last_error,
            "current_backoff_seconds": float(self._reconnect_delay),
            "backend": self._backend,
            "liquidation_stream_mode": self.liquidation_stream_mode,
            "rest_fallback_enabled": bool(self.rest_fallback_enabled),
            "rest_fallback_active": bool(self._rest_fallback_active),
            "rest_last_progress_ts_utc": self._rest_last_progress_ts_utc,
            "rest_last_error": self._rest_last_error,
            "db_path": self.db_path,
            "wal_size_mb": float(wal_size_mb),
            "wal_alert_threshold_mb": float(self.wal_alert_mb),
            "wal_alert": bool(wal_size_mb > self.wal_alert_mb),
        }

    def _write_heartbeat(self):
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._heartbeat_payload()
            self.heartbeat_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
            now = time.time()
            last_msg = None
            progress_ts = self._last_data_progress_ts_utc or self._last_message_ts_utc
            if progress_ts:
                try:
                    last_msg = datetime.fromisoformat(progress_ts).timestamp()
                except Exception:
                    last_msg = None
            lag = None if last_msg is None else max(0.0, now - float(last_msg))
            reconnects5m = len([x for x in self._recent_reconnect_ts if (now - x) <= 300.0])
            data_progressing = lag is not None and lag <= max(self.stall_timeout_sec, self.rest_poll_interval_sec * 4.0)
            status = "ok" if data_progressing else ("degraded" if self._connected else "degraded")
            component = {
                "status": status,
                "connected": bool(self._connected),
                "transport_connected": bool(self._connected),
                "last_progress_ts_utc": progress_ts,
                "progress_lag_sec": (None if lag is None else int(lag)),
                "reconnects_last_5m": int(reconnects5m),
                "errors_last_5m": int(reconnects5m if self._last_error else 0),
                "backend": self._backend,
                "liquidation_stream_mode": self.liquidation_stream_mode,
                "rest_fallback_enabled": bool(self.rest_fallback_enabled),
                "rest_fallback_active": bool(self._rest_fallback_active),
                "rest_last_progress_ts_utc": self._rest_last_progress_ts_utc,
                "required_streams_progressing": bool(data_progressing),
                "liquidation_transport_available": bool(self._last_message_ts_utc),
            }
            # Only writes its own collector.json component. logs/health/overall.json
            # (the canonical top-level operational-health verdict) is owned solely
            # by tools/heartbeat_watchdog.py -- see
            # reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md.
            # A prior version of this method also blind-overwrote overall.json
            # here (no read-first merge), which would have silently erased any
            # other component's contribution (e.g. execution/health_gate.py's
            # paper_trader) every time it ran.
            write_component_health("collector", component, root=self.health_root)
        except Exception:
            pass

    def _wal_size_mb(self) -> float:
        try:
            wal_path = Path(str(self.db_path) + "-wal")
            if not wal_path.exists():
                return 0.0
            return float(wal_path.stat().st_size) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _maybe_checkpoint(self) -> None:
        if self.conn is None:
            return
        now = time.time()
        if (now - float(self._last_checkpoint_ts or 0.0)) < self.checkpoint_interval_sec:
            return
        try:
            row = self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            self.conn.commit()
            self._last_checkpoint_ts = now
            self._last_checkpoint_ts_utc = datetime.now(timezone.utc).isoformat()
            print(f"  [CHECKPOINT] wal_checkpoint(TRUNCATE)={row}", flush=True)
        except Exception as e:
            self._last_error = f"checkpoint_error:{type(e).__name__}: {e}"
            print(f"  [WARN] {self._last_error}", flush=True)


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
    parser.add_argument("--checkpoint-interval-sec", type=float, default=300.0,
                        help="Run WAL checkpoint this often in seconds (default: 300).")
    parser.add_argument("--wal-alert-mb", type=float, default=2048.0,
                        help="Warn when WAL file exceeds this size in MB (default: 2048).")
    parser.add_argument("--heartbeat-path", type=str, default="logs/collector_heartbeat.json",
                        help="Heartbeat JSON path (default: logs/collector_heartbeat.json).")
    parser.add_argument("--stall-timeout-sec", type=float, default=45.0,
                        help="Force reconnect if no message received for this many seconds (default: 45).")
    parser.add_argument("--simulate-connection", action="store_true",
                        help="Run deterministic local connection simulation (no network).")
    parser.add_argument("--simulate-cycle-sec", type=float, default=20.0,
                        help="Simulation cycle length seconds (default: 20).")
    parser.add_argument("--simulate-down-sec", type=float, default=6.0,
                        help="Simulation disconnected duration per cycle seconds (default: 6).")
    parser.add_argument("--simulate-max-seconds", type=float, default=0.0,
                        help="Stop simulation after this many seconds (default: 0=no auto-stop).")
    parser.add_argument("--rest-fallback-enabled", action="store_true",
                        help="Enable public REST fallback for aggTrade and markPrice when WebSocket stalls.")
    parser.add_argument("--rest-fallback-disabled", action="store_true",
                        help="Disable public REST fallback even if enabled by default/env.")
    parser.add_argument("--rest-poll-interval-sec", type=float, default=5.0,
                        help="REST fallback polling interval in seconds (default: 5).")
    parser.add_argument("--liquidation-stream-mode", choices=["all_market_arr", "per_symbol"], default="all_market_arr",
                        help="Liquidation stream strategy: all_market_arr uses !forceOrder@arr, per_symbol uses <symbol>@forceOrder.")
    parser.add_argument("--heartbeat-write-interval-sec", type=float, default=15.0,
                        help="How often the heartbeat JSON file is refreshed, independent of --stats-interval (default: 15).")
    parser.add_argument("--health-root", type=str, default="logs/health",
                        help="Root dir for this process's own collector.json component (default: logs/health). "
                             "Isolated test runs should point this at a temp directory.")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("ERROR: No symbols specified")
        return

    collector = MicrostructureCollector(
        symbols=symbols,
        # the argparse default is the PRE-ROTATION path, deleted in Phase-4;
        # opening it would silently create a second, unread feed
        db_path=str(resolve_writer_db_path(args.db_path)),
        flush_interval=args.flush_interval,
        stats_interval=args.stats_interval,
        flush_fail_threshold=int(args.flush_fail_threshold),
        checkpoint_interval_sec=float(args.checkpoint_interval_sec),
        wal_alert_mb=float(args.wal_alert_mb),
        heartbeat_path=args.heartbeat_path,
        stall_timeout_sec=float(args.stall_timeout_sec),
        simulate_connection=bool(args.simulate_connection),
        simulate_cycle_sec=float(args.simulate_cycle_sec),
        simulate_down_sec=float(args.simulate_down_sec),
        simulate_max_seconds=float(args.simulate_max_seconds),
        rest_fallback_enabled=(bool(args.rest_fallback_enabled) and not bool(args.rest_fallback_disabled)),
        rest_poll_interval_sec=float(args.rest_poll_interval_sec),
        liquidation_stream_mode=str(args.liquidation_stream_mode),
        heartbeat_write_interval_sec=float(args.heartbeat_write_interval_sec),
        health_root=str(args.health_root),
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
