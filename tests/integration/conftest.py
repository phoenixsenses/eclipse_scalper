"""
Integration test infrastructure for Eclipse Scalper.

Provides:
- FakeExchange       — configurable mock implementing ExchangeAdapter interface
- FakeNotificationManager — records all notifications
- TimeController     — freeze / advance wall-clock for deterministic tests
- make_bot fixture   — wires everything into a minimal bot namespace
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# TimeController
# ═══════════════════════════════════════════════════════════════════════════


class TimeController:
    """Allows freezing and advancing time for deterministic testing.

    Usage:
        tc = TimeController()
        tc.freeze(1700000000.0)
        assert tc.now() == 1700000000.0
        tc.advance(10)
        assert tc.now() == 1700000010.0
    """

    def __init__(self) -> None:
        self._frozen_ts: Optional[float] = None

    def freeze(self, ts: float) -> None:
        """Set the current time to *ts* (epoch seconds)."""
        self._frozen_ts = float(ts)

    def advance(self, seconds: float) -> None:
        """Advance frozen time by *seconds*.  If not frozen, freezes at now()."""
        if self._frozen_ts is None:
            self._frozen_ts = time.time()
        self._frozen_ts += seconds

    def now(self) -> float:
        """Return current (possibly frozen) epoch time."""
        if self._frozen_ts is not None:
            return self._frozen_ts
        return time.time()

    def reset(self) -> None:
        """Unfreeze — revert to real wall-clock."""
        self._frozen_ts = None


# ═══════════════════════════════════════════════════════════════════════════
# FakeExchange
# ═══════════════════════════════════════════════════════════════════════════


class FakeExchange:
    """Configurable mock exchange that mirrors :class:`BinanceCosmicAdapter`.

    Every public method that the adapter exposes is implemented here.
    Calls are recorded in ``self.calls`` so tests can assert interaction
    history.  Use ``fail_next`` / ``delay_next`` to inject faults.
    """

    name: str = "fake"

    def __init__(self, time_controller: Optional[TimeController] = None) -> None:
        self._tc = time_controller or TimeController()

        # ---- configurable return data ----
        self._balance: Dict[str, Any] = {
            "total": {"USDT": 1000.0},
            "free": {"USDT": 1000.0},
            "used": {"USDT": 0.0},
            "info": {},
        }
        self._positions: List[Dict[str, Any]] = []
        self._tickers: Dict[str, Dict[str, Any]] = {}
        self._ohlcv: Dict[str, List[list]] = {}
        self._funding_rates: Dict[str, float] = {}

        # ---- order tracking ----
        self._open_orders: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._closed_orders: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # ---- call recording ----
        self.calls: List[Dict[str, Any]] = []

        # ---- fault injection ----
        self._fail_next: Dict[str, Exception] = {}
        self._delay_next: Dict[str, float] = {}

    # ------------------------------------------------------------------ config
    def set_balance(self, balance_dict: Dict[str, Any]) -> None:
        """Set what ``fetch_balance`` returns."""
        self._balance = balance_dict

    def set_positions(self, positions_list: List[Dict[str, Any]]) -> None:
        """Set what ``fetch_positions`` returns."""
        self._positions = list(positions_list)

    def set_ticker(self, symbol: str, ticker: Dict[str, Any]) -> None:
        """Set what ``fetch_ticker`` returns for *symbol*."""
        self._tickers[symbol] = ticker

    def set_ohlcv(self, symbol: str, candles: List[list]) -> None:
        """Set what ``fetch_ohlcv`` returns for *symbol*."""
        self._ohlcv[symbol] = candles

    def set_funding_rate(self, symbol: str, rate: float) -> None:
        """Set what ``fetch_funding_rate`` returns for *symbol*."""
        self._funding_rates[symbol] = rate

    # ------------------------------------------------------------------ faults
    def fail_next(self, method: str, error: Exception) -> None:
        """Make the next call to *method* raise *error*."""
        self._fail_next[method] = error

    def delay_next(self, method: str, seconds: float) -> None:
        """Simulate *seconds* of latency on the next call to *method*."""
        self._delay_next[method] = seconds

    # -------------------------------------------------------- internal helpers
    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append({"method": method, "ts": self._tc.now(), **kwargs})

    async def _maybe_fault(self, method: str) -> None:
        delay = self._delay_next.pop(method, None)
        if delay is not None:
            await asyncio.sleep(delay)

        err = self._fail_next.pop(method, None)
        if err is not None:
            raise err

    # ═══════════════════════════════════════════════════════════════════════
    # ExchangeAdapter interface
    # ═══════════════════════════════════════════════════════════════════════

    async def fetch_balance(self) -> Dict[str, Any]:
        await self._maybe_fault("fetch_balance")
        self._record("fetch_balance")
        return dict(self._balance)

    async def fetch_positions(self, symbols: Any = None) -> List[Dict[str, Any]]:
        await self._maybe_fault("fetch_positions")
        self._record("fetch_positions", symbols=symbols)
        if symbols:
            sym_set = set(symbols) if isinstance(symbols, (list, tuple)) else {symbols}
            return [p for p in self._positions if p.get("symbol") in sym_set]
        return list(self._positions)

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        await self._maybe_fault("fetch_ticker")
        self._record("fetch_ticker", symbol=symbol)
        if symbol in self._tickers:
            return dict(self._tickers[symbol])
        # sensible default
        return {
            "symbol": symbol,
            "last": 100.0,
            "bid": 99.95,
            "ask": 100.05,
            "high": 105.0,
            "low": 95.0,
            "volume": 1_000_000.0,
            "timestamp": int(self._tc.now() * 1000),
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Any = None,
        limit: Any = None,
        params: Optional[dict] = None,
    ) -> List[list]:
        await self._maybe_fault("fetch_ohlcv")
        self._record("fetch_ohlcv", symbol=symbol, timeframe=timeframe, since=since, limit=limit)
        if symbol in self._ohlcv:
            candles = self._ohlcv[symbol]
            if limit is not None:
                candles = candles[-int(limit):]
            return list(candles)
        # default: 10 flat candles at 100.0
        ts_base = int(self._tc.now() * 1000) - 600_000
        return [
            [ts_base + i * 60_000, 100.0, 100.5, 99.5, 100.0, 1000.0]
            for i in range(limit or 10)
        ]

    async def fetch_funding_rate(self, symbol: str) -> float:
        await self._maybe_fault("fetch_funding_rate")
        self._record("fetch_funding_rate", symbol=symbol)
        return self._funding_rates.get(symbol, 0.0001)

    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: Any,
        price: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self._maybe_fault("create_order")
        params = params or {}
        order_id = str(uuid.uuid4())[:12]
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": float(amount),
            "price": float(price) if price is not None else None,
            "filled": float(amount),  # assume immediate fill by default
            "remaining": 0.0,
            "status": "closed",
            "average": float(price) if price is not None else 100.0,
            "reduceOnly": bool(params.get("reduceOnly", False)),
            "timestamp": int(self._tc.now() * 1000),
            "info": {},
        }
        self._record("create_order", order=order)

        # If type is limit and no immediate fill simulation, put in open
        if type == "limit":
            order["filled"] = 0.0
            order["remaining"] = float(amount)
            order["status"] = "open"
            self._open_orders[symbol].append(order)
        else:
            self._closed_orders[symbol].append(order)

        return order

    async def cancel_order(self, id_: str, symbol: str) -> Dict[str, Any]:
        await self._maybe_fault("cancel_order")
        self._record("cancel_order", id=id_, symbol=symbol)

        # Move from open to closed if found
        remaining_open = []
        canceled = None
        for o in self._open_orders.get(symbol, []):
            if o["id"] == id_:
                o["status"] = "canceled"
                canceled = o
                self._closed_orders[symbol].append(o)
            else:
                remaining_open.append(o)
        self._open_orders[symbol] = remaining_open

        if canceled:
            return canceled
        return {"id": id_, "symbol": symbol, "status": "canceled", "info": {}}

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        await self._maybe_fault("fetch_open_orders")
        self._record("fetch_open_orders", symbol=symbol)
        if symbol is None:
            return []
        return list(self._open_orders.get(symbol, []))

    async def fetch_closed_orders(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        await self._maybe_fault("fetch_closed_orders")
        self._record("fetch_closed_orders", symbol=symbol, limit=limit)
        if symbol is None:
            return []
        orders = self._closed_orders.get(symbol, [])
        return list(orders[-limit:])

    async def fetch_markets(self) -> Dict[str, Any]:
        await self._maybe_fault("fetch_markets")
        self._record("fetch_markets")
        return {}

    async def set_position_mode(self, hedged: bool = True) -> Any:
        self._record("set_position_mode", hedged=hedged)
        return {"code": 0, "msg": "ok"}

    def price_to_precision(self, symbol: str, price: float) -> str:
        return str(round(float(price), 2))

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return str(round(float(amount), 4))

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        return symbol

    async def health_check(self) -> None:
        self._record("health_check")

    async def close(self) -> None:
        self._record("close")


# ═══════════════════════════════════════════════════════════════════════════
# FakeNotificationManager
# ═══════════════════════════════════════════════════════════════════════════


class FakeNotificationManager:
    """Records all notifications so tests can inspect what was sent.

    Usage:
        nm = FakeNotificationManager()
        await nm.send("Position opened", priority="info")
        assert len(nm.messages) == 1
    """

    def __init__(self, time_controller: Optional[TimeController] = None) -> None:
        self._tc = time_controller or TimeController()
        self.messages: List[Dict[str, Any]] = []

    async def send(self, text: str, priority: str = "info", **kwargs: Any) -> None:
        self.messages.append({
            "text": text,
            "priority": priority,
            "timestamp": self._tc.now(),
            **kwargs,
        })

    def clear(self) -> None:
        """Reset recorded messages."""
        self.messages.clear()

    @property
    def last(self) -> Optional[Dict[str, Any]]:
        """Convenience: return the last message or None."""
        return self.messages[-1] if self.messages else None

    def texts(self) -> List[str]:
        """Return just the text of every message."""
        return [m["text"] for m in self.messages]


# ═══════════════════════════════════════════════════════════════════════════
# Pytest fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def time_controller():
    """Provides a :class:`TimeController` frozen at a deterministic epoch."""
    tc = TimeController()
    tc.freeze(1_700_000_000.0)  # 2023-11-14T22:13:20Z
    return tc


@pytest.fixture
def fake_exchange(time_controller):
    """Provides a fresh :class:`FakeExchange` wired to the time controller."""
    return FakeExchange(time_controller=time_controller)


@pytest.fixture
def fake_notify(time_controller):
    """Provides a fresh :class:`FakeNotificationManager`."""
    return FakeNotificationManager(time_controller=time_controller)


@pytest.fixture
def make_bot(time_controller, fake_exchange, fake_notify):
    """Factory fixture that creates a minimal bot namespace.

    Returns a callable; each invocation produces a new bot with:
      - ``bot.cfg``      — config-like namespace with common fields
      - ``bot.state``    — mutable state namespace
      - ``bot.exchange`` — :class:`FakeExchange`
      - ``bot.notify``   — :class:`FakeNotificationManager`
      - ``bot.tc``       — :class:`TimeController` (for advancing time in tests)
    """
    def _factory(
        *,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        state_overrides: Optional[Dict[str, Any]] = None,
        exchange: Optional[FakeExchange] = None,
        notify: Optional[FakeNotificationManager] = None,
    ) -> SimpleNamespace:
        # ---- config ----
        cfg_defaults = dict(
            KILL_SWITCH_ENABLED=True,
            KILL_SWITCH_COOLDOWN_SEC=300.0,
            CIRCUIT_BREAKER_ENABLED=True,
            EVENT_JOURNAL_ENABLED=True,
            MAX_POSITIONS=5,
            MAX_LEVERAGE=10,
            DEFAULT_LEVERAGE=5,
            RISK_PER_TRADE=0.01,
            MAX_DAILY_LOSS=0.05,
            MAX_DRAWDOWN=0.10,
            PAPER_MODE=False,
            DRY_RUN=False,
            ACTIVE_SYMBOLS=["BTCUSDT", "ETHUSDT"],
            WATCHDOG_EVERY_SEC=60,
            EX_PROBE_EVERY_SEC=60,
        )
        if cfg_overrides:
            cfg_defaults.update(cfg_overrides)
        cfg = SimpleNamespace(**cfg_defaults)

        # ---- state ----
        state_defaults = dict(
            halt_until_ts=0.0,
            halt_reason="",
            halt_count=0,
            kill_metrics={},
            current_equity=1000.0,
            peak_equity=1000.0,
            start_of_day_equity=1000.0,
            daily_pnl=0.0,
            margin_ratio=0.0,
            positions={},
            run_context={},
            emergency_flag=False,
            emergency_reason="",
            emergency_forced=False,
            last_exit_time={},
            total_trades=0,
        )
        if state_overrides:
            state_defaults.update(state_overrides)
        state = SimpleNamespace(**state_defaults)

        # ---- assemble bot ----
        ex = exchange or fake_exchange
        ntf = notify or fake_notify

        return SimpleNamespace(
            cfg=cfg,
            state=state,
            exchange=ex,
            ex=ex,  # alias used by some modules
            notify=ntf,
            tc=time_controller,
            active_symbols=set(cfg.ACTIVE_SYMBOLS),
            data=SimpleNamespace(raw_symbol={}, ohlcv={}),
            session_peak_equity=1000.0,
        )

    return _factory
