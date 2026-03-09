"""
tests/test_bot_core.py — Unit tests for bot/core.py (EclipseEternal)
Guardian-safe: all dependencies mocked, no live calls, no real API keys.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import date, datetime, timezone
from types import SimpleNamespace

import sys
import os


# ---------------------------------------------------------------------------
# Pre-patch heavy dependencies so bot/core.py imports cleanly
# ---------------------------------------------------------------------------

# Mock ccxt
_fake_ccxt = MagicMock()
_fake_ccxt.async_support = MagicMock()
_fake_ccxt.async_support.AuthenticationError = type("AuthenticationError", (Exception,), {})
_fake_ccxt.async_support.RateLimitExceeded = type("RateLimitExceeded", (Exception,), {})
_fake_ccxt.async_support.RequestTimeout = type("RequestTimeout", (Exception,), {})
_fake_ccxt.async_support.ExchangeError = type("ExchangeError", (Exception,), {})
_fake_ccxt.async_support.binanceusdm = MagicMock(return_value=MagicMock())
sys.modules.setdefault("ccxt", _fake_ccxt)
sys.modules.setdefault("ccxt.async_support", _fake_ccxt.async_support)
sys.modules.setdefault("dotenv", MagicMock())


def _run(coro):
    """Run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_state():
    """Create a minimal mock PsycheState."""
    state = MagicMock()
    state.positions = {}
    state.current_equity = 1000.0
    state.peak_equity = 1000.0
    state.start_of_day_equity = 1000.0
    state.daily_pnl = 0.0
    state.total_trades = 0
    state.current_day = date.today()
    state.version = ""
    state.blacklist = {}
    state.last_exit_time = {}
    state.validate = MagicMock()
    state.recompute_derived = MagicMock()
    return state


def _make_bot():
    """Create a bot instance with all external deps mocked."""
    with patch("bot.core.get_exchange") as mock_get_ex, \
         patch("bot.core.DataCache") as mock_dc, \
         patch("bot.core.Notifier", return_value=None), \
         patch("bot.core.load_brain", new_callable=lambda: AsyncMock(return_value=False)), \
         patch("bot.core.save_brain", new_callable=lambda: AsyncMock(return_value=True)), \
         patch("bot.core.emergency_flat", new_callable=lambda: AsyncMock()), \
         patch.dict("os.environ", {
             "TELEGRAM_TOKEN": "",
             "TELEGRAM_CHAT_ID": "",
             "BINANCE_API_KEY": "test",
             "BINANCE_API_SECRET": "test",
             "SCALPER_DRY_RUN": "true",
         }, clear=False):

        mock_ex = MagicMock()
        mock_ex.load_markets = AsyncMock(return_value={})
        mock_ex.fetch_markets = AsyncMock(return_value={})
        mock_ex.fetch_balance = AsyncMock(return_value={"total": {"USDT": 500.0}})
        mock_ex.set_position_mode = AsyncMock()
        mock_ex.close = AsyncMock()
        mock_ex.fetch_closed_orders = AsyncMock(return_value=[])
        mock_get_ex.return_value = mock_ex

        mock_data = MagicMock()
        mock_data.load_cache = AsyncMock()
        mock_data.save_cache = AsyncMock()
        mock_data.get_cache_age = MagicMock(return_value=10.0)
        mock_data.price = {"BTCUSDT": 50000.0}
        mock_data.ohlcv = {}
        mock_data.poll_ohlcv = MagicMock(return_value=asyncio.sleep(0))
        mock_data.poll_ticker = MagicMock(return_value=asyncio.sleep(0))
        mock_data.get_df = MagicMock(return_value=None)
        mock_dc.return_value = mock_data

        from bot.core import EclipseEternal
        bot = EclipseEternal()

    # Replace state with our controllable mock
    bot.state = _make_mock_state()
    bot.data = mock_data
    bot.ex = mock_ex
    bot.exchange = mock_ex

    return bot


# ===========================================================================
# Tests
# ===========================================================================

class TestTradeAllowed(unittest.TestCase):
    """_trade_allowed() logic — wraps kill_switch.trade_allowed()."""

    def test_trade_allowed_true(self):
        bot = _make_bot()
        with patch("bot.core.ks_trade_allowed", return_value=True):
            result = _run(bot._trade_allowed())
        self.assertTrue(result)

    def test_trade_allowed_false(self):
        bot = _make_bot()
        with patch("bot.core.ks_trade_allowed", return_value=False):
            result = _run(bot._trade_allowed())
        self.assertFalse(result)

    def test_trade_allowed_exception_returns_false(self):
        """Guardian-safe: exception -> fail-closed (False)."""
        bot = _make_bot()
        with patch("bot.core.ks_trade_allowed", side_effect=RuntimeError("kaboom")):
            result = _run(bot._trade_allowed())
        self.assertFalse(result)

    def test_trade_allowed_async_variant(self):
        """If kill_switch returns a coroutine, bot awaits it."""
        bot = _make_bot()

        async def _async_allowed(b):
            return True

        with patch("bot.core.ks_trade_allowed", side_effect=_async_allowed):
            result = _run(bot._trade_allowed())
        self.assertTrue(result)


class TestHalted(unittest.TestCase):
    """_halted() logic — wraps kill_switch.is_halted()."""

    def test_halted_false(self):
        bot = _make_bot()
        with patch("bot.core.ks_is_halted", return_value=False):
            result = _run(bot._halted())
        self.assertFalse(result)

    def test_halted_true(self):
        bot = _make_bot()
        with patch("bot.core.ks_is_halted", return_value=True):
            result = _run(bot._halted())
        self.assertTrue(result)

    def test_halted_exception_returns_false(self):
        bot = _make_bot()
        with patch("bot.core.ks_is_halted", side_effect=Exception("oops")):
            result = _run(bot._halted())
        self.assertFalse(result)


class TestGuardianSafe(unittest.TestCase):
    """Functions don't raise — guardian-safe contract."""

    def test_speak_swallows_exception(self):
        bot = _make_bot()
        bot.notify = MagicMock()
        bot.notify.speak = AsyncMock(side_effect=Exception("telegram down"))
        # Should not raise
        _run(bot.speak("test message"))

    def test_maybe_set_data_ready_no_crash_on_bad_data(self):
        bot = _make_bot()
        bot.data.price = "not a dict"
        bot._maybe_set_data_ready()  # should not raise

    def test_resolve_raw_symbol_never_raises(self):
        bot = _make_bot()
        bot.data = None  # extreme case
        result = bot._resolve_raw_symbol("BTCUSDT", "fallback")
        self.assertEqual(result, "fallback")

    def test_shutdown_idempotent(self):
        bot = _make_bot()
        with patch("bot.core.emergency_flat", new_callable=lambda: AsyncMock()), \
             patch("bot.core.save_brain", new_callable=lambda: AsyncMock()):
            _run(bot.shutdown())
            _run(bot.shutdown())  # second call is no-op


class TestBotStateTransitions(unittest.TestCase):
    """State machine / bot state transitions."""

    def test_shutdown_sets_event(self):
        bot = _make_bot()
        self.assertFalse(bot._shutdown.is_set())
        with patch("bot.core.emergency_flat", new_callable=lambda: AsyncMock()), \
             patch("bot.core.save_brain", new_callable=lambda: AsyncMock()):
            _run(bot.shutdown())
        self.assertTrue(bot._shutdown.is_set())

    def test_data_ready_initially_unset(self):
        bot = _make_bot()
        self.assertFalse(bot.data_ready.is_set())

    def test_maybe_set_data_ready_with_prices(self):
        bot = _make_bot()
        bot.data.price = {"BTCUSDT": 50000.0}
        bot._maybe_set_data_ready()
        self.assertTrue(bot.data_ready.is_set())

    def test_maybe_set_data_ready_only_once(self):
        bot = _make_bot()
        bot.data.price = {"BTCUSDT": 50000.0}
        bot._maybe_set_data_ready()
        self.assertTrue(bot._data_ready_set_once)
        # Clear the event manually, calling again should NOT re-set it
        bot.data_ready.clear()
        bot._maybe_set_data_ready()
        self.assertFalse(bot.data_ready.is_set())  # already set once, won't fire again

    def test_shutdown_sets_data_ready(self):
        """Shutdown sets data_ready to unblock any waiter."""
        bot = _make_bot()
        with patch("bot.core.emergency_flat", new_callable=lambda: AsyncMock()), \
             patch("bot.core.save_brain", new_callable=lambda: AsyncMock()):
            _run(bot.shutdown())
        self.assertTrue(bot.data_ready.is_set())


class TestConfigurationLoading(unittest.TestCase):
    """Configuration loading and helpers."""

    def test_is_micro_mode_true(self):
        from bot.core import _is_micro_mode
        cfg = SimpleNamespace(CONFIG_VERSION="micro-v1.0")
        self.assertTrue(_is_micro_mode(cfg))

    def test_is_micro_mode_false(self):
        from bot.core import _is_micro_mode
        cfg = SimpleNamespace(CONFIG_VERSION="production-v4.0")
        self.assertFalse(_is_micro_mode(cfg))

    def test_is_micro_mode_none_cfg(self):
        from bot.core import _is_micro_mode
        self.assertFalse(_is_micro_mode(None))

    def test_parse_whitelist(self):
        from bot.core import _parse_whitelist
        result = _parse_whitelist("btcusdt, ethusdt, solusdt")
        self.assertEqual(result, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

    def test_parse_whitelist_empty(self):
        from bot.core import _parse_whitelist
        self.assertEqual(_parse_whitelist(""), [])
        self.assertEqual(_parse_whitelist(None), [])


class TestPublicHelpers(unittest.TestCase):
    """Public utility functions."""

    def test_extract_usdt_equity_total(self):
        from bot.core import _extract_usdt_equity
        bal = {"total": {"USDT": 1234.56}}
        self.assertAlmostEqual(_extract_usdt_equity(bal), 1234.56)

    def test_extract_usdt_equity_info_wallet(self):
        from bot.core import _extract_usdt_equity
        bal = {"total": {}, "info": {"totalWalletBalance": "5000.0"}}
        self.assertAlmostEqual(_extract_usdt_equity(bal), 5000.0)

    def test_extract_usdt_equity_none(self):
        from bot.core import _extract_usdt_equity
        self.assertEqual(_extract_usdt_equity(None), 0.0)
        self.assertEqual(_extract_usdt_equity("garbage"), 0.0)

    def test_safe_float(self):
        from bot.core import _safe_float
        self.assertEqual(_safe_float("3.14"), 3.14)
        self.assertEqual(_safe_float(None, 99.0), 99.0)
        self.assertEqual(_safe_float("not_a_number", -1.0), -1.0)
        # NaN check
        self.assertEqual(_safe_float(float("nan"), 0.0), 0.0)


class TestTaskTracking(unittest.TestCase):
    """Task tracking helpers."""

    def test_track_task_cancels_old(self):
        bot = _make_bot()

        async def _dummy():
            await asyncio.sleep(999)

        loop = asyncio.new_event_loop()
        try:
            async def _test():
                bot._track_task("t1", _dummy())
                old = bot._tasks["t1"]
                bot._track_task("t1", _dummy())
                await asyncio.sleep(0)  # let cancellation propagate
                self.assertTrue(old.cancelled() or old.done())
                # cleanup
                bot._tasks["t1"].cancel()

            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_cancel_all_tasks(self):
        bot = _make_bot()

        async def _test():
            async def _sleeper():
                await asyncio.sleep(999)

            bot._track_task("a", _sleeper())
            bot._track_task("b", _sleeper())
            self.assertEqual(len(bot._tasks), 2)
            await bot._cancel_all_tasks()
            self.assertEqual(len(bot._tasks), 0)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_test())
        finally:
            loop.close()


class TestCorrelationHeatCheck(unittest.TestCase):
    """Correlation heat check logic."""

    def test_no_equity_returns_false(self):
        bot = _make_bot()
        bot.state.current_equity = 0
        result = _run(bot._correlation_heat_check())
        self.assertFalse(result)

    def test_no_positions_returns_true(self):
        bot = _make_bot()
        bot.state.positions = {}
        bot.cfg = SimpleNamespace(LEVERAGE=1, CORRELATION_HEAT_CAP=0.35)
        result = _run(bot._correlation_heat_check())
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
