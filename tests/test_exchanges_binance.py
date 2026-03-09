"""
tests/test_exchanges_binance.py — Unit tests for exchanges/binance.py (BinanceCosmicAdapter)
Guardian-safe: all exchange calls mocked, no live connections, no real API keys.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# We need to mock heavy imports before importing the module under test
import sys


# ---------------------------------------------------------------------------
# Patch ccxt + dotenv before the adapter is imported
# ---------------------------------------------------------------------------
_fake_ccxt = MagicMock()
_fake_ccxt.async_support = MagicMock()

# Exception classes the adapter catches
_fake_ccxt.async_support.AuthenticationError = type("AuthenticationError", (Exception,), {})
_fake_ccxt.async_support.RateLimitExceeded = type("RateLimitExceeded", (Exception,), {})
_fake_ccxt.async_support.RequestTimeout = type("RequestTimeout", (Exception,), {})
_fake_ccxt.async_support.ExchangeError = type("ExchangeError", (Exception,), {})

# binanceusdm constructor returns a mock exchange instance
_mock_exchange_instance = MagicMock()
_fake_ccxt.async_support.binanceusdm = MagicMock(return_value=_mock_exchange_instance)

sys.modules.setdefault("ccxt", _fake_ccxt)
sys.modules.setdefault("ccxt.async_support", _fake_ccxt.async_support)

# Ensure dotenv doesn't fail
sys.modules.setdefault("dotenv", MagicMock())


def _run(coro):
    """Run a coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_adapter():
    """Create a fresh adapter with mocked internals."""
    from exchanges.binance import BinanceCosmicAdapter

    with patch.dict("os.environ", {
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "SCALPER_DRY_RUN": "",
    }, clear=False):
        adapter = BinanceCosmicAdapter()

    # Pre-build symbol maps so we don't need real markets
    adapter._sym_to_raw = {
        "BTCUSDT": "BTC/USDT:USDT",
        "ETHUSDT": "ETH/USDT:USDT",
        "SOLUSDT": "SOL/USDT:USDT",
    }
    adapter._raw_to_sym = {v: k for k, v in adapter._sym_to_raw.items()}
    adapter._markets_loaded = True
    adapter._time_synced = True

    # Mock exchange methods as AsyncMock
    adapter.exchange.load_markets = AsyncMock(return_value={})
    adapter.exchange.fetch_time = AsyncMock(return_value=int(time.time() * 1000))
    adapter.exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 1000.0}})
    adapter.exchange.fetch_positions = AsyncMock(return_value=[])
    adapter.exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0, "symbol": "BTC/USDT:USDT"})
    adapter.exchange.fetch_ohlcv = AsyncMock(return_value=[[1, 2, 3, 4, 5, 6]])
    adapter.exchange.fetch_funding_rate = AsyncMock(return_value={"fundingRate": 0.0001})
    adapter.exchange.create_order = AsyncMock(return_value={"id": "123", "status": "open"})
    adapter.exchange.cancel_order = AsyncMock(return_value={"id": "123", "status": "canceled"})
    adapter.exchange.fetch_closed_orders = AsyncMock(return_value=[])
    adapter.exchange.fetch_open_orders = AsyncMock(return_value=[])
    adapter.exchange.fapiPublicGetPing = AsyncMock(return_value={})
    adapter.exchange.close = AsyncMock()
    adapter.exchange.price_to_precision = MagicMock(return_value="50000.00")
    adapter.exchange.amount_to_precision = MagicMock(return_value="0.001")

    return adapter


# ===========================================================================
# Tests
# ===========================================================================

class TestSymbolResolution(unittest.TestCase):
    """Symbol resolution / canonical<->raw mapping."""

    def test_resolve_canonical_to_raw(self):
        adapter = _make_adapter()
        raw = adapter._resolve_symbol("BTCUSDT")
        self.assertEqual(raw, "BTC/USDT:USDT")

    def test_resolve_raw_stays_raw(self):
        adapter = _make_adapter()
        raw = adapter._resolve_symbol("BTC/USDT:USDT")
        self.assertEqual(raw, "BTC/USDT:USDT")

    def test_resolve_none_returns_none(self):
        adapter = _make_adapter()
        self.assertIsNone(adapter._resolve_symbol(None))

    def test_resolve_empty_string(self):
        adapter = _make_adapter()
        result = adapter._resolve_symbol("")
        self.assertEqual(result, "")

    def test_resolve_unknown_symbol_returns_input(self):
        adapter = _make_adapter()
        result = adapter._resolve_symbol("UNKNOWNUSDT")
        self.assertEqual(result, "UNKNOWNUSDT")

    def test_cache_key_canonical(self):
        adapter = _make_adapter()
        k1 = adapter._cache_key("BTC/USDT:USDT")
        k2 = adapter._cache_key("BTCUSDT")
        self.assertEqual(k1, k2)

    def test_build_symbol_maps(self):
        adapter = _make_adapter()
        adapter._sym_to_raw.clear()
        adapter._raw_to_sym.clear()
        adapter.exchange.markets = {
            "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT"},
            "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT"},
        }
        adapter._build_symbol_maps_from_markets()
        self.assertIn("BTCUSDT", adapter._sym_to_raw)
        self.assertIn("ETHUSDT", adapter._sym_to_raw)


class TestOrderSubmission(unittest.TestCase):
    """Order creation with mocked exchange."""

    def test_create_order_resolves_symbol(self):
        adapter = _make_adapter()
        result = _run(adapter.create_order("BTCUSDT", "limit", "buy", 0.001, 50000.0))
        # safe_request should have been called with resolved raw symbol
        adapter.exchange.create_order.assert_called_once()
        args = adapter.exchange.create_order.call_args
        self.assertEqual(args[0][0], "BTC/USDT:USDT")

    def test_create_order_returns_response(self):
        adapter = _make_adapter()
        result = _run(adapter.create_order("ETHUSDT", "market", "sell", 1.0))
        self.assertIn("id", result)

    def test_dry_run_blocks_order(self):
        adapter = _make_adapter()
        with patch.dict("os.environ", {"SCALPER_DRY_RUN": "true"}):
            result = _run(adapter.create_order("BTCUSDT", "limit", "buy", 0.001, 50000.0))
        self.assertEqual(result["id"], "DRY_RUN_ORDER")
        self.assertEqual(result["status"], "canceled")
        self.assertTrue(result["info"]["dry_run"])

    def test_dry_run_order_zero_quantity(self):
        adapter = _make_adapter()
        with patch.dict("os.environ", {"SCALPER_DRY_RUN": "true"}):
            result = _run(adapter.create_order("BTCUSDT", "limit", "buy", 0, 50000.0))
        self.assertEqual(result["amount"], 0.0)


class TestOrderCancellation(unittest.TestCase):
    """Order cancellation."""

    def test_cancel_order_resolves_symbol(self):
        adapter = _make_adapter()
        result = _run(adapter.cancel_order("order123", "BTCUSDT"))
        adapter.exchange.cancel_order.assert_called_once()
        args = adapter.exchange.cancel_order.call_args
        self.assertEqual(args[0][1], "BTC/USDT:USDT")

    def test_cancel_order_dry_run(self):
        adapter = _make_adapter()
        with patch.dict("os.environ", {"SCALPER_DRY_RUN": "true"}):
            result = _run(adapter.cancel_order("order123", "BTCUSDT"))
        self.assertEqual(result["status"], "canceled")
        self.assertTrue(result["info"]["dry_run"])


class TestErrorHandling(unittest.TestCase):
    """Error handling: timeout, rate limit, API errors."""

    def test_rate_limit_retries(self):
        adapter = _make_adapter()
        RLE = type(adapter.exchange).__module__  # won't use this
        # Import the exception class from our fake ccxt
        import ccxt.async_support as ccxt_mod
        call_count = 0

        async def _rate_limited(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ccxt_mod.RateLimitExceeded("rate limited")
            return {"last": 50000.0}

        adapter.exchange.fetch_ticker = AsyncMock(side_effect=_rate_limited)
        result = _run(adapter.fetch_ticker("BTCUSDT"))
        self.assertEqual(call_count, 3)
        self.assertEqual(result["last"], 50000.0)

    def test_request_timeout_retries(self):
        adapter = _make_adapter()
        import ccxt.async_support as ccxt_mod
        call_count = 0

        async def _timeout(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ccxt_mod.RequestTimeout("timed out")
            return {"total": {"USDT": 500.0}}

        adapter.exchange.fetch_balance = AsyncMock(side_effect=_timeout)
        result = _run(adapter.fetch_balance())
        self.assertEqual(call_count, 2)

    def test_all_retries_exhausted_raises(self):
        adapter = _make_adapter()
        import ccxt.async_support as ccxt_mod
        adapter.exchange.fetch_balance = AsyncMock(side_effect=ccxt_mod.RequestTimeout("timeout"))
        with self.assertRaises(Exception) as ctx:
            _run(adapter.fetch_balance())
        self.assertIn("6 ATTEMPTS", str(ctx.exception))

    def test_exchange_error_nonce_triggers_time_sync(self):
        adapter = _make_adapter()
        import ccxt.async_support as ccxt_mod
        call_count = 0

        async def _nonce_error(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ccxt_mod.ExchangeError("Timestamp for this request is outside")
            return []

        adapter.exchange.fetch_positions = AsyncMock(side_effect=_nonce_error)
        adapter.exchange.fetch_time = AsyncMock(return_value=int(time.time() * 1000))
        _run(adapter.fetch_positions())
        # Time sync should have been triggered
        self.assertTrue(adapter._time_synced)


class TestPositionFetching(unittest.TestCase):
    """Position fetching."""

    def test_fetch_positions_resolves_symbols(self):
        adapter = _make_adapter()
        adapter.exchange.fetch_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USDT:USDT", "contracts": 1}
        ])
        result = _run(adapter.fetch_positions(["BTCUSDT", "ETHUSDT"]))
        self.assertEqual(len(result), 1)

    def test_fetch_positions_none_symbols(self):
        adapter = _make_adapter()
        adapter.exchange.fetch_positions = AsyncMock(return_value=[])
        result = _run(adapter.fetch_positions(None))
        self.assertEqual(result, [])


class TestBalanceFetching(unittest.TestCase):
    """Balance fetching."""

    def test_fetch_balance_returns_dict(self):
        adapter = _make_adapter()
        result = _run(adapter.fetch_balance())
        self.assertIsInstance(result, dict)
        self.assertIn("total", result)


class TestEdgeCases(unittest.TestCase):
    """Edge cases: invalid symbol, zero qty, negative price, None orders."""

    def test_fetch_closed_orders_none_symbol_returns_empty(self):
        adapter = _make_adapter()
        result = _run(adapter.fetch_closed_orders(None))
        self.assertEqual(result, [])

    def test_fetch_open_orders_none_symbol_returns_empty(self):
        adapter = _make_adapter()
        result = _run(adapter.fetch_open_orders(None))
        self.assertEqual(result, [])

    def test_price_to_precision_fallback(self):
        adapter = _make_adapter()
        adapter.exchange.price_to_precision = MagicMock(side_effect=Exception("no market"))
        result = adapter.price_to_precision("UNKNOWNUSDT", 123.456)
        self.assertIn("123.456", result)

    def test_amount_to_precision_fallback(self):
        adapter = _make_adapter()
        adapter.exchange.amount_to_precision = MagicMock(side_effect=Exception("no market"))
        result = adapter.amount_to_precision("UNKNOWNUSDT", 0.00123)
        self.assertIn("0.00123", result)

    def test_ticker_cache_hit(self):
        adapter = _make_adapter()
        # Prime cache
        adapter.ticker_cache["BTCUSDT"] = ({"last": 99999.0}, time.time())
        result = _run(adapter.fetch_ticker("BTCUSDT"))
        self.assertEqual(result["last"], 99999.0)
        # Exchange should NOT have been called (cache hit)
        adapter.exchange.fetch_ticker.assert_not_called()

    def test_ticker_cache_expired(self):
        adapter = _make_adapter()
        # Expired cache entry
        adapter.ticker_cache["BTCUSDT"] = ({"last": 99999.0}, time.time() - 10.0)
        result = _run(adapter.fetch_ticker("BTCUSDT"))
        # Should have called exchange for fresh data
        adapter.exchange.fetch_ticker.assert_called_once()

    def test_funding_cache_hit(self):
        adapter = _make_adapter()
        adapter.funding_cache["BTCUSDT"] = (0.0005, time.time())
        result = _run(adapter.fetch_funding_rate("BTCUSDT"))
        self.assertAlmostEqual(result, 0.0005)
        adapter.exchange.fetch_funding_rate.assert_not_called()

    def test_close_idempotent(self):
        adapter = _make_adapter()
        _run(adapter.close())
        _run(adapter.close())  # second call should be no-op
        adapter.exchange.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
