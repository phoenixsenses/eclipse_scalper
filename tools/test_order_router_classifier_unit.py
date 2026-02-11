#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import types
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import order_router  # noqa: E402


class DummyEx:
    def __init__(self, *, create_behavior=None, market_meta=None, markets=None, dual_side=True):
        self._create_behavior = create_behavior
        self._market_meta = market_meta or {"contract": True}
        self.markets = markets or {}
        self._dual_side = dual_side
        self.create_calls = []
        self.id = "binance"

    async def create_order(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._create_behavior:
            return self._create_behavior(kwargs)
        return {"id": "ok", "status": "open", "params": kwargs}

    async def load_markets(self, _=True):
        return self.markets

    async def set_margin_mode(self, *_args, **_kwargs):
        return None

    async def set_leverage(self, *_args, **_kwargs):
        return None

    async def fapiPrivateGetPositionSideDual(self):
        return {"dualSidePosition": self._dual_side}

    def market(self, _sym):
        return self._market_meta


class DummyData:
    def __init__(self, raw_symbol):
        self.raw_symbol = raw_symbol


class DummyBot:
    def __init__(self, ex, data):
        self.ex = ex
        self.data = data
        self.cfg = types.SimpleNamespace()
        self.state = types.SimpleNamespace(positions={}, current_equity=0.0)


class OrderRouterClassifierTests(unittest.TestCase):
    def test_non_retryable_error_aborts_immediately(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(_kwargs):
            raise Exception("invalid symbol")

        ex = DummyEx(create_behavior=behavior, market_meta=market, markets={sym_raw: market}, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.ROUTER_RETRY_BASE_SEC = 0.01
        bot.cfg.ROUTER_RETRY_MAX_DELAY_SEC = 0.01
        bot.cfg.ROUTER_RETRY_JITTER_PCT = 0.0

        orig_validate = order_router._validate_and_normalize_order
        orig_sleep = order_router.asyncio.sleep

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        async def _noop_sleep(_delay):
            return None

        order_router._validate_and_normalize_order = _ok_validate
        order_router.asyncio.sleep = _noop_sleep
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    params={},
                    retries=6,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate
            order_router.asyncio.sleep = orig_sleep

        self.assertIsNone(res)
        self.assertGreaterEqual(len(ex.create_calls), 1)
        self.assertLessEqual(len(ex.create_calls), 12)

    def test_coinbase_error_classification(self):
        ex = types.SimpleNamespace(id="coinbase")
        retryable, reason, _code = order_router._classify_order_error(
            Exception("rate limit exceeded"), ex=ex, sym_raw="BTC/USD"
        )
        self.assertTrue(retryable)
        self.assertIn(str(reason or ""), ("exchange_busy", "unknown"))

        retryable2, reason2, _code2 = order_router._classify_order_error(
            Exception("insufficient funds"), ex=ex, sym_raw="BTC/USD"
        )
        self.assertIn(str(reason2 or ""), ("margin_insufficient", "unknown"))
        self.assertLessEqual(int(bool(retryable2)), int(bool(retryable)))

    def test_exchange_retry_policy_override(self):
        env = dict(os.environ)
        try:
            os.environ["ROUTER_RETRY_POLICY_COINBASE"] = "exchange_busy=9:0.9:9:3"
            bot = DummyBot(DummyEx(), DummyData({}))
            bot.ex.id = "coinbase"
            pol = order_router._build_retry_policies(bot)
            self.assertEqual(int(pol["exchange_busy"].get("max_attempts", 0)), 9)
            self.assertAlmostEqual(float(pol["exchange_busy"].get("base_delay", 0.0)), 0.9, places=3)
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_classify_order_error(self):
        margin = order_router._classify_order_error(Exception("Margin is insufficient"))
        timeout = order_router._classify_order_error(Exception("Request timed out"))
        price_filter = order_router._classify_order_error(Exception("Filter failure: PRICE_FILTER"))
        margin_code = order_router._classify_order_error(
            Exception("BinanceError: code=-2019, msg=Margin is insufficient.")
        )
        timestamp = order_router._classify_order_error(
            Exception("BinanceError: code=-1021, msg=Timestamp for this request is outside of the recvWindow.")
        )

        for retryable, _reason, code in (margin, timeout, price_filter, margin_code, timestamp):
            self.assertIsInstance(retryable, bool)
            self.assertTrue(str(code or "").startswith("ERR_"))

        self.assertGreaterEqual(int(bool(timeout[0])), int(bool(price_filter[0])))
        self.assertGreaterEqual(int(bool(timestamp[0])), int(bool(price_filter[0])))

    def test_error_class_policy(self):
        p1 = order_router._classify_order_error_policy(Exception("Request timed out"))
        self.assertIn(p1["error_class"], ("retryable", "retryable_with_modification"))
        p2 = order_router._classify_order_error_policy(Exception("invalid symbol"))
        self.assertIn(p2["error_class"], ("fatal", "retryable"))
        p3 = order_router._classify_order_error_policy(Exception("clientOrderId is duplicated -4116"))
        self.assertIn(p3["error_class"], ("retryable_with_modification", "retryable"))
        p4 = order_router._classify_order_error_policy(Exception("Unknown order -2011"))
        self.assertIn(p4["error_class"], ("idempotent_safe", "retryable_with_modification"))


if __name__ == "__main__":
    unittest.main()
