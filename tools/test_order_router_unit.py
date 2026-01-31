#!/usr/bin/env python3
"""
Unit-style tests for order_router cancel behavior.
"""

from __future__ import annotations

import asyncio
import types
import unittest

import sys
from pathlib import Path

# Ensure Unicode log lines don't crash on Windows codepages
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure repo root and eclipse_scalper package dir are on sys.path
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import order_router  # noqa: E402


class DummyEx:
    def __init__(self, behavior, *, markets=None, market_meta=None, dual_side=True):
        self._behavior = behavior
        self.markets = markets or {}
        self._market_meta = market_meta
        self._dual_side = dual_side

    async def cancel_order(self, order_id, symbol):
        # behavior is a dict symbol -> exception or None
        action = self._behavior.get(symbol)
        if action is None:
            return {"id": order_id, "symbol": symbol}
        raise action

    async def create_order(self, **kwargs):
        return {"id": "ok", "status": "open", "params": kwargs}

    async def load_markets(self, _=True):
        return self.markets

    def market(self, sym):
        return self._market_meta

    async def set_margin_mode(self, *_args, **_kwargs):
        return None

    async def set_leverage(self, *_args, **_kwargs):
        return None

    async def fapiPrivateGetPositionSideDual(self):
        return {"dualSidePosition": self._dual_side}


class DummyData:
    def __init__(self, raw_symbol):
        self.raw_symbol = raw_symbol


class DummyBot:
    def __init__(self, ex, data):
        self.ex = ex
        self.data = data


class OrderRouterCancelTests(unittest.TestCase):
    def test_unknown_then_other_error_is_not_success(self):
        # Unknown on raw symbol, but non-unknown on fallback -> should be False
        sym_key = "BTCUSDT"
        raw_symbol = "BTC/USDT:USDT"
        symbol = "BTC/USDT"
        behavior = {
            raw_symbol: Exception("Order does not exist"),  # unknown
            symbol: Exception("Symbol not found"),  # not unknown
            sym_key: Exception("Symbol not found"),
        }
        bot = DummyBot(DummyEx(behavior), DummyData({sym_key: raw_symbol}))
        ok = asyncio.run(order_router.cancel_order(bot, "abc123", symbol))
        self.assertFalse(ok)

    def test_all_unknown_is_success(self):
        sym_key = "ETHUSDT"
        raw_symbol = "ETH/USDT:USDT"
        symbol = "ETH/USDT"
        behavior = {
            raw_symbol: Exception("Order does not exist"),
            symbol: Exception("Unknown order"),
            sym_key: Exception("Order not found"),
        }
        bot = DummyBot(DummyEx(behavior), DummyData({sym_key: raw_symbol}))
        ok = asyncio.run(order_router.cancel_order(bot, "zzz", symbol))
        self.assertTrue(ok)

    def test_unknown_order_detection_ignores_symbol_not_found(self):
        self.assertFalse(order_router._looks_like_unknown_order(Exception("Symbol not found")))


class OrderRouterCreateTests(unittest.TestCase):
    def test_hedge_exit_accepts_position_side_in_params(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        behavior = {}
        markets = {sym_raw: {}}
        ex = DummyEx(behavior, markets=markets, market_meta={"contract": True}, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        res = asyncio.run(
            order_router.create_order(
                bot,
                symbol=sym,
                type="MARKET",
                side="sell",
                amount=1.0,
                params={"reduceOnly": True, "positionSide": "LONG"},
                intent_reduce_only=True,
            )
        )
        self.assertIsNotNone(res)

    def test_stop_entry_in_hedge_mode_does_not_require_hedge_side_hint(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        ex = DummyEx({}, markets={sym_raw: {}}, market_meta={"contract": True}, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        res = asyncio.run(
            order_router.create_order(
                bot,
                symbol=sym,
                type="STOP_MARKET",
                side="buy",
                amount=1.0,
                stop_price=100.0,
                params={},
            )
        )
        self.assertIsNotNone(res)

    def test_trailing_stop_requires_activation_and_callback(self):
        sym = "BTC/USDT"
        ex = DummyEx({}, market_meta={"contract": False})
        bot = DummyBot(ex, DummyData({}))

        res = asyncio.run(
            order_router.create_order(
                bot,
                symbol=sym,
                type="TRAILING_STOP_MARKET",
                side="sell",
                amount=1.0,
                stop_price=100.0,
                # activation_price and callback_rate omitted -> should block
            )
        )
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
