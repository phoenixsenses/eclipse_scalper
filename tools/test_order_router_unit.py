#!/usr/bin/env python3
"""
Unit-style tests for order_router cancel behavior.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
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
    def __init__(
        self,
        behavior,
        *,
        markets=None,
        market_meta=None,
        dual_side=True,
        create_behavior=None,
        price_prec_fn=None,
        amount_prec_fn=None,
        ticker=None,
        order_book=None,
        fetch_order_behavior=None,
    ):
        self._behavior = behavior
        self.markets = markets or {}
        self._market_meta = market_meta
        self._dual_side = dual_side
        self._create_behavior = create_behavior
        self._price_prec_fn = price_prec_fn
        self._amount_prec_fn = amount_prec_fn
        self._ticker = ticker
        self._order_book = order_book
        self._fetch_order_behavior = fetch_order_behavior or {}
        self.create_calls = []
        self.id = "binance"

    async def cancel_order(self, order_id, symbol):
        # behavior is a dict symbol -> exception or None
        action = self._behavior.get(symbol)
        if action is None:
            return {"id": order_id, "symbol": symbol}
        raise action

    async def create_order(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._create_behavior:
            return self._create_behavior(kwargs)
        return {"id": "ok", "status": "open", "params": kwargs}

    async def load_markets(self, _=True):
        return self.markets

    async def fetch_ticker(self, _sym):
        return self._ticker or {}

    async def fetch_order_book(self, _sym):
        return self._order_book or {}

    async def fetch_order(self, order_id, symbol):
        action = self._fetch_order_behavior.get((str(order_id), str(symbol)))
        if isinstance(action, Exception):
            raise action
        if isinstance(action, dict):
            return action
        return {"id": order_id, "symbol": symbol, "status": "closed"}

    def market(self, sym):
        return self._market_meta

    def price_to_precision(self, sym, price):
        if self._price_prec_fn:
            return self._price_prec_fn(sym, price)
        return price

    def amount_to_precision(self, sym, amount):
        if self._amount_prec_fn:
            return self._amount_prec_fn(sym, amount)
        return amount

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
        self.cfg = types.SimpleNamespace()
        self.state = types.SimpleNamespace(positions={}, current_equity=0.0)


class OrderRouterCancelTests(unittest.TestCase):
    def test_unknown_then_other_error_is_still_success(self):
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
        self.assertTrue(ok)

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

    def test_unknown_cancel_with_open_fetch_order_is_not_success(self):
        sym_key = "ETHUSDT"
        raw_symbol = "ETH/USDT:USDT"
        symbol = "ETH/USDT"
        behavior = {
            raw_symbol: Exception("Order does not exist"),
            symbol: Exception("Unknown order"),
            sym_key: Exception("Order not found"),
        }
        fetch_behavior = {
            ("zzz", raw_symbol): {"id": "zzz", "status": "open"},
        }
        bot = DummyBot(DummyEx(behavior, fetch_order_behavior=fetch_behavior), DummyData({sym_key: raw_symbol}))
        ok = asyncio.run(order_router.cancel_order(bot, "zzz", symbol))
        self.assertFalse(ok)

    def test_unknown_order_detection_ignores_symbol_not_found(self):
        self.assertFalse(order_router._looks_like_unknown_order(Exception("Symbol not found")))

    def test_unknown_order_detection_variants(self):
        samples = [
            Exception("Order does not exist"),
            Exception("Unknown order"),
            Exception("Order not found"),
            Exception("order_not_found"),
            Exception("cancel already order"),
            Exception("BINANCE -2011"),
        ]
        for err in samples:
            self.assertTrue(order_router._looks_like_unknown_order(err))


class OrderRouterCreateTests(unittest.TestCase):
    """Create-order reliability tests grouped by replace, classifier, and idempotency sections."""

    def test_retry_max_elapsed_caps_attempts(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(_kwargs):
            raise Exception("temporary failure")

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.ROUTER_RETRY_MAX_ELAPSED_SEC = 0.05
        bot.cfg.ROUTER_RETRY_BASE_SEC = 0.01
        bot.cfg.ROUTER_RETRY_MAX_DELAY_SEC = 0.01
        bot.cfg.ROUTER_RETRY_JITTER_PCT = 0.0

        orig_validate = order_router._validate_and_normalize_order
        orig_sleep = order_router.asyncio.sleep
        orig_monotonic = order_router.time.monotonic

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        async def _noop_sleep(_delay):
            return None

        tick = {"t": 0.0}

        def _fake_monotonic():
            tick["t"] += 0.03
            return tick["t"]

        order_router._validate_and_normalize_order = _ok_validate
        order_router.asyncio.sleep = _noop_sleep
        order_router.time.monotonic = _fake_monotonic
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    params={},
                    retries=10,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate
            order_router.asyncio.sleep = orig_sleep
            order_router.time.monotonic = orig_monotonic

        self.assertIsNone(res)
        self.assertLess(len(ex.create_calls), 10)


    def test_resolve_leverage_overrides_and_caps(self):
        env = dict(os.environ)
        try:
            os.environ["LEVERAGE"] = "20"
            os.environ["LEVERAGE_MAX"] = "25"
            os.environ["LEVERAGE_MIN"] = "2"
            os.environ["LEVERAGE_BY_SYMBOL"] = "BTCUSDT=30"
            os.environ["CORR_GROUPS"] = "MAJOR:BTCUSDT,ETHUSDT"
            os.environ["LEVERAGE_BY_GROUP"] = "MAJOR=12"
            os.environ["LEVERAGE_BTCUSDT"] = "18"

            bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
            lev = order_router._resolve_leverage(bot, "BTCUSDT", "BTC/USDT:USDT")
            self.assertEqual(lev, 12)
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_resolve_leverage_group_dynamic_scaling(self):
        env = dict(os.environ)
        try:
            os.environ["LEVERAGE"] = "20"
            os.environ["LEVERAGE_GROUP_DYNAMIC"] = "1"
            os.environ["LEVERAGE_GROUP_SCALE"] = "0.5"
            os.environ["LEVERAGE_GROUP_SCALE_MIN"] = "2"
            os.environ["CORR_GROUPS"] = "MEME:DOGEUSDT,SHIBUSDT"
            os.environ["LEVERAGE_GROUP_EXCLUDE_SELF"] = "1"

            bot = DummyBot(DummyEx({}), DummyData({"DOGEUSDT": "DOGE/USDT:USDT"}))
            bot.state.positions["SHIBUSDT"] = types.SimpleNamespace(size=1.0)
            lev = order_router._resolve_leverage(bot, "DOGEUSDT", "DOGE/USDT:USDT")
            self.assertEqual(lev, 10)
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_resolve_leverage_group_scaling_excludes_self(self):
        env = dict(os.environ)
        try:
            os.environ["LEVERAGE"] = "20"
            os.environ["LEVERAGE_GROUP_DYNAMIC"] = "1"
            os.environ["LEVERAGE_GROUP_SCALE"] = "0.5"
            os.environ["LEVERAGE_GROUP_SCALE_MIN"] = "2"
            os.environ["CORR_GROUPS"] = "MEME:DOGEUSDT,SHIBUSDT"
            os.environ["LEVERAGE_GROUP_EXCLUDE_SELF"] = "1"

            bot = DummyBot(DummyEx({}), DummyData({"DOGEUSDT": "DOGE/USDT:USDT"}))
            bot.state.positions["DOGEUSDT"] = types.SimpleNamespace(size=1.0)
            lev = order_router._resolve_leverage(bot, "DOGEUSDT", "DOGE/USDT:USDT")
            self.assertEqual(lev, 20)
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_resolve_leverage_group_exposure_scaling(self):
        env = dict(os.environ)
        try:
            os.environ["LEVERAGE"] = "20"
            os.environ["LEVERAGE_GROUP_DYNAMIC"] = "1"
            os.environ["LEVERAGE_GROUP_EXPOSURE"] = "1"
            os.environ["LEVERAGE_GROUP_EXPOSURE_REF_PCT"] = "0.10"
            os.environ["LEVERAGE_GROUP_SCALE"] = "0.5"
            os.environ["LEVERAGE_GROUP_SCALE_MIN"] = "2"
            os.environ["CORR_GROUPS"] = "MAJOR:BTCUSDT,ETHUSDT"

            bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
            bot.state.current_equity = 1000.0
            bot.state.positions["ETHUSDT"] = types.SimpleNamespace(size=2.0, entry_price=100.0)
            lev = order_router._resolve_leverage(bot, "BTCUSDT", "BTC/USDT:USDT")
            self.assertEqual(lev, 5)
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_resolve_leverage_applies_belief_guard_cap_for_entries_only(self):
        env = dict(os.environ)
        try:
            os.environ["LEVERAGE"] = "20"
            bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
            bot.state.guard_knobs = {"max_leverage": 7}
            lev_entry = order_router._resolve_leverage(bot, "BTCUSDT", "BTC/USDT:USDT", is_exit=False)
            lev_exit = order_router._resolve_leverage(bot, "BTCUSDT", "BTC/USDT:USDT", is_exit=True)
            self.assertEqual(lev_entry, 7)
            self.assertEqual(lev_exit, 20)
        finally:
            os.environ.clear()
            os.environ.update(env)

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

    def test_trailing_stop_requires_activation(self):
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
                callback_rate=0.5,
                # activation_price omitted -> should block
            )
        )
        self.assertIsNone(res)

    def test_trailing_stop_requires_callback(self):
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
                activation_price=100.0,
                # callback_rate omitted -> should block
            )
        )
        self.assertIsNone(res)

    def test_trailing_stop_callback_rate_normalizes(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: {}},
            market_meta={"contract": True},
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="TRAILING_STOP_MARKET",
                    side="sell",
                    amount=1.0,
                    activation_price=100.0,
                    callback_rate=45,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        self.assertEqual(ex.create_calls[-1]["params"]["callbackRate"], 0.45)

    def test_hedge_exit_requires_position_side(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        ex = DummyEx({}, markets={sym_raw: {}}, market_meta={"contract": True}, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        res = asyncio.run(
            order_router.create_order(
                bot,
                symbol=sym,
                type="MARKET",
                side="sell",
                amount=1.0,
                params={"reduceOnly": True},
                intent_reduce_only=True,
            )
        )
        self.assertIsNone(res)


    def test_push_variant_bounded_and_deduped(self):
        variants = []
        seen = set()
        for i in range(order_router._MAX_VARIANTS + 5):
            order_router._push_variant(variants, seen, "BTC/USDT", 1, None, {"i": i})
        self.assertEqual(len(variants), order_router._MAX_VARIANTS)

        # Duplicate should not increase size
        order_router._push_variant(variants, seen, "BTC/USDT", 1, None, {"i": 1})
        self.assertEqual(len(variants), order_router._MAX_VARIANTS)

    def test_validate_and_normalize_rounding(self):
        sym_raw = "BTC/USDT:USDT"
        market = {
            "limits": {"amount": {"min": 0.0001}, "cost": {"min": 0.1}},
            "info": {"filters": [{"filterType": "LOT_SIZE", "minQty": "0.0001"}, {"filterType": "MIN_NOTIONAL", "minNotional": "0.1"}]},
            "contract": True,
        }

        def price_prec(_sym, price):
            return "123.45"

        def amount_prec(_sym, amount):
            return "0.001"

        ex = DummyEx({}, markets={sym_raw: market}, market_meta=market, price_prec_fn=price_prec, amount_prec_fn=amount_prec)
        ok, amt_norm, px_norm, why = asyncio.run(
            order_router._validate_and_normalize_order(
                ex,
                sym_raw=sym_raw,
                amount=0.001234,
                price=123.4567,
                params={},
                log=lambda _s: None,
            )
        )
        self.assertTrue(ok)
        self.assertEqual(amt_norm, 0.001)
        self.assertEqual(px_norm, 123.45)

    def test_create_order_payload_smoke_close_position_stop(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="STOP_MARKET",
                    side="sell",
                    amount=1.0,
                    stop_price=123.45,
                    params={"closePosition": True, "reduceOnly": True},
                    intent_close_position=True,
                    hedge_side_hint="LONG",
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        sent = ex.create_calls[-1]
        self.assertEqual(sent["symbol"], sym_raw)
        self.assertEqual(sent["type"], "stop_market")
        self.assertEqual(sent["side"], "sell")
        self.assertEqual(sent["amount"], 0.0)
        self.assertNotIn("reduceOnly", sent["params"])
        self.assertEqual(sent["params"].get("closePosition"), True)
        self.assertEqual(sent["params"].get("stopPrice"), 123.45)
        self.assertEqual(sent["params"].get("positionSide"), "LONG")

    def test_stop_price_uses_trigger_price_when_stop_missing(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="STOP_MARKET",
                    side="sell",
                    amount=1.0,
                    trigger_price=111.11,
                    params={},
                    intent_reduce_only=True,
                    hedge_side_hint="LONG",
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        sent = ex.create_calls[-1]
        self.assertEqual(sent["params"].get("stopPrice"), 111.11)
        self.assertNotIn("triggerPrice", sent["params"])

    def test_stop_price_prefers_explicit_stop_over_trigger(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="STOP_MARKET",
                    side="sell",
                    amount=1.0,
                    stop_price=222.22,
                    trigger_price=111.11,
                    params={"stopPrice": 999.99},
                    intent_reduce_only=True,
                    hedge_side_hint="LONG",
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        sent = ex.create_calls[-1]
        self.assertEqual(sent["params"].get("stopPrice"), 222.22)
        self.assertNotIn("triggerPrice", sent["params"])

    def test_close_position_strips_reduce_only(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="STOP_MARKET",
                    side="sell",
                    amount=1.0,
                    stop_price=123.45,
                    params={"closePosition": True, "reduceOnly": True},
                    intent_close_position=True,
                    hedge_side_hint="LONG",
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        sent = ex.create_calls[-1]
        self.assertEqual(sent["amount"], 0.0)
        self.assertEqual(sent["params"].get("closePosition"), True)
        self.assertNotIn("reduceOnly", sent["params"])

    def test_reduce_only_preserved_when_not_close_position(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="sell",
                    amount=1.0,
                    params={"reduceOnly": True},
                    intent_reduce_only=True,
                    hedge_side_hint="LONG",
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        sent = ex.create_calls[-1]
        self.assertEqual(sent["params"].get("reduceOnly"), True)

    def test_first_live_safe_blocks_non_allowlisted_entry(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.FIRST_LIVE_SAFE = True
        bot.cfg.FIRST_LIVE_SYMBOLS = "ETHUSDT"

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    params={},
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNone(res)
        self.assertEqual(len(ex.create_calls), 0)

    def test_first_live_safe_blocks_notional_cap(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.FIRST_LIVE_SAFE = True
        bot.cfg.FIRST_LIVE_SYMBOLS = "BTCUSDT"
        bot.cfg.FIRST_LIVE_MAX_NOTIONAL_USDT = 5.0

        orig_validate = order_router._validate_and_normalize_order
        orig_fetch_last = order_router._fetch_last_price

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        async def _last_price(*_args, **_kwargs):
            return 10.0

        order_router._validate_and_normalize_order = _ok_validate
        order_router._fetch_last_price = _last_price
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    params={},
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate
            order_router._fetch_last_price = orig_fetch_last

        self.assertIsNone(res)
        self.assertEqual(len(ex.create_calls), 0)

    def test_first_live_safe_caps_not_applied_to_exits(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(kwargs):
            return {"id": "ok", "params": kwargs}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.FIRST_LIVE_SAFE = True
        bot.cfg.FIRST_LIVE_SYMBOLS = "BTCUSDT"
        bot.cfg.FIRST_LIVE_MAX_NOTIONAL_USDT = 0.01

        orig_validate = order_router._validate_and_normalize_order
        orig_fetch_last = order_router._fetch_last_price

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        async def _last_price(*_args, **_kwargs):
            return 1000.0

        order_router._validate_and_normalize_order = _ok_validate
        order_router._fetch_last_price = _last_price
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="sell",
                    amount=1.0,
                    params={"reduceOnly": True},
                    intent_reduce_only=True,
                    hedge_side_hint="LONG",
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate
            order_router._fetch_last_price = orig_fetch_last

        self.assertIsNotNone(res)
        self.assertEqual(len(ex.create_calls), 1)

    def test_detect_binance_futures_modes_cached(self):
        class ModeEx:
            def __init__(self):
                self.calls = {"dual": 0, "multi": 0}

            async def fapiPrivateGetPositionSideDual(self):
                self.calls["dual"] += 1
                return {"dualSidePosition": True}

            async def fapiPrivateGetMultiAssetsMargin(self):
                self.calls["multi"] += 1
                return {"multiAssetsMargin": False}

        ex = ModeEx()

        # reset cache
        order_router._MODE_CACHE["ts"] = 0.0
        order_router._MODE_CACHE["dualSidePosition"] = None
        order_router._MODE_CACHE["multiAssetsMargin"] = None

        dual1, multi1 = asyncio.run(order_router._detect_binance_futures_modes(ex))
        dual2, multi2 = asyncio.run(order_router._detect_binance_futures_modes(ex))

        self.assertTrue(dual1)
        self.assertFalse(multi1)
        self.assertTrue(dual2)
        self.assertFalse(multi2)
        self.assertEqual(ex.calls["dual"], 1)
        self.assertEqual(ex.calls["multi"], 1)

    def test_validate_rejects_amount_le_zero(self):
        sym_raw = "BTC/USDT:USDT"
        market = {"limits": {"amount": {"min": 0.001}}, "info": {}}
        ex = DummyEx({}, markets={sym_raw: market}, market_meta=market)

        ok, _amt, _px, why = asyncio.run(
            order_router._validate_and_normalize_order(
                ex,
                sym_raw=sym_raw,
                amount=0.0,
                price=100.0,
                params={},
                log=lambda _s: None,
            )
        )
        self.assertFalse(ok)
        self.assertIn("amount<=0", why)

    def test_validate_rejects_below_min_qty(self):
        sym_raw = "BTC/USDT:USDT"
        market = {
            "limits": {"amount": {"min": 0.01}},
            "info": {"filters": [{"filterType": "LOT_SIZE", "minQty": "0.02"}]},
        }
        ex = DummyEx({}, markets={sym_raw: market}, market_meta=market)

        ok, _amt, _px, why = asyncio.run(
            order_router._validate_and_normalize_order(
                ex,
                sym_raw=sym_raw,
                amount=0.015,
                price=100.0,
                params={},
                log=lambda _s: None,
            )
        )
        self.assertFalse(ok)
        self.assertIn("amount<", why)

    def test_validate_rejects_below_min_notional(self):
        sym_raw = "BTC/USDT:USDT"
        market = {
            "limits": {"cost": {"min": 10}},
            "info": {"filters": [{"filterType": "MIN_NOTIONAL", "minNotional": "10"}]},
        }

        class PriceEx(DummyEx):
            async def fetch_ticker(self, _sym):
                return {"last": 2.0}

        ex = PriceEx({}, markets={sym_raw: market}, market_meta=market)

        ok, _amt, _px, why = asyncio.run(
            order_router._validate_and_normalize_order(
                ex,
                sym_raw=sym_raw,
                amount=3.0,  # notional 6 < 10
                price=None,
                params={},
                log=lambda _s: None,
            )
        )
        self.assertFalse(ok)
        self.assertIn("notional<", why)

    def test_spread_guard_blocks_entry(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ticker = {"bid": 99.0, "ask": 101.0}

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            ticker=ticker,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.MAX_SPREAD_PCT = 0.001  # 0.1%

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    params={},
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNone(res)
        self.assertEqual(len(ex.create_calls), 0)

    def test_market_impact_guard_blocks(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ticker = {"bid": 99.0, "ask": 101.0}
        order_book = {
            "bids": [[99.0, 5.0]],
            "asks": [[101.0, 1.0], [110.0, 1.0]],
        }

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            ticker=ticker,
            order_book=order_book,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.MAX_SPREAD_PCT = 0.05  # allow spread
        bot.cfg.MAX_IMPACT_PCT = 0.01  # 1%

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.5, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.5,
                    params={},
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNone(res)
        self.assertEqual(len(ex.create_calls), 0)

    def test_spread_guard_symbol_override(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ticker = {"bid": 99.0, "ask": 101.0}  # 2% spread

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            ticker=ticker,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.MAX_SPREAD_PCT = 0.001  # default tight
        bot.cfg.MAX_SPREAD_PCT_BY_SYMBOL = "BTCUSDT=5"  # 5% override

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    params={},
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        self.assertEqual(len(ex.create_calls), 1)

    def test_impact_guard_symbol_override(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ticker = {"bid": 99.0, "ask": 101.0}
        order_book = {
            "bids": [[99.0, 5.0]],
            "asks": [[101.0, 1.0], [110.0, 1.0]],
        }

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            ticker=ticker,
            order_book=order_book,
        )
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
        bot.cfg.MAX_IMPACT_PCT = 0.01  # default tight
        bot.cfg.MAX_IMPACT_PCT_BY_SYMBOL = "BTCUSDT=10"  # 10% override
        bot.cfg.MAX_SPREAD_PCT = 0.05  # allow spread

        orig_validate = order_router._validate_and_normalize_order

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.5, None, "ok"

        order_router._validate_and_normalize_order = _ok_validate
        try:
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol=sym,
                    type="MARKET",
                    side="buy",
                    amount=1.5,
                    params={},
                    retries=1,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        self.assertEqual(len(ex.create_calls), 1)


if __name__ == "__main__":
    unittest.main()
