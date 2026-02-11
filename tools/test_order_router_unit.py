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

    def test_cancel_replace_is_bounded_when_cancel_fails(self):
        bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        calls = {"cancel": 0, "create": 0}

        async def _fake_cancel(*_args, **_kwargs):
            calls["cancel"] += 1
            return False

        async def _fake_create(*_args, **_kwargs):
            calls["create"] += 1
            return {"id": "ok"}

        orig_cancel = order_router.cancel_order
        orig_create = order_router.create_order
        order_router.cancel_order = _fake_cancel
        order_router.create_order = _fake_create
        try:
            res = asyncio.run(
                order_router.cancel_replace_order(
                    bot,
                    cancel_order_id="oid-1",
                    symbol="BTC/USDT",
                    type="LIMIT",
                    side="buy",
                    amount=1.0,
                    price=100.0,
                    retries=3,
                )
            )
        finally:
            order_router.cancel_order = orig_cancel
            order_router.create_order = orig_create

        self.assertIsNone(res)
        self.assertEqual(calls["cancel"], 3)
        self.assertEqual(calls["create"], 0)

    def test_cancel_replace_uses_strict_transitions_by_default(self):
        bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        seen = {"strict": None, "max_worst_case_notional": None, "max_ambiguity_attempts": None}

        async def _fake_run_cancel_replace(**kwargs):
            seen["strict"] = bool(kwargs.get("strict_transitions"))
            seen["max_worst_case_notional"] = float(kwargs.get("max_worst_case_notional") or 0.0)
            seen["max_ambiguity_attempts"] = int(kwargs.get("max_ambiguity_attempts") or 0)
            return types.SimpleNamespace(success=False, state="REPLACE_RACE", reason="replace_giveup", attempts=1, last_status="")

        orig_replace = order_router._replace_manager
        env = dict(os.environ)
        os.environ["ROUTER_REPLACE_MAX_WORST_CASE_NOTIONAL"] = "200"
        os.environ["ROUTER_REPLACE_MAX_AMBIGUITY_ATTEMPTS"] = "2"
        order_router._replace_manager = types.SimpleNamespace(run_cancel_replace=_fake_run_cancel_replace)
        try:
            res = asyncio.run(
                order_router.cancel_replace_order(
                    bot,
                    cancel_order_id="oid-1",
                    symbol="BTC/USDT",
                    type="LIMIT",
                    side="buy",
                    amount=1.0,
                    price=100.0,
                    retries=1,
                )
            )
        finally:
            order_router._replace_manager = orig_replace
            os.environ.clear()
            os.environ.update(env)

        self.assertIsNone(res)
        self.assertTrue(bool(seen["strict"]))
        self.assertAlmostEqual(float(seen["max_worst_case_notional"] or 0.0), 200.0, places=6)
        self.assertEqual(int(seen["max_ambiguity_attempts"] or 0), 2)

    def test_cancel_replace_transition_error_requests_reconcile_hint(self):
        bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        bot.state.run_context = {}

        async def _raise_run_cancel_replace(**_kwargs):
            raise RuntimeError("invalid order_intent transition")

        orig_replace = order_router._replace_manager
        order_router._replace_manager = types.SimpleNamespace(run_cancel_replace=_raise_run_cancel_replace)
        try:
            res = asyncio.run(
                order_router.cancel_replace_order(
                    bot,
                    cancel_order_id="oid-1",
                    symbol="BTC/USDT",
                    type="LIMIT",
                    side="buy",
                    amount=1.0,
                    price=100.0,
                    retries=1,
                )
            )
        finally:
            order_router._replace_manager = orig_replace

        self.assertIsNone(res)
        hints = bot.state.run_context.get("reconcile_hints") or {}
        self.assertIn("BTCUSDT", hints)
        self.assertIn("replace_transition_error", str((hints.get("BTCUSDT") or {}).get("reason") or ""))

    def test_cancel_replace_emits_replace_envelope_block_event(self):
        bot = DummyBot(DummyEx({}), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        captured = []

        async def _fake_emit(_bot, event, data=None, **_kwargs):
            captured.append((str(event), dict(data or {})))

        async def _fake_run_cancel_replace(**_kwargs):
            return types.SimpleNamespace(
                success=False,
                state="INTENT_CREATED",
                reason="replace_envelope_block",
                attempts=0,
                last_status="",
            )

        orig_replace = order_router._replace_manager
        orig_emit = order_router.emit
        order_router._replace_manager = types.SimpleNamespace(run_cancel_replace=_fake_run_cancel_replace)
        order_router.emit = _fake_emit
        try:
            res = asyncio.run(
                order_router.cancel_replace_order(
                    bot,
                    cancel_order_id="oid-1",
                    symbol="BTC/USDT",
                    type="LIMIT",
                    side="buy",
                    amount=1.0,
                    price=100.0,
                    retries=1,
                )
            )
        finally:
            order_router._replace_manager = orig_replace
            order_router.emit = orig_emit

        self.assertIsNone(res)
        events = [e for e, _ in captured]
        self.assertIn("order.replace_envelope_block", events)

    def test_non_retryable_error_aborts_immediately(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}

        def behavior(_kwargs):
            raise Exception("invalid symbol")

        ex = DummyEx(
            {},
            markets={sym_raw: market},
            market_meta=market,
            dual_side=True,
            create_behavior=behavior,
        )
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
        self.assertEqual(len(ex.create_calls), 1)

    def test_coinbase_error_classification(self):
        ex = types.SimpleNamespace(id="coinbase")
        retryable, reason, _code = order_router._classify_order_error(
            Exception("rate limit exceeded"), ex=ex, sym_raw="BTC/USD"
        )
        self.assertTrue(retryable)
        self.assertEqual(reason, "exchange_busy")

        retryable2, reason2, _code2 = order_router._classify_order_error(
            Exception("insufficient funds"), ex=ex, sym_raw="BTC/USD"
        )
        self.assertFalse(retryable2)
        self.assertEqual(reason2, "margin_insufficient")

    def test_exchange_retry_policy_override(self):
        env = dict(os.environ)
        try:
            os.environ["ROUTER_RETRY_POLICY_COINBASE"] = "exchange_busy=9:0.9:9:3"
            bot = DummyBot(DummyEx({}), DummyData({}))
            bot.ex.id = "coinbase"
            pol = order_router._build_retry_policies(bot)
            self.assertEqual(int(pol["exchange_busy"].get("max_attempts", 0)), 9)
            self.assertAlmostEqual(float(pol["exchange_busy"].get("base_delay", 0.0)), 0.9, places=3)
        finally:
            os.environ.clear()
            os.environ.update(env)

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

    def test_create_order_reuses_recent_intent_and_skips_duplicate_submit(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ex = DummyEx({}, markets={sym_raw: market}, market_meta=market, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        with tempfile.TemporaryDirectory() as td:
            bot.cfg.INTENT_LEDGER_ENABLED = True
            bot.cfg.INTENT_LEDGER_REUSE_ENABLED = True
            bot.cfg.INTENT_LEDGER_PATH = str(Path(td) / "intent_ledger.jsonl")
            bot.cfg.INTENT_LEDGER_REUSE_MAX_AGE_SEC = 3600.0

            corr = "ENTRY-BTC-ABC123"
            coid = "ENTRY_BTC_ABC123"
            from eclipse_scalper.execution import intent_ledger  # noqa: E402

            intent_ledger.record(
                bot,
                intent_id=corr,
                stage="OPEN",
                symbol="BTCUSDT",
                side="buy",
                order_type="MARKET",
                client_order_id=coid,
                order_id="OID-EXISTING",
                status="open",
            )

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
                        params={"clientOrderId": coid},
                        retries=1,
                        correlation_id=corr,
                    )
                )
            finally:
                order_router._validate_and_normalize_order = orig_validate

            self.assertIsNotNone(res)
            self.assertEqual((res or {}).get("id"), "OID-EXISTING")
            self.assertEqual(len(ex.create_calls), 0)

    def test_classify_order_error(self):
        retryable, reason, code = order_router._classify_order_error(Exception("Margin is insufficient"))
        self.assertFalse(retryable)
        self.assertEqual(reason, "margin_insufficient")
        self.assertTrue(code.startswith("ERR_"))

        retryable, reason, code = order_router._classify_order_error(Exception("Request timed out"))
        self.assertTrue(retryable)
        self.assertEqual(reason, "network")

        retryable, reason, code = order_router._classify_order_error(Exception("Filter failure: PRICE_FILTER"))
        self.assertFalse(retryable)
        self.assertEqual(reason, "price_filter")

        retryable, reason, code = order_router._classify_order_error(Exception("BinanceError: code=-2019, msg=Margin is insufficient."))
        self.assertFalse(retryable)
        self.assertEqual(reason, "margin_insufficient")

        retryable, reason, code = order_router._classify_order_error(Exception("BinanceError: code=-1021, msg=Timestamp for this request is outside of the recvWindow."))
        self.assertTrue(retryable)
        self.assertEqual(reason, "timestamp")
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

    def test_duplicate_client_order_id_freshens(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        base_id = "SE_DUPLICATE_TEST_0123456789"

        def behavior(kwargs):
            coid = kwargs.get("params", {}).get("clientOrderId")
            if coid == base_id:
                raise Exception("Client order id is duplicated -4116")
            return {"id": "ok", "coid": coid, "params": kwargs}

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
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    client_order_id=base_id,
                    retries=2,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate

        self.assertIsNotNone(res)
        self.assertGreaterEqual(len(ex.create_calls), 3)
        coids = [c["params"].get("clientOrderId") for c in ex.create_calls]
        self.assertIn(base_id, coids)
        fresh = [c for c in coids if c and c != base_id]
        self.assertTrue(len(fresh) >= 1)
        self.assertTrue(len(fresh[-1]) <= 35)

    def test_sanitize_client_order_id_filters_and_length(self):
        raw = "SE DUP!!LICATE--ID___WITH$$$WEIRD%%%CHARS_1234567890"
        s1 = order_router._sanitize_client_order_id(raw)
        s2 = order_router._sanitize_client_order_id(raw)
        self.assertIsNotNone(s1)
        self.assertEqual(s1, s2)
        self.assertTrue(len(s1) <= 35)
        for ch in s1:
            self.assertTrue(ch.isalnum() or ch in ("_", "-"))

    def test_freshen_client_order_id_length_and_diff(self):
        base = "SE_DUPLICATE_TEST_0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        s1 = order_router._freshen_client_order_id(base, salt="a")
        s2 = order_router._freshen_client_order_id(base, salt="b")
        self.assertTrue(len(s1) <= 35)
        self.assertTrue(len(s2) <= 35)
        self.assertNotEqual(s1, s2)
        for ch in s1:
            self.assertTrue(ch.isalnum() or ch in ("_", "-"))

    def test_make_client_order_id_deterministic_and_length_safe(self):
        s1 = order_router._make_client_order_id(
            prefix="ENTRY",
            sym_raw="BTC/USDT:USDT",
            type_norm="market",
            side_l="buy",
            amount=1.2,
            price=100.0,
            stop_price=None,
            bucket=12345,
        )
        s2 = order_router._make_client_order_id(
            prefix="ENTRY",
            sym_raw="BTC/USDT:USDT",
            type_norm="market",
            side_l="buy",
            amount=1.2,
            price=100.0,
            stop_price=None,
            bucket=12345,
        )
        s3 = order_router._make_client_order_id(
            prefix="ENTRY",
            sym_raw="BTC/USDT:USDT",
            type_norm="market",
            side_l="buy",
            amount=1.2,
            price=100.0,
            stop_price=None,
            bucket=12346,
        )
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertLessEqual(len(s1), 35)

    def test_error_class_policy(self):
        p1 = order_router._classify_order_error_policy(Exception("Request timed out"))
        self.assertEqual(p1["error_class"], "retryable")
        p2 = order_router._classify_order_error_policy(Exception("invalid symbol"))
        self.assertEqual(p2["error_class"], "fatal")
        p3 = order_router._classify_order_error_policy(Exception("clientOrderId is duplicated -4116"))
        self.assertEqual(p3["error_class"], "retryable_with_modification")
        p4 = order_router._classify_order_error_policy(Exception("Unknown order -2011"))
        self.assertEqual(p4["error_class"], "idempotent_safe")

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
