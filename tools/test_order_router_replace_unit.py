#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import time
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
    def __init__(self):
        self.id = "binance"


class DummyData:
    def __init__(self, raw_symbol):
        self.raw_symbol = raw_symbol


class DummyBot:
    def __init__(self, ex, data):
        self.ex = ex
        self.data = data
        self.cfg = types.SimpleNamespace()
        self.state = types.SimpleNamespace(positions={}, current_equity=0.0, run_context={})


class OrderRouterReplaceTests(unittest.TestCase):
    def test_cancel_replace_is_bounded_when_cancel_fails(self):
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
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
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        seen = {
            "strict": None,
            "max_worst_case_notional": None,
            "max_ambiguity_attempts": None,
            "verify_cancel_with_status": None,
        }

        async def _fake_run_cancel_replace(**kwargs):
            seen["strict"] = bool(kwargs.get("strict_transitions"))
            seen["max_worst_case_notional"] = float(kwargs.get("max_worst_case_notional") or 0.0)
            seen["max_ambiguity_attempts"] = int(kwargs.get("max_ambiguity_attempts") or 0)
            seen["verify_cancel_with_status"] = bool(kwargs.get("verify_cancel_with_status"))
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
        self.assertTrue(bool(seen["verify_cancel_with_status"]))

    def test_cancel_replace_transition_error_requests_reconcile_hint(self):
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
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
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
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

    def test_cancel_replace_reconcile_required_records_reconcile_first_pressure(self):
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        bot.state.kill_metrics = {}

        async def _fake_run_cancel_replace(**_kwargs):
            return types.SimpleNamespace(
                success=False,
                state="REPLACE_RACE",
                reason="replace_reconcile_required",
                attempts=2,
                ambiguity_count=2,
                last_status="unknown",
            )

        orig_replace = order_router._replace_manager
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

        self.assertIsNone(res)
        km = dict(getattr(bot.state, "kill_metrics", {}) or {})
        self.assertGreaterEqual(int(km.get("reconcile_first_gate_count", 0) or 0), 1)
        self.assertIn("replace_race", str(km.get("reconcile_first_gate_last_reason", "") or ""))
        events = list(km.get("reconcile_first_gate_events", []) or [])
        self.assertGreaterEqual(len(events), 1)

    def test_cancel_replace_budget_block_emits_and_hints(self):
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        bot.state.run_context = {
            "replace_budget": {
                "global": [time.time()],
                "by_symbol": {"BTCUSDT": [time.time()]},
            }
        }
        os.environ["ROUTER_REPLACE_BUDGET_WINDOW_SEC"] = "300"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_GLOBAL"] = "1"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_PER_SYMBOL"] = "1"
        captured = []

        async def _fake_emit(_bot, event, data=None, **_kwargs):
            captured.append((str(event), dict(data or {})))

        orig_emit = order_router.emit
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
            order_router.emit = orig_emit
            os.environ.pop("ROUTER_REPLACE_BUDGET_WINDOW_SEC", None)
            os.environ.pop("ROUTER_REPLACE_BUDGET_MAX_GLOBAL", None)
            os.environ.pop("ROUTER_REPLACE_BUDGET_MAX_PER_SYMBOL", None)

        self.assertIsNone(res)
        self.assertTrue(any(ev == "order.replace_budget_block" for ev, _ in captured))
        hints = bot.state.run_context.get("reconcile_hints") or {}
        self.assertIn("BTCUSDT", hints)
        self.assertIn("replace_budget_exceeded", str((hints.get("BTCUSDT") or {}).get("reason") or ""))

    def test_cancel_replace_budget_block_on_symbol_ambiguity_rate(self):
        bot = DummyBot(DummyEx(), DummyData({"BTCUSDT": "BTC/USDT:USDT"}))
        now = time.time()
        bot.state.run_context = {
            "replace_budget": {
                "global": [],
                "by_symbol": {"BTCUSDT": []},
                "ambiguity_by_symbol": {"BTCUSDT": [now, now - 1.0]},
            }
        }
        os.environ["ROUTER_REPLACE_BUDGET_WINDOW_SEC"] = "300"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_GLOBAL"] = "10"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_PER_SYMBOL"] = "10"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_AMBIGUITY_PER_SYMBOL"] = "2"
        captured = []

        async def _fake_emit(_bot, event, data=None, **_kwargs):
            captured.append((str(event), dict(data or {})))

        orig_emit = order_router.emit
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
            order_router.emit = orig_emit
            os.environ.pop("ROUTER_REPLACE_BUDGET_WINDOW_SEC", None)
            os.environ.pop("ROUTER_REPLACE_BUDGET_MAX_GLOBAL", None)
            os.environ.pop("ROUTER_REPLACE_BUDGET_MAX_PER_SYMBOL", None)
            os.environ.pop("ROUTER_REPLACE_BUDGET_MAX_AMBIGUITY_PER_SYMBOL", None)

        self.assertIsNone(res)
        self.assertTrue(any(ev == "order.replace_budget_block" for ev, _ in captured))
        payloads = [d for ev, d in captured if ev == "order.replace_budget_block"]
        self.assertTrue(payloads)
        self.assertTrue(bool(payloads[0].get("over_ambiguity_symbol")))


if __name__ == "__main__":
    unittest.main()
