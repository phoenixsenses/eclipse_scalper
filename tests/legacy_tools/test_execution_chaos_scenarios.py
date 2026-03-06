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

from eclipse_scalper.execution import belief_evidence, order_router, reconcile  # noqa: E402


class SequenceExchange:
    def __init__(self, create_sequence):
        self.id = "binance"
        self.create_sequence = list(create_sequence)
        self.create_calls = []
        self.markets = {"BTC/USDT:USDT": {}}

    async def create_order(self, **kwargs):
        self.create_calls.append(kwargs)
        if not self.create_sequence:
            return {"id": "fallback", "status": "open"}
        action = self.create_sequence.pop(0)
        if isinstance(action, Exception):
            raise action
        return dict(action)

    async def cancel_order(self, _order_id, _symbol):
        return {"id": "ok"}

    async def fetch_order(self, _order_id, _symbol):
        return {"status": "closed"}

    async def load_markets(self, _reload=True):
        return self.markets

    def market(self, _sym):
        return {"contract": True}

    async def fapiPrivateGetPositionSideDual(self):
        return {"dualSidePosition": True}


class AlwaysFailExchange(SequenceExchange):
    async def create_order(self, **kwargs):
        self.create_calls.append(kwargs)
        raise Exception("Request timed out")


class CancelUnknownExchange:
    def __init__(self):
        self.id = "binance"

    async def cancel_order(self, _order_id, _symbol):
        raise Exception("Unknown order -2011")

    async def fetch_order(self, _order_id, _symbol):
        return {"status": "closed"}


class DummyData:
    def __init__(self, raw_symbol):
        self.raw_symbol = raw_symbol


class DummyState:
    def __init__(self):
        self.positions = {}
        self.current_equity = 1000.0
        self.reconcile_metrics = {}
        self.kill_metrics = {}
        self.run_context = {}


class DummyBot:
    def __init__(self, ex, raw_map=None):
        self.ex = ex
        self.data = DummyData(raw_map or {"BTCUSDT": "BTC/USDT:USDT"})
        self.cfg = types.SimpleNamespace(
            RECONCILE_FULL_SCAN_ORPHANS=False,
            RECONCILE_PHANTOM_MISS_COUNT=1,
            RECONCILE_PHANTOM_GRACE_SEC=0.0,
        )
        self.state = DummyState()
        self._shutdown = asyncio.Event()
        self.active_symbols = {"BTCUSDT"}


class ExecutionChaosScenarioTests(unittest.TestCase):
    def test_router_recovers_timeout_and_duplicate_then_returns_partial_fill(self):
        env = dict(os.environ)
        ex = SequenceExchange(
            [
                Exception("Request timed out"),
                Exception("clientOrderId is duplicated -4116"),
                {"id": "oid-3", "status": "partially_filled"},
            ]
        )
        bot = DummyBot(ex)
        bot.cfg.ROUTER_RETRY_BASE_SEC = 0.01
        bot.cfg.ROUTER_RETRY_MAX_DELAY_SEC = 0.01
        bot.cfg.ROUTER_RETRY_JITTER_PCT = 0.0
        bot.cfg.ROUTER_RETRY_MAX_ELAPSED_SEC = 5.0

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        async def _noop_sleep(_delay):
            return None

        orig_validate = order_router._validate_and_normalize_order
        orig_sleep = order_router.asyncio.sleep
        try:
            os.environ["ROUTER_BINANCE_AUTO_CLIENT_ID"] = "1"
            order_router._validate_and_normalize_order = _ok_validate
            order_router.asyncio.sleep = _noop_sleep
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol="BTC/USDT",
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    retries=5,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate
            order_router.asyncio.sleep = orig_sleep
            os.environ.clear()
            os.environ.update(env)

        self.assertIsNotNone(res)
        self.assertEqual(str((res or {}).get("status")), "partially_filled")
        self.assertEqual(len(ex.create_calls), 3)
        for call in ex.create_calls:
            params = call.get("params") or {}
            cid = str(params.get("clientOrderId") or "")
            if cid:
                self.assertLessEqual(len(cid), 35)

    def test_cancel_after_fill_unknown_order_is_idempotent_success(self):
        bot = DummyBot(CancelUnknownExchange())
        ok = asyncio.run(order_router.cancel_order(bot, "oid-1", "BTC/USDT"))
        self.assertTrue(ok)

    def test_reconcile_contradictory_snapshot_escalates_belief_and_clears_phantom(self):
        class EmptyPositionsEx:
            id = "binance"

            async def fetch_positions(self, *_args, **_kwargs):
                return []

        bot = DummyBot(EmptyPositionsEx())
        bot.state.positions["BTCUSDT"] = types.SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0)

        asyncio.run(reconcile.reconcile_tick(bot))

        self.assertNotIn("BTCUSDT", bot.state.positions)
        rm = bot.state.reconcile_metrics
        self.assertGreater(int(rm.get("mismatch_streak", 0) or 0), 0)
        self.assertIsInstance(getattr(bot.state, "guard_knobs", {}), dict)
        self.assertIn("allow_entries", bot.state.guard_knobs)

    def test_retry_storm_pressure_is_bounded_by_router_attempts(self):
        env = dict(os.environ)
        ex = AlwaysFailExchange([])
        bot = DummyBot(ex)
        bot.cfg.ROUTER_RETRY_BASE_SEC = 0.01
        bot.cfg.ROUTER_RETRY_MAX_DELAY_SEC = 0.01
        bot.cfg.ROUTER_RETRY_JITTER_PCT = 0.0
        bot.cfg.ROUTER_RETRY_MAX_ELAPSED_SEC = 0.05

        async def _ok_validate(*_args, **_kwargs):
            return True, 1.0, None, "ok"

        async def _noop_sleep(_delay):
            return None

        orig_validate = order_router._validate_and_normalize_order
        orig_sleep = order_router.asyncio.sleep
        try:
            order_router._validate_and_normalize_order = _ok_validate
            order_router.asyncio.sleep = _noop_sleep
            res = asyncio.run(
                order_router.create_order(
                    bot,
                    symbol="BTC/USDT",
                    type="MARKET",
                    side="buy",
                    amount=1.0,
                    retries=5,
                )
            )
        finally:
            order_router._validate_and_normalize_order = orig_validate
            order_router.asyncio.sleep = orig_sleep
            os.environ.clear()
            os.environ.update(env)

        self.assertIsNone(res)
        # Even under persistent failure, attempts remain bounded (no infinite loop).
        self.assertLessEqual(len(ex.create_calls), 50)

    def test_ws_down_rest_healthy_degrades_without_collapsing_confidence(self):
        now = 1700000000.0
        bot = types.SimpleNamespace(
            state=types.SimpleNamespace(run_context={"ws_last_event_ts": now - 300.0, "rest_last_ok_ts": now - 2.0}, kill_metrics={}),
            ex=types.SimpleNamespace(last_health_check=now - 2.0),
            data=types.SimpleNamespace(last_poll_ts={"BTCUSDT": now - 2.0}),
        )
        out = belief_evidence.compute_belief_evidence(bot, types.SimpleNamespace(), now=now)
        self.assertLess(float(out.get("evidence_ws_score", 1.0)), 0.7)
        self.assertGreater(float(out.get("evidence_rest_score", 0.0)), 0.9)
        self.assertGreater(float(out.get("evidence_confidence", 0.0)), 0.3)

    def test_rest_stale_and_ws_lag_drive_high_degradation(self):
        now = 1700000000.0
        bot = types.SimpleNamespace(
            state=types.SimpleNamespace(
                run_context={
                    "ws_last_event_ts": now - 200.0,
                    "rest_last_ok_ts": now - 500.0,
                    "fills_last_ts": now - 500.0,
                },
                kill_metrics={},
            ),
            ex=types.SimpleNamespace(last_health_check=now - 500.0),
            data=types.SimpleNamespace(last_poll_ts={"BTCUSDT": now - 500.0}),
        )
        out = belief_evidence.compute_belief_evidence(bot, types.SimpleNamespace(), now=now)
        self.assertLess(float(out.get("evidence_confidence", 1.0)), 0.3)
        self.assertGreaterEqual(int(out.get("evidence_degraded_sources", 0)), 2)


if __name__ == "__main__":
    unittest.main()
