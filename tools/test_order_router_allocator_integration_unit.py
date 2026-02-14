#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import tempfile
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
from eclipse_scalper.execution import intent_ledger  # noqa: E402


class DummyEx:
    def __init__(self):
        self.id = "binance"
        self.markets = {"BTC/USDT:USDT": {"contract": True}}
        self.create_calls = []
        self.cancel_calls = []

    async def create_order(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"id": "OID-1", "status": "open", "info": {"orderId": "OID-1"}}

    async def cancel_order(self, order_id, symbol):
        self.cancel_calls.append((order_id, symbol))
        return {"id": order_id}

    async def fapiPrivateGetPositionSideDual(self):
        return {"dualSidePosition": True}

    def market(self, _sym):
        return {"contract": True}


class FakeAllocator:
    def __init__(self):
        self.allocate_calls = []
        self.derive_calls = []

    def allocate_intent_id(self, bot, *, component, intent_kind, symbol, is_exit=False):
        self.allocate_calls.append(
            {
                "component": component,
                "intent_kind": intent_kind,
                "symbol": symbol,
                "is_exit": bool(is_exit),
            }
        )
        return "I2.PHX01.1739318400.order_router_create.ENTRY.BTCUSDT.00000001"

    def derive_client_order_id(self, *, intent_id, prefix, symbol, max_len=35):
        self.derive_calls.append(
            {
                "intent_id": intent_id,
                "prefix": prefix,
                "symbol": symbol,
                "max_len": max_len,
            }
        )
        return "ENTRY_BTC_1234567890ABCDEF"


class RouterAllocatorIntegrationTests(unittest.TestCase):
    def _bot(self, ex, ledger_path: Path):
        journal_path = ledger_path.parent / "execution_journal.jsonl"
        return types.SimpleNamespace(
            ex=ex,
            data=types.SimpleNamespace(raw_symbol={"BTCUSDT": "BTC/USDT:USDT"}),
            cfg=types.SimpleNamespace(
                INTENT_LEDGER_ENABLED=True,
                INTENT_LEDGER_MAY_SEND_ENABLED=True,
                INTENT_LEDGER_REUSE_ENABLED=False,
                INTENT_LEDGER_PATH=str(ledger_path),
                EVENT_JOURNAL_PATH=str(journal_path),
                ROUTER_AUTO_CLIENT_ID=True,
            ),
            state=types.SimpleNamespace(positions={}, current_equity=0.0, run_context={}),
        )

    def test_create_order_uses_allocator_for_default_ids(self):
        ex = DummyEx()
        with tempfile.TemporaryDirectory() as td:
            bot = self._bot(ex, Path(td) / "intent_ledger.jsonl")
            fake_alloc = FakeAllocator()
            orig_alloc = order_router._intent_allocator
            orig_validate = order_router._validate_and_normalize_order
            orig_is_dry = order_router._is_dry_run

            async def _ok_validate(*_args, **_kwargs):
                return True, 1.0, None, "ok"

            order_router._intent_allocator = fake_alloc
            order_router._validate_and_normalize_order = _ok_validate
            order_router._is_dry_run = lambda _bot: False
            try:
                res = asyncio.run(
                    order_router.create_order(
                        bot,
                        symbol="BTC/USDT",
                        type="MARKET",
                        side="buy",
                        amount=1.0,
                        auto_client_order_id=True,
                        retries=1,
                    )
                )
            finally:
                order_router._intent_allocator = orig_alloc
                order_router._validate_and_normalize_order = orig_validate
                order_router._is_dry_run = orig_is_dry

            self.assertIsNotNone(res)
            self.assertEqual(len(fake_alloc.allocate_calls), 1)
            self.assertEqual(len(fake_alloc.derive_calls), 1)
            self.assertEqual(len(ex.create_calls), 1)
            params = ex.create_calls[0].get("params", {}) or {}
            self.assertEqual(params.get("clientOrderId"), "ENTRY_BTC_1234567890ABCDEF")

    def test_cancel_order_uses_allocator_when_no_correlation(self):
        ex = DummyEx()
        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            bot = self._bot(ex, ledger_path)
            fake_alloc = FakeAllocator()
            orig_alloc = order_router._intent_allocator
            order_router._intent_allocator = fake_alloc
            try:
                ok = asyncio.run(order_router.cancel_order(bot, "OID-CANCEL-1", "BTC/USDT"))
            finally:
                order_router._intent_allocator = orig_alloc

            self.assertTrue(ok)
            self.assertEqual(len(ex.cancel_calls), 1)
            summary = intent_ledger.summary(bot)
            self.assertGreaterEqual(int(summary.get("intents_total", 0)), 1)

    def test_create_order_may_send_blocks_duplicate_intent(self):
        ex = DummyEx()
        with tempfile.TemporaryDirectory() as td:
            bot = self._bot(ex, Path(td) / "intent_ledger.jsonl")
            fake_alloc = FakeAllocator()
            orig_alloc = order_router._intent_allocator
            orig_validate = order_router._validate_and_normalize_order
            orig_is_dry = order_router._is_dry_run

            async def _ok_validate(*_args, **_kwargs):
                return True, 1.0, None, "ok"

            order_router._intent_allocator = fake_alloc
            order_router._validate_and_normalize_order = _ok_validate
            order_router._is_dry_run = lambda _bot: False
            try:
                r1 = asyncio.run(
                    order_router.create_order(
                        bot,
                        symbol="BTC/USDT",
                        type="MARKET",
                        side="buy",
                        amount=1.0,
                        auto_client_order_id=True,
                        retries=1,
                    )
                )
                r2 = asyncio.run(
                    order_router.create_order(
                        bot,
                        symbol="BTC/USDT",
                        type="MARKET",
                        side="buy",
                        amount=1.0,
                        auto_client_order_id=True,
                        retries=1,
                        correlation_id="I2.PHX01.1739318400.order_router_create.ENTRY.BTCUSDT.00000001",
                    )
                )
            finally:
                order_router._intent_allocator = orig_alloc
                order_router._validate_and_normalize_order = orig_validate
                order_router._is_dry_run = orig_is_dry

            self.assertIsNotNone(r1)
            self.assertIsNotNone(r2)
            self.assertEqual(len(ex.create_calls), 1)
            self.assertTrue(bool((r2 or {}).get("info", {}).get("intent_blocked", False)))

    def test_create_order_blocks_when_allocator_missing_for_risk_increase(self):
        ex = DummyEx()
        with tempfile.TemporaryDirectory() as td:
            bot = self._bot(ex, Path(td) / "intent_ledger.jsonl")
            bot.cfg.INTENT_ALLOCATOR_REQUIRED = True
            orig_alloc = order_router._intent_allocator
            orig_validate = order_router._validate_and_normalize_order
            orig_is_dry = order_router._is_dry_run

            async def _ok_validate(*_args, **_kwargs):
                return True, 1.0, None, "ok"

            order_router._intent_allocator = None
            order_router._validate_and_normalize_order = _ok_validate
            order_router._is_dry_run = lambda _bot: False
            try:
                res = asyncio.run(
                    order_router.create_order(
                        bot,
                        symbol="BTC/USDT",
                        type="MARKET",
                        side="buy",
                        amount=1.0,
                        auto_client_order_id=True,
                        retries=1,
                    )
                )
            finally:
                order_router._intent_allocator = orig_alloc
                order_router._validate_and_normalize_order = orig_validate
                order_router._is_dry_run = orig_is_dry

            self.assertEqual(len(ex.create_calls), 0)
            self.assertTrue(bool((res or {}).get("info", {}).get("intent_blocked", False)))
            self.assertEqual(str((res or {}).get("info", {}).get("reason") or ""), "allocator_unavailable")

    def test_create_order_blocks_when_may_send_gate_disabled_in_strict_mode(self):
        ex = DummyEx()
        with tempfile.TemporaryDirectory() as td:
            bot = self._bot(ex, Path(td) / "intent_ledger.jsonl")
            bot.cfg.INTENT_LEDGER_MAY_SEND_ENABLED = False
            bot.cfg.INTENT_LEDGER_STRICT_MAY_SEND = True
            fake_alloc = FakeAllocator()
            orig_alloc = order_router._intent_allocator
            orig_validate = order_router._validate_and_normalize_order
            orig_is_dry = order_router._is_dry_run

            async def _ok_validate(*_args, **_kwargs):
                return True, 1.0, None, "ok"

            order_router._intent_allocator = fake_alloc
            order_router._validate_and_normalize_order = _ok_validate
            order_router._is_dry_run = lambda _bot: False
            try:
                res = asyncio.run(
                    order_router.create_order(
                        bot,
                        symbol="BTC/USDT",
                        type="MARKET",
                        side="buy",
                        amount=1.0,
                        auto_client_order_id=True,
                        retries=1,
                    )
                )
            finally:
                order_router._intent_allocator = orig_alloc
                order_router._validate_and_normalize_order = orig_validate
                order_router._is_dry_run = orig_is_dry

            self.assertEqual(len(ex.create_calls), 0)
            self.assertTrue(bool((res or {}).get("info", {}).get("intent_blocked", False)))
            self.assertEqual(str((res or {}).get("info", {}).get("reason") or ""), "may_send_disabled")


if __name__ == "__main__":
    unittest.main()
