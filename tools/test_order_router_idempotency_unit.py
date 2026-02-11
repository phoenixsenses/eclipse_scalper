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


class OrderRouterIdempotencyTests(unittest.TestCase):
    def test_create_order_reuses_recent_intent_and_skips_duplicate_submit(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ex = DummyEx(markets={sym_raw: market}, market_meta=market, dual_side=True)
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

    def test_duplicate_client_order_id_freshens(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        base_id = "SE_DUPLICATE_TEST_0123456789"

        def behavior(kwargs):
            coid = kwargs.get("params", {}).get("clientOrderId")
            if coid == base_id:
                raise Exception("Client order id is duplicated -4116")
            return {"id": "ok", "coid": coid, "params": kwargs}

        ex = DummyEx(create_behavior=behavior, markets={sym_raw: {}}, market_meta={"contract": True}, dual_side=True)
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

    def test_create_order_blocks_duplicate_submit_when_pending_unknown_exists(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ex = DummyEx(markets={sym_raw: market}, market_meta=market, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        with tempfile.TemporaryDirectory() as td:
            bot.cfg.INTENT_LEDGER_ENABLED = True
            bot.cfg.INTENT_LEDGER_REUSE_ENABLED = True
            bot.cfg.INTENT_LEDGER_PATH = str(Path(td) / "intent_ledger.jsonl")
            bot.cfg.INTENT_LEDGER_UNKNOWN_MAX_AGE_SEC = 600.0

            corr = "ENTRY-BTC-UNKNOWN"
            coid = "ENTRY_BTC_UNKNOWN_1"
            from eclipse_scalper.execution import intent_ledger  # noqa: E402

            intent_ledger.record(
                bot,
                intent_id=corr,
                stage="SUBMITTED_UNKNOWN",
                symbol="BTCUSDT",
                side="buy",
                order_type="MARKET",
                client_order_id=coid,
                order_id="",
                status="unknown",
                reason="timeout",
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
            self.assertEqual(len(ex.create_calls), 0)
            info = dict((res or {}).get("info") or {})
            self.assertTrue(bool(info.get("intent_pending_unknown", False)))
            self.assertEqual(str(info.get("stage") or ""), "SUBMITTED_UNKNOWN")

    def test_restart_recovers_pending_unknown_from_journal_and_blocks_duplicate(self):
        sym_raw = "BTC/USDT:USDT"
        sym = "BTC/USDT"
        market = {"contract": True}
        ex = DummyEx(markets={sym_raw: market}, market_meta=market, dual_side=True)
        bot = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))

        with tempfile.TemporaryDirectory() as td:
            ledger_path = Path(td) / "intent_ledger.jsonl"
            journal_path = Path(td) / "execution_journal.jsonl"
            bot.cfg.INTENT_LEDGER_ENABLED = True
            bot.cfg.INTENT_LEDGER_REUSE_ENABLED = True
            bot.cfg.INTENT_LEDGER_PATH = str(ledger_path)
            bot.cfg.EVENT_JOURNAL_PATH = str(journal_path)
            bot.cfg.INTENT_LEDGER_UNKNOWN_MAX_AGE_SEC = 600.0

            corr = "ENTRY-BTC-RESTART-UNK"
            coid = "ENTRY_BTC_RESTART_UNK_1"
            from eclipse_scalper.execution import intent_ledger  # noqa: E402

            intent_ledger.record(
                bot,
                intent_id=corr,
                stage="SUBMITTED_UNKNOWN",
                symbol="BTCUSDT",
                side="buy",
                order_type="MARKET",
                client_order_id=coid,
                status="unknown",
                reason="timeout",
            )
            self.assertTrue(journal_path.exists())
            if ledger_path.exists():
                ledger_path.unlink()

            # fresh runtime process with empty run_context; must recover from journal mirror
            bot2 = DummyBot(ex, DummyData({"BTCUSDT": sym_raw}))
            bot2.cfg.INTENT_LEDGER_ENABLED = True
            bot2.cfg.INTENT_LEDGER_REUSE_ENABLED = True
            bot2.cfg.INTENT_LEDGER_PATH = str(ledger_path)
            bot2.cfg.EVENT_JOURNAL_PATH = str(journal_path)
            bot2.cfg.INTENT_LEDGER_UNKNOWN_MAX_AGE_SEC = 600.0

            orig_validate = order_router._validate_and_normalize_order

            async def _ok_validate(*_args, **_kwargs):
                return True, 1.0, None, "ok"

            order_router._validate_and_normalize_order = _ok_validate
            try:
                res = asyncio.run(
                    order_router.create_order(
                        bot2,
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
            self.assertEqual(len(ex.create_calls), 0)
            info = dict((res or {}).get("info") or {})
            self.assertTrue(bool(info.get("intent_pending_unknown", False)))

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


if __name__ == "__main__":
    unittest.main()
