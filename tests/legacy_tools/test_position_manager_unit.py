#!/usr/bin/env python3
"""
Unit-style tests for position_manager helpers.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from types import SimpleNamespace
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

from eclipse_scalper.execution import position_manager  # noqa: E402
from eclipse_scalper.execution import exit as exit_mod  # noqa: E402
from eclipse_scalper.execution import reconcile  # noqa: E402


class DummyEx:
    def __init__(self):
        self.calls = 0

    async def fetch_open_orders(self, _sym_raw):
        self.calls += 1
        return [{"id": str(self.calls)}]


class DummyBot:
    def __init__(self, ex):
        self.ex = ex


class PositionManagerTests(unittest.TestCase):
    def test_stable_stop_client_id_base_no_entry_ts(self):
        out1 = position_manager._stable_stop_client_id_base(
            k="BTCUSDT",
            hedge_hint="LONG",
            entry_ts=0.0,
            entry_px=100.0,
            size=1.5,
        )
        out2 = position_manager._stable_stop_client_id_base(
            k="BTCUSDT",
            hedge_hint="LONG",
            entry_ts=0.0,
            entry_px=100.0,
            size=1.5,
        )
        self.assertEqual(out1, out2)

    def test_open_orders_cache_invalidates_on_sym_change(self):
        position_manager._OPEN_ORDERS_CACHE.clear()
        ex = DummyEx()
        bot = DummyBot(ex)

        asyncio.run(
            position_manager._fetch_open_orders_best_effort(
                bot,
                "BTC/USDT:USDT",
                min_interval_sec=999.0,
                cache_key="BTCUSDT",
            )
        )
        asyncio.run(
            position_manager._fetch_open_orders_best_effort(
                bot,
                "BTC/USDT",
                min_interval_sec=999.0,
                cache_key="BTCUSDT",
            )
        )
        self.assertEqual(ex.calls, 2)

    def test_find_existing_stop_accepts_opposite_side(self):
        orders = [
            {"id": "1", "type": "stop_market", "side": "sell", "stopPrice": 100.0},
        ]
        oid, sp = position_manager._find_existing_stop(orders, "long")
        self.assertEqual(oid, "1")
        self.assertEqual(sp, 100.0)

    def test_exit_symkey_trims_usdtusdt(self):
        self.assertEqual(exit_mod._symkey("BTC/USDT:USDT"), "BTCUSDT")
        self.assertEqual(exit_mod._symkey("BTCUSDTUSDT"), "BTCUSDT")

    def test_reconcile_resolve_raw_symbol_prefers_data_map(self):
        class DummyData:
            def __init__(self):
                self.raw_symbol = {"BTCUSDT": "BTC/USDT:USDT"}

        class DummyEx:
            options = {"defaultType": "future"}

        class DummyBot:
            def __init__(self):
                self.data = DummyData()
                self.ex = DummyEx()

        bot = DummyBot()
        raw = reconcile._resolve_raw_symbol(bot, "BTCUSDT")
        self.assertEqual(raw, "BTC/USDT:USDT")

    def test_reconcile_phantom_grace_blocks_early_clear(self):
        class DummyEx:
            async def fetch_positions(self, *_args, **_kwargs):
                return []

        class DummyState:
            def __init__(self):
                self.positions = {"BTCUSDT": SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0)}
                self.run_context = {}

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = DummyState()
                self._shutdown = asyncio.Event()
                self.active_symbols = {"BTCUSDT"}
                self.cfg = SimpleNamespace(
                    RECONCILE_FULL_SCAN_ORPHANS=True,
                    RECONCILE_PHANTOM_MISS_COUNT=2,
                    RECONCILE_PHANTOM_GRACE_SEC=999.0,
                )

        bot = DummyBot()
        asyncio.run(reconcile.reconcile_tick(bot))
        self.assertIn("BTCUSDT", bot.state.positions)

    def test_posmgr_does_not_recreate_stop_with_recent_id(self):
        class DummyData:
            def __init__(self):
                self.price = {"DOGEUSDT": 0.1}
                self.raw_symbol = {"DOGEUSDT": "DOGE/USDT:USDT"}

            def get_price(self, k, in_position=False):
                return 0.1

        class DummyState:
            def __init__(self):
                self.positions = {
                    "DOGEUSDT": SimpleNamespace(
                        side="long",
                        size=50.0,
                        entry_price=0.1,
                        atr=0.001,
                        entry_ts=0.0,
                        hard_stop_order_id="abc123",
                        hard_stop_order_ts=position_manager._now(),
                    )
                }
                self.symbol_performance = {}

        class DummyBot:
            def __init__(self):
                self.ex = None
                self.data = DummyData()
                self.state = DummyState()
                self.cfg = SimpleNamespace(
                    POSMGR_ENABLED=True,
                    POSMGR_STOP_CHECK_SEC=0.0,
                    POSMGR_STOP_RESTORE_COOLDOWN_SEC=300.0,
                    POSMGR_OPEN_ORDERS_MIN_INTERVAL_SEC=0.0,
                )

        bot = DummyBot()
        calls = {"stops": 0}

        async def _fake_fetch_open_orders_best_effort(*_args, **_kwargs):
            return []

        async def _fake_place_stop_ladder_router(*_args, **_kwargs):
            calls["stops"] += 1
            return "newstop"

        # Patch
        orig_fetch = position_manager._fetch_open_orders_best_effort
        orig_place = position_manager._place_stop_ladder_router
        position_manager._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        position_manager._place_stop_ladder_router = _fake_place_stop_ladder_router
        try:
            asyncio.run(position_manager.position_manager_tick(bot))
        finally:
            position_manager._fetch_open_orders_best_effort = orig_fetch
            position_manager._place_stop_ladder_router = orig_place

        self.assertEqual(calls["stops"], 0)


if __name__ == "__main__":
    unittest.main()
