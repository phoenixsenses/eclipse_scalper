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

    def test_reconcile_tracks_mismatch_streak_for_kill_switch(self):
        class DummyEx:
            async def fetch_positions(self, *_args, **_kwargs):
                return []

        class DummyState:
            def __init__(self):
                self.positions = {"BTCUSDT": SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0)}
                self.run_context = {}
                self.kill_metrics = {}

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = DummyState()
                self._shutdown = asyncio.Event()
                self.active_symbols = {"BTCUSDT"}
                self.cfg = SimpleNamespace(
                    RECONCILE_FULL_SCAN_ORPHANS=True,
                    RECONCILE_PHANTOM_MISS_COUNT=1,
                    RECONCILE_PHANTOM_GRACE_SEC=0.0,
                )

        bot = DummyBot()
        asyncio.run(reconcile.reconcile_tick(bot))
        self.assertGreaterEqual(int(bot.state.kill_metrics.get("reconcile_mismatch_streak", 0) or 0), 1)

    def test_reconcile_stop_repair_cooldown_prevents_thrashing(self):
        class DummyEx:
            pass

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = SimpleNamespace(
                    positions={},
                    run_context={"protection_refresh": {"BTCUSDT": {"qty": 1.0, "ts": reconcile._now()}}},
                    reconcile_metrics={},
                )
                self.cfg = SimpleNamespace(
                    GUARDIAN_ENSURE_STOP=True,
                    GUARDIAN_RESPECT_KILL_SWITCH=False,
                    RECONCILE_STOP_THROTTLE_SEC=0.0,
                    RECONCILE_REPAIR_COOLDOWN_SEC=999.0,
                    STOP_ATR_MULT=1.0,
                    GUARDIAN_STOP_BUFFER_ATR_MULT=0.0,
                )

        pos = SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=1.0)
        bot = DummyBot()
        calls = {"place": 0}

        async def _fake_fetch_open_orders_best_effort(*_args, **_kwargs):
            return []

        async def _fake_place_stop_ladder_router(*_args, **_kwargs):
            calls["place"] += 1
            return f"oid-{calls['place']}"

        orig_fetch = reconcile._fetch_open_orders_best_effort
        orig_place = reconcile._place_stop_ladder_router
        orig_halt = reconcile.is_halted
        reconcile._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        reconcile._place_stop_ladder_router = _fake_place_stop_ladder_router
        reconcile.is_halted = lambda _bot: False
        try:
            out1 = asyncio.run(reconcile._ensure_protective_stop(bot, "BTCUSDT", pos, "long", 1.0))
            out2 = asyncio.run(reconcile._ensure_protective_stop(bot, "BTCUSDT", pos, "long", 1.0))
        finally:
            reconcile._fetch_open_orders_best_effort = orig_fetch
            reconcile._place_stop_ladder_router = orig_place
            reconcile.is_halted = orig_halt

        self.assertEqual(out1, "restored")
        self.assertEqual(out2, "repair_cooldown")
        self.assertEqual(calls["place"], 1)

    def test_reconcile_stop_refresh_deferred_avoids_duplicate_stop(self):
        class DummyEx:
            pass

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = SimpleNamespace(
                    positions={},
                    run_context={"protection_refresh": {"BTCUSDT": {"qty": 1.0, "ts": reconcile._now()}}},
                    reconcile_metrics={},
                )
                self.cfg = SimpleNamespace(
                    GUARDIAN_ENSURE_STOP=True,
                    GUARDIAN_RESPECT_KILL_SWITCH=False,
                    RECONCILE_STOP_THROTTLE_SEC=0.0,
                    RECONCILE_REPAIR_COOLDOWN_SEC=0.0,
                    STOP_ATR_MULT=1.0,
                    GUARDIAN_STOP_BUFFER_ATR_MULT=0.0,
                    RECONCILE_STOP_MIN_COVERAGE_RATIO=0.98,
                    RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO=0.20,
                    RECONCILE_STOP_REFRESH_MIN_DELTA_ABS=0.20,
                    RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC=300.0,
                    RECONCILE_STOP_REFRESH_FORCE_COVERAGE_RATIO=0.80,
                )

        pos = SimpleNamespace(side="long", size=1.05, entry_price=100.0, atr=1.0)
        bot = DummyBot()
        calls = {"place": 0, "replace": 0}

        async def _fake_fetch_open_orders_best_effort(*_args, **_kwargs):
            return [{"id": "stop-1", "type": "STOP_MARKET", "params": {"reduceOnly": True}, "amount": 1.0}]

        async def _fake_place_stop_ladder_router(*_args, **_kwargs):
            calls["place"] += 1
            return "new-stop"

        async def _fake_cancel_replace(*_args, **_kwargs):
            calls["replace"] += 1
            return {"id": "replaced-stop"}

        orig_fetch = reconcile._fetch_open_orders_best_effort
        orig_place = reconcile._place_stop_ladder_router
        orig_replace = reconcile.cancel_replace_order
        orig_halt = reconcile.is_halted
        reconcile._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        reconcile._place_stop_ladder_router = _fake_place_stop_ladder_router
        reconcile.cancel_replace_order = _fake_cancel_replace
        reconcile.is_halted = lambda _bot: False
        try:
            out = asyncio.run(reconcile._ensure_protective_stop(bot, "BTCUSDT", pos, "long", 1.05))
        finally:
            reconcile._fetch_open_orders_best_effort = orig_fetch
            reconcile._place_stop_ladder_router = orig_place
            reconcile.cancel_replace_order = orig_replace
            reconcile.is_halted = orig_halt

        self.assertEqual(out, "refresh_deferred")
        self.assertEqual(calls["replace"], 0)
        self.assertEqual(calls["place"], 0)

    def test_reconcile_stop_refresh_force_ratio_triggers_replace(self):
        class DummyEx:
            pass

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = SimpleNamespace(
                    positions={},
                    run_context={},
                    reconcile_metrics={},
                )
                self.cfg = SimpleNamespace(
                    GUARDIAN_ENSURE_STOP=True,
                    GUARDIAN_RESPECT_KILL_SWITCH=False,
                    RECONCILE_STOP_THROTTLE_SEC=0.0,
                    RECONCILE_REPAIR_COOLDOWN_SEC=0.0,
                    STOP_ATR_MULT=1.0,
                    GUARDIAN_STOP_BUFFER_ATR_MULT=0.0,
                    RECONCILE_STOP_MIN_COVERAGE_RATIO=0.98,
                    RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO=0.50,
                    RECONCILE_STOP_REFRESH_MIN_DELTA_ABS=1.0,
                    RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC=300.0,
                    RECONCILE_STOP_REFRESH_FORCE_COVERAGE_RATIO=0.80,
                )

        pos = SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=1.0)
        bot = DummyBot()
        calls = {"replace": 0}

        async def _fake_fetch_open_orders_best_effort(*_args, **_kwargs):
            return [{"id": "stop-1", "type": "STOP_MARKET", "params": {"reduceOnly": True}, "amount": 0.4}]

        async def _fake_cancel_replace(*_args, **_kwargs):
            calls["replace"] += 1
            return {"id": "stop-2"}

        orig_fetch = reconcile._fetch_open_orders_best_effort
        orig_replace = reconcile.cancel_replace_order
        orig_halt = reconcile.is_halted
        reconcile._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        reconcile.cancel_replace_order = _fake_cancel_replace
        reconcile.is_halted = lambda _bot: False
        try:
            out = asyncio.run(reconcile._ensure_protective_stop(bot, "BTCUSDT", pos, "long", 1.0))
        finally:
            reconcile._fetch_open_orders_best_effort = orig_fetch
            reconcile.cancel_replace_order = orig_replace
            reconcile.is_halted = orig_halt

        self.assertEqual(out, "restored")
        self.assertEqual(calls["replace"], 1)

    def test_reconcile_tp_refresh_deferred_avoids_duplicate_tp(self):
        class DummyEx:
            pass

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = SimpleNamespace(
                    positions={},
                    run_context={"protection_refresh_tp": {"BTCUSDT": {"qty": 1.0, "ts": reconcile._now()}}},
                    reconcile_metrics={},
                )
                self.cfg = SimpleNamespace(
                    GUARDIAN_ENSURE_TP=True,
                    RECONCILE_TP_MIN_COVERAGE_RATIO=0.98,
                    RECONCILE_TP_REFRESH_MIN_DELTA_RATIO=0.20,
                    RECONCILE_TP_REFRESH_MIN_DELTA_ABS=0.20,
                    RECONCILE_TP_REFRESH_MAX_INTERVAL_SEC=300.0,
                    RECONCILE_TP_REFRESH_FORCE_COVERAGE_RATIO=0.80,
                    TP_ATR_MULT=1.8,
                    RECONCILE_TP_FALLBACK_PCT=0.005,
                )

        pos = SimpleNamespace(side="long", size=1.05, entry_price=100.0, atr=1.0)
        bot = DummyBot()
        calls = {"place": 0, "replace": 0}

        async def _fake_fetch_open_orders_best_effort(*_args, **_kwargs):
            return [{"id": "tp-1", "type": "TAKE_PROFIT_MARKET", "params": {"reduceOnly": True}, "amount": 1.0}]

        async def _fake_place_tp_ladder_router(*_args, **_kwargs):
            calls["place"] += 1
            return "new-tp"

        async def _fake_cancel_replace(*_args, **_kwargs):
            calls["replace"] += 1
            return {"id": "replaced-tp"}

        orig_fetch = reconcile._fetch_open_orders_best_effort
        orig_place_tp = reconcile._place_tp_ladder_router
        orig_replace = reconcile.cancel_replace_order
        reconcile._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        reconcile._place_tp_ladder_router = _fake_place_tp_ladder_router
        reconcile.cancel_replace_order = _fake_cancel_replace
        try:
            out = asyncio.run(reconcile._ensure_protective_tp(bot, "BTCUSDT", pos, "long", 1.05))
        finally:
            reconcile._fetch_open_orders_best_effort = orig_fetch
            reconcile._place_tp_ladder_router = orig_place_tp
            reconcile.cancel_replace_order = orig_replace

        self.assertEqual(out, "tp_refresh_deferred")
        self.assertEqual(calls["replace"], 0)
        self.assertEqual(calls["place"], 0)

    def test_reconcile_tp_refresh_force_ratio_triggers_replace(self):
        class DummyEx:
            pass

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = SimpleNamespace(
                    positions={},
                    run_context={},
                    reconcile_metrics={},
                )
                self.cfg = SimpleNamespace(
                    GUARDIAN_ENSURE_TP=True,
                    RECONCILE_TP_MIN_COVERAGE_RATIO=0.98,
                    RECONCILE_TP_REFRESH_MIN_DELTA_RATIO=0.50,
                    RECONCILE_TP_REFRESH_MIN_DELTA_ABS=1.0,
                    RECONCILE_TP_REFRESH_MAX_INTERVAL_SEC=300.0,
                    RECONCILE_TP_REFRESH_FORCE_COVERAGE_RATIO=0.80,
                    TP_ATR_MULT=1.8,
                    RECONCILE_TP_FALLBACK_PCT=0.005,
                )

        pos = SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=1.0)
        bot = DummyBot()
        calls = {"replace": 0}

        async def _fake_fetch_open_orders_best_effort(*_args, **_kwargs):
            return [{"id": "tp-1", "type": "TAKE_PROFIT_MARKET", "params": {"reduceOnly": True}, "amount": 0.4}]

        async def _fake_cancel_replace(*_args, **_kwargs):
            calls["replace"] += 1
            return {"id": "tp-2"}

        orig_fetch = reconcile._fetch_open_orders_best_effort
        orig_replace = reconcile.cancel_replace_order
        reconcile._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        reconcile.cancel_replace_order = _fake_cancel_replace
        try:
            out = asyncio.run(reconcile._ensure_protective_tp(bot, "BTCUSDT", pos, "long", 1.0))
        finally:
            reconcile._fetch_open_orders_best_effort = orig_fetch
            reconcile.cancel_replace_order = orig_replace

        self.assertEqual(out, "tp_restored")
        self.assertEqual(calls["replace"], 1)

    def test_reconcile_emits_belief_state_event(self):
        class DummyEx:
            async def fetch_positions(self, *_args, **_kwargs):
                return []

        class DummyState:
            def __init__(self):
                self.positions = {"BTCUSDT": SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0)}
                self.run_context = {}
                self.kill_metrics = {}
                self.reconcile_metrics = {}

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = DummyState()
                self._shutdown = asyncio.Event()
                self.active_symbols = {"BTCUSDT"}
                self.cfg = SimpleNamespace(
                    RECONCILE_FULL_SCAN_ORPHANS=True,
                    RECONCILE_PHANTOM_MISS_COUNT=1,
                    RECONCILE_PHANTOM_GRACE_SEC=0.0,
                )

        bot = DummyBot()
        events = []

        async def _fake_emit(_bot, event, data=None, **_kwargs):
            events.append({"event": event, "data": dict(data or {})})

        orig_emit = reconcile._tel_emit
        reconcile._tel_emit = _fake_emit
        try:
            asyncio.run(reconcile.reconcile_tick(bot))
        finally:
            reconcile._tel_emit = orig_emit

        belief = next((e for e in events if e.get("event") == "execution.belief_state"), None)
        self.assertIsNotNone(belief)
        self.assertIn("belief_debt_sec", belief["data"])
        self.assertIn("evidence_confidence", belief["data"])
        self.assertIn("intent_unknown_count", belief["data"])

    def test_reconcile_tracks_protection_gap_ttl_breach_metrics(self):
        class DummyEx:
            async def fetch_positions(self, *_args, **_kwargs):
                return [{"symbol": "BTCUSDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}]

        class DummyState:
            def __init__(self):
                self.positions = {
                    "BTCUSDT": SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0)
                }
                self.run_context = {
                    "protection_gap_state": {"BTCUSDT": {"gap_first_ts": reconcile._now() - 120.0, "ttl_breached": False}}
                }
                self.kill_metrics = {}
                self.reconcile_metrics = {}

        class DummyBot:
            def __init__(self):
                self.ex = DummyEx()
                self.state = DummyState()
                self._shutdown = asyncio.Event()
                self.active_symbols = {"BTCUSDT"}
                self.cfg = SimpleNamespace(
                    RECONCILE_FULL_SCAN_ORPHANS=True,
                    RECONCILE_PROTECTION_GAP_TTL_SEC=60.0,
                    RECONCILE_ALERT_COOLDOWN_SEC=0.0,
                )

        bot = DummyBot()
        orig_stop = reconcile._ensure_protective_stop
        orig_tp = reconcile._ensure_protective_tp
        reconcile._ensure_protective_stop = lambda *_a, **_k: asyncio.sleep(0, result="failed")
        reconcile._ensure_protective_tp = lambda *_a, **_k: asyncio.sleep(0, result="tp_disabled")
        try:
            asyncio.run(reconcile.reconcile_tick(bot))
        finally:
            reconcile._ensure_protective_stop = orig_stop
            reconcile._ensure_protective_tp = orig_tp

        rm = getattr(bot.state, "reconcile_metrics", {}) or {}
        self.assertGreater(float(rm.get("protection_coverage_gap_seconds", 0.0) or 0.0), 0.0)
        self.assertGreaterEqual(int(rm.get("protection_coverage_ttl_breaches", 0) or 0), 1)

    def test_posmgr_stop_ladder_delegates_to_protection_manager(self):
        calls = {"n": 0}

        async def _fake_pm(*_args, **_kwargs):
            calls["n"] += 1
            return "oid-stop-posmgr"

        orig_pm = position_manager._pm_place_stop_ladder_router
        position_manager._pm_place_stop_ladder_router = _fake_pm
        try:
            out = asyncio.run(
                position_manager._place_stop_ladder_router(
                    SimpleNamespace(),
                    sym_raw="DOGE/USDT:USDT",
                    side="long",
                    qty=10.0,
                    stop_price=0.09,
                    hedge_side_hint="LONG",
                    k="DOGEUSDT",
                    stop_client_id_base="SE_STOP_DOGEUSDT_LONG_1",
                )
            )
        finally:
            position_manager._pm_place_stop_ladder_router = orig_pm

        self.assertEqual(out, "oid-stop-posmgr")
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
