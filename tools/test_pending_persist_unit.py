#!/usr/bin/env python3
"""
Unit tests for Fix #6 (pending-block persistence) and Fix #5 (kill-switch / intent tags).
"""

from __future__ import annotations

import inspect
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import entry_loop  # noqa: E402


def _make_bot():
    """Minimal bot with state.run_context for pending persistence."""
    bot = SimpleNamespace()
    bot.state = SimpleNamespace()
    bot.state.run_context = {}
    return bot


class TestPendingPersistence(unittest.TestCase):
    """Fix #6: _set_pending / _pending_active / _clear_pending persist to run_context."""

    def setUp(self):
        # Clear in-memory state between tests
        entry_loop._PENDING_UNTIL.clear()
        entry_loop._PENDING_ORDER_ID.clear()

    def test_set_pending_persists_to_run_context(self):
        bot = _make_bot()
        entry_loop._set_pending(bot, "BTCUSDT", sec=30.0)
        store = bot.state.run_context.get("entry_pending", {})
        self.assertIn("BTCUSDT", store)
        self.assertGreater(store["BTCUSDT"], time.time())

    def test_pending_active_reads_from_run_context(self):
        bot = _make_bot()
        # Directly inject into run_context (simulating restart hydration)
        bot.state.run_context["entry_pending"] = {"ETHUSDT": time.time() + 60.0}
        # Clear in-memory so only persisted state can answer
        entry_loop._PENDING_UNTIL.pop("ETHUSDT", None)
        self.assertTrue(entry_loop._pending_active(bot, "ETHUSDT"))

    def test_pending_active_expired_returns_false(self):
        bot = _make_bot()
        bot.state.run_context["entry_pending"] = {"ETHUSDT": time.time() - 10.0}
        entry_loop._PENDING_UNTIL.pop("ETHUSDT", None)
        self.assertFalse(entry_loop._pending_active(bot, "ETHUSDT"))

    def test_clear_pending_removes_both(self):
        bot = _make_bot()
        entry_loop._set_pending(bot, "SOLUSDT", sec=30.0)
        self.assertTrue(entry_loop._pending_active(bot, "SOLUSDT"))
        entry_loop._clear_pending(bot, "SOLUSDT")
        self.assertFalse(entry_loop._pending_active(bot, "SOLUSDT"))
        store = bot.state.run_context.get("entry_pending", {})
        self.assertNotIn("SOLUSDT", store)

    def test_survives_simulated_restart(self):
        """Persist, clear in-memory, then check that bot.state still provides the block."""
        bot = _make_bot()
        entry_loop._set_pending(bot, "XRPUSDT", sec=60.0)
        # Simulate restart: nuke in-memory dict
        entry_loop._PENDING_UNTIL.clear()
        entry_loop._PENDING_ORDER_ID.clear()
        # Still active via persisted state
        self.assertTrue(entry_loop._pending_active(bot, "XRPUSDT"))

    def test_legacy_signature_still_works(self):
        """Old call pattern _set_pending(k, sec=…) and _pending_active(k) still work."""
        entry_loop._set_pending("DOTUSDT", sec=30.0)
        self.assertTrue(entry_loop._pending_active("DOTUSDT"))


class TestKillSwitchIntentTags(unittest.TestCase):
    """Fix #5: verify intent tags are present in reconcile create_order calls."""

    def test_reconcile_stop_ladder_has_intent_tag(self):
        src = inspect.getsource(entry_loop)
        # Not in entry_loop — check reconcile
        from eclipse_scalper.execution import reconcile
        src_r = inspect.getsource(reconcile._place_stop_ladder_router)
        self.assertIn('intent_component="reconcile"', src_r)
        self.assertIn('intent_kind="STOP_RESTORE"', src_r)

    def test_reconcile_tp_ladder_has_intent_tag(self):
        from eclipse_scalper.execution import reconcile
        src_r = inspect.getsource(reconcile._place_tp_ladder_router)
        self.assertIn('intent_component="reconcile"', src_r)
        self.assertIn('intent_kind="TP_RESTORE"', src_r)

    def test_entry_loop_create_order_has_kill_switch(self):
        src = inspect.getsource(entry_loop.entry_loop)
        # Both LIMIT and MARKET entry create_order calls should have kill switch
        self.assertIn("respect_kill_switch=True", src)


class TestPhantomClearingGuard(unittest.TestCase):
    """Fix #3: phantom clearing defers when mismatch_streak > 0."""

    def test_phantom_clearing_source_checks_mismatch_streak(self):
        from eclipse_scalper.execution import reconcile
        src = inspect.getsource(reconcile.reconcile_tick)
        self.assertIn("mismatch_streak", src)
        self.assertIn("deferring clear due to mismatch_streak", src)

    def test_phantom_clearing_does_verification_fetch(self):
        from eclipse_scalper.execution import reconcile
        src = inspect.getsource(reconcile.reconcile_tick)
        self.assertIn("phantom_verify_found", src)
        self.assertIn("_fetch_positions_best_effort", src)


class TestEntryWAL(unittest.TestCase):
    """Fix #1: entry WAL written to run_context before order submission."""

    def test_entry_loop_writes_wal_before_submit(self):
        src = inspect.getsource(entry_loop.entry_loop)
        # WAL write must appear before create_order
        wal_idx = src.find('entry_wal')
        create_idx = src.find('create_order')
        self.assertGreater(wal_idx, -1, "entry_wal not found in entry_loop")
        self.assertGreater(create_idx, -1, "create_order not found in entry_loop")
        self.assertLess(wal_idx, create_idx, "entry_wal should appear before create_order")

    def test_entry_loop_clears_wal_after_submit(self):
        src = inspect.getsource(entry_loop.entry_loop)
        # Should clear WAL after ORDER SUBMITTED log
        submitted_idx = src.find("ORDER SUBMITTED")
        clear_idx = src.find('entry_wal', submitted_idx)
        self.assertGreater(clear_idx, submitted_idx, "entry_wal clear should appear after ORDER SUBMITTED")


class TestBrainSaveAfterEntry(unittest.TestCase):
    """Fix #8: force brain save after successful entry submission."""

    def test_entry_loop_saves_brain_after_submit(self):
        src = inspect.getsource(entry_loop.entry_loop)
        self.assertIn("save_brain", src)
        # save should appear after ORDER SUBMITTED
        submitted_idx = src.find("ORDER SUBMITTED")
        save_idx = src.find("save_brain", submitted_idx)
        self.assertGreater(save_idx, submitted_idx)


class TestCircuitBreakerWired(unittest.TestCase):
    """Fix #7: circuit breaker wired into order_router."""

    def test_order_router_checks_circuit_breaker(self):
        from eclipse_scalper.execution import order_router
        src = inspect.getsource(order_router.create_order)
        self.assertIn("CircuitBreaker", src)
        self.assertIn("allow_request", src)

    def test_order_router_records_success_and_failure(self):
        from eclipse_scalper.execution import order_router
        src = inspect.getsource(order_router.create_order)
        self.assertIn("record_success", src)
        self.assertIn("record_failure", src)

    def test_circuit_breaker_skips_reduce_only(self):
        """Circuit breaker should NOT block reduce_only (exit/protective) orders."""
        from eclipse_scalper.execution import order_router
        src = inspect.getsource(order_router.create_order)
        # The check should be gated on "not intent_reduce_only"
        self.assertIn("not intent_reduce_only", src)

    def test_circuit_breaker_requires_explicit_cfg(self):
        """Circuit breaker only activates when bot.cfg.CIRCUIT_BREAKER_ENABLED is set."""
        from eclipse_scalper.execution import order_router
        src = inspect.getsource(order_router.create_order)
        self.assertIn("CIRCUIT_BREAKER_ENABLED", src)


class TestPartialFillSafetyNet(unittest.TestCase):
    """Fix #10: if partial fill flatten fails, stage1 emergency stop is placed."""

    def test_partial_fill_fallback_to_stage1(self):
        src = inspect.getsource(entry_loop.entry_loop)
        # After flatten fail, should call _maybe_place_stage1_emergency_stop
        idx = src.find('not pf.get("flatten_ok")')
        self.assertGreater(idx, -1, "flatten_ok check not found")
        stage1_idx = src.find("_maybe_place_stage1_emergency_stop", idx)
        self.assertGreater(stage1_idx, idx, "stage1 emergency stop should be placed when flatten fails")


if __name__ == "__main__":
    unittest.main()
