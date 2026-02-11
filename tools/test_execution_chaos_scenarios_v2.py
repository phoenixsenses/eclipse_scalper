#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
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

from eclipse_scalper.execution import rebuild, reconcile, replace_manager, order_router  # noqa: E402
from eclipse_scalper.execution.belief_controller import BeliefController  # noqa: E402
from tools import replay_trade  # noqa: E402


def _load_journal_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_journal_has_events(testcase: unittest.TestCase, rows: list[dict], required: set[str]) -> None:
    seen = {str(r.get("event") or "") for r in rows}
    testcase.assertTrue(required.issubset(seen), f"missing events: {sorted(required - seen)}")


class _RestartExchange:
    id = "binance"

    def __init__(self, *, positions=None, orders=None, trades=None):
        self._positions = list(positions or [])
        self._orders = list(orders or [])
        self._trades = list(trades or [])

    async def fetch_positions(self, *_args, **_kwargs):
        return list(self._positions)

    async def fetch_open_orders(self, *_args, **_kwargs):
        return list(self._orders)

    async def fetch_my_trades(self, symbol=None, _since=None):
        if symbol is None:
            return list(self._trades)
        return [t for t in self._trades if str(t.get("symbol") or "") == str(symbol)]


class _DummyBot:
    def __init__(self, ex):
        self.ex = ex
        self.state = types.SimpleNamespace(
            positions={},
            run_context={},
            kill_metrics={},
            reconcile_metrics={},
            guard_knobs={},
            belief_controller=BeliefController(),
            halt=False,
        )
        self.data = types.SimpleNamespace(raw_symbol={"BTCUSDT": "BTC/USDT:USDT"})
        self.cfg = types.SimpleNamespace(
            RECONCILE_FULL_SCAN_ORPHANS=False,
            RECONCILE_PHANTOM_MISS_COUNT=1,
            RECONCILE_PHANTOM_GRACE_SEC=0.0,
            RECONCILE_ADOPT_ORPHANS=True,
        )
        self._shutdown = asyncio.Event()
        self.active_symbols = {"BTCUSDT", "DOGEUSDT"}


class ChaosScenariosV2Tests(unittest.TestCase):
    def test_restart_rebuild_adopts_live_position_and_reconcile_keeps_it_consistent(self):
        ex = _RestartExchange(
            positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
            orders=[],
            trades=[{"symbol": "BTC/USDT:USDT", "timestamp": 1700000000000}],
        )
        bot = _DummyBot(ex)

        out = asyncio.run(rebuild.rebuild_local_state(bot, symbols=["BTC/USDT:USDT"]))
        self.assertTrue(out.get("ok"))
        self.assertIn("BTCUSDT", bot.state.positions)

        asyncio.run(reconcile.reconcile_tick(bot))
        self.assertIn("BTCUSDT", bot.state.positions)
        self.assertIsInstance(getattr(bot.state, "guard_knobs", {}), dict)

    def test_restart_rebuild_orphan_entry_can_force_safe_freeze(self):
        ex = _RestartExchange(
            positions=[],
            orders=[{"id": "entry-1", "symbol": "DOGE/USDT:USDT", "status": "open", "type": "LIMIT", "params": {}}],
            trades=[],
        )
        bot = _DummyBot(ex)

        out = asyncio.run(
            rebuild.rebuild_local_state(
                bot,
                symbols=["DOGE/USDT:USDT"],
                adopt_orphans=False,
                freeze_on_orphans=True,
            )
        )
        self.assertTrue(out.get("ok"))
        self.assertTrue(bool(out.get("halted", False)))
        self.assertTrue(bool(bot.state.halt))
        self.assertEqual(int((out.get("orphan_action_counts") or {}).get("FREEZE", 0) or 0), 1)
        self.assertEqual(int(out.get("orphans", 0) or 0), 1)
        orphan = dict((out.get("orphans_list") or [{}])[0] or {})
        self.assertEqual(str(orphan.get("class") or ""), "unknown_position_exposure")
        self.assertEqual(str(orphan.get("action") or ""), "FREEZE")
        self.assertTrue(bool(str(orphan.get("reason") or "").strip()))

    def test_restart_rebuild_does_not_expand_local_exposure_from_stale_state(self):
        ex = _RestartExchange(
            positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
            orders=[],
            trades=[],
        )
        bot = _DummyBot(ex)
        bot.state.positions = {
            "BTCUSDT": types.SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0),
            "ETHUSDT": types.SimpleNamespace(side="long", size=2.0, entry_price=200.0, atr=0.0),
        }

        out = asyncio.run(rebuild.rebuild_local_state(bot, symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"]))
        self.assertTrue(out.get("ok"))
        self.assertEqual(int(out.get("positions_prev", 0) or 0), 2)
        self.assertEqual(int(out.get("positions_rebuilt", 0) or 0), 1)
        self.assertIn("BTCUSDT", bot.state.positions)
        self.assertNotIn("ETHUSDT", bot.state.positions)
        btc = bot.state.positions["BTCUSDT"]
        self.assertLessEqual(float(getattr(btc, "size", 0.0) or 0.0), 1.0)

    def test_crash_mid_replace_filled_after_cancel_rebuild_reconcile_stays_single_exposure(self):
        create_calls = {"n": 0}

        async def _cancel_fn(_order_id: str, _symbol: str) -> bool:
            return False

        async def _status_fn(_order_id: str, _symbol: str) -> str:
            return "closed"

        async def _create_fn():
            create_calls["n"] += 1
            return {"id": "should-not-create", "status": "open"}

        outcome = asyncio.run(
            replace_manager.run_cancel_replace(
                cancel_order_id="old-1",
                symbol="BTC/USDT:USDT",
                max_attempts=2,
                cancel_fn=_cancel_fn,
                create_fn=_create_fn,
                status_fn=_status_fn,
                strict_transitions=True,
            )
        )
        self.assertFalse(bool(getattr(outcome, "success", False)))
        self.assertEqual(str(getattr(outcome, "reason", "")), "filled_after_cancel")
        self.assertEqual(int(create_calls["n"]), 0)

        with tempfile.TemporaryDirectory() as td:
            jpath = Path(td) / "journal.jsonl"
            ex = _RestartExchange(
                positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
                orders=[],
                trades=[{"symbol": "BTC/USDT:USDT", "timestamp": 1700000000000}],
            )
            bot = _DummyBot(ex)
            bot.cfg.EVENT_JOURNAL_PATH = str(jpath)

            out = asyncio.run(rebuild.rebuild_local_state(bot, symbols=["BTC/USDT:USDT"]))
            self.assertTrue(out.get("ok"))
            asyncio.run(reconcile.reconcile_tick(bot))

            self.assertEqual(len(bot.state.positions), 1)
            self.assertIn("BTCUSDT", bot.state.positions)
            btc = bot.state.positions["BTCUSDT"]
            self.assertLessEqual(float(getattr(btc, "size", 0.0) or 0.0), 1.0)

            replay = replay_trade.replay(jpath, symbol="BTCUSDT")
            self.assertGreaterEqual(int(replay.get("count", 0) or 0), 1)
            self.assertEqual(str(replay.get("last_state") or ""), "OPEN_CONFIRMED")
            rows = _load_journal_rows(jpath)
            _assert_journal_has_events(self, rows, {"rebuild.summary", "state.transition"})

    def test_restart_stale_open_order_with_live_position_freezes_when_policy_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            jpath = Path(td) / "journal.jsonl"
            ex = _RestartExchange(
                positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
                orders=[{"id": "stale-entry-1", "symbol": "BTC/USDT:USDT", "status": "open", "type": "LIMIT", "params": {}}],
                trades=[],
            )
            bot = _DummyBot(ex)
            bot.cfg.EVENT_JOURNAL_PATH = str(jpath)

            out = asyncio.run(
                rebuild.rebuild_local_state(
                    bot,
                    symbols=["BTC/USDT:USDT"],
                    adopt_orphans=False,
                    freeze_on_orphans=True,
                )
            )
            self.assertTrue(out.get("ok"))
            self.assertTrue(bool(out.get("halted", False)))
            self.assertTrue(bool(bot.state.halt))
            self.assertEqual(int((out.get("orphan_action_counts") or {}).get("FREEZE", 0) or 0), 1)
            orphan = dict((out.get("orphans_list") or [{}])[0] or {})
            self.assertEqual(str(orphan.get("class") or ""), "unknown_position_exposure")
            self.assertEqual(str(orphan.get("action") or ""), "FREEZE")

            rows = _load_journal_rows(jpath)
            _assert_journal_has_events(self, rows, {"rebuild.summary", "state.transition", "rebuild.orphan_decision"})
            orphan_events = [r for r in rows if str(r.get("event") or "") == "rebuild.orphan_decision"]
            self.assertEqual(len(orphan_events), 1)
            data = dict(orphan_events[0].get("data") or {})
            self.assertEqual(str(data.get("class") or ""), "unknown_position_exposure")
            self.assertEqual(str(data.get("action") or ""), "FREEZE")

            replay = replay_trade.replay(jpath, symbol="BTCUSDT")
            self.assertGreaterEqual(int(replay.get("count", 0) or 0), 1)
            self.assertEqual(str(replay.get("last_state") or ""), "OPEN_CONFIRMED")

    def test_restart_stale_open_order_with_live_position_cancels_when_policy_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            jpath = Path(td) / "journal.jsonl"
            ex = _RestartExchange(
                positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
                orders=[{"id": "stale-entry-2", "symbol": "BTC/USDT:USDT", "status": "open", "type": "LIMIT", "params": {}}],
                trades=[],
            )
            bot = _DummyBot(ex)
            bot.cfg.EVENT_JOURNAL_PATH = str(jpath)
            cancel_calls = {"n": 0}

            async def _cancel_order(order_id, _symbol=None):
                cancel_calls["n"] += 1
                return {"id": str(order_id), "status": "canceled"}

            bot.ex.cancel_order = _cancel_order

            out = asyncio.run(
                rebuild.rebuild_local_state(
                    bot,
                    symbols=["BTC/USDT:USDT"],
                    adopt_orphans=False,
                    freeze_on_orphans=False,
                )
            )
            self.assertTrue(out.get("ok"))
            self.assertFalse(bool(out.get("halted", False)))
            self.assertEqual(int((out.get("orphan_action_counts") or {}).get("CANCEL", 0) or 0), 1)
            self.assertEqual(int(cancel_calls["n"]), 1)
            orphan = dict((out.get("orphans_list") or [{}])[0] or {})
            self.assertEqual(str(orphan.get("class") or ""), "orphan_entry_order")
            self.assertEqual(str(orphan.get("action") or ""), "CANCEL")
            self.assertTrue(bool(orphan.get("cancel_ok")))

            rows = _load_journal_rows(jpath)
            _assert_journal_has_events(self, rows, {"rebuild.summary", "state.transition", "rebuild.orphan_decision"})
            orphan_events = [r for r in rows if str(r.get("event") or "") == "rebuild.orphan_decision"]
            self.assertEqual(len(orphan_events), 1)
            data = dict(orphan_events[0].get("data") or {})
            self.assertEqual(str(data.get("class") or ""), "orphan_entry_order")
            self.assertEqual(str(data.get("action") or ""), "CANCEL")
            self.assertTrue(bool(data.get("cancel_ok")))

            replay = replay_trade.replay(jpath, symbol="BTCUSDT")
            self.assertGreaterEqual(int(replay.get("count", 0) or 0), 1)
            self.assertEqual(str(replay.get("last_state") or ""), "OPEN_CONFIRMED")

    def test_missing_transition_coverage_flips_runtime_reconcile_first_posture(self):
        with tempfile.TemporaryDirectory() as td:
            gate_path = Path(td) / "reliability_gate.txt"
            gate_path.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=999",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=0.000",
                        "replay_mismatch_ids:",
                        "- CID-LOST-1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ex = _RestartExchange(positions=[], orders=[], trades=[])
            bot = _DummyBot(ex)
            bot.cfg.RELIABILITY_GATE_PATH = str(gate_path)
            bot.cfg.RELIABILITY_GATE_MAX_REPLAY_MISMATCH = 0
            bot.cfg.RELIABILITY_GATE_MAX_INVALID_TRANSITIONS = 0
            bot.cfg.RELIABILITY_GATE_MIN_JOURNAL_COVERAGE = 0.90

            asyncio.run(reconcile.reconcile_tick(bot))
            knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
            if not knobs:
                # CI can occasionally see first-tick guard hydration lag; second tick should converge.
                asyncio.run(reconcile.reconcile_tick(bot))
                knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
            self.assertTrue(bool(knobs))
            self.assertFalse(bool(knobs.get("allow_entries", True)))
            self.assertTrue(bool(knobs.get("runtime_gate_degraded", False)))
            self.assertIn("runtime_gate_degraded", str(knobs.get("reason") or ""))

    def test_contradiction_spike_in_runtime_gate_freezes_entries(self):
        with tempfile.TemporaryDirectory() as td:
            gate_path = Path(td) / "reliability_gate.txt"
            gate_path.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=1.000",
                        "evidence_contradiction_count=3",
                        "replay_mismatch_categories={\"contradiction\":3}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ex = _RestartExchange(positions=[], orders=[], trades=[])
            bot = _DummyBot(ex)
            bot.cfg.RELIABILITY_GATE_PATH = str(gate_path)
            bot.cfg.RELIABILITY_GATE_MAX_REPLAY_MISMATCH = 0
            bot.cfg.RELIABILITY_GATE_MAX_INVALID_TRANSITIONS = 0
            bot.cfg.RELIABILITY_GATE_MIN_JOURNAL_COVERAGE = 0.90
            bot.cfg.RELIABILITY_GATE_MAX_EVIDENCE_CONTRADICTION_COUNT = 1

            asyncio.run(reconcile.reconcile_tick(bot))
            knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
            if not knobs:
                asyncio.run(reconcile.reconcile_tick(bot))
                knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
            self.assertTrue(bool(knobs))
            self.assertFalse(bool(knobs.get("allow_entries", True)))
            self.assertTrue(bool(knobs.get("runtime_gate_degraded", False)))
            self.assertIn("contradiction", str(knobs.get("runtime_gate_reason") or "").lower())
            # Exit safety invariant remains separate from entry freeze logic.
            self.assertIn(str(knobs.get("recovery_stage") or ""), ("RUNTIME_GATE_DEGRADED", "ORANGE_RECOVERY"))

    def test_runtime_gate_freeze_then_warmup_reenable_is_constrained(self):
        with tempfile.TemporaryDirectory() as td:
            gate_path = Path(td) / "reliability_gate.txt"
            ex = _RestartExchange(positions=[], orders=[], trades=[])
            bot = _DummyBot(ex)
            bot.cfg.RELIABILITY_GATE_PATH = str(gate_path)
            bot.cfg.RELIABILITY_GATE_MAX_REPLAY_MISMATCH = 0
            bot.cfg.RELIABILITY_GATE_MAX_INVALID_TRANSITIONS = 0
            bot.cfg.RELIABILITY_GATE_MIN_JOURNAL_COVERAGE = 0.90
            bot.cfg.BELIEF_RUNTIME_GATE_RECOVER_SEC = 120.0
            bot.cfg.BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE = 0.5
            bot.cfg.BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE = 0.5
            bot.cfg.FIXED_NOTIONAL_USDT = 100.0
            bot.cfg.LEVERAGE = 20
            bot.cfg.ENTRY_MIN_CONFIDENCE = 0.2

            # Tick 1: degraded runtime gate should freeze entries.
            gate_path.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=1",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=1.000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            asyncio.run(reconcile.reconcile_tick(bot))
            k1 = dict(getattr(bot.state, "guard_knobs", {}) or {})
            if not k1:
                asyncio.run(reconcile.reconcile_tick(bot))
                k1 = dict(getattr(bot.state, "guard_knobs", {}) or {})
            self.assertTrue(bool(k1))
            self.assertFalse(bool(k1.get("allow_entries", True)))
            self.assertTrue(bool(k1.get("runtime_gate_degraded", False)))
            if "max_notional_usdt" in k1:
                self.assertEqual(float(k1.get("max_notional_usdt", 0.0)), 0.0)

            # Tick 2: gate clears; controller should move to runtime warmup, not full unlock.
            gate_path.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=1.000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            asyncio.run(reconcile.reconcile_tick(bot))
            k2 = dict(getattr(bot.state, "guard_knobs", {}) or {})
            self.assertTrue(bool(k2.get("allow_entries", False)))
            self.assertIn("runtime_gate_warmup", str(k2.get("reason") or ""))
            self.assertEqual(str(k2.get("recovery_stage") or ""), "RUNTIME_GATE_WARMUP")
            if "max_notional_usdt" in k2:
                self.assertLessEqual(float(k2.get("max_notional_usdt", 999.0)), 50.0)
            if "max_leverage" in k2:
                self.assertLessEqual(int(k2.get("max_leverage", 999)), 10)

    def test_replace_race_spike_auto_degrades_entry_posture(self):
        ex = _RestartExchange(positions=[], orders=[], trades=[])
        bot = _DummyBot(ex)
        bot.cfg.BELIEF_RECONCILE_FIRST_GATE_COUNT_THRESHOLD = 1
        bot.cfg.BELIEF_RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD = 0.80
        bot.cfg.BELIEF_RECONCILE_FIRST_GATE_STREAK_THRESHOLD = 1
        bot.cfg.BELIEF_RUNTIME_GATE_RECOVER_SEC = 120.0

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
        asyncio.run(reconcile.reconcile_tick(bot))
        knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
        if not knobs:
            asyncio.run(reconcile.reconcile_tick(bot))
            knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
        self.assertTrue(bool(knobs))
        self.assertFalse(bool(knobs.get("allow_entries", True)))
        self.assertTrue(
            bool(knobs.get("reconcile_first_gate_degraded", False))
            or bool(knobs.get("runtime_gate_degraded", False))
        )
        reason = str(knobs.get("reason") or "").lower()
        self.assertTrue(("reconcile_first" in reason) or ("runtime_gate_degraded" in reason))

    def test_runtime_gate_critical_categories_freeze_then_warmup(self):
        with tempfile.TemporaryDirectory() as td:
            gate_path = Path(td) / "reliability_gate.txt"
            ex = _RestartExchange(positions=[], orders=[], trades=[])
            bot = _DummyBot(ex)
            bot.cfg.RELIABILITY_GATE_PATH = str(gate_path)
            bot.cfg.RELIABILITY_GATE_MAX_REPLAY_MISMATCH = 0
            bot.cfg.RELIABILITY_GATE_MAX_INVALID_TRANSITIONS = 0
            bot.cfg.RELIABILITY_GATE_MIN_JOURNAL_COVERAGE = 0.90
            bot.cfg.BELIEF_RUNTIME_GATE_RECOVER_SEC = 120.0
            bot.cfg.BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD = 2.0
            bot.cfg.BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD = 1.0
            bot.cfg.FIXED_NOTIONAL_USDT = 100.0
            bot.cfg.LEVERAGE = 20
            bot.cfg.ENTRY_MIN_CONFIDENCE = 0.2

            gate_path.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=1.000",
                        "replay_mismatch_categories={\"position\":1,\"replace_race\":1}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            asyncio.run(reconcile.reconcile_tick(bot))
            k1 = dict(getattr(bot.state, "guard_knobs", {}) or {})
            self.assertFalse(bool(k1.get("allow_entries", True)))
            self.assertTrue(bool(k1.get("runtime_gate_degraded", False)))
            self.assertIn("runtime_gate_degraded", str(k1.get("reason") or ""))

            gate_path.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=1.000",
                        "replay_mismatch_categories={}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            asyncio.run(reconcile.reconcile_tick(bot))
            k2 = dict(getattr(bot.state, "guard_knobs", {}) or {})
            self.assertTrue(bool(k2.get("allow_entries", False)))
            self.assertIn("runtime_gate_warmup", str(k2.get("reason") or ""))

    def test_reconcile_emits_posture_transition_audit_event(self):
        with tempfile.TemporaryDirectory() as td:
            tele_path = Path(td) / "telemetry.jsonl"
            ex = _RestartExchange(positions=[], orders=[], trades=[])
            bot = _DummyBot(ex)
            bot.cfg.TELEMETRY_PATH = str(tele_path)
            bot.cfg.BELIEF_RED_SCORE = 0.01
            bot.cfg.BELIEF_ORANGE_SCORE = 0.005
            bot.cfg.BELIEF_YELLOW_SCORE = 0.001
            bot.cfg.BELIEF_RED_GROWTH = 99.0
            bot.cfg.BELIEF_ORANGE_GROWTH = 99.0
            bot.cfg.BELIEF_YELLOW_GROWTH = 99.0
            bot.cfg.BELIEF_MODE_PERSIST_SEC = 0.0
            bot.state.positions = {
                "BTCUSDT": types.SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0),
            }

            asyncio.run(reconcile.reconcile_tick(bot))
            self.assertTrue(tele_path.exists())
            events = [json.loads(line) for line in tele_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            transitions = [ev for ev in events if str(ev.get("event") or "") == "execution.posture_transition"]
            self.assertTrue(transitions)
            data = dict((transitions[-1].get("data") or {}))
            self.assertTrue(str(data.get("transition") or "").strip())
            self.assertTrue(str(data.get("new_mode") or "").strip())

    def test_replace_storm_hits_budget_and_stops_retry_spam(self):
        ex = _RestartExchange(positions=[], orders=[], trades=[])
        bot = _DummyBot(ex)
        bot.state.run_context = {}
        env = dict(os.environ)
        os.environ["ROUTER_REPLACE_BUDGET_WINDOW_SEC"] = "300"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_GLOBAL"] = "2"
        os.environ["ROUTER_REPLACE_BUDGET_MAX_PER_SYMBOL"] = "2"
        calls = {"n": 0}

        async def _fake_run_cancel_replace(**_kwargs):
            calls["n"] += 1
            return types.SimpleNamespace(
                success=False,
                state="REPLACE_RACE",
                reason="replace_reconcile_required",
                attempts=1,
                ambiguity_count=1,
                last_status="unknown",
            )

        orig_replace = order_router._replace_manager
        order_router._replace_manager = types.SimpleNamespace(run_cancel_replace=_fake_run_cancel_replace)
        try:
            for _ in range(4):
                asyncio.run(
                    order_router.cancel_replace_order(
                        bot,
                        cancel_order_id="oid-storm",
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

        # Budget limit should cap replace-manager invocations.
        self.assertLessEqual(int(calls["n"]), 2)
        hints = dict((bot.state.run_context or {}).get("reconcile_hints") or {})
        self.assertIn("BTCUSDT", hints)
        self.assertIn(
            "replace_budget_exceeded",
            str((hints.get("BTCUSDT") or {}).get("reason") or ""),
        )

    def test_protection_gap_ttl_breach_freezes_entries_in_runtime_loop(self):
        ex = _RestartExchange(
            positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
            orders=[],
            trades=[],
        )
        bot = _DummyBot(ex)
        bot.state.positions = {
            "BTCUSDT": types.SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0),
        }
        bot.state.run_context = {
            "protection_gap_state": {"BTCUSDT": {"gap_first_ts": reconcile._now() - 120.0, "ttl_breached": False}}
        }
        bot.cfg.RECONCILE_PROTECTION_GAP_TTL_SEC = 60.0
        bot.cfg.RECONCILE_ALERT_COOLDOWN_SEC = 0.0
        bot.cfg.BELIEF_YELLOW_SCORE = 99.0
        bot.cfg.BELIEF_ORANGE_SCORE = 199.0
        bot.cfg.BELIEF_RED_SCORE = 299.0
        bot.cfg.BELIEF_YELLOW_GROWTH = 99.0
        bot.cfg.BELIEF_ORANGE_GROWTH = 199.0
        bot.cfg.BELIEF_RED_GROWTH = 299.0
        bot.cfg.BELIEF_PROTECTION_GAP_TRIP_SEC = 60.0

        orig_stop = reconcile._ensure_protective_stop
        orig_tp = reconcile._ensure_protective_tp
        reconcile._ensure_protective_stop = lambda *_a, **_k: asyncio.sleep(0, result="failed")
        reconcile._ensure_protective_tp = lambda *_a, **_k: asyncio.sleep(0, result="tp_disabled")
        try:
            asyncio.run(reconcile.reconcile_tick(bot))
        finally:
            reconcile._ensure_protective_stop = orig_stop
            reconcile._ensure_protective_tp = orig_tp

        knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
        self.assertTrue(bool(knobs))
        self.assertFalse(bool(knobs.get("allow_entries", True)))
        self.assertIn("protection_gap_degraded", str(knobs.get("reason") or ""))

        rm = dict(getattr(bot.state, "reconcile_metrics", {}) or {})
        self.assertGreater(float(rm.get("protection_coverage_gap_seconds", 0.0) or 0.0), 0.0)
        self.assertGreaterEqual(int(rm.get("protection_coverage_ttl_breaches", 0) or 0), 1)

    def test_dribble_fill_budget_saturation_with_ttl_breach_forces_refresh_and_clamps_entries(self):
        ex = _RestartExchange(
            positions=[{"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long", "entryPrice": 100.0}],
            orders=[],
            trades=[],
        )
        bot = _DummyBot(ex)
        bot.state.positions = {
            "BTCUSDT": types.SimpleNamespace(side="long", size=1.0, entry_price=100.0, atr=0.0, stop_order_id="old-stop")
        }
        bot.state.run_context = {
            "protection_gap_state": {"BTCUSDT": {"ttl_breached": True}},
            "protection_refresh": {
                "BTCUSDT": {
                    "qty": 1.0,
                    "ts": reconcile._now(),
                    "refresh_events": [reconcile._now() - 1.0, reconcile._now() - 2.0, reconcile._now() - 3.0],
                }
            },
        }
        bot.state.reconcile_metrics = {"protection_refresh_budget_blocked_count": 4}
        bot.cfg.RECONCILE_STOP_REFRESH_MAX_PER_WINDOW = 1
        bot.cfg.RECONCILE_STOP_REFRESH_BUDGET_WINDOW_SEC = 60.0
        bot.cfg.RECONCILE_STOP_REFRESH_FORCE_COVERAGE_RATIO = 0.80
        bot.cfg.BELIEF_PROTECTION_REFRESH_ENTRY_BLOCK_THRESHOLD = 3
        bot.cfg.FIXED_NOTIONAL_USDT = 100.0
        bot.cfg.LEVERAGE = 20
        bot.cfg.ENTRY_MIN_CONFIDENCE = 0.2
        bot.cfg.RECONCILE_ALERT_COOLDOWN_SEC = 0.0

        async def _fake_fetch_open_orders_best_effort(_bot, _sym_raw):
            return []

        def _fake_assess_stop_coverage(_orders, **_kwargs):
            return {
                "covered": False,
                "needs_refresh": True,
                "order_id": "old-stop",
                "existing_qty": 1.0,
                "coverage_ratio": 1.0,
                "reason": "qty_under_covered",
            }

        async def _fake_cancel_replace_order(*_args, **_kwargs):
            return {"id": "new-stop"}

        orig_fetch = reconcile._fetch_open_orders_best_effort
        orig_assess = reconcile._assess_stop_coverage
        orig_cancel_replace = reconcile.cancel_replace_order
        orig_should_refresh = reconcile._should_refresh_protection
        orig_tp = reconcile._ensure_protective_tp
        reconcile._fetch_open_orders_best_effort = _fake_fetch_open_orders_best_effort
        reconcile._assess_stop_coverage = _fake_assess_stop_coverage
        reconcile.cancel_replace_order = _fake_cancel_replace_order
        reconcile._should_refresh_protection = lambda **_kwargs: True
        reconcile._ensure_protective_tp = lambda *_a, **_k: asyncio.sleep(0, result="tp_disabled")
        try:
            asyncio.run(reconcile.reconcile_tick(bot))
        finally:
            reconcile._fetch_open_orders_best_effort = orig_fetch
            reconcile._assess_stop_coverage = orig_assess
            reconcile.cancel_replace_order = orig_cancel_replace
            reconcile._should_refresh_protection = orig_should_refresh
            reconcile._ensure_protective_tp = orig_tp

        knobs = dict(getattr(bot.state, "guard_knobs", {}) or {})
        self.assertTrue(bool(knobs))
        self.assertFalse(bool(knobs.get("allow_entries", True)))
        self.assertIn("protection_refresh_budget_hard_block", str(knobs.get("reason") or ""))

        pos = bot.state.positions.get("BTCUSDT")
        self.assertIsNotNone(pos)
        self.assertEqual(str(getattr(pos, "stop_order_id", "")), "new-stop")
        rm = dict(getattr(bot.state, "reconcile_metrics", {}) or {})
        self.assertGreaterEqual(int(rm.get("protection_refresh_budget_force_override_count", 0) or 0), 1)

    def test_refresh_hard_block_release_path_progresses_warmup_then_release(self):
        class _Clock:
            def __init__(self, start: float = 0.0):
                self.t = float(start)

            def now(self) -> float:
                return float(self.t)

            def tick(self, sec: float) -> None:
                self.t += float(sec)

        cfg = types.SimpleNamespace(
            BELIEF_DEBT_REF_SEC=300.0,
            BELIEF_YELLOW_SCORE=99.0,
            BELIEF_ORANGE_SCORE=199.0,
            BELIEF_RED_SCORE=299.0,
            BELIEF_YELLOW_GROWTH=99.0,
            BELIEF_ORANGE_GROWTH=99.0,
            BELIEF_RED_GROWTH=99.0,
            BELIEF_MODE_PERSIST_SEC=0.0,
            BELIEF_MODE_RECOVER_SEC=1.0,
            BELIEF_DOWN_HYST=0.99,
            FIXED_NOTIONAL_USDT=100.0,
            LEVERAGE=20,
            ENTRY_MIN_CONFIDENCE=0.2,
            BELIEF_PROTECTION_REFRESH_ENTRY_BLOCK_THRESHOLD=2,
            BELIEF_PROTECTION_REFRESH_DECAY_SEC=2.0,
            BELIEF_PROTECTION_REFRESH_RECOVER_SEC=10.0,
            BELIEF_PROTECTION_REFRESH_WARMUP_NOTIONAL_SCALE=0.8,
            BELIEF_PROTECTION_REFRESH_WARMUP_LEVERAGE_SCALE=0.8,
            BELIEF_PROTECTION_REFRESH_BLOCKED_WEIGHT=0.0,
            BELIEF_PROTECTION_REFRESH_FORCE_WEIGHT=0.0,
        )
        clock = _Clock(1.0)
        ctl = BeliefController(clock=clock.now)

        hard_block = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "protection_refresh_budget_blocked_count": 3,
            },
            cfg,
        )
        self.assertFalse(hard_block.allow_entries)
        self.assertIn(
            str(getattr(hard_block, "recovery_stage", "")),
            ("PROTECTION_REFRESH_HARD_BLOCK", "ORANGE_RECOVERY", "YELLOW_WATCH", "GREEN"),
        )
        self.assertIn("protection_refresh_budget_hard_block", str(hard_block.reason))

        clock.tick(3.0)
        warmup = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "protection_refresh_budget_blocked_count": 0,
            },
            cfg,
        )
        self.assertTrue(warmup.allow_entries)
        self.assertEqual(str(getattr(warmup, "recovery_stage", "")), "PROTECTION_REFRESH_WARMUP")
        self.assertIn("protection_refresh_budget_warmup", str(warmup.reason))
        self.assertLess(float(warmup.max_notional_usdt), 100.0)

        clock.tick(15.0)
        green = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "protection_refresh_budget_blocked_count": 0,
            },
            cfg,
        )
        self.assertTrue(green.allow_entries)
        self.assertIn(str(getattr(green, "recovery_stage", "")), ("YELLOW_WATCH", "GREEN"))
        self.assertNotIn("protection_refresh_budget_warmup", str(green.reason))

        clock.tick(120.0)
        full_clear = ctl.update(
            {
                "belief_debt_sec": 0.0,
                "belief_debt_symbols": 0,
                "mismatch_streak": 0,
                "protection_refresh_budget_blocked_count": 0,
            },
            cfg,
        )
        self.assertTrue(full_clear.allow_entries)
        self.assertIn(str(getattr(full_clear, "recovery_stage", "")), ("YELLOW_WATCH", "GREEN"))
        self.assertNotIn("protection_refresh_budget_warmup", str(full_clear.reason))


if __name__ == "__main__":
    unittest.main()
