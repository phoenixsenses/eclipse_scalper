#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
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

from eclipse_scalper.execution import rebuild, reconcile, replace_manager  # noqa: E402
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
            halt=False,
        )
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
            self.assertTrue(bool(knobs))
            self.assertFalse(bool(knobs.get("allow_entries", True)))
            self.assertTrue(bool(knobs.get("runtime_gate_degraded", False)))
            self.assertIn("runtime_gate_degraded", str(knobs.get("reason") or ""))


if __name__ == "__main__":
    unittest.main()
