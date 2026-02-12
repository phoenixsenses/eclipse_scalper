#!/usr/bin/env python3
"""
Unit-style tests for entry_loop helpers.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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

from eclipse_scalper.execution import entry_loop  # noqa: E402


class EntryLoopTelemetryTests(unittest.TestCase):
    def tearDown(self):
        for key in (
            "CORR_GROUPS",
            "CORR_GROUP_MAX_POSITIONS",
            "CORR_GROUP_MAX_NOTIONAL_USDT",
            "ENTRY_BUDGET_ENABLED",
            "ENTRY_GLOBAL_BUDGET_USDT",
            "ENTRY_BUDGET_INCLUDE_OPEN_EXPOSURE",
            "ENTRY_BUDGET_MIN_SHARE",
            "ENTRY_BUDGET_MAX_SHARE",
        ):
            if key in entry_loop.os.environ:
                entry_loop.os.environ.pop(key, None)

    def test_recent_router_blocks_counts(self):
        now = 1000.0
        bot = SimpleNamespace()
        bot.state = SimpleNamespace()
        bot.state.telemetry = SimpleNamespace(
            recent=[
                {"ts": now - 5, "event": "order.blocked", "symbol": "BTCUSDT"},
                {"ts": now - 5, "event": "order.blocked", "data": {"k": "BTCUSDT"}},
                {"ts": now - 70, "event": "order.blocked", "symbol": "BTCUSDT"},
                {"ts": now - 3, "event": "order.create", "symbol": "BTCUSDT"},
            ]
        )

        self.assertEqual(entry_loop._recent_router_blocks(bot, "BTCUSDT", 60, now_ts=now), 2)
        self.assertEqual(entry_loop._recent_router_blocks(bot, "ETHUSDT", 60, now_ts=now), 0)

    def test_corr_group_caps_block(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(
            positions={
                "DOGEUSDT": SimpleNamespace(size=10, entry_price=1.0),
                "SHIBUSDT": SimpleNamespace(size=5, entry_price=1.0),
            }
        )
        entry_loop.os.environ["CORR_GROUPS"] = "MEME:DOGEUSDT,SHIBUSDT,PEPEUSDT"
        entry_loop.os.environ["CORR_GROUP_MAX_POSITIONS"] = "2"
        entry_loop.os.environ["CORR_GROUP_MAX_NOTIONAL_USDT"] = "20"

        reason, meta = entry_loop._check_corr_group(bot, "PEPEUSDT", planned_notional=5.0)
        self.assertIsNotNone(reason)
        self.assertIn("group MEME", reason)
        self.assertEqual(meta.get("group"), "MEME")

    def test_corr_group_scale_applies(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(
            positions={"DOGEUSDT": SimpleNamespace(size=10, entry_price=1.0)}
        )
        entry_loop.os.environ["CORR_GROUPS"] = "MEME:DOGEUSDT,PEPEUSDT"
        entry_loop.os.environ["CORR_GROUP_SCALE_ENABLED"] = "1"
        entry_loop.os.environ["CORR_GROUP_SCALE"] = "0.5"

        _, meta = entry_loop._check_corr_group(bot, "PEPEUSDT", planned_notional=10.0)
        scale, reason = entry_loop._corr_group_scale(bot, meta)
        self.assertLess(scale, 1.0)
        self.assertIn("corr_group_scale", reason)

    def test_guard_block_reason_code_runtime_gate(self):
        reason, code = entry_loop._guard_block_reason_code(
            {
                "reason": "stable | runtime_gate_degraded:mismatch>0",
                "runtime_gate_degraded": True,
            }
        )
        self.assertEqual(reason, "runtime_gate_reconcile_first")
        self.assertEqual(code, "ERR_RELIABILITY_GATE")

    def test_guard_block_reason_code_default(self):
        reason, code = entry_loop._guard_block_reason_code({"reason": "orange threshold"})
        self.assertEqual(reason, "belief_controller_block")
        self.assertEqual(code, "ERR_ROUTER_BLOCK")

    def test_guard_block_reason_code_reconcile_first_pressure(self):
        reason, code = entry_loop._guard_block_reason_code(
            {
                "reason": "stable | reconcile_first_spike",
                "reconcile_first_gate_degraded": True,
            }
        )
        self.assertEqual(reason, "reconcile_first_pressure")
        self.assertEqual(code, "ERR_RELIABILITY_GATE")

    def test_record_reconcile_first_gate_updates_kill_metrics(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(kill_metrics={})
        entry_loop._record_reconcile_first_gate(
            bot,
            "BTCUSDT",
            0.9,
            "runtime_gate",
            corr={"corr_pressure": 0.88, "corr_regime": "STRESS", "corr_reason_tags": "tail_coupling"},
        )
        entry_loop._record_reconcile_first_gate(
            bot,
            "BTCUSDT",
            0.95,
            "runtime_gate",
            corr={"corr_pressure": 0.91, "corr_regime": "STRESS", "corr_reason_tags": "tail_coupling"},
        )
        km = dict(getattr(bot.state, "kill_metrics", {}) or {})
        self.assertGreaterEqual(int(km.get("reconcile_first_gate_count", 0)), 2)
        self.assertGreaterEqual(int(km.get("reconcile_first_gate_current_streak", 0)), 2)
        self.assertGreaterEqual(int(km.get("reconcile_first_gate_max_streak", 0)), 2)
        self.assertEqual(str(km.get("reconcile_first_gate_last_corr_regime", "")), "STRESS")
        self.assertGreater(float(km.get("reconcile_first_gate_last_corr_pressure", 0.0) or 0.0), 0.0)
        events = list(km.get("reconcile_first_gate_events", []) or [])
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(str((events[-1] or {}).get("corr_regime", "")), "STRESS")
        self.assertGreater(float((events[-1] or {}).get("corr_pressure", 0.0) or 0.0), 0.0)

    def test_resolve_symbol_guard_applies_override(self):
        knobs = {
            "allow_entries": True,
            "min_entry_conf": 0.2,
            "per_symbol": {
                "BTCUSDT": {
                    "allow_entries": False,
                    "min_entry_conf": 0.5,
                    "reason": "symbol_debt=120.0s",
                }
            },
        }
        out = entry_loop._resolve_symbol_guard(knobs, "BTCUSDT")
        self.assertFalse(bool(out.get("allow_entries")))
        self.assertAlmostEqual(float(out.get("min_entry_conf") or 0.0), 0.5, places=6)
        self.assertIn("symbol_debt", str(out.get("reason") or ""))

    def test_entry_budget_snapshot_with_open_exposure(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace(ENTRY_GLOBAL_BUDGET_USDT=0.0)
        bot.state = SimpleNamespace(
            positions={
                "BTCUSDT": SimpleNamespace(size=1.0, entry_price=100.0),
                "ETHUSDT": SimpleNamespace(size=0.5, entry_price=200.0),
            }
        )
        entry_loop.os.environ["ENTRY_BUDGET_ENABLED"] = "1"
        entry_loop.os.environ["ENTRY_GLOBAL_BUDGET_USDT"] = "500"
        enabled, total, remaining, reason = entry_loop._entry_budget_snapshot(bot, {})
        self.assertTrue(enabled)
        self.assertAlmostEqual(total, 500.0, places=6)
        self.assertAlmostEqual(remaining, 300.0, places=6)
        self.assertIn("entry_budget", reason)

    def test_entry_budget_symbol_cap_confidence_weighting(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        entry_loop.os.environ["ENTRY_BUDGET_MIN_SHARE"] = "0.1"
        entry_loop.os.environ["ENTRY_BUDGET_MAX_SHARE"] = "0.6"
        low = entry_loop._entry_budget_symbol_cap(bot, confidence=0.2, min_conf=0.2, remaining=100.0)
        high = entry_loop._entry_budget_symbol_cap(bot, confidence=0.9, min_conf=0.2, remaining=100.0)
        self.assertGreaterEqual(low, 10.0)
        self.assertLessEqual(high, 60.0)
        self.assertGreater(high, low)

    def test_corr_snapshot_reads_reconcile_metrics(self):
        bot = SimpleNamespace()
        bot.state = SimpleNamespace(
            reconcile_metrics={
                "corr_pressure": 0.82,
                "corr_regime": "STRESS",
                "corr_confidence": 0.61,
                "corr_reason_tags": "tail_coupling,belief_uplift",
                "corr_worst_group": "MAJOR",
            }
        )
        snap = entry_loop._corr_snapshot(bot, None)
        self.assertAlmostEqual(float(snap.get("corr_pressure", 0.0)), 0.82, places=6)
        self.assertEqual(str(snap.get("corr_regime", "")), "STRESS")
        self.assertEqual(str(snap.get("corr_worst_group", "")), "MAJOR")

    def test_emit_entry_decision_includes_corr_context(self):
        bot = SimpleNamespace()
        bot.state = SimpleNamespace(
            reconcile_metrics={
                "corr_pressure": 0.77,
                "corr_regime": "TIGHTENING",
                "corr_confidence": 0.72,
                "corr_reason_tags": "downside_corr",
            }
        )
        emitted: list[dict] = []
        prev_emit = entry_loop.emit
        prev_emit_throttled = entry_loop.emit_throttled

        async def _fake_emit(_bot, event, data=None, symbol=None, level="info"):
            emitted.append({"event": event, "data": dict(data or {}), "symbol": symbol, "level": level})

        try:
            entry_loop.emit = _fake_emit
            entry_loop.emit_throttled = None
            rec = entry_loop.compute_entry_decision(
                symbol="BTCUSDT",
                signal={"action": "buy", "confidence": 0.8},
                guard_knobs={"allow_entries": True, "min_entry_conf": 0.2},
                min_confidence=0.2,
                amount=1.0,
                order_type="market",
                price=None,
                planned_notional=10.0,
                stage="propose",
            )
            asyncio.run(entry_loop._emit_entry_decision(bot, rec))
        finally:
            entry_loop.emit = prev_emit
            entry_loop.emit_throttled = prev_emit_throttled

        self.assertTrue(emitted)
        payload = emitted[-1]["data"]
        self.assertAlmostEqual(float(payload.get("corr_pressure", 0.0)), 0.77, places=6)
        self.assertEqual(str(payload.get("corr_regime", "")), "TIGHTENING")


if __name__ == "__main__":
    unittest.main()
