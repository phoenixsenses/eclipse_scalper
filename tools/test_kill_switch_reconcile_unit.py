#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
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

from eclipse_scalper.risk import kill_switch  # noqa: E402


class KillSwitchReconcileTests(unittest.TestCase):
    def test_reconcile_mismatch_streak_trips_halt(self):
        bot = SimpleNamespace()
        bot.ex = None
        bot.cfg = SimpleNamespace(
            KILL_SWITCH_ENABLED=True,
            KILL_RECONCILE_MISMATCH_STREAK_MAX=3,
            KILL_RECONCILE_MISMATCH_HALT_SEC=30.0,
            KILL_MAX_DATA_STALENESS_SEC=0.0,
            KILL_MAX_API_ERROR_RATE=1.0,
            KILL_MAX_API_ERROR_BURST=99999,
            KILL_MIN_REQ_WINDOW=99999,
            MAX_DAILY_LOSS_PCT=0.0,
            MAX_DRAWDOWN_PCT=0.0,
            KILL_MIN_EQUITY=0.0,
            KILL_SWITCH_COOLDOWN_SEC=120.0,
        )
        bot.state = SimpleNamespace(
            run_context={},
            kill_metrics={"reconcile_mismatch_streak": 4},
            current_equity=1000.0,
            peak_equity=1000.0,
            start_of_day_equity=1000.0,
            halt_until=0.0,
            halt_reason="",
        )

        ok, reason = asyncio.run(kill_switch.evaluate(bot))
        self.assertFalse(ok)
        self.assertIn("RECONCILE MISMATCH STORM", str(reason))
        self.assertTrue(kill_switch.is_halted(bot))

    def test_reconcile_belief_debt_limit_trips_halt(self):
        bot = SimpleNamespace()
        bot.ex = None
        bot.cfg = SimpleNamespace(
            KILL_SWITCH_ENABLED=True,
            KILL_RECONCILE_MISMATCH_STREAK_MAX=0,
            KILL_RECONCILE_BELIEF_DEBT_SEC_MAX=120.0,
            KILL_RECONCILE_BELIEF_DEBT_MIN_SYMBOLS=2,
            KILL_RECONCILE_BELIEF_DEBT_HALT_SEC=45.0,
            KILL_MAX_DATA_STALENESS_SEC=0.0,
            KILL_MAX_API_ERROR_RATE=1.0,
            KILL_MAX_API_ERROR_BURST=99999,
            KILL_MIN_REQ_WINDOW=99999,
            MAX_DAILY_LOSS_PCT=0.0,
            MAX_DRAWDOWN_PCT=0.0,
            KILL_MIN_EQUITY=0.0,
            KILL_SWITCH_COOLDOWN_SEC=120.0,
        )
        bot.state = SimpleNamespace(
            run_context={},
            kill_metrics={
                "reconcile_mismatch_streak": 0,
                "reconcile_belief_debt_sec": 200.0,
                "reconcile_belief_debt_symbols": 3,
            },
            current_equity=1000.0,
            peak_equity=1000.0,
            start_of_day_equity=1000.0,
            halt_until=0.0,
            halt_reason="",
        )

        ok, reason = asyncio.run(kill_switch.evaluate(bot))
        self.assertFalse(ok)
        self.assertIn("RECONCILE BELIEF DEBT LIMIT", str(reason))
        self.assertTrue(kill_switch.is_halted(bot))

    def test_reconcile_belief_debt_growth_burst_trips_halt(self):
        bot = SimpleNamespace()
        bot.ex = None
        bot.cfg = SimpleNamespace(
            KILL_SWITCH_ENABLED=True,
            KILL_RECONCILE_MISMATCH_STREAK_MAX=0,
            KILL_RECONCILE_BELIEF_DEBT_SEC_MAX=0.0,
            KILL_RECONCILE_BELIEF_DEBT_GROWTH_MIN_SEC=30.0,
            KILL_RECONCILE_BELIEF_DEBT_GROWTH_BURST=2,
            KILL_RECONCILE_BELIEF_DEBT_HALT_SEC=45.0,
            KILL_MAX_DATA_STALENESS_SEC=0.0,
            KILL_MAX_API_ERROR_RATE=1.0,
            KILL_MAX_API_ERROR_BURST=99999,
            KILL_MIN_REQ_WINDOW=99999,
            MAX_DAILY_LOSS_PCT=0.0,
            MAX_DRAWDOWN_PCT=0.0,
            KILL_MIN_EQUITY=0.0,
            KILL_SWITCH_COOLDOWN_SEC=120.0,
        )
        bot.state = SimpleNamespace(
            run_context={},
            kill_metrics={
                "reconcile_mismatch_streak": 0,
                "reconcile_belief_debt_sec": 200.0,
                "reconcile_belief_debt_symbols": 1,
                "reconcile_belief_debt_prev_sec": 100.0,
                "reconcile_belief_debt_growth_hits": 1,
            },
            current_equity=1000.0,
            peak_equity=1000.0,
            start_of_day_equity=1000.0,
            halt_until=0.0,
            halt_reason="",
        )

        ok, reason = asyncio.run(kill_switch.evaluate(bot))
        self.assertFalse(ok)
        self.assertIn("RECONCILE BELIEF DEBT GROWTH", str(reason))
        self.assertTrue(kill_switch.is_halted(bot))


if __name__ == "__main__":
    unittest.main()
