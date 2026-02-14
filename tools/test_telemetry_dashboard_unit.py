#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.tools import telemetry_dashboard as td  # noqa: E402


class TelemetryDashboardUnitTests(unittest.TestCase):
    def test_protection_refresh_budget_summary_uses_reconcile_summary_max(self):
        events = [
            {
                "event": "reconcile.summary",
                "data": {
                    "protection_refresh_budget_blocked_count": 1,
                    "protection_refresh_budget_force_override_count": 0,
                    "protection_refresh_stop_budget_blocked_count": 1,
                    "protection_refresh_tp_budget_blocked_count": 0,
                    "protection_refresh_stop_force_override_count": 0,
                    "protection_refresh_tp_force_override_count": 0,
                },
            },
            {
                "event": "reconcile.summary",
                "data": {
                    "protection_refresh_budget_blocked_count": 3,
                    "protection_refresh_budget_force_override_count": 2,
                    "protection_refresh_stop_budget_blocked_count": 2,
                    "protection_refresh_tp_budget_blocked_count": 1,
                    "protection_refresh_stop_force_override_count": 1,
                    "protection_refresh_tp_force_override_count": 1,
                },
            },
            {
                "event": "entry.partial_fill_escalation",
                "data": {"protection_refresh_budget_blocked_count": 99},
            },
        ]
        out = td._protection_refresh_budget_summary(events)
        self.assertEqual(int(out["protection_refresh_budget_blocked_count"]), 3)
        self.assertEqual(int(out["protection_refresh_budget_force_override_count"]), 2)
        self.assertEqual(int(out["protection_refresh_stop_budget_blocked_count"]), 2)
        self.assertEqual(int(out["protection_refresh_tp_budget_blocked_count"]), 1)
        self.assertEqual(int(out["protection_refresh_stop_force_override_count"]), 1)
        self.assertEqual(int(out["protection_refresh_tp_force_override_count"]), 1)

    def test_correlation_contribution_summary_counts_entry_impact(self):
        events = [
            {
                "event": "entry.decision",
                "symbol": "BTCUSDT",
                "data": {"action": "SCALE", "corr_pressure": 0.81, "corr_regime": "TIGHTENING", "corr_reason_tags": "downside_corr"},
            },
            {
                "event": "entry.blocked",
                "symbol": "ETHUSDT",
                "data": {"reason": "belief_controller_block", "corr_pressure": 0.92, "corr_regime": "STRESS", "corr_reason_tags": "tail_coupling"},
            },
            {
                "event": "execution.correlation_state",
                "data": {"corr_regime": "STRESS", "corr_pressure": 0.88, "corr_reason_tags": "tail_coupling,belief_uplift"},
            },
        ]
        out = td._correlation_contribution_summary(events)
        self.assertEqual(int(out.get("scaled", 0) or 0), 1)
        self.assertEqual(int(out.get("blocked", 0) or 0), 1)
        self.assertEqual(int(out.get("stress", 0) or 0), 1)
        self.assertEqual(str(out.get("latest_regime", "")), "STRESS")


if __name__ == "__main__":
    unittest.main()
