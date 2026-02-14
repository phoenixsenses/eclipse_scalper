#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import protection_manager as pm  # noqa: E402


class ProtectionManagerTests(unittest.TestCase):
    def test_assess_stop_coverage_close_position_is_covered(self):
        orders = [
            {"id": "s1", "type": "STOP_MARKET", "params": {"closePosition": True}},
        ]
        out = pm.assess_stop_coverage(orders, required_qty=3.0)
        self.assertTrue(out["covered"])
        self.assertEqual(out["reason"], "close_position_stop")

    def test_assess_stop_coverage_qty_under(self):
        orders = [
            {"id": "s1", "type": "STOP_MARKET", "params": {"reduceOnly": True}, "amount": 1.0},
        ]
        out = pm.assess_stop_coverage(orders, required_qty=2.0, min_coverage_ratio=0.9)
        self.assertFalse(out["covered"])
        self.assertTrue(out["needs_refresh"])
        self.assertEqual(out["order_id"], "s1")
        self.assertGreater(float(out["coverage_shortfall_ratio"]), 0.0)

    def test_assess_tp_coverage_qty_under(self):
        orders = [
            {"id": "t1", "type": "TAKE_PROFIT_MARKET", "params": {"reduceOnly": True}, "amount": 1.0},
        ]
        out = pm.assess_tp_coverage(orders, required_qty=2.0, min_coverage_ratio=0.9)
        self.assertFalse(out["covered"])
        self.assertTrue(out["needs_refresh"])
        self.assertEqual(out["order_id"], "t1")

    def test_should_refresh_protection_by_delta_or_interval(self):
        self.assertTrue(
            pm.should_refresh_protection(
                previous_qty=1.0,
                new_qty=1.25,
                last_refresh_ts=100.0,
                min_delta_ratio=0.1,
                max_refresh_interval_sec=60.0,
                now_ts=105.0,
            )
        )
        self.assertTrue(
            pm.should_refresh_protection(
                previous_qty=1.0,
                new_qty=1.01,
                last_refresh_ts=100.0,
                min_delta_ratio=0.2,
                max_refresh_interval_sec=3.0,
                now_ts=104.5,
            )
        )
        self.assertFalse(
            pm.should_refresh_protection(
                previous_qty=1.0,
                new_qty=1.01,
                last_refresh_ts=100.0,
                min_delta_ratio=0.2,
                max_refresh_interval_sec=10.0,
                now_ts=104.0,
            )
        )
        self.assertTrue(
            pm.should_refresh_protection(
                previous_qty=10.0,
                new_qty=10.08,
                last_refresh_ts=100.0,
                min_delta_ratio=0.2,
                min_delta_abs=0.05,
                max_refresh_interval_sec=999.0,
                now_ts=101.0,
            )
        )

    def test_update_coverage_gap_state_trips_ttl_and_increments_breach(self):
        state = {"gap_first_ts": 10.0, "ttl_breached": False}
        out = pm.update_coverage_gap_state(
            state,
            required_qty=2.0,
            covered=False,
            ttl_sec=30.0,
            now_ts=50.0,
            reason="failed",
            coverage_ratio=0.0,
        )
        self.assertTrue(out["active"])
        self.assertTrue(out["ttl_breached"])
        self.assertTrue(out["new_ttl_breach"])
        self.assertEqual(int(out["breach_count"]), 1)

    def test_update_coverage_gap_state_resolves_on_coverage_restore(self):
        state = {
            "active": True,
            "gap_first_ts": 10.0,
            "gap_seconds": 20.0,
            "ttl_breached": True,
            "breach_count": 2,
        }
        out = pm.update_coverage_gap_state(
            state,
            required_qty=2.0,
            covered=True,
            ttl_sec=30.0,
            now_ts=40.0,
            reason="present",
            coverage_ratio=1.0,
        )
        self.assertFalse(out["active"])
        self.assertFalse(out["ttl_breached"])
        self.assertEqual(float(out["gap_seconds"]), 0.0)
        self.assertEqual(int(out["breach_count"]), 2)

    def test_refresh_budget_blocks_when_limit_reached(self):
        state = {"refresh_events": [100.0, 120.0, 130.0]}
        out = pm.should_allow_refresh_budget(
            state,
            now_ts=140.0,
            window_sec=60.0,
            max_refresh_per_window=3,
            force=False,
        )
        self.assertFalse(out["allowed"])
        self.assertEqual(int(out["count"]), 3)
        self.assertEqual(int(out["limit"]), 3)

    def test_refresh_budget_force_override_and_record(self):
        state = {"refresh_events": [100.0, 120.0, 130.0]}
        out = pm.should_allow_refresh_budget(
            state,
            now_ts=140.0,
            window_sec=60.0,
            max_refresh_per_window=3,
            force=True,
        )
        self.assertTrue(out["allowed"])
        pm.record_refresh_budget_event(state, now_ts=140.0)
        self.assertEqual(len(state.get("refresh_events", [])), 4)


if __name__ == "__main__":
    unittest.main()
