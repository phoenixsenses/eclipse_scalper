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

from eclipse_scalper.tools import reliability_gate as rg  # noqa: E402


class ReliabilityGateTests(unittest.TestCase):
    def test_build_report_detects_replay_mismatch(self):
        telemetry = [
            {"event": "order.retry", "data": {"correlation_id": "CID-1"}},
            {"event": "order.create_failed", "data": {"correlation_id": "CID-2"}},
        ]
        journal = [
            {
                "event": "state.transition",
                "data": {"machine": "order_intent", "state_from": "INTENT_CREATED", "state_to": "SUBMITTED", "correlation_id": "CID-1"},
            }
        ]
        out = rg.build_report(telemetry, journal)
        self.assertEqual(int(out.get("telemetry_corr_ids", 0)), 2)
        self.assertEqual(int(out.get("replay_mismatch_count", 0)), 1)
        self.assertLess(float(out.get("journal_coverage_ratio", 1.0)), 1.0)
        cats = dict(out.get("replay_mismatch_categories") or {})
        self.assertEqual(int(cats.get("unknown", 0) or 0), 1)

    def test_invalid_transition_count(self):
        telemetry = []
        journal = [
            {
                "event": "state.transition",
                "data": {"machine": "order_intent", "state_from": "DONE", "state_to": "OPEN", "correlation_id": "CID-X"},
            }
        ]
        out = rg.build_report(telemetry, journal)
        self.assertGreaterEqual(int(out.get("invalid_transition_count", 0)), 1)

    def test_threshold_eval(self):
        report = {
            "replay_mismatch_count": 1,
            "invalid_transition_count": 0,
            "journal_coverage_ratio": 0.5,
        }
        self.assertFalse(
            rg._passes(
                report,
                max_replay_mismatch=0,
                max_invalid_transitions=0,
                min_journal_coverage=0.9,
            )
        )
        self.assertTrue(
            rg._passes(
                report,
                max_replay_mismatch=1,
                max_invalid_transitions=0,
                min_journal_coverage=0.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
