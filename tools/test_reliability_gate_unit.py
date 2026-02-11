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

    def test_build_report_emits_extended_category_counts(self):
        telemetry = [
            {"event": "rebuild.orphan_decision", "data": {"symbol": "DOGEUSDT", "action": "FREEZE"}},
            {"event": "order.replace_envelope_block", "data": {"reason": "replace_envelope_block"}},
            {
                "event": "execution.belief_state",
                "data": {"mismatch_streak": 2, "belief_debt_symbols": 1, "evidence_contradiction_score": 0.8},
            },
        ]
        journal = []
        out = rg.build_report(telemetry, journal)
        self.assertGreaterEqual(int(out.get("position_mismatch_count", 0) or 0), 1)
        self.assertGreaterEqual(int(out.get("orphan_count", 0) or 0), 1)
        self.assertGreaterEqual(int(out.get("replace_race_count", 0) or 0), 1)
        self.assertGreaterEqual(int(out.get("evidence_contradiction_count", 0) or 0), 1)

    def test_render_includes_critical_contributors(self):
        report = {
            "telemetry_corr_ids": 4,
            "journal_corr_ids": 3,
            "replay_mismatch_count": 1,
            "journal_coverage_ratio": 0.75,
            "invalid_transition_count": 0,
            "position_mismatch_count": 2,
            "orphan_count": 1,
            "protection_coverage_gap_seconds": 9.0,
            "replace_race_count": 3,
            "evidence_contradiction_count": 4,
            "replay_mismatch_ids": ["CID-X"],
            "replay_mismatch_categories": {
                "ledger": 0,
                "transition": 0,
                "belief": 0,
                "position": 2,
                "orphan": 1,
                "coverage_gap": 1,
                "replace_race": 3,
                "contradiction": 4,
                "unknown": 0,
            },
        }
        text = rg._render(
            report,
            telemetry_path=Path("logs/telemetry.jsonl"),
            journal_path=Path("logs/execution_journal.jsonl"),
        )
        self.assertIn("top_contributors=contradiction:4,replace_race:3,position:2", text)
        self.assertIn("critical_contributors=contradiction:4,replace_race:3,position:2", text)


if __name__ == "__main__":
    unittest.main()
