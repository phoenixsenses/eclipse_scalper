#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools import telemetry_dashboard_page as tdp  # noqa: E402


class TelemetryDashboardPageTests(unittest.TestCase):
    def test_notify_state_summary_renders_current_and_previous(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry_dashboard_notify_state.json"
            p.write_text(
                json.dumps(
                    {
                        "level": "critical",
                        "previous_level": "normal",
                        "last_decision_reason": "level_transition:normal->critical",
                        "last_decision_sent": True,
                        "reliability_mismatch": 2,
                        "previous_reliability_mismatch": 0,
                        "reliability_coverage": 0.75,
                        "previous_reliability_coverage": 1.0,
                        "reconcile_gate_max_severity": 0.95,
                        "previous_reconcile_gate_max_severity": 0.10,
                        "reconcile_gate_max_streak": 3,
                        "previous_reconcile_gate_max_streak": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            html = tdp._notify_state_summary(p)
            self.assertIn("Notifier State", html)
            self.assertIn("level: critical (prev normal)", html)
            self.assertIn("last_decision: send (level_transition:normal->critical)", html)
            self.assertIn("replay_mismatch: 2 (prev 0)", html)
            self.assertIn("journal_coverage: 0.750 (prev 1.000)", html)
            self.assertIn("mismatch_categories: ledger=0 (prev 0) transition=0 (prev 0)", html)
            self.assertIn("position=0 (prev 0)", html)
            self.assertIn("orphan=0 (prev 0)", html)
            self.assertIn("reconcile_max_streak: 3 (prev 0)", html)

    def test_reliability_gate_section_renders_category_breakdown(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "replay_mismatch_count=2",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=0.95",
                        'replay_mismatch_categories={"ledger":1,"transition":1,"belief":0,"position":0,"orphan":0,"coverage_gap":0,"replace_race":0,"contradiction":2,"unknown":0}',
                        "replay_mismatch_ids:",
                        "- CID-1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            html = tdp._read_reliability_gate_section(p)
            self.assertIn(
                "mismatch_categories: ledger=1 transition=1 belief=0 position=0 orphan=0 coverage_gap=0 replace_race=0 contradiction=2 unknown=0",
                html,
            )
            self.assertIn("critical_contributors: contradiction=2", html)
            self.assertIn("top_missing_ids: CID-1", html)


if __name__ == "__main__":
    unittest.main()
