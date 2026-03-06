#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.tools import telemetry_dashboard_page as tdp  # noqa: E402
from eclipse_scalper.tools import telemetry_alert_summary as tas  # noqa: E402


class TelemetryBeliefStateTests(unittest.TestCase):
    def _events(self):
        return [
            {
                "event": "execution.belief_state",
                "data": {
                    "belief_debt_sec": 120.0,
                    "belief_debt_symbols": 2,
                    "belief_confidence": 0.85,
                    "mismatch_streak": 3,
                    "repair_actions": 1,
                    "repair_skipped": 2,
                },
            },
            {
                "event": "execution.belief_state",
                "data": {
                    "belief_debt_sec": 410.0,
                    "belief_debt_symbols": 3,
                    "belief_confidence": 0.61,
                    "mismatch_streak": 5,
                    "repair_actions": 2,
                    "repair_skipped": 1,
                },
            },
        ]

    def test_dashboard_belief_state_summary_renders(self):
        html = tdp._belief_state_summary(self._events())
        self.assertIn("Execution Belief State", html)
        self.assertIn("events=2", html)
        self.assertIn("confidence=0.61", html)

    def test_alert_summary_belief_state_lines_and_latest(self):
        lines = tas._belief_state_lines(self._events())
        self.assertTrue(lines)
        self.assertIn("latest debt=410.0s", lines[0])
        latest = tas._latest_belief_state(self._events())
        self.assertIsNotNone(latest)
        self.assertAlmostEqual(float(latest["belief_confidence"]), 0.61, places=3)

    def test_dashboard_journal_replay_summary_renders(self):
        events = [
            {"event": "order.retry", "data": {"correlation_id": "CID-1", "k": "BTCUSDT"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "execution_journal.jsonl"
            rows = [
                {
                    "ts": 1.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-1",
                        "state_from": "INTENT_CREATED",
                        "state_to": "SUBMITTED",
                        "reason": "send",
                        "correlation_id": "CID-1",
                    },
                },
                {
                    "ts": 2.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-1",
                        "state_from": "SUBMITTED",
                        "state_to": "DONE",
                        "reason": "terminal",
                        "correlation_id": "CID-1",
                    },
                },
            ]
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            html = tdp._journal_replay_summary(events, path, replay_limit=4)
            self.assertIn("Execution Journal Replay", html)
            self.assertIn("last_state=DONE", html)

    def test_alert_summary_journal_replay_lines_renders(self):
        events = [
            {"event": "order.retry", "data": {"correlation_id": "CID-2", "k": "ETHUSDT"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "execution_journal.jsonl"
            rows = [
                {
                    "ts": 1.0,
                    "event": "state.transition",
                    "data": {
                        "machine": "order_intent",
                        "entity": "CID-2",
                        "state_from": "INTENT_CREATED",
                        "state_to": "SUBMITTED",
                        "reason": "send",
                        "correlation_id": "CID-2",
                    },
                }
            ]
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            lines = tas._journal_replay_lines(events, path, limit=3)
            self.assertTrue(lines)
            self.assertIn("last_state=SUBMITTED", "\n".join(lines))

    def test_dashboard_severity_debt_trend_summary_renders(self):
        events = [
            {
                "ts": 1700000000.0,
                "event": "entry.reconcile_first_gate",
                "data": {"reconcile_first_severity": 0.90},
            },
            {
                "ts": 1700000300.0,
                "event": "entry.reconcile_first_gate",
                "data": {"reconcile_first_severity": 0.95},
            },
            {
                "ts": 1700000600.0,
                "event": "execution.belief_state",
                "data": {"runtime_gate_degrade_score": 0.70},
            },
        ]
        html = tdp._severity_debt_trend_summary(events, severity_threshold=0.85)
        self.assertIn("Severity Debt Trend", html)
        self.assertIn("threshold=0.85", html)
        self.assertIn("1h: count=", html)

    def test_dashboard_recovery_stage_summary_renders(self):
        events = [
            {
                "event": "execution.belief_state",
                "data": {
                    "guard_recovery_stage": "RED_LOCK",
                    "worst_symbols": [["BTCUSDT", 400.0], ["ETHUSDT", 200.0]],
                },
            },
            {
                "event": "execution.belief_state",
                "data": {
                    "guard_recovery_stage": "POST_RED_WARMUP",
                    "worst_symbols": [["BTCUSDT", 120.0]],
                },
            },
        ]
        html = tdp._recovery_stage_summary(events)
        self.assertIn("Recovery Stage Timeline", html)
        self.assertIn("RED_LOCK: 1", html)
        self.assertIn("POST_RED_WARMUP: 1", html)
        self.assertIn("BTCUSDT: 2", html)


if __name__ == "__main__":
    unittest.main()
