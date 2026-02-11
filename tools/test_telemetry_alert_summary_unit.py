#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools import telemetry_alert_summary as tas  # noqa: E402


class TelemetryAlertSummaryTests(unittest.TestCase):
    def test_reliability_gate_lines_parse_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=2",
                        "journal_coverage_ratio=0.750",
                        "invalid_transition_count=1",
                        "position_mismatch_count=1",
                        "orphan_count=2",
                        "protection_coverage_gap_seconds=12.5",
                        "replace_race_count=3",
                        "evidence_contradiction_count=4",
                        'replay_mismatch_categories={"ledger":1,"transition":2,"belief":0,"position":1,"orphan":0,"coverage_gap":0,"replace_race":0,"contradiction":0,"unknown":0}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            lines, metrics = tas._reliability_gate_lines(p)
            self.assertTrue(lines)
            self.assertEqual(int(metrics.get("replay_mismatch_count", 0)), 2)
            self.assertEqual(int(metrics.get("invalid_transition_count", 0)), 1)
            self.assertAlmostEqual(float(metrics.get("journal_coverage_ratio", 0.0)), 0.75, places=3)
            self.assertEqual(int(metrics.get("position_mismatch_count", 0)), 1)
            self.assertEqual(int(metrics.get("orphan_count", 0)), 2)
            self.assertAlmostEqual(float(metrics.get("protection_coverage_gap_seconds", 0.0)), 12.5, places=3)
            self.assertEqual(int(metrics.get("replace_race_count", 0)), 3)
            self.assertEqual(int(metrics.get("evidence_contradiction_count", 0)), 4)
            self.assertTrue(any("top_contributors:" in s for s in lines))

    def test_reliability_gate_lines_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.txt"
            lines, metrics = tas._reliability_gate_lines(p)
            self.assertEqual(lines, [])
            self.assertEqual(metrics, {})

    def test_reconcile_first_gate_lines(self):
        events = [
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "BTCUSDT",
                "data": {"reason": "mismatch>0", "reconcile_first_severity": 0.6},
            },
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "ETHUSDT",
                "data": {"reason": "coverage<0.90", "reconcile_first_severity": 0.8},
            },
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "BTCUSDT",
                "data": {"reason": "mismatch>0", "reconcile_first_severity": 0.7},
            },
        ]
        lines = tas._reconcile_first_gate_lines(events, limit=5)
        self.assertTrue(lines)
        self.assertIn("- events: 3", lines[0])
        self.assertTrue(any("mismatch>0: 2" in s for s in lines))
        self.assertTrue(any("BTCUSDT: 2" in s for s in lines))
        self.assertTrue(any("top severity symbols:" in s for s in lines))
        self.assertTrue(any("ETHUSDT: 0.80" in s for s in lines))

    def test_reconcile_first_gate_max_severity(self):
        events = [
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "BTCUSDT",
                "data": {"reason": "mismatch>0", "runtime_gate_degrade_score": 0.40},
            },
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "ETHUSDT",
                "data": {"reason": "coverage<0.90", "runtime_gate_degrade_score": 0.85},
            },
        ]
        mx = tas._reconcile_first_gate_max_severity(events)
        self.assertAlmostEqual(mx, 0.85, places=6)

    def test_reconcile_first_gate_max_streak(self):
        events = [
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "BTCUSDT",
                "data": {"runtime_gate_degrade_score": 0.91},
            },
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "ETHUSDT",
                "data": {"runtime_gate_degrade_score": 0.92},
            },
            {
                "event": "entry.reconcile_first_gate",
                "symbol": "SOLUSDT",
                "data": {"runtime_gate_degrade_score": 0.50},
            },
        ]
        mx = tas._reconcile_first_gate_max_streak(events, severity_threshold=0.90)
        self.assertEqual(mx, 2)

    def test_entry_budget_lines(self):
        events = [
            {
                "event": "entry.blocked",
                "symbol": "BTCUSDT",
                "data": {"reason": "entry_budget_depleted"},
            },
            {
                "event": "entry.notional_scaled",
                "symbol": "ETHUSDT",
                "data": {"reason": "entry_budget_allocator"},
            },
            {
                "event": "entry.notional_scaled",
                "symbol": "ETHUSDT",
                "data": {"reason": "entry_budget_allocator"},
            },
        ]
        lines, stats = tas._entry_budget_lines(events, limit=5)
        self.assertTrue(lines)
        self.assertEqual(int(stats.get("depleted_total", 0)), 1)
        self.assertEqual(int(stats.get("scaled_total", 0)), 2)
        self.assertTrue(any("ETHUSDT: 2" in s for s in lines))

    def test_replace_envelope_lines(self):
        events = [
            {
                "event": "order.replace_envelope_block",
                "symbol": "BTCUSDT",
                "data": {"reason": "replace_envelope_block"},
            },
            {
                "event": "order.replace_envelope_block",
                "symbol": "BTCUSDT",
                "data": {"reason": "replace_ambiguity_cap"},
            },
        ]
        lines, total = tas._replace_envelope_lines(events, limit=5)
        self.assertTrue(lines)
        self.assertEqual(total, 2)
        self.assertTrue(any("replace_envelope_block: 1" in s for s in lines))
        self.assertTrue(any("BTCUSDT: 2" in s for s in lines))

    def test_rebuild_orphan_lines(self):
        events = [
            {
                "event": "rebuild.orphan_decision",
                "data": {"symbol": "DOGEUSDT", "class": "orphan_entry_order", "action": "CANCEL"},
            },
            {
                "event": "rebuild.orphan_decision",
                "data": {"symbol": "BTCUSDT", "class": "unknown_position_exposure", "action": "FREEZE"},
            },
        ]
        lines, stats = tas._rebuild_orphan_lines(events, limit=5)
        self.assertTrue(lines)
        self.assertEqual(int(stats.get("total", 0)), 2)
        self.assertEqual(int(stats.get("freeze_count", 0)), 1)
        self.assertTrue(any("FREEZE: 1" in s for s in lines))


if __name__ == "__main__":
    unittest.main()
