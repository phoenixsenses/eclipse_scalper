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
                        "position_mismatch_count_peak=3",
                        "orphan_count=2",
                        "intent_collision_count=1",
                        "protection_coverage_gap_seconds=12.5",
                        "protection_coverage_gap_seconds_peak=20.0",
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
            self.assertEqual(int(metrics.get("position_mismatch_count_peak", 0)), 3)
            self.assertEqual(int(metrics.get("orphan_count", 0)), 2)
            self.assertEqual(int(metrics.get("intent_collision_count", 0)), 1)
            self.assertAlmostEqual(float(metrics.get("protection_coverage_gap_seconds", 0.0)), 12.5, places=3)
            self.assertAlmostEqual(float(metrics.get("protection_coverage_gap_seconds_peak", 0.0)), 20.0, places=3)
            self.assertEqual(int(metrics.get("replace_race_count", 0)), 3)
            self.assertEqual(int(metrics.get("evidence_contradiction_count", 0)), 4)
            self.assertTrue(any("position_mismatch_count: 1 (peak 3)" in s for s in lines))
            self.assertTrue(any("intent_collision_count: 1" in s for s in lines))
            self.assertTrue(any("protection_coverage_gap_seconds: 12.5 (peak 20.0)" in s for s in lines))
            self.assertTrue(any("top_contributors:" in s for s in lines))
            self.assertTrue(any("critical_contributors:" in s for s in lines))
            self.assertTrue(any("position=1" in s for s in lines if "critical_contributors:" in s))

    def test_reliability_gate_lines_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.txt"
            lines, metrics = tas._reliability_gate_lines(p)
            self.assertEqual(lines, [])
            self.assertEqual(metrics, {})

    def test_intent_collision_policy_lines_include_thresholds_and_state(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "notify_state.json"
            state.write_text(
                '{"level":"critical","intent_collision_streak":2,"reliability_intent_collision_count":1}\n',
                encoding="utf-8",
            )
            lines, metrics = tas._intent_collision_policy_lines(
                reliability_metrics={"intent_collision_count": 1},
                notify_state_path=state,
                reliability_max_intent_collision_count=0,
                intent_collision_critical_threshold=1,
                intent_collision_critical_streak=2,
            )
            self.assertTrue(any("current_count: 1" in s for s in lines))
            self.assertTrue(any("reliability_max=0 critical=1 critical_streak=2" in s for s in lines))
            self.assertTrue(any("notifier_streak: 2" in s for s in lines))
            self.assertTrue(any("notifier_level: critical" in s for s in lines))
            self.assertEqual(int(metrics.get("intent_collision_count", 0)), 1)
            self.assertEqual(int(metrics.get("intent_collision_streak", 0)), 2)

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

    def test_belief_state_lines_include_unlock_remaining_from_snapshot(self):
        events = [
            {
                "event": "execution.belief_state",
                "data": {
                    "belief_debt_sec": 10.0,
                    "belief_debt_symbols": 1,
                    "belief_confidence": 0.80,
                    "evidence_coverage_ratio": 0.82,
                    "evidence_ws_coverage_ratio": 0.80,
                    "evidence_rest_coverage_ratio": 0.90,
                    "evidence_fill_coverage_ratio": 0.75,
                    "belief_envelope_symbols": 2,
                    "belief_envelope_ambiguous_symbols": 1,
                    "belief_position_interval_width_sum": 1.5,
                    "belief_position_interval_width_max": 1.5,
                    "belief_live_unknown_symbols": 1,
                    "belief_envelope_worst_symbol": "BTCUSDT",
                    "mismatch_streak": 2,
                    "repair_actions": 1,
                    "repair_skipped": 0,
                    "runtime_gate_cause_summary": "position_peak=2 current=0",
                    "guard_unlock_snapshot": {
                        "healthy_ticks_current": 1,
                        "healthy_ticks_required": 3,
                        "journal_coverage_current": 0.90,
                        "journal_coverage_required": 0.95,
                        "contradiction_clear_current_sec": 20.0,
                        "contradiction_clear_required_sec": 60.0,
                        "protection_gap_current_sec": 0.5,
                        "protection_gap_max_sec": 0.0,
                    },
                },
            }
        ]
        lines = tas._belief_state_lines(events)
        self.assertTrue(any("unlock_snapshot healthy_ticks=1/3" in s for s in lines))
        self.assertTrue(any("unlock_remaining healthy_ticks=2" in s for s in lines))
        self.assertTrue(any("evidence_coverage=0.820 (ws=0.800 rest=0.900 fill=0.750)" in s for s in lines))
        self.assertTrue(any("envelope symbols=2 ambiguous=1 width_sum=1.500 width_max=1.500 unknown=1 worst=BTCUSDT" in s for s in lines))
        self.assertTrue(any("journal_coverage=0.050" in s for s in lines))
        self.assertTrue(any("contradiction_clear=40s" in s for s in lines))
        self.assertTrue(any("runtime_gate_cause_summary=position_peak=2 current=0" in s for s in lines))

    def test_correlation_lines_summarize_entry_impact(self):
        events = [
            {
                "event": "entry.decision",
                "symbol": "BTCUSDT",
                "data": {"action": "SCALE", "corr_pressure": 0.72, "corr_regime": "TIGHTENING", "corr_reason_tags": "downside_corr"},
            },
            {
                "event": "entry.blocked",
                "symbol": "ETHUSDT",
                "data": {"reason": "belief_controller_block", "corr_pressure": 0.91, "corr_regime": "STRESS", "corr_reason_tags": "tail_coupling"},
            },
            {
                "event": "execution.correlation_state",
                "data": {"corr_regime": "STRESS", "corr_pressure": 0.89, "corr_reason_tags": "tail_coupling,belief_uplift"},
            },
        ]
        lines, stats = tas._correlation_lines(events, limit=5)
        self.assertTrue(lines)
        self.assertEqual(int(stats.get("scaled", 0) or 0), 1)
        self.assertEqual(int(stats.get("blocked", 0) or 0), 1)
        self.assertEqual(int(stats.get("stress", 0) or 0), 1)
        self.assertTrue(any("latest regime=STRESS" in s for s in lines))

    def test_corr_vs_exit_quality_lines(self):
        events = [
            {"event": "execution.correlation_state", "ts": 100.0, "data": {"corr_regime": "NORMAL", "corr_pressure": 0.25}},
            {"event": "position.closed", "ts": 101.0, "symbol": "BTCUSDT", "data": {"pnl_usdt": 1.2, "duration_sec": 120}},
            {"event": "execution.correlation_state", "ts": 200.0, "data": {"corr_regime": "STRESS", "corr_pressure": 0.88}},
            {"event": "position.closed", "ts": 201.0, "symbol": "ETHUSDT", "data": {"pnl_usdt": -0.4, "duration_sec": 80}},
            {"event": "position.closed", "ts": 202.0, "symbol": "SOLUSDT", "data": {"pnl_usdt": 0.0, "duration_sec": 60}},
        ]
        lines, stats = tas._corr_vs_exit_quality_lines(events, limit=4)
        self.assertTrue(lines)
        self.assertTrue(any("NORMAL" in s for s in lines))
        self.assertTrue(any("STRESS" in s for s in lines))
        self.assertEqual(int(stats.get("stress_count", 0)), 2)
        self.assertEqual(int(stats.get("normal_count", 0)), 1)
        self.assertGreater(float(stats.get("stress_vs_normal_pnl_delta", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
