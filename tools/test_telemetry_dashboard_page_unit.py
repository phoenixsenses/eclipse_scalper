#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

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
                        "unlock_next_sec": 42.0,
                        "unlock_healthy_ticks_remaining": 2,
                        "unlock_journal_coverage_remaining": 0.05,
                        "unlock_contradiction_clear_remaining_sec": 40.0,
                        "unlock_protection_gap_remaining_sec": 1.5,
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
            self.assertIn("intent_collision=0 (prev 0)", html)
            self.assertIn("reconcile_max_streak: 3 (prev 0)", html)
            self.assertIn("unlock_remaining: next=42.0s healthy_ticks=2 journal_coverage=0.050 contradiction_clear=40s protection_gap=1.5s", html)

    def test_reliability_gate_section_renders_category_breakdown(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "replay_mismatch_count=2",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=0.95",
                        "position_mismatch_count=1",
                        "position_mismatch_count_peak=4",
                        "orphan_count=2",
                        "intent_collision_count=3",
                        "protection_coverage_gap_seconds=3.0",
                        "protection_coverage_gap_seconds_peak=7.5",
                        "replace_race_count=1",
                        "evidence_contradiction_count=2",
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
            self.assertIn("position_mismatch_count: 1 (peak 4)", html)
            self.assertIn("orphan_count: 2", html)
            self.assertIn("intent_collision_count: 3", html)
            self.assertIn("protection_coverage_gap_seconds: 3.0 (peak 7.5)", html)
            self.assertIn("replace_race_count: 1", html)
            self.assertIn("evidence_contradiction_count: 2", html)
            self.assertIn("critical_contributors: contradiction=2", html)
            self.assertIn("top_missing_ids: CID-1", html)

    def test_protection_refresh_budget_summary_renders_counts(self):
        events = [
            {
                "event": "reconcile.summary",
                "data": {
                    "protection_refresh_budget_blocked_count": 2,
                    "protection_refresh_budget_force_override_count": 1,
                    "protection_refresh_stop_budget_blocked_count": 2,
                    "protection_refresh_tp_budget_blocked_count": 0,
                    "protection_refresh_stop_force_override_count": 1,
                    "protection_refresh_tp_force_override_count": 0,
                },
            }
        ]
        html = tdp._protection_refresh_budget_summary(events)
        self.assertIn("Protection Refresh Budget", html)
        self.assertIn("blocked_total: 2 (stop=2, tp=0)", html)
        self.assertIn("force_override_total: 1 (stop=1, tp=0)", html)

    def test_top_reliability_strip_renders_latest_metrics(self):
        events = [
            {"event": "execution.belief_state", "data": {"guard_mode": "YELLOW", "allow_entries": False, "belief_debt_sec": 42.0, "mismatch_streak": 3, "guard_refresh_blocked_level": 1.2, "guard_refresh_force_level": 0.4, "corr_regime": "TIGHTENING", "corr_pressure": 0.74, "runtime_gate_intent_collision_count": 2, "intent_collision_streak": 3}},
            {"event": "reconcile.summary", "data": {"protection_coverage_gap_seconds": 9.5, "protection_refresh_budget_blocked_count": 2, "protection_refresh_budget_force_override_count": 1, "runtime_gate_intent_collision_count": 2}},
        ]
        html = tdp._top_reliability_strip(events)
        self.assertIn("mode", html)
        self.assertIn("YELLOW", html)
        self.assertIn("allow_entries", html)
        self.assertIn("false", html)
        self.assertIn("42.0", html)
        self.assertIn("9.5", html)
        self.assertIn("refresh_blocked", html)
        self.assertIn("force_override", html)
        self.assertIn("2 (1.20)", html)
        self.assertIn("1 (0.40)", html)
        self.assertIn("intent_collision", html)
        self.assertIn("collision_streak", html)
        self.assertIn(">2</div>", html)
        self.assertIn(">3</div>", html)
        self.assertIn("class='metric warn'><strong>mode</strong>", html)
        self.assertIn("class='metric crit'><strong>allow_entries</strong>", html)
        self.assertIn("class='metric crit'><strong>coverage_gap_s</strong>", html)
        self.assertIn("class='metric crit'><strong>intent_collision</strong>", html)
        self.assertIn("class='metric crit'><strong>collision_streak</strong>", html)
        self.assertIn("corr_state", html)
        self.assertIn("TIGHTENING (0.74)", html)
        self.assertIn("class='metric warn'><strong>corr_state</strong>", html)
        self.assertIn("corr_state thresholds: warn>=0.60, crit>=0.80", html)
        self.assertIn("TELEMETRY_DASHBOARD_CORR_WARN / TELEMETRY_DASHBOARD_CORR_CRIT", html)

    def test_top_reliability_strip_uses_env_threshold_overrides(self):
        events = [
            {"event": "execution.belief_state", "data": {"guard_mode": "GREEN", "allow_entries": True, "belief_debt_sec": 6.0, "mismatch_streak": 0}},
            {"event": "reconcile.summary", "data": {"protection_coverage_gap_seconds": 0.0, "protection_refresh_budget_blocked_count": 0, "protection_refresh_budget_force_override_count": 0}},
        ]
        with patch.dict(
            "os.environ",
            {
                "TELEMETRY_DASHBOARD_DEBT_WARN_SEC": "5",
                "TELEMETRY_DASHBOARD_DEBT_CRIT_SEC": "10",
            },
            clear=False,
        ):
            html = tdp._top_reliability_strip(events)
        self.assertIn("class='metric warn'><strong>debt_sec</strong>", html)

    def test_top_reliability_strip_uses_intent_collision_env_threshold_overrides(self):
        events = [
            {
                "event": "execution.belief_state",
                "data": {
                    "guard_mode": "GREEN",
                    "allow_entries": True,
                    "belief_debt_sec": 0.0,
                    "mismatch_streak": 0,
                    "runtime_gate_intent_collision_count": 1,
                    "intent_collision_streak": 1,
                },
            },
            {
                "event": "reconcile.summary",
                "data": {
                    "protection_coverage_gap_seconds": 0.0,
                    "protection_refresh_budget_blocked_count": 0,
                    "protection_refresh_budget_force_override_count": 0,
                    "runtime_gate_intent_collision_count": 1,
                },
            },
        ]
        with patch.dict(
            "os.environ",
            {
                "TELEMETRY_DASHBOARD_INTENT_COLLISION_WARN": "1",
                "TELEMETRY_DASHBOARD_INTENT_COLLISION_CRIT": "2",
                "TELEMETRY_DASHBOARD_INTENT_COLLISION_STREAK_WARN": "1",
                "TELEMETRY_DASHBOARD_INTENT_COLLISION_STREAK_CRIT": "2",
            },
            clear=False,
        ):
            html = tdp._top_reliability_strip(events)
        self.assertIn("class='metric warn'><strong>intent_collision</strong>", html)
        self.assertIn("class='metric warn'><strong>collision_streak</strong>", html)

    def test_refresh_pressure_trend_summary_renders_levels_and_delta(self):
        events = [
            {"event": "execution.belief_state", "data": {"guard_refresh_blocked_level": 2.0, "guard_refresh_force_level": 1.0, "allow_entries": False, "guard_mode": "ORANGE", "guard_recovery_stage": "RUNTIME_GATE_DEGRADED"}},
            {"event": "execution.belief_state", "data": {"guard_refresh_blocked_level": 1.5, "guard_refresh_force_level": 0.8, "allow_entries": True, "guard_mode": "YELLOW", "guard_recovery_stage": "PROTECTION_REFRESH_WARMUP"}},
        ]
        html = tdp._refresh_pressure_trend_summary(events)
        self.assertIn("Refresh Pressure Trend", html)
        self.assertIn("latest blocked_level=1.50 force_level=0.80", html)
        self.assertIn("delta blocked=-0.50 force=-0.20", html)
        self.assertIn("stage=PROTECTION_REFRESH_WARMUP", html)

    def test_belief_state_summary_renders_unlock_snapshot_when_present(self):
        events = [
            {
                "event": "execution.belief_state",
                "data": {
                    "belief_debt_sec": 10.0,
                    "belief_debt_symbols": 1,
                    "belief_confidence": 0.8,
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
                    "guard_mode": "YELLOW",
                    "allow_entries": True,
                    "guard_recovery_stage": "YELLOW_WATCH",
                    "runtime_gate_cause_summary": "position_peak=2 current=0",
                    "guard_cause_tags": "runtime_gate,runtime_gate_position_peak",
                    "guard_dominant_contributors": "position=1.5",
                    "guard_unlock_conditions": "stable",
                    "guard_next_unlock_sec": 0.0,
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
        html = tdp._belief_state_summary(events)
        self.assertIn("unlock snapshot:", html)
        self.assertIn("latest gate_cause_summary=position_peak=2 current=0", html)
        self.assertIn("latest evidence_coverage=0.820 (ws=0.800 rest=0.900 fill=0.750)", html)
        self.assertIn("latest envelope symbols=2 ambiguous=1 width_sum=1.500 width_max=1.500 unknown=1 worst=BTCUSDT", html)
        self.assertIn("latest cause_tags=runtime_gate,runtime_gate_position_peak", html)
        self.assertIn("latest dominant_contributors=position=1.5", html)
        self.assertIn("latest top_contributors=position=1.5", html)
        self.assertIn("healthy_ticks=1/3", html)
        self.assertIn("journal_coverage=0.900/0.950", html)
        self.assertIn("contradiction_clear=20s/60s", html)
        self.assertIn("unlock_remaining healthy_ticks=2", html)
        self.assertIn("journal_coverage=0.050", html)

    def test_correlation_contribution_summary_renders_entry_impact(self):
        events = [
            {
                "event": "entry.decision",
                "symbol": "BTCUSDT",
                "data": {"action": "SCALE", "corr_pressure": 0.74, "corr_regime": "TIGHTENING", "corr_reason_tags": "downside_corr"},
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
        html = tdp._correlation_contribution_summary(events)
        self.assertIn("Correlation Contribution", html)
        self.assertIn("entry impact blocked=1 scaled=1", html)
        self.assertIn("latest regime=STRESS pressure=0.88", html)
        self.assertIn("tail_coupling", html)

    def test_correlation_symbol_table_renders_blocked_scaled_and_tag(self):
        events = [
            {
                "event": "entry.decision",
                "symbol": "BTCUSDT",
                "data": {"action": "SCALE", "corr_pressure": 0.74, "corr_regime": "TIGHTENING", "corr_reason_tags": "downside_corr"},
            },
            {
                "event": "entry.blocked",
                "symbol": "BTCUSDT",
                "data": {"reason": "belief_controller_block", "corr_pressure": 0.92, "corr_regime": "STRESS", "corr_reason_tags": "tail_coupling"},
            },
            {
                "event": "entry.blocked",
                "symbol": "ETHUSDT",
                "data": {"reason": "belief_controller_block", "corr_pressure": 0.81, "corr_regime": "TIGHTENING", "corr_reason_tags": "downside_corr"},
            },
        ]
        html = tdp._correlation_symbol_table(events, limit=8)
        self.assertIn("Correlation Impact By Symbol", html)
        self.assertIn("SYMBOL", html)
        self.assertIn("BTCUSDT", html)
        self.assertIn("ETHUSDT", html)
        self.assertIn("tail_coupling", html)

    def test_corr_vs_exit_quality_summary_groups_by_regime(self):
        events = [
            {"event": "execution.correlation_state", "ts": 100.0, "data": {"corr_regime": "NORMAL", "corr_pressure": 0.25}},
            {"event": "position.closed", "ts": 101.0, "symbol": "BTCUSDT", "data": {"pnl_usdt": 1.5, "duration_sec": 120}},
            {"event": "execution.correlation_state", "ts": 200.0, "data": {"corr_regime": "STRESS", "corr_pressure": 0.88}},
            {"event": "position.closed", "ts": 201.0, "symbol": "ETHUSDT", "data": {"pnl_usdt": -0.5, "duration_sec": 80}},
            {"event": "position.closed", "ts": 202.0, "symbol": "SOLUSDT", "data": {"pnl_usdt": 0.2, "duration_sec": 60}},
        ]
        html = tdp._corr_vs_exit_quality_summary(events, limit=8)
        self.assertIn("Corr vs Exit Quality", html)
        self.assertIn("NORMAL", html)
        self.assertIn("STRESS", html)
        self.assertIn("50.0%", html)


if __name__ == "__main__":
    unittest.main()
