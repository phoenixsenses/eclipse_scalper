#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
import sys
import json
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools import telemetry_dashboard_notify as tdn  # noqa: E402


class TelemetryDashboardNotifyTests(unittest.TestCase):
    def test_extract_unlock_metrics_parses_expected_values(self):
        out = tdn._extract_unlock_metrics(
            "healthy_ticks 1/3; journal_coverage 0.900/0.950; "
            "contradiction_clear 10s/60s; protection_gap 5.0s/30.0s",
            42.0,
        )
        self.assertAlmostEqual(float(out.get("unlock_next_sec", 0.0)), 42.0, places=6)
        self.assertEqual(float(out.get("unlock_healthy_ticks_remaining", 0.0)), 2.0)
        self.assertAlmostEqual(float(out.get("unlock_journal_coverage_remaining", 0.0)), 0.05, places=6)
        self.assertEqual(float(out.get("unlock_contradiction_clear_remaining_sec", 0.0)), 50.0)
        self.assertEqual(float(out.get("unlock_protection_gap_remaining_sec", 0.0)), 0.0)

    def test_extract_unlock_metrics_prefers_structured_snapshot_when_present(self):
        out = tdn._extract_unlock_metrics(
            "",
            30.0,
            {
                "healthy_ticks_current": 2,
                "healthy_ticks_required": 5,
                "journal_coverage_current": 0.91,
                "journal_coverage_required": 0.95,
                "contradiction_clear_current_sec": 20.0,
                "contradiction_clear_required_sec": 60.0,
                "protection_gap_current_sec": 4.0,
                "protection_gap_max_sec": 1.0,
            },
        )
        self.assertEqual(float(out.get("unlock_healthy_ticks_remaining", 0.0)), 3.0)
        self.assertAlmostEqual(float(out.get("unlock_journal_coverage_remaining", 0.0)), 0.04, places=6)
        self.assertEqual(float(out.get("unlock_contradiction_clear_remaining_sec", 0.0)), 40.0)
        self.assertEqual(float(out.get("unlock_protection_gap_remaining_sec", 0.0)), 3.0)

    def test_decide_notify_reasons(self):
        send, reason = tdn._decide_notify({}, {"level": "normal"})
        self.assertTrue(send)
        self.assertEqual(reason, "initial_state")

        send, reason = tdn._decide_notify({"level": "normal"}, {"level": "critical"})
        self.assertTrue(send)
        self.assertEqual(reason, "level_transition:normal->critical")

        prev = {"level": "critical", "reconcile_gate_count": 1}
        curr = {"level": "critical", "reconcile_gate_count": 2}
        send, reason = tdn._decide_notify(prev, curr)
        self.assertTrue(send)
        self.assertEqual(reason, "critical_worsened")

        send, reason = tdn._decide_notify({"level": "critical"}, {"level": "critical"})
        self.assertFalse(send)
        self.assertEqual(reason, "critical_unchanged")

        send, reason = tdn._decide_notify({"level": "normal"}, {"level": "normal"})
        self.assertFalse(send)
        self.assertEqual(reason, "normal_unchanged")

    def test_is_worsened_when_runtime_gate_cause_summary_degrades(self):
        prev = {"runtime_gate_cause_summary": "stable", "reliability_coverage": 1.0}
        curr = {"runtime_gate_cause_summary": "position_peak=2 current=0", "reliability_coverage": 1.0}
        self.assertTrue(tdn._is_worsened(prev, curr))

    def test_reliability_gate_snippet_parses_metrics(self):
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
                        "intent_collision_count=2",
                        "protection_coverage_gap_seconds=4.0",
                        "protection_coverage_gap_seconds_peak=9.5",
                        "replace_race_count=2",
                        "evidence_contradiction_count=1",
                        'replay_mismatch_categories={"ledger":1,"transition":1,"belief":0,"position":2,"orphan":0,"coverage_gap":0,"replace_race":0,"contradiction":0,"unknown":0}',
                        "replay_mismatch_ids:",
                        "- CID-A",
                        "- CID-B",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, degraded, metrics = tdn._reliability_gate_snippet(p)
            self.assertIn("status=DEGRADED", snippet)
            self.assertIn("replay_mismatch=2", snippet)
            self.assertIn("invalid_transitions=1", snippet)
            self.assertIn("journal_coverage=0.750", snippet)
            self.assertIn("position_mismatch=1 (peak=3) orphan=2 intent_collision=2", snippet)
            self.assertIn("coverage_gap_sec=4.0 (peak=9.5) replace_race=2 contradiction=1", snippet)
            self.assertIn("mismatch_categories: ledger=1 transition=1 belief=0 position=2 orphan=0", snippet)
            self.assertIn("critical_contributors: position=2", snippet)
            self.assertIn("missing_ids: CID-A, CID-B", snippet)
            self.assertTrue(degraded)
            self.assertEqual(int(metrics.get("replay_mismatch_count", 0)), 2)
            self.assertEqual(int(metrics.get("position_mismatch_count", 0)), 1)
            self.assertEqual(int(metrics.get("position_mismatch_count_peak", 0)), 3)
            self.assertEqual(int(metrics.get("orphan_count", 0)), 2)
            self.assertEqual(int(metrics.get("intent_collision_count", 0)), 2)
            self.assertAlmostEqual(float(metrics.get("protection_coverage_gap_seconds", 0.0)), 4.0, places=6)
            self.assertAlmostEqual(float(metrics.get("protection_coverage_gap_seconds_peak", 0.0)), 9.5, places=6)
            self.assertEqual(int(metrics.get("replace_race_count", 0)), 2)
            self.assertEqual(int(metrics.get("evidence_contradiction_count", 0)), 1)
            self.assertEqual(int(metrics.get("replay_mismatch_cat_ledger", 0)), 1)
            self.assertEqual(int(metrics.get("replay_mismatch_cat_position", 0)), 2)

    def test_reliability_gate_snippet_empty_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "missing.txt"
            snippet, degraded, metrics = tdn._reliability_gate_snippet(p)
            self.assertEqual(snippet, "")
            self.assertFalse(degraded)
            self.assertEqual(metrics, {})

    def test_reconcile_first_gate_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.40}}',
                        '{"event":"entry.reconcile_first_gate","symbol":"ETHUSDT","data":{"reason":"coverage<0.90","runtime_gate_degrade_score":0.85}}',
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.50}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, count, max_sev, max_streak = tdn._reconcile_first_gate_snippet(p, severity_threshold=0.80)
            self.assertIn("Reconcile-first gate: events=3", snippet)
            self.assertIn("max_severity=0.85", snippet)
            self.assertIn("max_streak=1", snippet)
            self.assertIn("mismatch>0: 2", snippet)
            self.assertIn("BTCUSDT: 2", snippet)
            self.assertEqual(count, 3)
            self.assertAlmostEqual(max_sev, 0.85, places=6)
            self.assertEqual(max_streak, 1)
            self.assertIn("top severity symbols:", snippet)
            self.assertIn("ETHUSDT: 0.85", snippet)

    def test_recovery_stage_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"execution.belief_state","data":{"guard_recovery_stage":"RED_LOCK","guard_unlock_conditions":"wait stability window"}}',
                        '{"event":"execution.belief_state","data":{"guard_recovery_stage":"POST_RED_WARMUP","guard_unlock_conditions":"post-red warmup remaining 120s"}}',
                        '{"event":"execution.belief_state","data":{"guard_recovery_stage":"POST_RED_WARMUP","guard_unlock_conditions":"post-red warmup remaining 90s"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, latest = tdn._recovery_stage_snippet(p)
            self.assertIn("Recovery stages:", snippet)
            self.assertIn("POST_RED_WARMUP:2", snippet)
            self.assertIn("latest=POST_RED_WARMUP", snippet)
            self.assertEqual(latest, "POST_RED_WARMUP")

    def test_entry_budget_pressure_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"entry.blocked","symbol":"BTCUSDT","data":{"reason":"entry_budget_depleted"}}',
                        '{"event":"entry.notional_scaled","symbol":"ETHUSDT","data":{"reason":"entry_budget_allocator"}}',
                        '{"event":"entry.notional_scaled","symbol":"ETHUSDT","data":{"reason":"entry_budget_allocator"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, depleted = tdn._entry_budget_pressure_snippet(p)
            self.assertIn("Entry budget pressure: depleted=1 scaled=2", snippet)
            self.assertIn("ETHUSDT: 2", snippet)
            self.assertEqual(depleted, 1)

    def test_replace_envelope_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"order.replace_envelope_block","symbol":"BTCUSDT","data":{"reason":"replace_envelope_block"}}',
                        '{"event":"order.replace_envelope_block","symbol":"ETHUSDT","data":{"reason":"replace_ambiguity_cap"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, total = tdn._replace_envelope_snippet(p)
            self.assertIn("Replace envelope blocks: events=2", snippet)
            self.assertIn("replace_envelope_block: 1", snippet)
            self.assertEqual(total, 2)

    def test_correlation_contribution_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"entry.decision","symbol":"BTCUSDT","data":{"action":"SCALE","corr_pressure":0.74,"corr_regime":"TIGHTENING","corr_reason_tags":"downside_corr"}}',
                        '{"event":"entry.blocked","symbol":"ETHUSDT","data":{"reason":"belief_controller_block","corr_pressure":0.92,"corr_regime":"STRESS","corr_reason_tags":"tail_coupling"}}',
                        '{"event":"execution.correlation_state","data":{"corr_regime":"STRESS","corr_pressure":0.88,"corr_reason_tags":"tail_coupling,belief_uplift"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, metrics = tdn._correlation_contribution_snippet(p)
            self.assertIn("Correlation contribution: blocked=1 scaled=1 stress=1 tightening=1", snippet)
            self.assertIn("latest regime=STRESS pressure=0.88", snippet)
            self.assertEqual(int(metrics.get("corr_blocked", 0)), 1)
            self.assertEqual(int(metrics.get("corr_scaled", 0)), 1)
            self.assertEqual(int(metrics.get("corr_stress", 0)), 1)

    def test_corr_vs_exit_quality_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"execution.correlation_state","ts":100.0,"data":{"corr_regime":"NORMAL","corr_pressure":0.25}}',
                        '{"event":"position.closed","ts":101.0,"symbol":"BTCUSDT","data":{"pnl_usdt":1.2,"duration_sec":120}}',
                        '{"event":"execution.correlation_state","ts":200.0,"data":{"corr_regime":"STRESS","corr_pressure":0.88}}',
                        '{"event":"position.closed","ts":201.0,"symbol":"ETHUSDT","data":{"pnl_usdt":-0.4,"duration_sec":80}}',
                        '{"event":"position.closed","ts":202.0,"symbol":"SOLUSDT","data":{"pnl_usdt":0.0,"duration_sec":60}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, metrics = tdn._corr_vs_exit_quality_snippet(p, limit=4)
            self.assertIn("Corr vs exit quality:", snippet)
            self.assertIn("NORMAL", snippet)
            self.assertIn("STRESS", snippet)
            self.assertEqual(int(metrics.get("corr_exit_stress_count", 0)), 2)
            self.assertEqual(int(metrics.get("corr_exit_normal_count", 0)), 1)
            self.assertGreater(float(metrics.get("corr_exit_stress_vs_normal_pnl_delta", 0.0)), 0.0)

    def test_protection_refresh_budget_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "telemetry.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"event":"reconcile.summary","data":{"protection_refresh_budget_blocked_count":1,"protection_refresh_budget_force_override_count":0}}',
                        '{"event":"reconcile.summary","data":{"protection_refresh_budget_blocked_count":3,"protection_refresh_budget_force_override_count":2,"protection_refresh_stop_budget_blocked_count":2,"protection_refresh_tp_budget_blocked_count":1,"protection_refresh_stop_force_override_count":1,"protection_refresh_tp_force_override_count":1}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, blocked, forced = tdn._protection_refresh_budget_snippet(p)
            self.assertIn("Protection refresh budget:", snippet)
            self.assertIn("latest blocked=3", snippet)
            self.assertIn("latest force_override=2", snippet)
            self.assertIn("peak blocked=3 force_override=2", snippet)
            self.assertEqual(blocked, 3)
            self.assertEqual(forced, 2)

    def test_reliability_gate_snippet_ok_with_relaxed_thresholds(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=1",
                        "journal_coverage_ratio=0.900",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, degraded, metrics = tdn._reliability_gate_snippet(
                p,
                max_replay_mismatch=1,
                max_invalid_transitions=0,
                min_journal_coverage=0.90,
            )
            self.assertIn("status=OK", snippet)
            self.assertFalse(degraded)
            self.assertAlmostEqual(float(metrics.get("journal_coverage_ratio", 0.0)), 0.9, places=6)

    def test_reliability_gate_snippet_degrades_on_intent_collision_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                        "intent_collision_count=1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippet, degraded, _metrics = tdn._reliability_gate_snippet(
                p,
                max_replay_mismatch=0,
                max_invalid_transitions=0,
                min_journal_coverage=0.90,
                max_intent_collision_count=0,
            )
            self.assertIn("status=DEGRADED", snippet)
            self.assertTrue(degraded)

    def test_main_sends_critical_when_reliability_degraded(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text("", encoding="utf-8")
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=2",
                        "journal_coverage_ratio=0.750",
                        "invalid_transition_count=1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertGreaterEqual(len(args), 2)
                self.assertIn("Execution reliability gate:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_normal_when_reliability_healthy(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text("", encoding="utf-8")
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertGreaterEqual(len(args), 2)
                self.assertIn("Execution reliability gate:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "normal")

    def test_main_includes_runtime_gate_cause_summary_in_notification_body(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"execution.belief_state","data":{"guard_mode":"YELLOW","allow_entries":false,'
                        '"runtime_gate_cause_summary":"position_peak=2 current=0; coverage_gap_peak=12.0s current=0.0s"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, _kwargs = send_alert.call_args
                self.assertGreaterEqual(len(args), 2)
                body = str(args[1])
                self.assertIn("Runtime gate cause summary:", body)
                self.assertIn("position_peak=2 current=0; coverage_gap_peak=12.0s current=0.0s", body)

    def test_main_sends_critical_on_reconcile_first_spike_even_when_reliability_ok(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0"}}',
                        '{"event":"entry.reconcile_first_gate","symbol":"ETHUSDT","data":{"reason":"coverage<0.90"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reconcile-first-gate-critical-threshold",
                        "2",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                _, kwargs = send_alert.call_args
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_reconcile_first_severity_streak(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.92}}',
                        '{"event":"entry.reconcile_first_gate","symbol":"ETHUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.93}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reconcile-first-gate-critical-threshold",
                        "99",
                        "--reconcile-first-gate-severity-threshold",
                        "0.90",
                        "--reconcile-first-gate-severity-streak-threshold",
                        "2",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                _, kwargs = send_alert.call_args
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_reconcile_first_severity_even_with_low_count(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.95}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reconcile-first-gate-critical-threshold",
                        "5",
                        "--reconcile-first-gate-severity-threshold",
                        "0.90",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                _, kwargs = send_alert.call_args
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_entry_budget_depleted_spike(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.blocked","symbol":"BTCUSDT","data":{"reason":"entry_budget_depleted"}}',
                        '{"event":"entry.blocked","symbol":"ETHUSDT","data":{"reason":"entry_budget_depleted"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--entry-budget-depleted-critical-threshold",
                        "2",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertIn("Entry budget pressure:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_correlation_stress_even_when_reliability_ok(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.blocked","symbol":"BTCUSDT","data":{"reason":"belief_controller_block","corr_pressure":0.93,"corr_regime":"STRESS","corr_reason_tags":"tail_coupling"}}',
                        '{"event":"execution.correlation_state","data":{"corr_regime":"STRESS","corr_pressure":0.93,"corr_reason_tags":"tail_coupling"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--corr-stress-threshold",
                        "1",
                        "--corr-pressure-threshold",
                        "0.90",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertIn("Correlation contribution:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_corr_exit_quality_drop(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"execution.correlation_state","ts":100.0,"data":{"corr_regime":"NORMAL","corr_pressure":0.25}}',
                        '{"event":"position.closed","ts":101.0,"symbol":"BTCUSDT","data":{"pnl_usdt":1.2,"duration_sec":120}}',
                        '{"event":"execution.correlation_state","ts":200.0,"data":{"corr_regime":"STRESS","corr_pressure":0.88}}',
                        '{"event":"position.closed","ts":201.0,"symbol":"ETHUSDT","data":{"pnl_usdt":-0.4,"duration_sec":80}}',
                        '{"event":"position.closed","ts":202.0,"symbol":"SOLUSDT","data":{"pnl_usdt":0.0,"duration_sec":60}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--corr-stress-threshold",
                        "99",
                        "--corr-pressure-threshold",
                        "0.99",
                        "--corr-exit-pnl-drop-threshold",
                        "0.10",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertIn("Corr vs exit quality:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_replace_envelope_spike(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"order.replace_envelope_block","symbol":"BTCUSDT","data":{"reason":"replace_envelope_block"}}',
                        '{"event":"order.replace_envelope_block","symbol":"ETHUSDT","data":{"reason":"replace_ambiguity_cap"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--replace-envelope-critical-threshold",
                        "2",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertIn("Replace envelope blocks:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_sends_critical_on_refresh_budget_spike(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"reconcile.summary","data":{"protection_refresh_budget_blocked_count":2,"protection_refresh_budget_force_override_count":0,"protection_refresh_stop_budget_blocked_count":2}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--protection-refresh-budget-blocked-critical-threshold",
                        "2",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                args, kwargs = send_alert.call_args
                self.assertIn("Protection refresh budget:", str(args[1]))
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_refresh_budget_release_uses_latest_not_peak(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"reconcile.summary","data":{"protection_refresh_budget_blocked_count":2,"protection_refresh_budget_force_override_count":1}}',
                        '{"event":"reconcile.summary","data":{"protection_refresh_budget_blocked_count":0,"protection_refresh_budget_force_override_count":0}}',
                        '{"event":"execution.belief_state","data":{"allow_entries":true,"guard_recovery_stage":"GREEN","trace":{"reason":"stable"}}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--protection-refresh-budget-blocked-critical-threshold",
                        "2",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(send_alert.called)
                _, kwargs = send_alert.call_args
                self.assertEqual(kwargs.get("level"), "normal")

    def test_main_dedups_unchanged_critical_state(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.95}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc1 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reconcile-first-gate-critical-threshold",
                        "5",
                        "--reconcile-first-gate-severity-threshold",
                        "0.90",
                    ]
                )
                rc2 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reconcile-first-gate-critical-threshold",
                        "5",
                        "--reconcile-first-gate-severity-threshold",
                        "0.90",
                    ]
                )
                self.assertEqual(rc1, 0)
                self.assertEqual(rc2, 0)
                self.assertEqual(send_alert.call_count, 1)

    def test_main_transition_normal_to_critical_persists_decision_reason(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"

            tele.write_text(
                "\n".join(
                    [
                        '{"event":"entry.reconcile_first_gate","symbol":"BTCUSDT","data":{"reason":"mismatch>0","runtime_gate_degrade_score":0.95}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            state.write_text(
                json.dumps(
                    {
                        "level": "normal",
                        "reliability_mismatch": 0,
                        "reliability_invalid": 0,
                        "reliability_coverage": 1.0,
                        "reconcile_gate_count": 0,
                        "reconcile_gate_max_severity": 0.0,
                        "reconcile_gate_max_streak": 0,
                        "entry_budget_depleted": 0,
                        "replace_envelope_count": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reconcile-first-gate-critical-threshold",
                        "5",
                        "--reconcile-first-gate-severity-threshold",
                        "0.90",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertEqual(send_alert.call_count, 1)

            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved.get("level"), "critical")
            self.assertEqual(saved.get("previous_level"), "normal")
            self.assertEqual(saved.get("last_decision_reason"), "level_transition:normal->critical")
            self.assertTrue(saved.get("last_decision_sent"))

    def test_main_persists_state_even_without_notifier(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text("", encoding="utf-8")
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=None), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertFalse(send_alert.called)
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertIn("level", saved)
            self.assertIn("updated_at", saved)

    def test_main_no_notify_still_persists_state(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text("", encoding="utf-8")
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--no-notify",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertFalse(send_alert.called)
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertIn("level", saved)
            self.assertIn("updated_at", saved)

    def test_main_recovery_red_lock_streak_escalates_to_critical(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                "\n".join(
                    [
                        '{"event":"execution.belief_state","data":{"guard_recovery_stage":"RED_LOCK","guard_unlock_conditions":"wait stability window"}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc1 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--recovery-red-lock-critical-streak",
                        "2",
                    ]
                )
                rc2 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--recovery-red-lock-critical-streak",
                        "2",
                    ]
                )
                self.assertEqual(rc1, 0)
                self.assertEqual(rc2, 0)
                self.assertGreaterEqual(send_alert.call_count, 2)
                _, kwargs = send_alert.call_args
                self.assertEqual(kwargs.get("level"), "critical")

            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved.get("recovery_stage_latest"), "RED_LOCK")
            self.assertEqual(saved.get("recovery_red_lock_streak"), 2)
            self.assertEqual(saved.get("level"), "critical")

    def test_main_recovery_red_lock_streak_resets_when_stage_changes(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert"):
                tele.write_text(
                    '{"event":"execution.belief_state","data":{"guard_recovery_stage":"RED_LOCK"}}\n',
                    encoding="utf-8",
                )
                rc1 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--recovery-red-lock-critical-streak",
                        "3",
                    ]
                )
                self.assertEqual(rc1, 0)
                saved1 = json.loads(state.read_text(encoding="utf-8"))
                self.assertEqual(saved1.get("recovery_red_lock_streak"), 1)

                tele.write_text(
                    '{"event":"execution.belief_state","data":{"guard_recovery_stage":"POST_RED_WARMUP"}}\n',
                    encoding="utf-8",
                )
                rc2 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--recovery-red-lock-critical-streak",
                        "3",
                    ]
                )
                self.assertEqual(rc2, 0)
                saved2 = json.loads(state.read_text(encoding="utf-8"))
                self.assertEqual(saved2.get("recovery_red_lock_streak"), 0)

    def test_main_intent_collision_streak_escalates_to_critical(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text("", encoding="utf-8")
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                        "intent_collision_count=1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=object()), \
                mock.patch.object(tdn, "_send_alert") as send_alert:
                rc1 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reliability-max-intent-collision-count",
                        "99",
                        "--intent-collision-critical-threshold",
                        "1",
                        "--intent-collision-critical-streak",
                        "2",
                    ]
                )
                self.assertEqual(rc1, 0)
                saved1 = json.loads(state.read_text(encoding="utf-8"))
                self.assertEqual(int(saved1.get("intent_collision_streak") or 0), 1)
                self.assertEqual(str(saved1.get("level") or ""), "critical")

                rc2 = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                        "--reliability-max-intent-collision-count",
                        "99",
                        "--intent-collision-critical-threshold",
                        "1",
                        "--intent-collision-critical-streak",
                        "2",
                    ]
                )
                self.assertEqual(rc2, 0)
                saved2 = json.loads(state.read_text(encoding="utf-8"))
                self.assertEqual(int(saved2.get("intent_collision_streak") or 0), 2)
                self.assertEqual(str(saved2.get("level") or ""), "critical")
                kwargs = send_alert.call_args.kwargs if send_alert.call_args else {}
                self.assertEqual(kwargs.get("level"), "critical")

    def test_main_persists_unlock_condition_fields(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            gate = Path(td) / "reliability_gate.txt"
            state = Path(td) / "notify_state.json"
            tele.write_text(
                (
                    '{"event":"execution.belief_state","data":{"guard_recovery_stage":"YELLOW_WATCH",'
                    '"guard_unlock_conditions":"healthy_ticks 1/3; journal_coverage 0.900/0.950; '
                    'contradiction_clear 10s/60s; protection_gap 5.0s/30.0s",'
                    '"guard_next_unlock_sec":42.0}}\n'
                ),
                encoding="utf-8",
            )
            gate.write_text(
                "\n".join(
                    [
                        "Execution Reliability Gate",
                        "replay_mismatch_count=0",
                        "journal_coverage_ratio=1.000",
                        "invalid_transition_count=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(tdn, "_run_dashboard", return_value=("dashboard-ok", 0)), \
                mock.patch.object(tdn, "_build_notifier", return_value=None), \
                mock.patch.object(tdn, "_send_alert"):
                rc = tdn.main(
                    [
                        "--path",
                        str(tele),
                        "--reliability-gate",
                        str(gate),
                        "--state-path",
                        str(state),
                    ]
                )
                self.assertEqual(rc, 0)
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(saved.get("unlock_next_sec", 0.0)), 42.0, places=6)
            self.assertEqual(float(saved.get("unlock_healthy_ticks_remaining", 0.0)), 2.0)
            self.assertAlmostEqual(float(saved.get("unlock_journal_coverage_remaining", 0.0)), 0.05, places=6)
            self.assertEqual(float(saved.get("unlock_contradiction_clear_remaining_sec", 0.0)), 50.0)

    def test_belief_state_snippet_uses_latest_rows_when_file_is_large(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            lines = [
                '{"event":"execution.belief_state","data":{"guard_recovery_stage":"YELLOW_WATCH","guard_unlock_conditions":"healthy_ticks 0/3"}}'
                for _ in range(5000)
            ]
            lines.append(
                (
                    '{"event":"execution.belief_state","data":{"guard_recovery_stage":"YELLOW_WATCH",'
                    '"guard_unlock_conditions":"stale text",'
                    '"guard_unlock_snapshot":{"healthy_ticks_current":1,"healthy_ticks_required":3,'
                    '"journal_coverage_current":0.91,"journal_coverage_required":0.95,'
                    '"contradiction_clear_current_sec":20.0,"contradiction_clear_required_sec":60.0,'
                    '"protection_gap_current_sec":4.0,"protection_gap_max_sec":1.0}}}'
                )
            )
            tele.write_text("\n".join(lines) + "\n", encoding="utf-8")
            snippet, latest = tdn._belief_state_snippet(tele)
            self.assertIn("unlock_snapshot healthy_ticks=1/3", snippet)
            self.assertIn("unlock_remaining healthy_ticks=2", snippet)
            self.assertEqual(str(latest.get("recovery_stage") or ""), "YELLOW_WATCH")

    def test_belief_state_snippet_renders_cause_tags_and_dominant_contributors(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry.jsonl"
            tele.write_text(
                (
                    '{"event":"execution.belief_state","data":{'
                    '"guard_mode":"ORANGE","allow_entries":false,'
                    '"runtime_gate_cause_summary":"position_peak=2 current=0; coverage_gap_peak=12.0s current=0.0s",'
                    '"guard_cause_tags":"runtime_gate,runtime_gate_position_peak,runtime_gate_coverage_gap_peak",'
                    '"guard_dominant_contributors":"position=2.0,coverage_gap=1.5"}}\n'
                ),
                encoding="utf-8",
            )
            snippet, _latest = tdn._belief_state_snippet(tele)
            self.assertIn(
                "gate_cause_summary=position_peak=2 current=0; coverage_gap_peak=12.0s current=0.0s",
                snippet,
            )
            self.assertIn("cause_tags=runtime_gate,runtime_gate_position_peak,runtime_gate_coverage_gap_peak", snippet)
            self.assertIn("dominant_contributors=position=2.0,coverage_gap=1.5", snippet)
            self.assertIn("top_contributors=position=2.0,coverage_gap=1.5", snippet)


if __name__ == "__main__":
    unittest.main()
