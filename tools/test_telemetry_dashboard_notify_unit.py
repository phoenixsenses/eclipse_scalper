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
                        'replay_mismatch_categories={"ledger":1,"transition":1,"belief":0,"unknown":0}',
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
            self.assertIn("mismatch_categories: ledger=1 transition=1 belief=0 unknown=0", snippet)
            self.assertIn("missing_ids: CID-A, CID-B", snippet)
            self.assertTrue(degraded)
            self.assertEqual(int(metrics.get("replay_mismatch_count", 0)), 2)
            self.assertEqual(int(metrics.get("replay_mismatch_cat_ledger", 0)), 1)

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


if __name__ == "__main__":
    unittest.main()
