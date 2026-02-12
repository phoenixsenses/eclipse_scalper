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

    def test_build_report_uses_intent_ledger_ids_for_coverage(self):
        telemetry = [
            {"event": "order.retry", "data": {"correlation_id": "CID-LEDGER-1"}},
        ]
        journal = [
            {
                "event": "intent.ledger",
                "data": {"intent_id": "CID-LEDGER-1", "stage": "SUBMITTED_UNKNOWN"},
            }
        ]
        out = rg.build_report(telemetry, journal)
        self.assertEqual(int(out.get("replay_mismatch_count", 0)), 0)
        self.assertAlmostEqual(float(out.get("journal_coverage_ratio", 0.0)), 1.0, places=6)

    def test_build_report_ignores_restart_unknown_corr_ids(self):
        telemetry = [
            {"event": "order.retry", "data": {"correlation_id": "ENTRY-BTC-RESTART-UNK"}},
            {"event": "order.retry", "data": {"correlation_id": "CID-OK"}},
        ]
        journal = [
            {
                "event": "state.transition",
                "data": {
                    "machine": "order_intent",
                    "state_from": "INTENT_CREATED",
                    "state_to": "SUBMITTED",
                    "correlation_id": "CID-OK",
                },
            }
        ]
        out = rg.build_report(telemetry, journal, ignore_corr_tokens=rg._resolve_ignore_tokens(""))
        self.assertEqual(int(out.get("telemetry_corr_ids_raw", 0)), 2)
        self.assertEqual(int(out.get("telemetry_corr_ids_ignored", 0)), 1)
        self.assertEqual(int(out.get("replay_mismatch_count", 0)), 0)

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
                max_intent_collision_count=0,
            )
        )
        self.assertTrue(
            rg._passes(
                report,
                max_replay_mismatch=1,
                max_invalid_transitions=0,
                min_journal_coverage=0.5,
                max_intent_collision_count=0,
            )
        )

    def test_threshold_eval_intent_collision_count(self):
        report = {
            "replay_mismatch_count": 0,
            "invalid_transition_count": 0,
            "journal_coverage_ratio": 1.0,
            "intent_collision_count": 2,
        }
        self.assertFalse(
            rg._passes(
                report,
                max_replay_mismatch=0,
                max_invalid_transitions=0,
                min_journal_coverage=1.0,
                max_intent_collision_count=0,
            )
        )
        self.assertTrue(
            rg._passes(
                report,
                max_replay_mismatch=0,
                max_invalid_transitions=0,
                min_journal_coverage=1.0,
                max_intent_collision_count=2,
            )
        )

    def test_build_report_emits_extended_category_counts(self):
        telemetry = [
            {"event": "rebuild.orphan_decision", "data": {"symbol": "DOGEUSDT", "action": "FREEZE"}},
            {"event": "rebuild.summary", "data": {"intent_collision_count": 2}},
            {"event": "order.replace_envelope_block", "data": {"reason": "replace_envelope_block"}},
            {"event": "reconcile.summary", "data": {"protection_coverage_gap_seconds": 9.0}},
            {"event": "reconcile.summary", "data": {"protection_coverage_gap_seconds": 2.0}},
            {
                "event": "execution.belief_state",
                "data": {"mismatch_streak": 2, "belief_debt_symbols": 1, "evidence_contradiction_score": 0.8},
            },
        ]
        journal = []
        out = rg.build_report(telemetry, journal)
        self.assertGreaterEqual(int(out.get("position_mismatch_count", 0) or 0), 1)
        self.assertGreaterEqual(int(out.get("orphan_count", 0) or 0), 1)
        self.assertGreaterEqual(int(out.get("intent_collision_count", 0) or 0), 2)
        self.assertGreaterEqual(int(out.get("replace_race_count", 0) or 0), 1)
        self.assertGreaterEqual(int(out.get("evidence_contradiction_count", 0) or 0), 1)
        self.assertAlmostEqual(float(out.get("protection_coverage_gap_seconds", 0.0) or 0.0), 2.0, places=6)
        self.assertAlmostEqual(float(out.get("protection_coverage_gap_seconds_peak", 0.0) or 0.0), 9.0, places=6)

    def test_replace_race_count_dedups_same_correlation_id(self):
        telemetry = [
            {"event": "order.replace_reconcile_required", "data": {"correlation_id": "REP-1"}},
            {"event": "order.replace_reconcile_required", "data": {"correlation_id": "REP-1"}},
            {"event": "order.replace_reconcile_required", "data": {"correlation_id": "REP-2"}},
        ]
        out = rg.build_report(telemetry, [])
        self.assertEqual(int(out.get("replace_race_count", 0) or 0), 2)

    def test_position_mismatch_count_uses_latest_not_event_count(self):
        telemetry = [
            {"event": "execution.belief_state", "data": {"mismatch_streak": 2, "belief_debt_symbols": 3}},
            {"event": "execution.belief_state", "data": {"mismatch_streak": 0, "belief_debt_symbols": 0}},
        ]
        out = rg.build_report(telemetry, [])
        self.assertEqual(int(out.get("position_mismatch_count", 0) or 0), 0)
        self.assertEqual(int(out.get("position_mismatch_count_peak", 0) or 0), 3)

    def test_render_includes_critical_contributors(self):
        report = {
            "telemetry_corr_ids": 4,
            "journal_corr_ids": 3,
            "replay_mismatch_count": 1,
            "journal_coverage_ratio": 0.75,
            "invalid_transition_count": 0,
            "position_mismatch_count": 2,
            "orphan_count": 1,
            "intent_collision_count": 3,
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
        self.assertIn("intent_collision_count=3", text)

    def test_build_report_window_filters_stale_events(self):
        telemetry = [
            {"event": "order.retry", "ts": 10.0, "data": {"correlation_id": "CID-OLD"}},
            {"event": "order.retry", "ts": 200.0, "data": {"correlation_id": "CID-NEW"}},
        ]
        journal = [
            {
                "event": "state.transition",
                "ts": 200.0,
                "data": {
                    "machine": "order_intent",
                    "state_from": "INTENT_CREATED",
                    "state_to": "SUBMITTED",
                    "correlation_id": "CID-NEW",
                },
            }
        ]
        out_all = rg.build_report(telemetry, journal)
        self.assertEqual(int(out_all.get("replay_mismatch_count", 0)), 1)
        out_win = rg.build_report(telemetry, journal, window_seconds=30.0, now_ts=200.0)
        self.assertEqual(int(out_win.get("replay_mismatch_count", 0)), 0)
        self.assertEqual(int(out_win.get("telemetry_corr_ids", 0)), 1)

    def test_render_shows_window_seconds(self):
        report = {
            "window_seconds": 3600.0,
            "telemetry_corr_ids": 1,
            "journal_corr_ids": 1,
            "replay_mismatch_count": 0,
            "journal_coverage_ratio": 1.0,
            "invalid_transition_count": 0,
            "position_mismatch_count": 0,
            "orphan_count": 0,
            "protection_coverage_gap_seconds": 0.0,
            "replace_race_count": 0,
            "evidence_contradiction_count": 0,
        }
        text = rg._render(
            report,
            telemetry_path=Path("logs/telemetry.jsonl"),
            journal_path=Path("logs/execution_journal.jsonl"),
        )
        self.assertIn("window_seconds=3600.0", text)


if __name__ == "__main__":
    unittest.main()
