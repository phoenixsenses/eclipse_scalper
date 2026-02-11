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

from tools import telemetry_smoke_assert as tsa  # noqa: E402


class TelemetrySmokeAssertUnitTests(unittest.TestCase):
    def test_refresh_budget_requires_entry_clamp(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(
                json.dumps(
                    {
                        "protection_refresh_budget_blocked_count": 2,
                        "belief_allow_entries_latest": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = tsa.main(
                [
                    "--state",
                    str(p),
                    "--min-refresh-budget-blocked",
                    "2",
                    "--require-entry-clamped-on-refresh-budget",
                ]
            )
            self.assertEqual(rc, 0)

    def test_refresh_consistency_fails_when_entries_allowed_outside_warmup_or_green(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(
                json.dumps(
                    {
                        "protection_refresh_budget_blocked_count": 1,
                        "belief_allow_entries_latest": True,
                        "recovery_stage_latest": "ORANGE_RECOVERY",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = tsa.main(
                [
                    "--state",
                    str(p),
                    "--require-refresh-consistency",
                ]
            )
            self.assertEqual(rc, 1)

    def test_refresh_consistency_allows_warmup_with_entries(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(
                json.dumps(
                    {
                        "protection_refresh_budget_blocked_count": 1,
                        "belief_allow_entries_latest": True,
                        "recovery_stage_latest": "PROTECTION_REFRESH_WARMUP",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = tsa.main(
                [
                    "--state",
                    str(p),
                    "--require-refresh-consistency",
                ]
            )
            self.assertEqual(rc, 0)

    def test_require_unlock_fields_fails_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(json.dumps({"level": "normal"}) + "\n", encoding="utf-8")
            rc = tsa.main(["--state", str(p), "--require-unlock-fields"])
            self.assertEqual(rc, 1)

    def test_require_unlock_fields_passes_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(
                json.dumps(
                    {
                        "unlock_next_sec": 10.0,
                        "unlock_healthy_ticks_current": 1.0,
                        "unlock_healthy_ticks_required": 3.0,
                        "unlock_healthy_ticks_remaining": 2.0,
                        "unlock_journal_coverage_current": 0.9,
                        "unlock_journal_coverage_required": 0.95,
                        "unlock_journal_coverage_remaining": 0.05,
                        "unlock_contradiction_clear_current_sec": 10.0,
                        "unlock_contradiction_clear_required_sec": 60.0,
                        "unlock_contradiction_clear_remaining_sec": 50.0,
                        "unlock_protection_gap_current_sec": 0.0,
                        "unlock_protection_gap_max_sec": 30.0,
                        "unlock_protection_gap_remaining_sec": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = tsa.main(["--state", str(p), "--require-unlock-fields"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
