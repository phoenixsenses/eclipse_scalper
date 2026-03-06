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


class TelemetrySmokeAssertTests(unittest.TestCase):
    def test_skip_without_expectations(self):
        rc = tsa.main(["--state", "missing.json"])
        self.assertEqual(rc, 0)

    def test_fail_when_state_missing_and_expectation_set(self):
        rc = tsa.main(["--state", "missing.json", "--expected-level", "critical"])
        self.assertEqual(rc, 2)

    def test_pass_on_matching_expectations(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(
                json.dumps(
                    {
                        "level": "critical",
                        "recovery_stage_latest": "RED_LOCK",
                        "recovery_red_lock_streak": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = tsa.main(
                [
                    "--state",
                    str(p),
                    "--expected-level",
                    "critical",
                    "--expected-stage",
                    "RED_LOCK",
                    "--expected-red-lock-streak",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)

    def test_fail_on_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(
                json.dumps(
                    {
                        "level": "normal",
                        "recovery_stage_latest": "POST_RED_WARMUP",
                        "recovery_red_lock_streak": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rc = tsa.main(
                [
                    "--state",
                    str(p),
                    "--expected-level",
                    "critical",
                    "--expected-stage",
                    "RED_LOCK",
                    "--expected-red-lock-streak",
                    "2",
                ]
            )
            self.assertEqual(rc, 1)

    def test_invalid_expected_streak(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(json.dumps({"level": "normal"}) + "\n", encoding="utf-8")
            rc = tsa.main(
                [
                    "--state",
                    str(p),
                    "--expected-red-lock-streak",
                    "abc",
                ]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
