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


class TelemetrySmokeWorkflowTests(unittest.TestCase):
    def test_smoke_workflow_has_chained_assertions(self):
        wf = PKG / ".github" / "workflows" / "telemetry-smoke.yml"
        text = wf.read_text(encoding="utf-8")
        required = [
            "Telemetry Smoke Assertions",
            "workflow_dispatch",
            "telemetry_smoke_result.txt",
            "tools/telemetry_dashboard_notify.py",
            "tools/telemetry_smoke_assert.py",
            "--expected-level critical",
            "--expected-stage RED_LOCK",
            "--expected-level normal",
            "--expected-stage POST_RED_WARMUP",
            "phase1=PASS escalation",
            "phase2=PASS reset",
            "Upload smoke artifacts",
        ]
        for token in required:
            self.assertIn(token, text, f"missing workflow token: {token}")


if __name__ == "__main__":
    unittest.main()
