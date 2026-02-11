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


class TelemetryWorkflowFlagsTests(unittest.TestCase):
    def test_telemetry_dashboard_workflow_has_required_reconcile_flags(self):
        wf = PKG / ".github" / "workflows" / "telemetry-dashboard.yml"
        text = wf.read_text(encoding="utf-8")
        required = [
            "--reconcile-first-gate-critical-threshold",
            "--reconcile-first-gate-severity-threshold",
            "--reconcile-first-gate-severity-streak-threshold",
            "--recovery-red-lock-critical-streak",
        ]
        for flag in required:
            self.assertIn(flag, text, f"missing required workflow flag: {flag}")


if __name__ == "__main__":
    unittest.main()
