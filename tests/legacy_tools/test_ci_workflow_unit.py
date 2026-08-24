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


class CiWorkflowTests(unittest.TestCase):
    def test_ci_workflow_contains_required_chaos_gate_and_pr_comment(self):
        wf = PKG / ".github" / "workflows" / "ci-tests.yml"
        text = wf.read_text(encoding="utf-8")
        required = [
            "Chaos Required - ack-after-fill-recovery",
            "Chaos Required - cancel-unknown-idempotent",
            "Chaos Required - replace-race-single-exposure",
            "Execution Invariants and Gate",
            "PR Reliability Comment",
            "Post sticky PR comment",
            "execution-reliability-pr-summary",
            "chaos-nightly",
            "Restore nightly chaos drift baseline",
            "Compare nightly chaos drift",
            "Save nightly chaos drift baseline",
            "Upload nightly chaos artifacts",
        ]
        for token in required:
            self.assertIn(token, text, f"missing workflow token: {token}")


if __name__ == "__main__":
    unittest.main()
