#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools import policy_alignment_check as pac  # noqa: E402


class PolicyAlignmentCheckTests(unittest.TestCase):
    def test_validate_passes_on_consistent_thresholds(self):
        kv = {
            "RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT": "0",
            "INTENT_COLLISION_CRITICAL_THRESHOLD": "1",
            "INTENT_COLLISION_CRITICAL_STREAK": "2",
            "summary.arg.reliability_max_intent_collision_count": "0",
            "summary.arg.intent_collision_critical_threshold": "1",
            "summary.arg.intent_collision_critical_streak": "2",
            "gate.intent_collision_count": "0",
            "notify.reliability_intent_collision_count": "0",
            "notify.intent_collision_streak": "0",
            "notify.level": "normal",
        }
        errs = pac._validate(kv)
        self.assertEqual(errs, [])

    def test_validate_fails_on_mismatch(self):
        kv = {
            "RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT": "0",
            "INTENT_COLLISION_CRITICAL_THRESHOLD": "1",
            "INTENT_COLLISION_CRITICAL_STREAK": "2",
            "summary.arg.reliability_max_intent_collision_count": "1",
            "summary.arg.intent_collision_critical_threshold": "1",
            "summary.arg.intent_collision_critical_streak": "2",
            "gate.intent_collision_count": "0",
            "notify.reliability_intent_collision_count": "0",
            "notify.intent_collision_streak": "0",
            "notify.level": "normal",
        }
        errs = pac._validate(kv)
        self.assertTrue(any("RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT" in e for e in errs))

    def test_parse_kv_reads_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "policy_alignment.txt"
            p.write_text("A=1\nB=2\n", encoding="utf-8")
            kv = pac._parse_kv(p)
            self.assertEqual(kv.get("A"), "1")
            self.assertEqual(kv.get("B"), "2")


if __name__ == "__main__":
    unittest.main()
