#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import reliability_gate_runtime as rgr  # noqa: E402


class ReliabilityGateRuntimeTests(unittest.TestCase):
    def test_missing_gate_file_is_available_false(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = type("Cfg", (), {"RELIABILITY_GATE_PATH": str(Path(td) / "missing.txt")})()
            out = rgr.get_runtime_gate(cfg)
            self.assertFalse(bool(out.get("available")))
            self.assertFalse(bool(out.get("degraded")))

    def test_degraded_when_threshold_breached(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "replay_mismatch_count=2",
                        "invalid_transition_count=1",
                        "journal_coverage_ratio=0.80",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = type(
                "Cfg",
                (),
                {
                    "RELIABILITY_GATE_PATH": str(p),
                    "RELIABILITY_GATE_MAX_REPLAY_MISMATCH": 0,
                    "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS": 0,
                    "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE": 0.90,
                },
            )()
            out = rgr.get_runtime_gate(cfg)
            self.assertTrue(bool(out.get("available")))
            self.assertTrue(bool(out.get("degraded")))
            self.assertIn("mismatch>0", str(out.get("reason")))

    def test_ok_when_within_thresholds(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "replay_mismatch_count=1",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=0.95",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = type(
                "Cfg",
                (),
                {
                    "RELIABILITY_GATE_PATH": str(p),
                    "RELIABILITY_GATE_MAX_REPLAY_MISMATCH": 1,
                    "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS": 0,
                    "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE": 0.90,
                },
            )()
            out = rgr.get_runtime_gate(cfg)
            self.assertTrue(bool(out.get("available")))
            self.assertFalse(bool(out.get("degraded")))
            self.assertEqual(str(out.get("reason") or ""), "ok")

    def test_parses_mismatch_ids_and_severity(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "replay_mismatch_count=3",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=0.80",
                        "replay_mismatch_ids:",
                        "- CID-1",
                        "- CID-2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = type(
                "Cfg",
                (),
                {
                    "RELIABILITY_GATE_PATH": str(p),
                    "RELIABILITY_GATE_MAX_REPLAY_MISMATCH": 1,
                    "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS": 0,
                    "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE": 0.90,
                },
            )()
            out = rgr.get_runtime_gate(cfg)
            self.assertTrue(bool(out.get("degraded")))
            self.assertGreater(float(out.get("mismatch_severity", 0.0) or 0.0), 0.0)
            self.assertGreater(float(out.get("coverage_severity", 0.0) or 0.0), 0.0)
            self.assertGreater(float(out.get("degrade_score", 0.0) or 0.0), 0.0)
            mids = list(out.get("replay_mismatch_ids") or [])
            self.assertIn("CID-1", mids)
            self.assertIn("CID-2", mids)

    def test_category_score_can_trigger_degrade_even_without_count_breach(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reliability_gate.txt"
            p.write_text(
                "\n".join(
                    [
                        "replay_mismatch_count=0",
                        "invalid_transition_count=0",
                        "journal_coverage_ratio=1.00",
                        'replay_mismatch_categories={"ledger":2,"transition":0,"belief":0,"unknown":0}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = type(
                "Cfg",
                (),
                {
                    "RELIABILITY_GATE_PATH": str(p),
                    "RELIABILITY_GATE_MAX_REPLAY_MISMATCH": 0,
                    "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS": 0,
                    "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE": 0.90,
                    "RELIABILITY_GATE_CATEGORY_DEGRADE_SCORE": 0.8,
                },
            )()
            out = rgr.get_runtime_gate(cfg)
            self.assertTrue(bool(out.get("degraded")))
            cats = dict(out.get("replay_mismatch_categories") or {})
            self.assertEqual(int(cats.get("ledger", 0) or 0), 2)
            self.assertGreaterEqual(float(out.get("mismatch_category_score", 0.0) or 0.0), 0.8)
            self.assertIn("ledger=2", str(out.get("reason") or ""))
            self.assertIn("cat_score>=0.80", str(out.get("reason") or ""))


if __name__ == "__main__":
    unittest.main()
