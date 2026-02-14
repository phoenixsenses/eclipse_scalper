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

from tools import gen_peak_fixture as gpf  # noqa: E402


class GenPeakFixtureTests(unittest.TestCase):
    def test_builders_include_peak_fields(self):
        event = gpf.build_telemetry_event(123.0)
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        self.assertIn("runtime_gate_position_peak", str(data.get("guard_cause_tags", "")))
        self.assertIn("runtime_gate_coverage_gap_peak", str(data.get("guard_cause_tags", "")))
        gate = gpf.build_reliability_gate_text()
        self.assertIn("position_mismatch_count_peak=2", gate)
        self.assertIn("protection_coverage_gap_seconds_peak=12.0", gate)

    def test_main_writes_fixture_files(self):
        with tempfile.TemporaryDirectory() as td:
            tele = Path(td) / "telemetry_peak_fixture.jsonl"
            gate = Path(td) / "reliability_gate_peak_fixture.txt"
            rc = gpf.main(
                [
                    "--telemetry-path",
                    str(tele),
                    "--gate-path",
                    str(gate),
                    "--ts",
                    "42.0",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(tele.exists())
            self.assertTrue(gate.exists())

            payload = json.loads(tele.read_text(encoding="utf-8").strip())
            self.assertEqual(float(payload.get("ts", 0.0)), 42.0)
            self.assertIn("runtime_gate_position_peak", str(payload.get("data", {}).get("guard_cause_tags", "")))

            gate_text = gate.read_text(encoding="utf-8")
            self.assertIn("position_mismatch_count_peak=2", gate_text)
            self.assertIn("protection_coverage_gap_seconds_peak=12.0", gate_text)


if __name__ == "__main__":
    unittest.main()
