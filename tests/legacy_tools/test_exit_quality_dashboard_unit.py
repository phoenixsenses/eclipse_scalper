#!/usr/bin/env python3
"""
Unit tests for exit_quality_dashboard helper.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.tools import exit_quality_dashboard as eqd


class ExitQualityDashboardTests(unittest.TestCase):
    def test_summary_output_written(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "telemetry.jsonl"
            out = Path(td) / "exit_quality.txt"
            now = __import__("time").time()
            events = [
                {
                    "ts": now,
                    "event": "position.closed",
                    "symbol": "BTCUSDT",
                    "data": {"pnl_usdt": 10.0, "pnl_pct": 0.05, "duration_sec": 120, "exit_reason": "time"},
                },
                {
                    "ts": now,
                    "event": "position.closed",
                    "symbol": "ETHUSDT",
                    "data": {"pnl_usdt": -5.0, "pnl_pct": -0.02, "duration_sec": 200, "exit_reason": "stagnation"},
                },
            ]
            with path.open("w", encoding="utf-8") as fh:
                for ev in events:
                    fh.write(json.dumps(ev) + "\n")
            eqd.main(["--path", str(path), "--since-min", "9999", "--output", str(out)])
            text = out.read_text(encoding="utf-8")
            self.assertIn("Exit quality dashboard", text)
            self.assertIn("BTCUSDT", text)
            self.assertIn("ETHUSDT", text)


if __name__ == "__main__":
    unittest.main()
