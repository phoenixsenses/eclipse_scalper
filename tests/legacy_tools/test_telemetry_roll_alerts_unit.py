#!/usr/bin/env python3
"""
Unit test for telemetry_roll_alerts helper.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

# Ensure repo root and eclipse_scalper package dir are on sys.path
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.tools import telemetry_roll_alerts as tra


class RollAlertsTests(unittest.TestCase):
    def test_roll_alert_summary(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "telemetry.jsonl"
            events = [
                {"event": "data.quality.roll_alert", "symbol": "BTCUSDT", "ts": 1, "data": {"roll": 50}},
                {"event": "data.quality.roll_alert", "symbol": "ETHUSDT", "ts": 2, "data": {"roll": 49}},
            ]
            with path.open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
            self.assertEqual(tra.main(["--path", str(path), "--since-min", "0"]), 0)


if __name__ == "__main__":
    unittest.main()
