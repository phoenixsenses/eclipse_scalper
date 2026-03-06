#!/usr/bin/env python3
"""
Unit test for telemetry_threshold_alerts helper.
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

from eclipse_scalper.tools import telemetry_threshold_alerts as tta


class ThresholdAlertsTests(unittest.TestCase):
    def test_threshold_alert_triggers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "telemetry.jsonl"
            events = [
                {"event": "entry.blocked", "ts": 1},
                {"event": "entry.blocked", "ts": 2},
                {"event": "order.create.retry_alert", "ts": 3},
            ]
            with path.open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
            self.assertEqual(tta.main(["--path", str(path), "--since-min", "0", "--thresholds", "entry.blocked=2"]), 0)


if __name__ == "__main__":
    unittest.main()
