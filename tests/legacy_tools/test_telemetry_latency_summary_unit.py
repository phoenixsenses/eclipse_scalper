#!/usr/bin/env python3
"""
Unit test for telemetry_latency_summary helper.
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

from eclipse_scalper.tools import telemetry_latency_summary as tls


class LatencySummaryTests(unittest.TestCase):
    def test_latency_summary_prints(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "telemetry.jsonl"
            events = [
                {"event": "telemetry.latency", "symbol": "BTCUSDT", "data": {"symbol": "BTCUSDT", "stage": "signal", "duration_ms": 25}, "ts": 1},
                {"event": "telemetry.latency", "symbol": "BTCUSDT", "data": {"symbol": "BTCUSDT", "stage": "order_router", "duration_ms": 150}, "ts": 2},
            ]
            with path.open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
            self.assertEqual(tls.main(["--path", str(path), "--since-min", "0"]), 0)


if __name__ == "__main__":
    unittest.main()
