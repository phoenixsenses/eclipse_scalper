#!/usr/bin/env python3
"""
Unit-style tests for telemetry_error_classes.
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

from eclipse_scalper.tools import telemetry_error_classes as tec


class TelemetryErrorClassesTests(unittest.TestCase):
    def test_classify_from_code_and_reason(self):
        ev1 = {"event": "order.create_failed", "data": {"code": "ERR_MARGIN"}}
        ev2 = {"event": "order.create_failed", "data": {"reason": "network"}}
        ev3 = {"event": "order.blocked", "data": {"err": "Filter failure: PRICE_FILTER"}}
        self.assertEqual(tec._classify_event(ev1), "margin")
        self.assertEqual(tec._classify_event(ev2), "network")
        self.assertEqual(tec._classify_event(ev3), "price_filter")

    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "telemetry.jsonl"
            events = [
                {"event": "entry.blocked", "symbol": "BTCUSDT", "data": {"code": "ERR_STALE_DATA"}},
                {"event": "order.create_failed", "symbol": "ETHUSDT", "data": {"reason": "network"}},
            ]
            with path.open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

            loaded = tec._load_jsonl(path, limit=10)
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
