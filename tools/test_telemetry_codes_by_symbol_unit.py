#!/usr/bin/env python3
"""
Unit-style tests for telemetry_codes_by_symbol.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys
from pathlib import Path

# Ensure repo root and eclipse_scalper package dir are on sys.path
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.tools import telemetry_codes_by_symbol as tcs


class TelemetryCodesBySymbolTests(unittest.TestCase):
    def test_parse_codes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "telemetry.jsonl"
            events = [
                {"event": "entry.blocked", "symbol": "BTCUSDT", "data": {"code": "ERR_STALE_DATA"}},
                {"event": "exit.success", "symbol": "BTCUSDT", "data": {"code": "EXIT_TIME"}},
                {"event": "entry.blocked", "symbol": "ETHUSDT", "data": {"code": "ERR_SPREAD"}},
            ]
            with path.open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

            loaded = tcs._load_jsonl(path, limit=10)
            self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()
