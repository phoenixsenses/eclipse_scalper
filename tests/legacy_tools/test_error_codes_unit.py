#!/usr/bin/env python3
"""
Unit-style tests for standardized error codes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure Unicode log lines don't crash on Windows codepages
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure repo root and eclipse_scalper package dir are on sys.path
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import error_codes  # noqa: E402


class ErrorCodeTests(unittest.TestCase):
    def test_map_reason(self):
        self.assertEqual(error_codes.map_reason("stale data"), error_codes.ERR_STALE_DATA)
        self.assertEqual(error_codes.map_reason("slippage 0.2%"), error_codes.ERR_SLIPPAGE)
        self.assertEqual(error_codes.map_reason("spread too wide"), error_codes.ERR_SPREAD)
        self.assertEqual(error_codes.map_reason("margin insufficient"), error_codes.ERR_MARGIN)
        self.assertEqual(error_codes.map_reason("risk_amount low"), error_codes.ERR_RISK)


if __name__ == "__main__":
    unittest.main()
