#!/usr/bin/env python3
"""
Tiny unit test to ensure diagnostics helpers do not raise.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure Unicode log lines do not crash on Windows codepages
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

from eclipse_scalper.execution import diagnostics  # noqa: E402


class DiagnosticsTests(unittest.TestCase):
    def test_print_diagnostics_does_not_raise(self):
        diagnostics.print_diagnostics()


if __name__ == "__main__":
    unittest.main()
