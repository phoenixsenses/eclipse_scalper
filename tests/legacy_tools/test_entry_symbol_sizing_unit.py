"""
Unit test for per-symbol sizing overrides.
"""

from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.disable(logging.CRITICAL)


class EntrySymbolSizingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_symbol_notional_override(self):
        from eclipse_scalper.execution import entry_loop as el

        os.environ["FIXED_NOTIONAL_USDT"] = "10"
        os.environ["FIXED_NOTIONAL_USDT_DOGE"] = "5"

        qty, notional = el._resolve_symbol_sizing(SimpleNamespace(cfg=None), "DOGEUSDT")
        self.assertEqual(notional, 5.0)


if __name__ == "__main__":
    unittest.main()
