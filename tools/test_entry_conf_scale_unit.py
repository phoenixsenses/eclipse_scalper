"""
Unit test for confidence-based entry scaling.
"""

from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.disable(logging.CRITICAL)


class EntryConfScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_conf_scale_midpoint(self):
        from eclipse_scalper.execution import entry_loop as el

        os.environ["ENTRY_CONF_SCALE_ENABLED"] = "1"
        os.environ["ENTRY_CONF_SCALE_MIN_CONF"] = "0.40"
        os.environ["ENTRY_CONF_SCALE_MAX_CONF"] = "0.80"
        os.environ["ENTRY_CONF_SCALE_MIN"] = "0.50"
        os.environ["ENTRY_CONF_SCALE_MAX"] = "1.00"

        scale, reason = el._confidence_notional_scale(None, 0.60)

        self.assertAlmostEqual(scale, 0.75, places=3)
        self.assertIn("confidence", reason)


if __name__ == "__main__":
    unittest.main()
