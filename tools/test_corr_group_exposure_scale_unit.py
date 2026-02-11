"""
Unit tests for correlation group exposure scaling.
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


class CorrGroupExposureScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_exposure_scale_applies(self):
        from eclipse_scalper.execution import entry_loop as el

        os.environ["CORR_GROUP_EXPOSURE_SCALE_ENABLED"] = "1"
        os.environ["CORR_GROUP_EXPOSURE_SCALE"] = "0.7"
        os.environ["CORR_GROUP_EXPOSURE_SCALE_MIN"] = "0.3"
        os.environ["CORR_GROUP_EXPOSURE_REF_NOTIONAL"] = "100"

        meta = {"group": "MAJOR", "group_notional": 80.0, "group_count": 1}
        scale, reason = el._corr_group_exposure_scale(None, meta, planned_notional=50.0)

        self.assertTrue(scale < 1.0)
        self.assertIn("corr_group_exposure", reason)


if __name__ == "__main__":
    unittest.main()
