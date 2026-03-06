"""
Unit test for adaptive guard scaling of fixed qty sizing.
"""

from __future__ import annotations

import asyncio
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


class EntryQtyScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_fixed_qty_scales(self):
        from eclipse_scalper.execution import entry_loop as el

        events: list[dict] = []

        async def _emit_throttled(_bot, event, **kwargs):
            events.append({"event": event, "data": kwargs.get("data", {})})

        orig_emit = el.emit_throttled
        orig_task = el.asyncio.create_task
        el.emit_throttled = _emit_throttled
        el.get_adaptive_notional_scale = lambda _sym: (0.5, "guard_history")

        def _run_immediate(coro):
            asyncio.run(coro)

        el.asyncio.create_task = _run_immediate

        os.environ["FIXED_QTY"] = "10"
        os.environ["ADAPTIVE_GUARD_QTY_SCALE"] = "1"

        bot = SimpleNamespace(cfg=None)
        try:
        qty = el._sizing_fallback_amount(bot, "BTCUSDT")
        finally:
            el.emit_throttled = orig_emit
            el.asyncio.create_task = orig_task

        self.assertAlmostEqual(qty or 0.0, 5.0, places=6)
        self.assertTrue(events, "Expected entry.qty_scaled event")
        found = next((e for e in events if e["event"] == "entry.qty_scaled"), None)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(float(found["data"].get("scale") or 0.0), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
