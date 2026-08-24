"""
Unit tests for ATR-based exit scaling.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import logging

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCALPER_ROOT = ROOT / "eclipse_scalper"
if str(SCALPER_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALPER_ROOT))

logging.disable(logging.CRITICAL)


class DummyPos:
    def __init__(self, *, entry_price: float, atr: float, side: str = "long", size: float = 1.0):
        self.entry_ts = time.time()
        self.entry_price = float(entry_price)
        self.atr = float(atr)
        self.side = side
        self.size = float(size)


class ExitAtrScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_exit_atr_scale_emits(self):
        import importlib
        from eclipse_scalper.execution import exit as exit_mod

        events: list[dict] = []

        async def _emit_throttled(_bot, event, **kwargs):
            events.append({"event": event, "data": kwargs.get("data", {})})

        exit_mod.emit_throttled = _emit_throttled

        os.environ["EXIT_ATR_SCALE_ENABLED"] = "1"
        os.environ["EXIT_ATR_SCALE_REF_PCT"] = "0.003"
        os.environ["EXIT_ATR_SCALE_MIN"] = "0.6"
        os.environ["EXIT_ATR_SCALE_MAX"] = "1.6"
        os.environ["EXIT_MAX_HOLD_SEC"] = "120"
        os.environ["EXIT_STAGNATION_SEC"] = "0"
        os.environ["EXIT_TICK_SEC"] = "0.1"

        pos = DummyPos(entry_price=1.0, atr=0.0015, side="long", size=1.0)
        state = SimpleNamespace(positions={"DOGEUSDT": pos}, run_context={})
        data = SimpleNamespace(raw_symbol={})
        bot = SimpleNamespace(state=state, data=data, ex=None)

        async def _run_once():
            task = asyncio.create_task(exit_mod.exit_loop(bot))
            try:
                await asyncio.sleep(0.35)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run_once())

        self.assertTrue(events, "Expected exit.atr_scaled event to be emitted")
        found = next((e for e in events if e["event"] == "exit.atr_scaled"), None)
        self.assertIsNotNone(found, "Missing exit.atr_scaled event")
        scale = float(found["data"].get("scale") or 0.0)
        self.assertAlmostEqual(scale, 0.6, places=2)


if __name__ == "__main__":
    unittest.main()
