"""
Unit tests for exit per-symbol overrides.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.disable(logging.CRITICAL)


class DummyPos:
    def __init__(self, *, entry_price: float, atr: float, side: str = "long", size: float = 1.0):
        self.entry_ts = time.time() - 10.0
        self.entry_price = float(entry_price)
        self.atr = float(atr)
        self.side = side
        self.size = float(size)


class ExitSymbolOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_exit_max_hold_symbol_override_triggers(self):
        from eclipse_scalper.execution import exit as exit_mod

        called: list[dict] = []

        async def _create_order(*args, **kwargs):
            called.append({"args": args, "kwargs": kwargs})
            return {"id": "test-exit"}

        exit_mod.create_order = _create_order  # type: ignore[assignment]
        exit_mod.emit = None
        exit_mod.emit_throttled = None

        os.environ["EXIT_ENABLED"] = "1"
        os.environ["EXIT_TICK_SEC"] = "0.1"
        os.environ["EXIT_TIME_COOLDOWN_SEC"] = "0"
        os.environ["EXIT_MAX_HOLD_SEC"] = "999"
        os.environ["EXIT_MAX_HOLD_SEC_DOGE"] = "5"
        os.environ["EXIT_STAGNATION_SEC"] = "0"
        os.environ["EXIT_MOM_ENABLED"] = "0"
        os.environ["EXIT_VWAP_ENABLED"] = "0"

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

        self.assertTrue(called, "Expected time exit create_order to be called")


if __name__ == "__main__":
    unittest.main()
