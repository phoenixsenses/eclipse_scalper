"""
Unit tests for entry partial-fill state resolution.
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


class EntryPartialFillStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_partial_forced_flatten_outcome(self):
        from eclipse_scalper.execution import entry_loop as el

        events: list[dict] = []
        calls: list[dict] = []

        async def _emit(_bot, event, data=None, symbol=None, level=None):
            events.append({"event": event, "data": data or {}, "symbol": symbol, "level": level})

        async def _cancel(_bot, order_id, symbol):
            calls.append({"kind": "cancel", "order_id": order_id, "symbol": symbol})
            return True

        async def _create(*_args, **kwargs):
            calls.append({"kind": "create", "kwargs": kwargs})
            return {"id": "flat-1", "filled": kwargs.get("amount")}

        el.emit = _emit
        el.cancel_order = _cancel
        el.create_order = _create

        os.environ["ENTRY_PARTIAL_CANCEL"] = "1"
        os.environ["ENTRY_PARTIAL_FORCE_FLATTEN"] = "1"

        out = asyncio.run(
            el._resolve_partial_fill_state(
                SimpleNamespace(),
                symbol="DOGEUSDT",
                sym_raw="DOGE/USDT:USDT",
                action="buy",
                otype="limit",
                order_id="oid-1",
                requested=10.0,
                filled=2.0,
                min_ratio=0.5,
                hedge_side_hint="long",
            )
        )

        self.assertEqual(out.get("outcome"), "partial_forced_flatten")
        self.assertTrue(out.get("cancel_ok"))
        self.assertTrue(out.get("flatten_ok"))
        self.assertTrue(any(ev.get("event") == "entry.partial_fill_state" for ev in events))
        self.assertTrue(any(c.get("kind") == "create" for c in calls))

    def test_partial_stuck_when_flatten_disabled(self):
        from eclipse_scalper.execution import entry_loop as el

        async def _emit(*_args, **_kwargs):
            return None

        async def _cancel(*_args, **_kwargs):
            return True

        async def _create(*_args, **_kwargs):
            return {"id": "ignored"}

        el.emit = _emit
        el.cancel_order = _cancel
        el.create_order = _create

        os.environ["ENTRY_PARTIAL_CANCEL"] = "1"
        os.environ["ENTRY_PARTIAL_FORCE_FLATTEN"] = "0"

        out = asyncio.run(
            el._resolve_partial_fill_state(
                SimpleNamespace(),
                symbol="DOGEUSDT",
                sym_raw="DOGE/USDT:USDT",
                action="buy",
                otype="limit",
                order_id="oid-1",
                requested=10.0,
                filled=2.0,
                min_ratio=0.5,
                hedge_side_hint="long",
            )
        )

        self.assertEqual(out.get("outcome"), "partial_stuck")
        self.assertTrue(out.get("cancel_ok"))
        self.assertFalse(out.get("flatten_attempted"))


if __name__ == "__main__":
    unittest.main()
