#!/usr/bin/env python3
"""
Unit tests for execution.exit telemetry helper.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution import exit as exit_mod  # noqa: E402


class ExitTelemetryHelperTests(unittest.TestCase):
    def setUp(self):
        self._orig_emit = exit_mod.emit
        self._orig_emit_throttled = exit_mod.emit_throttled
        self._orig_create_task = exit_mod.asyncio.create_task
        self._log: list = []

        async def fake_emit(bot, event, data=None, symbol=None, level=None):
            self._log.append(("emit", event, symbol, level, tuple(sorted((data or {}).items()))))

        async def fake_emit_throttled(bot, event, key, cooldown_sec, data=None, symbol=None, level=None):
            self._log.append(("throttled", event, key, symbol, level, tuple(sorted((data or {}).items()))))

        def run_immediate(coro):
            asyncio.run(coro)

        exit_mod.emit = fake_emit
        exit_mod.emit_throttled = fake_emit_throttled
        exit_mod.asyncio.create_task = run_immediate

    def tearDown(self):
        exit_mod.emit = self._orig_emit
        exit_mod.emit_throttled = self._orig_emit_throttled
        exit_mod.asyncio.create_task = self._orig_create_task

    def test_schedule_throttled_exit_event(self):
        bot = SimpleNamespace(name="bot")
        exit_mod._schedule_exit_event(
            bot,
            "exit.success",
            symbol="DOGEUSDT",
            reason="velocity",
            level="info",
            throttle_sec=5.0,
            key="DOGEUSDT:velocity",
        )
        self.assertEqual(len(self._log), 1)
        row = self._log[0]
        self.assertEqual(row[0], "throttled")
        self.assertEqual(row[1], "exit.success")
        self.assertIn(("code", exit_mod.map_reason("velocity")), row[5])

    def test_schedule_plain_emit_exit_event(self):
        exit_mod.emit_throttled = None
        bot = SimpleNamespace(name="bot")
        exit_mod._schedule_exit_event(
            bot,
            "exit.blocked",
            symbol="BTCUSDT",
            reason="trailing_failed",
            level="warning",
        )
        self.assertEqual(len(self._log), 1)
        row = self._log[0]
        self.assertEqual(row[0], "emit")
        self.assertEqual(row[1], "exit.blocked")


if __name__ == "__main__":
    unittest.main()
