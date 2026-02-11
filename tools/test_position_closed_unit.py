#!/usr/bin/env python3
"""
Unit tests for position.closed emission.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logging
logging.disable(logging.CRITICAL)

from eclipse_scalper.execution import exit as exit_mod  # noqa: E402


class PositionClosedTests(unittest.TestCase):
    def setUp(self):
        self._orig_emit = exit_mod.emit
        self._orig_emit_throttled = exit_mod.emit_throttled
        self._orig_create_task = exit_mod.asyncio.create_task
        self._log: list = []

        def fake_emit(bot, event, data=None, symbol=None, level=None):
            self._log.append((event, symbol, data or {}, level))

        def run_immediate(coro):
            asyncio.run(coro)

        exit_mod.emit = fake_emit
        exit_mod.emit_throttled = None
        exit_mod.asyncio.create_task = run_immediate

    def tearDown(self):
        exit_mod.emit = self._orig_emit
        exit_mod.emit_throttled = self._orig_emit_throttled
        exit_mod.asyncio.create_task = self._orig_create_task

    def test_position_closed_emits(self):
        pos = SimpleNamespace(
            side="long",
            size=1.0,
            entry_price=100.0,
            atr=1.0,
            leverage=1,
            entry_ts=time.time() - 120,
            confidence=0.7,
        )
        state = SimpleNamespace(
            positions={"BTCUSDT": pos},
            known_exit_order_ids=set(),
            known_exit_trade_ids=set(),
            last_exit_time={},
            total_wins=0,
            total_trades=1,
            win_streak=0,
            consecutive_losses={},
            symbol_performance={},
            blacklist={},
            run_context={
                "last_entry_signal": {
                    "BTCUSDT": {
                        "confidence": 0.7,
                        "side": "long",
                        "ts": time.time() - 120,
                        "entry_price": 100.0,
                    }
                }
            },
        )
        cfg = SimpleNamespace(
            BLACKLIST_AUTO_RESET_ON_PROFIT=False,
            CONSECUTIVE_LOSS_BLACKLIST_COUNT=3,
            SYMBOL_BLACKLIST_DURATION_HOURS=1,
            BREAKEVEN_BUFFER_ATR_MULT=0.1,
            VELOCITY_DRAWDOWN_PCT=0.02,
            VELOCITY_MINUTES=1,
            TRAILING_ACTIVATION_RR=0,
        )
        bot = SimpleNamespace(state=state, cfg=cfg, ex=None, data=SimpleNamespace(raw_symbol={}))
        order = {
            "id": "oid-1",
            "symbol": "BTCUSDT",
            "side": "sell",
            "filled": 1.0,
            "average": 110.0,
            "info": {"realizedPnl": 10.0},
        }

        asyncio.run(exit_mod.handle_exit(bot, order))
        events = [row for row in self._log if row[0] == "position.closed"]
        self.assertEqual(len(events), 1)
        _, symbol, data, _level = events[0]
        self.assertEqual(symbol, "BTCUSDT")
        self.assertIn("pnl_usdt", data)
        self.assertIn("duration_sec", data)


if __name__ == "__main__":
    unittest.main()
