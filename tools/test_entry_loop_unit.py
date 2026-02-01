#!/usr/bin/env python3
"""
Unit-style tests for entry_loop helpers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Ensure Unicode log lines don't crash on Windows codepages
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

from eclipse_scalper.execution import entry_loop  # noqa: E402


class EntryLoopTelemetryTests(unittest.TestCase):
    def test_recent_router_blocks_counts(self):
        now = 1000.0
        bot = SimpleNamespace()
        bot.state = SimpleNamespace()
        bot.state.telemetry = SimpleNamespace(
            recent=[
                {"ts": now - 5, "event": "order.blocked", "symbol": "BTCUSDT"},
                {"ts": now - 5, "event": "order.blocked", "data": {"k": "BTCUSDT"}},
                {"ts": now - 70, "event": "order.blocked", "symbol": "BTCUSDT"},
                {"ts": now - 3, "event": "order.create", "symbol": "BTCUSDT"},
            ]
        )

        self.assertEqual(entry_loop._recent_router_blocks(bot, "BTCUSDT", 60, now_ts=now), 2)
        self.assertEqual(entry_loop._recent_router_blocks(bot, "ETHUSDT", 60, now_ts=now), 0)


if __name__ == "__main__":
    unittest.main()
