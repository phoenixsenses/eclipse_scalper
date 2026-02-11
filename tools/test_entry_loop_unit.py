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
    def tearDown(self):
        for key in ("CORR_GROUPS", "CORR_GROUP_MAX_POSITIONS", "CORR_GROUP_MAX_NOTIONAL_USDT"):
            if key in entry_loop.os.environ:
                entry_loop.os.environ.pop(key, None)

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

    def test_corr_group_caps_block(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(
            positions={
                "DOGEUSDT": SimpleNamespace(size=10, entry_price=1.0),
                "SHIBUSDT": SimpleNamespace(size=5, entry_price=1.0),
            }
        )
        entry_loop.os.environ["CORR_GROUPS"] = "MEME:DOGEUSDT,SHIBUSDT,PEPEUSDT"
        entry_loop.os.environ["CORR_GROUP_MAX_POSITIONS"] = "2"
        entry_loop.os.environ["CORR_GROUP_MAX_NOTIONAL_USDT"] = "20"

        reason, meta = entry_loop._check_corr_group(bot, "PEPEUSDT", planned_notional=5.0)
        self.assertIsNotNone(reason)
        self.assertIn("group MEME", reason)
        self.assertEqual(meta.get("group"), "MEME")

    def test_corr_group_scale_applies(self):
        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(
            positions={"DOGEUSDT": SimpleNamespace(size=10, entry_price=1.0)}
        )
        entry_loop.os.environ["CORR_GROUPS"] = "MEME:DOGEUSDT,PEPEUSDT"
        entry_loop.os.environ["CORR_GROUP_SCALE_ENABLED"] = "1"
        entry_loop.os.environ["CORR_GROUP_SCALE"] = "0.5"

        _, meta = entry_loop._check_corr_group(bot, "PEPEUSDT", planned_notional=10.0)
        scale, reason = entry_loop._corr_group_scale(bot, meta)
        self.assertLess(scale, 1.0)
        self.assertIn("corr_group_scale", reason)


if __name__ == "__main__":
    unittest.main()
