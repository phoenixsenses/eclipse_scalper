#!/usr/bin/env python3
"""
Unit-style tests for entry correlation guards.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace

import sys
from pathlib import Path

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

from eclipse_scalper.execution import entry  # noqa: E402


class DummyData:
    def __init__(self):
        self.raw_symbol = {}

    def get_df(self, _sym, _tf="1m"):
        return None


class DummyBot:
    def __init__(self):
        self.data = DummyData()
        self.active_symbols = {"DOGEUSDT"}
        self.cfg = SimpleNamespace(
            SYMBOL_COOLDOWN_MINUTES=0,
            MAX_CONCURRENT_POSITIONS=10,
            MAX_PORTFOLIO_HEAT=10.0,
            SESSION_EQUITY_PEAK_PROTECTION_PCT=1.0,
            MAX_DAILY_LOSS_PCT=1.0,
            MIN_CONFIDENCE=0.0,
        )
        self.state = SimpleNamespace(
            blacklist={},
            positions={},
            last_exit_time={},
            current_equity=1000.0,
            start_of_day_equity=1000.0,
            daily_pnl=0.0,
            run_context={},
        )


class EntryCorrelationGuardTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_group_max_positions_blocks(self):
        bot = DummyBot()

        # existing MEME position
        bot.state.positions = {
            "SHIBUSDT": SimpleNamespace(size=1.0, entry_price=0.00001)
        }

        # set group limits
        os.environ["CORR_GROUP_MAX_POSITIONS"] = "1"
        bot.cfg.CORR_GROUP_MAX_POSITIONS = 1

        entry._get_corr_groups = lambda _bot: {"MEME": ["DOGEUSDT", "SHIBUSDT"]}  # type: ignore

        async def _ok_trade_allowed(_bot):
            return True

        entry.trade_allowed = _ok_trade_allowed  # type: ignore

        called = {"signal": False}

        def _fake_signal(*_a, **_kw):
            called["signal"] = True
            return (False, False, 0.0)

        entry.generate_signal = _fake_signal  # type: ignore

        # ensure create_order isn't called
        async def _fail_create_order(*_a, **_kw):
            raise AssertionError("create_order should not be called")

        entry.create_order = _fail_create_order  # type: ignore

        reasons = []

        def _collect_skip(_bot, _k, _side, reason):
            reasons.append(str(reason))

        entry._skip = _collect_skip  # type: ignore
        entry._skip(bot, "DOGEUSDT", "long", "probe")  # type: ignore

        groups = entry._get_corr_groups(bot)  # type: ignore
        self.assertIn("MEME", groups)
        self.assertIn("DOGEUSDT", groups["MEME"])
        self.assertIn("SHIBUSDT", groups["MEME"])
        gm = int(entry._safe_float(os.getenv("CORR_GROUP_MAX_POSITIONS", 0), 0))  # type: ignore
        self.assertEqual(gm, 1)
        self.assertTrue(any("probe" in r for r in reasons))

        asyncio.run(entry.try_enter(bot, "DOGEUSDT", "long"))

        self.assertTrue(any("group MEME max positions" in r for r in reasons))
        self.assertFalse(called["signal"])

    def test_group_max_notional_blocks(self):
        import pandas as pd

        class RichDummyBot(DummyBot):
            def __init__(self):
                super().__init__()
                self.cfg = SimpleNamespace(
                    SYMBOL_COOLDOWN_MINUTES=0,
                    MAX_CONCURRENT_POSITIONS=10,
                    MAX_PORTFOLIO_HEAT=10.0,
                    SESSION_EQUITY_PEAK_PROTECTION_PCT=1.0,
                    MAX_DAILY_LOSS_PCT=1.0,
                    MIN_CONFIDENCE=0.0,
                    TIMEFRAME="1m",
                    MIN_ATR_PCT_FOR_ENTRY=0.0,
                    MAX_FUNDING_LONG=1.0,
                    MIN_FUNDING_SHORT=-1.0,
                    SLIPPAGE_USE_FRESH_TICKER=False,
                    MAX_STOP_PCT=0.2,
                    MAX_RISK_PER_TRADE=0.02,
                    ADAPTIVE_RISK_SCALING=False,
                    CONFIDENCE_SCALING=False,
                    MIN_RISK_DOLLARS=0.0,
                    STOP_ATR_MULT=1.0,
                    MIN_FILL_RATIO=0.1,
                    TP1_RR_MULT=1.0,
                    TP2_RR_MULT=2.0,
                    DYNAMIC_TRAILING_FULL=False,
                    DUAL_TRAILING=False,
                    TRAILING_ACTIVATION_RR=0.0,
                    TRAILING_CALLBACK_RATE=1.0,
                    LEVERAGE=1,
                )
                self.state = SimpleNamespace(
                    blacklist={},
                    positions={},
                    last_exit_time={},
                    current_equity=1000.0,
                    start_of_day_equity=1000.0,
                    daily_pnl=0.0,
                    run_context={},
                    total_trades=0,
                    win_streak=0,
                )

        bot = RichDummyBot()

        # build a simple 200-bar dataframe with positive ATR
        n = 220
        base = 100.0
        data = {
            "c": [base + (i * 0.01) for i in range(n)],
            "h": [base + (i * 0.01) + 0.05 for i in range(n)],
            "l": [base + (i * 0.01) - 0.05 for i in range(n)],
        }
        df = pd.DataFrame(data)

        bot.data.get_df = lambda _sym, _tf="1m": df  # type: ignore

        os.environ["CORR_GROUP_MAX_NOTIONAL_USDT"] = "1"
        bot.cfg.CORR_GROUP_MAX_NOTIONAL_USDT = 1

        entry._get_corr_groups = lambda _bot: {"MEME": ["DOGEUSDT"]}  # type: ignore

        async def _ok_trade_allowed(_bot):
            return True

        entry.trade_allowed = _ok_trade_allowed  # type: ignore

        def _fake_signal(*_a, **_kw):
            return (True, False, 1.0)

        entry.generate_signal = _fake_signal  # type: ignore

        # ensure create_order isn't called
        async def _fail_create_order(*_a, **_kw):
            raise AssertionError("create_order should not be called")

        entry.create_order = _fail_create_order  # type: ignore

        reasons = []

        def _collect_skip(_bot, _k, _side, reason):
            reasons.append(str(reason))

        entry._skip = _collect_skip  # type: ignore

        asyncio.run(entry.try_enter(bot, "DOGEUSDT", "long"))

        self.assertTrue(any("group MEME notional" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
