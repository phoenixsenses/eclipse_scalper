#!/usr/bin/env python3
"""
Unit-style tests for exit handler edge cases.
"""

from __future__ import annotations

import asyncio
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

from eclipse_scalper.execution import exit as exit_mod  # noqa: E402


class DummyEx:
    async def fetch_markets(self):
        return {}

    async def load_markets(self):
        return {}


class DummyState:
    def __init__(self):
        self.positions = {}
        self.known_exit_order_ids = set()
        self.known_exit_trade_ids = set()
        self.symbol_performance = {}
        self.total_wins = 0
        self.win_streak = 0
        self.total_trades = 0
        self.win_rate = 0.0
        self.consecutive_losses = {}
        self.blacklist = {}
        self.last_exit_time = {}
        self.current_equity = 0.0


class DummyBot:
    def __init__(self):
        self.ex = DummyEx()
        self.state = DummyState()
        self.min_amounts = {}
        self.cfg = SimpleNamespace(
            BREAKEVEN_BUFFER_ATR_MULT=0.0,
            VELOCITY_DRAWDOWN_PCT=999.0,
            VELOCITY_MINUTES=999.0,
            BLACKLIST_AUTO_RESET_ON_PROFIT=False,
            CONSECUTIVE_LOSS_BLACKLIST_COUNT=9999,
            SYMBOL_BLACKLIST_DURATION_HOURS=0,
            TRAILING_ACTIVATION_RR=0.0,
            TRAILING_REBUILD_DEBOUNCE_SEC=999999,
        )


class ExitTests(unittest.TestCase):
    def test_handle_exit_accepts_opposite_side_without_reduce_only(self):
        bot = DummyBot()
        bot.state.positions["BTCUSDT"] = SimpleNamespace(
            side="long",
            size=1.0,
            entry_price=100.0,
            entry_ts=0.0,
            atr=0.0,
            hard_stop_order_id=None,
        )
        order = {
            "id": "123",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "filled": 1.0,
            "info": {},
        }

        asyncio.run(exit_mod.handle_exit(bot, order))
        self.assertEqual(bot.state.positions.get("BTCUSDT"), None)

    def test_momentum_exit_signal_long(self):
        import pandas as pd

        class DummyData:
            def __init__(self, df_5m, df_15m):
                self._df_5m = df_5m
                self._df_15m = df_15m
                self.raw_symbol = {}

            def get_df(self, _sym, tf="1m"):
                if tf == "5m":
                    return self._df_5m
                if tf == "15m":
                    return self._df_15m
                return None

        bot = DummyBot()
        df_5m = pd.DataFrame(
            [
                [1, 100, 100, 100, 100, 1],
                [2, 100, 100, 100, 100, 1],
                [3, 100, 100, 100, 95, 1],
            ],
            columns=["ts", "o", "h", "l", "c", "v"],
        )
        df_15m = pd.DataFrame(
            [
                [1, 100, 100, 100, 100, 1],
                [2, 100, 100, 100, 100, 1],
                [3, 100, 100, 100, 94, 1],
            ],
            columns=["ts", "o", "h", "l", "c", "v"],
        )
        bot.data = DummyData(df_5m, df_15m)

        hit, mom5, mom15 = exit_mod._momentum_exit_signal(
            bot,
            "BTCUSDT",
            "BTCUSDT",
            "long",
            min_mom=0.001,
            require_both=True,
            tf_fast="5m",
            tf_slow="15m",
        )
        self.assertTrue(hit)
        self.assertLess(mom5, 0.0)
        self.assertLess(mom15, 0.0)

    def test_vwap_cross_exit_signal_long(self):
        import pandas as pd

        class DummyData:
            def __init__(self, df_5m):
                self._df_5m = df_5m
                self.raw_symbol = {}

            def get_df(self, _sym, tf="1m"):
                if tf == "5m":
                    return self._df_5m
                return None

        bot = DummyBot()
        rows = []
        for i in range(10):
            rows.append([i + 1, 110, 111, 109, 110, 10])
        rows.append([11, 90, 91, 89, 90, 10])
        df_5m = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
        bot.data = DummyData(df_5m)

        hit, vwap, px = exit_mod._vwap_cross_exit_signal(
            bot,
            "BTCUSDT",
            "BTCUSDT",
            "long",
            tf="5m",
            window=10,
            require_cross=True,
        )
        self.assertTrue(hit)
        self.assertGreater(vwap, 0.0)
        self.assertGreater(px, 0.0)


if __name__ == "__main__":
    unittest.main()
