#!/usr/bin/env python3
"""
Unit-style tests for data_quality helpers.
"""

from __future__ import annotations

import sys
import time
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

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

from eclipse_scalper.execution import data_quality  # noqa: E402


class DummyData:
    def __init__(self, ohlcv_rows):
        self.ohlcv = {"BTCUSDT": ohlcv_rows}

    def get_df(self, _sym, _tf="1m"):
        if not self.ohlcv["BTCUSDT"]:
            return None
        df = pd.DataFrame(self.ohlcv["BTCUSDT"], columns=["ts", "o", "h", "l", "c", "v"])
        return df


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_staleness_check(self):
        now_ms = int(time.time() * 1000)
        rows = [[now_ms - 10_000, 1, 1, 1, 1, 1]]
        bot = SimpleNamespace(data=DummyData(rows))
        ok, age, _src = data_quality.staleness_check(bot, "BTCUSDT", max_sec=1.0)
        self.assertFalse(ok)
        self.assertGreater(age, 1.0)

    def test_quality_score_degrades_on_missing(self):
        bot = SimpleNamespace(data=DummyData([]))
        score, _details = data_quality.quality_score(bot, "BTCUSDT")
        self.assertLess(score, 100.0)

    def test_quality_history_rolls(self):
        os.environ["ENTRY_DATA_QUALITY_HISTORY_MAX"] = "5"
        os.environ["ENTRY_DATA_QUALITY_ROLL_SEC"] = "10000"

        base_ts = 1000.0
        rows = []
        for i in range(20):
            ts_ms = int((base_ts - (19 - i) * 60) * 1000)
            rows.append([ts_ms, 100, 101, 99, 100, 10])

        bot = SimpleNamespace(data=DummyData(rows), state=SimpleNamespace())

        orig_now = data_quality._now
        try:
            data_quality._now = lambda: base_ts
            s1 = data_quality.update_quality_state(bot, "BTCUSDT", max_sec=9999, window=20, emit_sec=0)

            data_quality._now = lambda: base_ts + 100
            bot.data.ohlcv["BTCUSDT"] = []
            s2 = data_quality.update_quality_state(bot, "BTCUSDT", max_sec=9999, window=20, emit_sec=0)
        finally:
            data_quality._now = orig_now

        dq = bot.state.data_quality.get("BTCUSDT")
        hist = bot.state.data_quality_history.get("BTCUSDT")
        self.assertEqual(len(hist), 2)
        self.assertAlmostEqual(dq["roll"], (s1 + s2) / 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
