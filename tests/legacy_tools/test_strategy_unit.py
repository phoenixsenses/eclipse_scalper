#!/usr/bin/env python3
"""
Unit-style tests for strategy guards (stale data, session filter, cooldown).
"""

from __future__ import annotations

import os
import time
import unittest

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

from eclipse_scalper.strategies import eclipse_scalper  # noqa: E402


class DummyData:
    def __init__(self, ohlcv):
        self.ohlcv = ohlcv
        self.raw_symbol = {}


class DummyState:
    def __init__(self, losses=0, last_exit=0.0, k="BTCUSDT"):
        self.consecutive_losses = {k: losses}
        self.last_exit_time = {k: last_exit}


class DummyBot:
    def __init__(self, data, state):
        self.data = data
        self.state = state


def _make_rows(age_sec: float, n: int = 20):
    ts = int((time.time() - age_sec) * 1000)
    return [[ts, 1, 1, 1, 1, 1] for _ in range(n)]


def _make_trend_rows(n: int = 120, step: float = 0.5):
    now = time.time()
    rows = []
    price = 100.0
    for i in range(n):
        ts = int((now - (n - 1 - i) * 60) * 1000)
        o = price
        c = price + step
        h = c + (step * 0.4)
        l = o - (step * 0.4)
        v = 100.0 + (i * 2.0)
        rows.append([ts, o, h, l, c, v])
        price = c
    if rows:
        rows[-1][5] = rows[-1][5] * 25.0
    return rows


def _make_flat_rows(n: int = 800):
    now = time.time()
    rows = []
    for i in range(n):
        ts = int((now - (n - 1 - i) * 60) * 1000)
        o = 100.0
        c = 100.0
        h = 100.1
        l = 99.9
        v = 100.0
        rows.append([ts, o, h, l, c, v])
    return rows


class StrategyGuardTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_stale_data_guard_blocks(self):
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "1"
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"

        data = DummyData({"BTCUSDT": _make_rows(age_sec=10)})
        long_sig, short_sig, conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertFalse(long_sig)
        self.assertFalse(short_sig)
        self.assertEqual(conf, 0.0)

    def test_session_filter_blocks(self):
        now_hour = time.gmtime().tm_hour
        off_hour = (now_hour + 2) % 24
        os.environ["SCALPER_SESSION_UTC"] = f"{off_hour}-{off_hour}"
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"

        data = DummyData({"BTCUSDT": _make_rows(age_sec=1)})
        long_sig, short_sig, conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertFalse(long_sig)
        self.assertFalse(short_sig)
        self.assertEqual(conf, 0.0)

    def test_cooldown_blocks_after_losses(self):
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "2"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "10"

        data = DummyData({"BTCUSDT": _make_rows(age_sec=1)})
        state = DummyState(losses=2, last_exit=time.time() - 60, k="BTCUSDT")
        bot = DummyBot(data, state)

        long_sig, short_sig, conf = eclipse_scalper.scalper_signal(bot, "BTCUSDT")
        self.assertFalse(long_sig)
        self.assertFalse(short_sig)
        self.assertEqual(conf, 0.0)

    def test_confidence_max_clamps(self):
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"
        os.environ["SCALPER_MIN_BARS"] = "20"
        os.environ["SCALPER_FAST_BACKTEST"] = "1"
        os.environ["SCALPER_DYN_MOM_MULT"] = "0.1"
        os.environ["SCALPER_DYN_MOM_FLOOR"] = "0"
        os.environ["SCALPER_VOL_Z_TH"] = "0.5"
        os.environ["SCALPER_VOL_Z_SOFT_TH"] = "0.2"
        os.environ["SCALPER_VWAP_BASE_DIST"] = "0.0001"
        os.environ["SCALPER_VWAP_ATR_MULT"] = "0.5"
        os.environ["SCALPER_CONFIDENCE_POWER"] = "1.0"
        os.environ["SCALPER_CONFIDENCE_MIN"] = "0.0"
        os.environ["SCALPER_CONFIDENCE_MAX"] = "0.2"

        data = DummyData({"BTCUSDT": _make_trend_rows(n=120)})
        _long_sig, _short_sig, conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertEqual(conf, 0.2)

    def test_confidence_diag_does_not_raise(self):
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"
        os.environ["SCALPER_MIN_BARS"] = "20"
        os.environ["SCALPER_FAST_BACKTEST"] = "1"
        os.environ["SCALPER_DYN_MOM_MULT"] = "0.1"
        os.environ["SCALPER_DYN_MOM_FLOOR"] = "0"
        os.environ["SCALPER_VOL_Z_TH"] = "0.5"
        os.environ["SCALPER_VOL_Z_SOFT_TH"] = "0.2"
        os.environ["SCALPER_VWAP_BASE_DIST"] = "0.0001"
        os.environ["SCALPER_VWAP_ATR_MULT"] = "0.5"
        os.environ["SCALPER_CONFIDENCE_POWER"] = "1.0"
        os.environ["SCALPER_CONFIDENCE_MIN"] = "0.0"
        os.environ["SCALPER_CONFIDENCE_MAX"] = "1.0"
        os.environ["SCALPER_CONFIDENCE_DIAG"] = "1"

        data = DummyData({"BTCUSDT": _make_trend_rows(n=120)})
        _long_sig, _short_sig, conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertIsInstance(conf, float)

    def test_volume_ratio_confidence_boosts(self):
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"
        os.environ["SCALPER_MIN_BARS"] = "20"
        os.environ["SCALPER_FAST_BACKTEST"] = "1"
        os.environ["SCALPER_DYN_MOM_MULT"] = "0.1"
        os.environ["SCALPER_DYN_MOM_FLOOR"] = "0"
        os.environ["SCALPER_VOL_Z_TH"] = "0.5"
        os.environ["SCALPER_VOL_Z_SOFT_TH"] = "0.2"
        os.environ["SCALPER_VWAP_BASE_DIST"] = "0.0001"
        os.environ["SCALPER_VWAP_ATR_MULT"] = "0.2"
        os.environ["SCALPER_CONFIDENCE_POWER"] = "1.0"
        os.environ["SCALPER_CONFIDENCE_MIN"] = "0.0"
        os.environ["SCALPER_CONFIDENCE_MAX"] = "1.0"
        os.environ["SCALPER_VOL_RATIO_WEIGHT"] = "1.0"
        os.environ["SCALPER_VOL_RATIO_MAX_DROP"] = "0.1"
        os.environ["SCALPER_AUDIT"] = "1"

        data = DummyData({"BTCUSDT": _make_trend_rows(n=120)})
        captured = []
        original_audit = eclipse_scalper._audit_emit

        def fake_audit(k, outcome, confidence, audit_data, blockers):
            captured.append(audit_data)

        eclipse_scalper._audit_emit = fake_audit
        try:
            os.environ["SCALPER_VOL_RATIO_MAX_BOOST"] = "0.0"
            _, _, base_conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
            self.assertTrue(captured)
            base_multiplier = captured[-1].get("vol_conf_mult")
            captured.clear()

            os.environ["SCALPER_VOL_RATIO_MAX_BOOST"] = "0.5"
            _, _, boosted_conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
            self.assertTrue(captured)
            boosted_multiplier = captured[-1].get("vol_conf_mult")
        finally:
            eclipse_scalper._audit_emit = original_audit

        self.assertAlmostEqual(base_multiplier, 1.0, places=3)
        self.assertGreater(boosted_multiplier, base_multiplier)
        self.assertGreaterEqual(boosted_conf, base_conf)

    def test_session_momentum_blocks(self):
        now_hour = time.gmtime().tm_hour
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_SESSION_MOM_UTC"] = f"{now_hour}-{now_hour}"
        os.environ["SCALPER_SESSION_MOM_MIN"] = "0.20"
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"
        os.environ["SCALPER_MIN_BARS"] = "20"
        os.environ["SCALPER_FAST_BACKTEST"] = "0"
        os.environ["SCALPER_DYN_MOM_MULT"] = "0.1"
        os.environ["SCALPER_DYN_MOM_FLOOR"] = "0"
        os.environ["SCALPER_VOL_Z_TH"] = "0.5"
        os.environ["SCALPER_VOL_Z_SOFT_TH"] = "0.2"
        os.environ["SCALPER_VWAP_BASE_DIST"] = "0.0001"
        os.environ["SCALPER_VWAP_ATR_MULT"] = "0.2"

        data = DummyData({"BTCUSDT": _make_trend_rows(n=240, step=0.2)})
        long_sig, short_sig, conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertFalse(long_sig)
        self.assertFalse(short_sig)
        self.assertEqual(conf, 0.0)

    def test_vol_regime_guard_blocks_low(self):
        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"
        os.environ["SCALPER_MIN_BARS"] = "20"
        os.environ["SCALPER_FAST_BACKTEST"] = "0"
        os.environ["SCALPER_VOL_REGIME_ENABLED"] = "1"
        os.environ["SCALPER_VOL_REGIME_GUARD"] = "1"
        os.environ["SCALPER_VOL_REGIME_LOW_ATR_PCT"] = "0.01"
        os.environ["SCALPER_VOL_REGIME_LOW_BB_PCT"] = "0.05"

        data = DummyData({"BTCUSDT": _make_flat_rows(n=800)})
        long_sig, short_sig, conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertFalse(long_sig)
        self.assertFalse(short_sig)
        self.assertEqual(conf, 0.0)

    def test_trend_confirm_gate_blocks_forced_entry(self):
        class TrendConfirmData(DummyData):
            def __init__(self, df_1m, df_1h):
                super().__init__({})
                self._df_1m = df_1m
                self._df_1h = df_1h

            def get_df(self, _sym, tf="1m"):
                if tf == "1h":
                    return self._df_1h
                return self._df_1m

        import pandas as pd

        os.environ["SCALPER_SESSION_UTC"] = ""
        os.environ["SCALPER_DATA_MAX_STALE_SEC"] = "9999"
        os.environ["SCALPER_COOLDOWN_LOSSES"] = "0"
        os.environ["SCALPER_COOLDOWN_MINUTES"] = "0"
        os.environ["SCALPER_MIN_BARS"] = "20"
        os.environ["SCALPER_FAST_BACKTEST"] = "1"
        os.environ["SCALPER_DYN_MOM_MULT"] = "0.05"
        os.environ["SCALPER_DYN_MOM_FLOOR"] = "0"
        os.environ["SCALPER_VWAP_BASE_DIST"] = "0.0001"
        os.environ["SCALPER_VWAP_ATR_MULT"] = "0.2"
        os.environ["SCALPER_ATR_PCT_SOFT_TH"] = "0.00001"
        os.environ["SCALPER_FORCE_ENTRY_TEST"] = "1"
        os.environ["SCALPER_FORCE_ENTRY_MIN_CONF"] = "0.0"
        os.environ["SCALPER_FORCE_ENTRY_MAX_PER_SYMBOL"] = "1"
        os.environ["SCALPER_TREND_CONFIRM"] = "1"
        os.environ["SCALPER_TREND_CONFIRM_MODE"] = "gate"
        os.environ["SCALPER_TREND_CONFIRM_TF"] = "1h"
        os.environ["SCALPER_TREND_CONFIRM_FAST"] = "10"
        os.environ["SCALPER_TREND_CONFIRM_SLOW"] = "30"

        df_1m = pd.DataFrame(_make_trend_rows(n=120, step=0.5), columns=["ts", "o", "h", "l", "c", "v"])
        df_1h = pd.DataFrame(_make_trend_rows(n=80, step=-0.2), columns=["ts", "o", "h", "l", "c", "v"])

        data = TrendConfirmData(df_1m, df_1h)
        long_sig, short_sig, _conf = eclipse_scalper.scalper_signal("BTCUSDT", data=data, cfg=None)
        self.assertFalse(long_sig)
        self.assertFalse(short_sig)


if __name__ == "__main__":
    unittest.main()
