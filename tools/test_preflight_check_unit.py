#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
import tempfile
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.tools import preflight_check as pc  # noqa: E402


def _base_env() -> dict[str, str]:
    return {
        "SCALPER_DRY_RUN": "1",
        "ACTIVE_SYMBOLS": "DOGEUSDT",
        "FIXED_NOTIONAL_USDT": "6",
        "LEVERAGE": "1",
        "MARGIN_MODE": "isolated",
        "FIRST_LIVE_SAFE": "1",
        "FIRST_LIVE_SYMBOLS": "DOGEUSDT",
        "FIRST_LIVE_MAX_NOTIONAL_USDT": "6",
        "MAX_DAILY_LOSS_PCT": "0.02",
        "MAX_DRAWDOWN_PCT": "0.06",
        "CORR_GROUPS": "MEME:DOGEUSDT,SHIBUSDT",
        "CORR_GROUP_MAX_POSITIONS": "1",
        "CORR_GROUP_MAX_NOTIONAL_USDT": "25",
        "RUNTIME_RELIABILITY_COUPLING": "1",
    }


class PreflightCheckTests(unittest.TestCase):
    def test_validate_env_happy_path(self):
        errors, warnings, summary = pc.validate_env(_base_env(), safe_profile_max_leverage=1.0)
        self.assertEqual(errors, [])
        self.assertIsInstance(warnings, list)
        self.assertTrue(bool(summary.get("active_symbols")))
        self.assertTrue(bool(summary.get("runtime_reliability_coupling")))

    def test_invalid_scalper_dry_run_rejected(self):
        env = _base_env()
        env["SCALPER_DRY_RUN"] = "2"
        errors, _, _ = pc.validate_env(env)
        self.assertTrue(any("SCALPER_DRY_RUN" in e for e in errors))

    def test_first_live_notional_cap_enforced(self):
        env = _base_env()
        env["FIXED_NOTIONAL_USDT"] = "10"
        env["FIRST_LIVE_MAX_NOTIONAL_USDT"] = "6"
        errors, _, _ = pc.validate_env(env)
        self.assertTrue(any("FIXED_NOTIONAL_USDT must be <=" in e for e in errors))

    def test_leverage_bound_enforced(self):
        env = _base_env()
        env["LEVERAGE"] = "3"
        errors, _, _ = pc.validate_env(env, safe_profile_max_leverage=1.0)
        self.assertTrue(any("SAFE_PROFILE_MAX_LEVERAGE" in e for e in errors))

    def test_margin_mode_must_be_isolated(self):
        env = _base_env()
        env["MARGIN_MODE"] = "cross"
        errors, _, _ = pc.validate_env(env)
        self.assertTrue(any("MARGIN_MODE" in e for e in errors))

    def test_daily_and_drawdown_safe_range(self):
        env = _base_env()
        env["MAX_DAILY_LOSS_PCT"] = "0.30"
        env["MAX_DRAWDOWN_PCT"] = "0.10"
        errors, _, _ = pc.validate_env(env, daily_loss_max=0.05, drawdown_max=0.20)
        self.assertTrue(any("MAX_DAILY_LOSS_PCT" in e for e in errors))

    def test_corr_groups_malformed_rejected(self):
        env = _base_env()
        env["CORR_GROUPS"] = "BROKEN_GROUP"
        errors, _, _ = pc.validate_env(env)
        self.assertTrue(any("CORR_GROUPS token" in e for e in errors))

    def test_reliability_gate_stale_warning(self):
        env = _base_env()
        with tempfile.TemporaryDirectory() as td:
            gate = Path(td) / "reliability_gate.txt"
            gate.write_text("ok\n", encoding="utf-8")
            old_ts = time.time() - 3600.0
            gate.touch()
            os.utime(gate, (old_ts, old_ts))
            env["RELIABILITY_GATE_PATH"] = str(gate)
            env["RELIABILITY_GATE_STALE_SECONDS"] = "900"
            errors, warnings, summary = pc.validate_env(env)
            self.assertEqual(errors, [])
            self.assertTrue(any("appears stale" in w for w in warnings))
            self.assertGreater(float(summary.get("reliability_gate_age_seconds", -1.0)), 900.0)


if __name__ == "__main__":
    unittest.main()
