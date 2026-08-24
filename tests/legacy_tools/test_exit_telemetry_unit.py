#!/usr/bin/env python3
"""
Unit-style tests for exit-side telemetry emits.
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


class ExitTelemetryTests(unittest.TestCase):
    def test_cancel_order_safe_emits(self):
        calls = []

        async def _fake_emit_throttled(*_args, **kwargs):
            calls.append({"args": _args, "kwargs": kwargs})

        exit_mod.emit_throttled = _fake_emit_throttled  # type: ignore

        async def _fake_cancel(_bot, _oid, _sym):
            raise RuntimeError("cancel failed")

        exit_mod.cancel_order = _fake_cancel  # type: ignore

        bot = SimpleNamespace(state=SimpleNamespace())
        asyncio.run(exit_mod._cancel_order_safe(bot, "oid", "BTCUSDT"))
        self.assertTrue(calls)

    def test_velocity_exit_emits_success(self):
        calls = []

        async def _fake_emit_throttled(*_args, **kwargs):
            calls.append({"kwargs": kwargs})

        created = {"ok": False}

        async def _fake_create_order(*_args, **_kwargs):
            created["ok"] = True
            return {"id": "ok"}

        exit_mod.emit_throttled = _fake_emit_throttled  # type: ignore
        exit_mod.create_order = _fake_create_order  # type: ignore

        bot = SimpleNamespace(
            ex=None,
            data=SimpleNamespace(get_price=lambda *_a, **_k: 90.0, price={}),
            state=SimpleNamespace(
                positions={"BTCUSDT": SimpleNamespace(
                    side="long",
                    size=1.0,
                    entry_price=100.0,
                    entry_ts=__import__("time").time(),
                    atr=0.0,
                    hard_stop_order_id=None,
                )},
                known_exit_order_ids=set(),
                symbol_performance={},
                total_wins=0,
                win_streak=0,
                total_trades=0,
                win_rate=0.0,
                consecutive_losses={},
                blacklist={},
                last_exit_time={},
                current_equity=0.0,
            ),
            cfg=SimpleNamespace(
                BREAKEVEN_BUFFER_ATR_MULT=0.0,
                VELOCITY_DRAWDOWN_PCT=0.001,
                VELOCITY_MINUTES=999.0,
                BLACKLIST_AUTO_RESET_ON_PROFIT=False,
                CONSECUTIVE_LOSS_BLACKLIST_COUNT=9999,
                SYMBOL_BLACKLIST_DURATION_HOURS=0,
                TRAILING_ACTIVATION_RR=0.0,
                TRAILING_REBUILD_DEBOUNCE_SEC=999999,
            ),
        )

        order = {
            "id": "123",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "filled": 0.1,
            "info": {},
        }

        async def _run():
            await exit_mod.handle_exit(bot, order)
            await asyncio.sleep(0.01)

        asyncio.run(_run())
        def _reason(call):
            if "kwargs" in call:
                return (call["kwargs"].get("data") or {}).get("reason")
            return None

        self.assertTrue(created["ok"])
        self.assertTrue(any(_reason(c) == "velocity" for c in calls))


if __name__ == "__main__":
    unittest.main()
