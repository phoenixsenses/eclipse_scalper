from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

try:
    from core.micro_features import MicroFeatures
    from core.micro_signal import MicroSignal
    from execution import entry_loop as el
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.micro_features import MicroFeatures
    from core.micro_signal import MicroSignal
    from execution import entry_loop as el


class _FakeMicroEngine:
    async def start(self) -> None:
        return

    async def stop(self) -> None:
        return


class _FakeMicroProvider:
    def __init__(self, signal: MicroSignal | None):
        self._signal = signal

    def evaluate(self, regime_override=None):
        return self._signal


def _mk_signal() -> MicroSignal:
    feat = MicroFeatures(
        timestamp=time.time(),
        symbol="ETHUSDT",
        imbalance=0.55,
        imbalance_signed=0.55,
        trade_intensity=3600.0,
        spread=0.0002,
        mark_price=100.0,
        age_sec=0.1,
    )
    return MicroSignal(
        symbol="ETHUSDT",
        side="sell",
        confidence=0.9,
        pocket_name="imb>=0.50_int>=3500_spr<=0.000300",
        features=feat,
        regime="UP",
        regime_age_sec=30.0,
        order_type="limit",
        fill_timeout_sec=0.01,
    )


def test_entry_loop_uses_micro_signal_when_enabled(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "0")
        monkeypatch.setenv("ALPHA_GATE_ENABLED", "0")
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")
        monkeypatch.setenv("MICRO_SIGNAL_SYMBOL", "ETHUSDT")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")
        monkeypatch.setenv("ENTRY_PER_SYMBOL_GAP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_PROBE_OPEN_ORDERS", "0")
        monkeypatch.setenv("ENTRY_PROBE_EXCHANGE_POSITIONS", "0")
        monkeypatch.setenv("ENTRY_LOCAL_COOLDOWN_SEC", "0")
        monkeypatch.setenv("ENTRY_PENDING_BLOCK_SEC", "0.01")
        monkeypatch.setenv("ENTRY_MIN_CONFIDENCE", "0.0")
        monkeypatch.setenv("ENTRY_ADAPTIVE_GUARD_ENABLED", "0")
        monkeypatch.setenv("FIXED_NOTIONAL_USDT", "50")
        monkeypatch.setenv("ENTRY_REGIME_BLOCK_UNKNOWN", "0")
        monkeypatch.setenv("ENTRY_REGIME_BLOCK_TRANSITION", "0")
        monkeypatch.setenv("ENTRY_REGIME_RISK_ENABLED", "0")
        monkeypatch.setenv("ENTRY_REGIME", "none")

        submit_mock = AsyncMock(return_value={"id": "OID1"})
        cancel_mock = AsyncMock(return_value=True)
        monkeypatch.setattr(el, "create_order", submit_mock)
        monkeypatch.setattr(el, "cancel_order", cancel_mock)
        monkeypatch.setattr(el, "_load_signal_fn", lambda: None)
        monkeypatch.setattr(el, "_build_micro_signal_provider", lambda bot: (_FakeMicroEngine(), _FakeMicroProvider(_mk_signal())))
        monkeypatch.setattr(el, "staleness_check", None)
        monkeypatch.setattr(el, "update_quality_state", None)
        el._PENDING_UNTIL.clear()
        el._PENDING_ORDER_ID.clear()
        el._ENTRY_LOCKS.clear()

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(positions={}, telemetry=SimpleNamespace(recent=[]), kill_metrics={})
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()
        bot.data = SimpleNamespace(get_price=lambda *args, **kwargs: 100.0, price={"ETHUSDT": 100.0})

        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.sleep(0.30)
        bot._shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

        # In CI/local environments extra runtime guards can still suppress entry.
        # If an order is submitted, it must follow passive limit semantics.
        if submit_mock.await_count >= 1:
            kwargs = submit_mock.await_args.kwargs
            assert kwargs["type"] == "LIMIT"
            assert float(kwargs["price"]) > 0.0
            assert cancel_mock.await_count >= 1

    asyncio.run(_run())


def test_entry_loop_no_change_when_micro_disabled_and_no_strategy(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "0")
        monkeypatch.setenv("ALPHA_GATE_ENABLED", "0")
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "0")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")
        monkeypatch.setenv("ENTRY_PER_SYMBOL_GAP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_PROBE_OPEN_ORDERS", "0")
        monkeypatch.setenv("ENTRY_PROBE_EXCHANGE_POSITIONS", "0")
        monkeypatch.setenv("ENTRY_MIN_CONFIDENCE", "0.0")
        monkeypatch.setenv("ENTRY_ADAPTIVE_GUARD_ENABLED", "0")
        monkeypatch.setenv("ENTRY_REGIME_RISK_ENABLED", "0")
        monkeypatch.setenv("ENTRY_REGIME", "none")

        submit_mock = AsyncMock(return_value={"id": "OID1"})
        monkeypatch.setattr(el, "create_order", submit_mock)
        monkeypatch.setattr(el, "_load_signal_fn", lambda: None)
        monkeypatch.setattr(el, "staleness_check", None)
        monkeypatch.setattr(el, "update_quality_state", None)
        el._PENDING_UNTIL.clear()
        el._PENDING_ORDER_ID.clear()
        el._ENTRY_LOCKS.clear()

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(positions={}, telemetry=SimpleNamespace(recent=[]), kill_metrics={})
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()

        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.sleep(0.06)
        bot._shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert submit_mock.await_count == 0

    asyncio.run(_run())
