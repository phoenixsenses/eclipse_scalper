from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


def _mk_signal(*, pocket_name: str = "pocket_a", regime: str = "UP") -> MicroSignal:
    feat = MicroFeatures(
        timestamp=time.time(),
        symbol="ETHUSDT",
        imbalance=0.90,
        imbalance_signed=0.90,
        trade_intensity=7200.0,
        spread=0.00015,
        mark_price=100.0,
        age_sec=0.1,
    )
    return MicroSignal(
        symbol="ETHUSDT",
        side="buy",
        confidence=0.95,
        pocket_name=pocket_name,
        features=feat,
        regime=regime,
        regime_age_sec=10.0,
        order_type="limit",
        fill_timeout_sec=0.01,
    )


def _base_env(monkeypatch) -> None:
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
    monkeypatch.setenv("REGIME_SIZER_ENABLED", "1")


def _mk_bot(*, pocket_scheduler=None):
    bot = SimpleNamespace()
    bot.cfg = SimpleNamespace()
    bot.state = SimpleNamespace(
        positions={},
        telemetry=SimpleNamespace(recent=[]),
        kill_metrics={},
        run_context={},
    )
    if pocket_scheduler is not None:
        bot.state.run_context["pocket_scheduler"] = pocket_scheduler
    bot.active_symbols = ["ETHUSDT"]
    bot._shutdown = asyncio.Event()
    bot.data_ready = asyncio.Event()
    bot.data_ready.set()
    bot.data = SimpleNamespace(get_price=lambda *args, **kwargs: 100.0, price={"ETHUSDT": 100.0})
    return bot


def _reset_entry_loop_state() -> None:
    el._PENDING_UNTIL.clear()
    el._PENDING_ORDER_ID.clear()
    el._ENTRY_LOCKS.clear()


def test_entry_loop_blocks_submit_when_pocket_scheduler_blocks(monkeypatch) -> None:
    async def _run() -> None:
        _base_env(monkeypatch)

        class _BlockedScheduler:
            def can_fire(self, pocket_name: str, now: float):
                return False, "cooldown 30s remaining"

            def record_fire(self, pocket_name: str, now: float) -> None:
                raise AssertionError("record_fire should not be called when blocked")

            def reset_daily(self) -> None:
                return

        submit_mock = AsyncMock(return_value={"id": "OID1"})
        blocked_mock = AsyncMock()

        monkeypatch.setattr(el, "create_order", submit_mock)
        monkeypatch.setattr(el, "_emit_entry_blocked", blocked_mock)
        monkeypatch.setattr(el, "_load_signal_fn", lambda: None)
        monkeypatch.setattr(
            el,
            "_build_micro_signal_provider",
            lambda bot: (_FakeMicroEngine(), _FakeMicroProvider(_mk_signal(pocket_name="blocked_pocket"))),
        )
        monkeypatch.setattr(el, "staleness_check", None)
        monkeypatch.setattr(el, "update_quality_state", None)
        _reset_entry_loop_state()

        bot = _mk_bot(pocket_scheduler=_BlockedScheduler())
        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.sleep(0.08)
        bot._shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

        assert submit_mock.await_count == 0
        assert blocked_mock.await_count >= 1
        assert blocked_mock.await_args.args[1] == "ETHUSDT"
        assert blocked_mock.await_args.args[2] == "pocket_scheduler:cooldown 30s remaining"

    asyncio.run(_run())


def test_entry_loop_applies_regime_size_scale_before_submit(monkeypatch) -> None:
    async def _run() -> None:
        _base_env(monkeypatch)
        monkeypatch.setenv("REGIME_SIZER_UP_BUY", "1.25")

        async def _fake_create_order(bot, **kwargs):
            bot._shutdown.set()
            return {"id": "OID1"}

        submit_mock = AsyncMock(side_effect=_fake_create_order)
        cancel_mock = AsyncMock(return_value=True)

        monkeypatch.setattr(el, "create_order", submit_mock)
        monkeypatch.setattr(el, "cancel_order", cancel_mock)
        monkeypatch.setattr(el, "_load_signal_fn", lambda: None)
        monkeypatch.setattr(
            el,
            "_build_micro_signal_provider",
            lambda bot: (_FakeMicroEngine(), _FakeMicroProvider(_mk_signal(regime="UP"))),
        )
        monkeypatch.setattr(el, "staleness_check", None)
        monkeypatch.setattr(el, "update_quality_state", None)
        _reset_entry_loop_state()

        bot = _mk_bot()
        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.wait_for(task, timeout=2.0)

        assert submit_mock.await_count == 1
        kwargs = submit_mock.await_args.kwargs
        assert kwargs["type"] == "LIMIT"
        assert kwargs["side"] == "buy"
        assert float(kwargs["amount"]) == pytest.approx(0.625)

    asyncio.run(_run())
