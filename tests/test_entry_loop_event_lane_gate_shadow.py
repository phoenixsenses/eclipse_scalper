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
        pocket_name="imb>=0.85_int>=7000_spr<=0.000200",
        features=feat,
        regime="UP",
        regime_age_sec=10.0,
        order_type="limit",
        fill_timeout_sec=0.01,
    )


def test_entry_loop_event_lane_gate_shadow_emits_without_blocking(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "0")
        monkeypatch.setenv("ALPHA_GATE_ENABLED", "0")
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")
        monkeypatch.setenv("MICRO_SIGNAL_SYMBOL", "ETHUSDT")
        monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_ENABLED", "1")
        monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_SHADOW", "1")
        monkeypatch.setenv("ENTRY_EVENT_LANE_GATE_DB", "data/microstructure.db")
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

        async def _fake_create_order(bot, **kwargs):
            bot._shutdown.set()
            return {"id": "OID1"}

        submit_mock = AsyncMock(side_effect=_fake_create_order)
        cancel_mock = AsyncMock(return_value=True)
        blocked_mock = AsyncMock()
        gate_emit_mock = AsyncMock()

        monkeypatch.setattr(el, "create_order", submit_mock)
        monkeypatch.setattr(el, "cancel_order", cancel_mock)
        monkeypatch.setattr(el, "_emit_entry_blocked", blocked_mock)
        monkeypatch.setattr(el, "emit_throttled", gate_emit_mock)
        monkeypatch.setattr(el, "_load_signal_fn", lambda: None)
        monkeypatch.setattr(
            el,
            "_build_micro_signal_provider",
            lambda bot: (_FakeMicroEngine(), _FakeMicroProvider(_mk_signal())),
        )
        monkeypatch.setattr(el, "staleness_check", None)
        monkeypatch.setattr(el, "update_quality_state", None)
        monkeypatch.setattr(
            el.event_lane_gate,
            "load_current_event_gate",
            lambda **kwargs: {
                "gate": "blocked",
                "allow_trade": False,
                "blocked_lanes": ["book_proxy_pressure"],
                "latest_ts_ms": 123,
                "latest_abs_imbalance": 0.91,
                "lanes": {
                    "book_proxy_pressure": {"rule_fired": True, "severity": "high"},
                    "volatility_burst": {"rule_fired": False, "severity": "none"},
                },
            },
        )
        monkeypatch.setattr(
            el.event_lane_gate,
            "should_block_event_gate",
            lambda *args, **kwargs: (
                True,
                "event_lane_gate_blocked",
                {"blocking_lanes": ["book_proxy_pressure"]},
            ),
        )
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
        await asyncio.wait_for(task, timeout=2.0)

        assert submit_mock.await_count >= 1
        assert blocked_mock.await_count >= 1
        _, symbol, reason = blocked_mock.await_args.args[:3]
        assert symbol == "ETHUSDT"
        assert reason == "event_lane_gate_shadow"
        assert blocked_mock.await_args.kwargs["data"]["blocking_lanes"] == ["book_proxy_pressure"]
        assert gate_emit_mock.await_count >= 1
        assert gate_emit_mock.await_args.args[1] == "entry.event_lane_gate"
        assert gate_emit_mock.await_args.kwargs["data"]["decision"] == "would_block"

    asyncio.run(_run())
