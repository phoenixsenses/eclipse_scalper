from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

try:
    from execution import entry_loop as el
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry_loop as el


def test_entry_loop_blocks_submit_when_alpha_gated(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "0")
        monkeypatch.setenv("ALPHA_GATE_ENABLED", "1")
        monkeypatch.setenv("ENTRY_ALPHA_BLOCK_SLEEP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")

        submit_mock = AsyncMock(return_value={"id": "X"})
        monkeypatch.setattr(el, "create_order", submit_mock)

        class _A:
            blocked = True
            reason = "alpha_negative_edge"
            details = {"pnl_net_per_fill": -0.001, "decision_to_fill_rate": 0.1}

        monkeypatch.setattr(el, "evaluate_alpha_gate_from_env", lambda now_ts=None: _A())
        logs = []
        monkeypatch.setattr(el.log_core, "critical", lambda msg: logs.append(str(msg)))

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(positions={}, telemetry=SimpleNamespace(recent=[]), kill_metrics={})
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()

        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.sleep(0.08)
        bot._shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

        assert submit_mock.await_count == 0
        assert any("[GATE] alpha halted" in x for x in logs)

    asyncio.run(_run())


def test_entry_loop_blocks_submit_when_alpha_gated_stability_mode(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "0")
        monkeypatch.setenv("ALPHA_GATE_ENABLED", "1")
        monkeypatch.setenv("ALPHA_GATE_MODE", "stability")
        monkeypatch.setenv("ENTRY_ALPHA_BLOCK_SLEEP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")

        submit_mock = AsyncMock(return_value={"id": "X"})
        monkeypatch.setattr(el, "create_order", submit_mock)

        class _A:
            blocked = True
            reason = "alpha_unstable"
            details = {"alpha_gate_mode": "stability", "pos_slices_frac": 0.2}

        monkeypatch.setattr(el, "evaluate_alpha_gate_from_env", lambda now_ts=None: _A())
        logs = []
        monkeypatch.setattr(el.log_core, "critical", lambda msg: logs.append(str(msg)))

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(positions={}, telemetry=SimpleNamespace(recent=[]), kill_metrics={})
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()

        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.sleep(0.08)
        bot._shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

        assert submit_mock.await_count == 0
        assert any("[GATE] alpha halted" in x for x in logs)

    asyncio.run(_run())


def test_entry_loop_blocks_submit_when_alpha_both_regime_env(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "0")
        monkeypatch.setenv("ALPHA_GATE_ENABLED", "1")
        monkeypatch.setenv("ALPHA_GATE_MODE", "both_regime")
        monkeypatch.setenv("ALPHA_GATE_METRICS_PATH", "runs/latest/metrics.json")
        monkeypatch.setenv("ALPHA_GATE_STABILITY_PATH", "runs/latest/stability.csv")
        monkeypatch.setenv("ALPHA_GATE_STABILITY_UP_PATH", "runs/latest/stability_up.csv")
        monkeypatch.setenv("ALPHA_GATE_STABILITY_DOWN_PATH", "runs/latest/stability_down.csv")
        monkeypatch.setenv("ENTRY_ALPHA_BLOCK_SLEEP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")

        submit_mock = AsyncMock(return_value={"id": "X"})
        monkeypatch.setattr(el, "create_order", submit_mock)
        logs = []
        monkeypatch.setattr(el.log_core, "critical", lambda msg: logs.append(str(msg)))

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.state = SimpleNamespace(positions={}, telemetry=SimpleNamespace(recent=[]), kill_metrics={})
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()

        task = asyncio.create_task(el.entry_loop(bot))
        await asyncio.sleep(0.08)
        bot._shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

        assert submit_mock.await_count == 0
        assert any("[GATE] alpha halted" in x for x in logs)

    asyncio.run(_run())
