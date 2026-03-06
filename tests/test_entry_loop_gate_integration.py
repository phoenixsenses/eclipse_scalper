from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock


try:
    from execution import entry_loop as el
    from execution.health_gate import GateDecision, GateState
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry_loop as el
    from execution.health_gate import GateDecision, GateState


def test_entry_loop_blocks_submit_when_health_gated(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_HEALTH_GATE_ENABLED", "1")
        monkeypatch.setenv("ENTRY_HEALTH_BLOCK_SLEEP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")

        submit_mock = AsyncMock(return_value={"id": "X"})
        monkeypatch.setattr(el, "create_order", submit_mock)

        def fake_eval(*args, **kwargs):
            return GateDecision(
                allow=False,
                reason="collector_disconnected",
                state="degraded",
                collector_connected=False,
                collector_lag_sec=99,
                reconnects_last_5m=0,
                errors_last_5m=0,
            )

        write_calls = []
        log_calls = []

        monkeypatch.setattr(el, "evaluate_health_gate", fake_eval)
        monkeypatch.setattr(el, "load_overall_health", lambda *_a, **_k: {"state": "degraded", "components": {"collector": {"connected": False}}})
        monkeypatch.setattr(el, "write_paper_trader_health", lambda decision, reason: write_calls.append((decision, reason)))
        monkeypatch.setattr(el.log_core, "critical", lambda msg: log_calls.append(str(msg)))

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
        assert any("[GATE] paper_trader halted" in x and "collector_disconnected" in x for x in log_calls)
        assert len(write_calls) >= 1
        assert any(reason == "collector_disconnected" for _, reason in write_calls)

    asyncio.run(_run())

