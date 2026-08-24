from __future__ import annotations

import asyncio
from types import SimpleNamespace


try:
    from execution import entry_loop_full as elf
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry_loop_full as elf


def test_entry_loop_full_shutdown_logs_reason(monkeypatch) -> None:
    async def _run() -> None:
        logs = []
        monkeypatch.setattr(elf.log_core, "critical", lambda msg: logs.append(str(msg)))
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot._shutdown = asyncio.Event()
        bot._shutdown.set()
        bot.data_ready = asyncio.Event()
        bot._shutdown_reason = "unit_reason"
        bot._shutdown_source = "unit_source"
        bot._shutdown_fatal = True
        bot._shutdown_ts = 1.0

        await elf.entry_loop_full(bot)

        joined = "\n".join(logs)
        assert "ENTRY_LOOP_FULL OFFLINE - shutdown flag set reason=unit_reason" in joined
        assert "source=unit_source" in joined
        assert "fatal=1" in joined

    asyncio.run(_run())
