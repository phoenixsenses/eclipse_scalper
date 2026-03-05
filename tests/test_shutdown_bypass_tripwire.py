from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from pathlib import Path


try:
    from execution import entry_loop_full as elf
    from execution.shutdown_control import TracedShutdownEvent
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry_loop_full as elf
    from execution.shutdown_control import TracedShutdownEvent


def test_shutdown_bypass_tripwire_writes_forensics(monkeypatch) -> None:
    async def _run() -> None:
        tdir = Path("localtests/shutdown_bypass_tripwire")
        tdir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tdir)
        logs = []
        monkeypatch.setattr(elf.log_core, "critical", lambda msg: logs.append(str(msg)))

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot._shutdown = TracedShutdownEvent(bot)
        bot._shutdown.set()
        bot.data_ready = asyncio.Event()

        await elf.entry_loop_full(bot)

        text = "\n".join(logs)
        assert "[SHUTDOWN_BYPASS_DETECTED]" in text
        jf = Path("logs/last_shutdown.json")
        assert jf.exists()
        payload = json.loads(jf.read_text(encoding="utf-8"))
        assert payload.get("reason") == "bypass_detected"
        assert payload.get("source") == "unknown_setter"
        trace = str(payload.get("trace") or "")
        assert "test_shutdown_bypass_tripwire.py" in trace
        assert "_tripwire_shutdown_bypass" not in trace

    asyncio.run(_run())


def test_guardian_uses_request_shutdown_not_direct_set() -> None:
    import pathlib

    src = pathlib.Path("execution/guardian.py").read_text(encoding="utf-8")
    assert "request_shutdown(" in src
    assert "shutdown_ev.set()" not in src


def test_shutdown_teardown_whitelist_not_bypass(monkeypatch) -> None:
    async def _run() -> None:
        tdir = Path("localtests/shutdown_bypass_teardown")
        tdir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tdir)
        logs = []
        monkeypatch.setattr(elf.log_core, "critical", lambda msg: logs.append(str(msg)))

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot._shutdown = TracedShutdownEvent(bot)
        bot._teardown_in_progress = True
        bot._shutdown.set()
        bot.data_ready = asyncio.Event()

        await elf.entry_loop_full(bot)

        text = "\n".join(logs)
        assert "[SHUTDOWN_BYPASS_DETECTED]" not in text
        assert "reason=teardown_cleanup" in text
        jf = Path("logs/last_shutdown.json")
        assert jf.exists() is False

    asyncio.run(_run())
