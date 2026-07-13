from __future__ import annotations

import asyncio
from types import SimpleNamespace


try:
    from execution.shutdown_control import request_shutdown
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.shutdown_control import request_shutdown


def test_request_shutdown_records_first_reason(monkeypatch, tmp_path) -> None:
    # request_shutdown(..., fatal=True) below writes a real, unisolated
    # logs/last_shutdown.json (execution/shutdown_control.py hardcodes
    # this relative path) -- without chdir isolation this leaks a
    # fresh fatal-shutdown marker into the real repo logs/ dir, which
    # risk/kill_switch.py's POST-CRASH COOLDOWN then applies to any other
    # test instantiating kill-switch state within the next 300s
    # (observed: intermittent entry_loop test failures depending on
    # run order/timing). chdir to an isolated tmp dir keeps this test's
    # relative-path writes off the real repo state entirely.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCALPER_DRY_RUN", "0")
    monkeypatch.setenv("PAPER_ALLOW_NONFATAL_SHUTDOWN", "0")

    bot = SimpleNamespace()
    bot._shutdown = asyncio.Event()
    bot.state = SimpleNamespace()

    applied = request_shutdown(bot, "first", source="t1", fatal=False)
    assert applied is True
    assert bot._shutdown.is_set()
    assert str(getattr(bot, "_shutdown_reason", "")) == "first"
    assert str(getattr(bot, "_shutdown_source", "")) == "t1"

    applied2 = request_shutdown(bot, "second", source="t2", fatal=True)
    assert applied2 is True
    assert str(getattr(bot, "_shutdown_reason", "")) == "first"
    assert str(getattr(bot, "_shutdown_source", "")) == "t1"


def test_request_shutdown_nonfatal_paper_can_continue(monkeypatch, tmp_path) -> None:
    # See test_request_shutdown_records_first_reason above: isolate
    # request_shutdown()'s real, unisolated logs/last_shutdown.json write.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCALPER_DRY_RUN", "1")
    monkeypatch.setenv("PAPER_ALLOW_NONFATAL_SHUTDOWN", "1")

    bot = SimpleNamespace()
    bot._shutdown = asyncio.Event()
    bot.state = SimpleNamespace()

    applied = request_shutdown(bot, "paper transient", source="paper.test", fatal=False)
    assert applied is False
    assert bot._shutdown.is_set() is False
