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


def test_request_shutdown_records_first_reason(monkeypatch) -> None:
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


def test_request_shutdown_nonfatal_paper_can_continue(monkeypatch) -> None:
    monkeypatch.setenv("SCALPER_DRY_RUN", "1")
    monkeypatch.setenv("PAPER_ALLOW_NONFATAL_SHUTDOWN", "1")

    bot = SimpleNamespace()
    bot._shutdown = asyncio.Event()
    bot.state = SimpleNamespace()

    applied = request_shutdown(bot, "paper transient", source="paper.test", fatal=False)
    assert applied is False
    assert bot._shutdown.is_set() is False
