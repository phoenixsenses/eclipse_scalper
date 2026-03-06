from __future__ import annotations

import asyncio
from pathlib import Path

try:
    import tools.push_status as ps
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tools.push_status as ps


class _NotifierFail:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    async def speak(self, text: str, priority: str = "normal", silent: bool = False) -> bool:
        return False


def test_push_status_requires_chat(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_TOKEN", "x")
    rc = asyncio.run(ps._run(silent=False))
    out = capsys.readouterr().out
    assert rc == 2
    assert "TELEGRAM_CHAT_ID is required" in out


def test_push_status_fail_does_not_print_status_sent(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("TELEGRAM_TOKEN", "x")
    monkeypatch.setattr(ps, "Notifier", _NotifierFail)
    rc = asyncio.run(ps._run(silent=False))
    out = capsys.readouterr().out
    assert rc == 3
    assert "status_sent" not in out
    assert "telegram send failed" in out
