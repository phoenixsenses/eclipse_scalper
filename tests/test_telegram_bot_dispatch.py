from __future__ import annotations

from pathlib import Path

try:
    from tools.telegram_bot import _dispatch, _is_allowed
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.telegram_bot import _dispatch, _is_allowed


def test_dispatch_help_and_unknown() -> None:
    assert "Commands:" in _dispatch("/help")
    assert "Unknown command" in _dispatch("/unknown")


def test_allowed_chat_ids() -> None:
    assert _is_allowed(123, {"123"}) is True
    assert _is_allowed(124, {"123"}) is False
    assert _is_allowed(124, set()) is True

