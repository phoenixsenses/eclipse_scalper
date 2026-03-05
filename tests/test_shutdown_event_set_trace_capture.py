from __future__ import annotations

from types import SimpleNamespace


try:
    from execution.shutdown_control import TracedShutdownEvent
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.shutdown_control import TracedShutdownEvent


def _direct_set(ev):
    ev.set()


def test_traced_shutdown_event_captures_setter_stack() -> None:
    bot = SimpleNamespace()
    ev = TracedShutdownEvent(bot)
    _direct_set(ev)
    trace = str(getattr(bot, "_shutdown_set_trace", "") or "")
    assert trace
    assert "_direct_set" in trace
    assert "test_shutdown_event_set_trace_capture.py" in trace
    assert float(getattr(bot, "_shutdown_set_ts", 0.0) or 0.0) > 0.0
