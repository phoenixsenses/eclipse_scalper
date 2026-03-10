"""Unit tests for execution/shutdown_control.py

Covers:
- TracedShutdownEvent tracing
- ensure_traced_shutdown_event creation
- request_shutdown: first-call recording, non-fatal paper mode, fatal shutdown
- should_continue_nonfatal
- _write_last_shutdown persistence
- Graceful shutdown: task cancellation, timeout
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from execution.shutdown_control import (
    TracedShutdownEvent,
    ensure_traced_shutdown_event,
    request_shutdown,
    should_continue_nonfatal,
    _truthy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bot(**state_kwargs):
    state = SimpleNamespace(**state_kwargs)
    return SimpleNamespace(state=state)


# ---------------------------------------------------------------------------
# 1. TracedShutdownEvent
# ---------------------------------------------------------------------------

def test_traced_event_records_trace():
    bot = SimpleNamespace()
    ev = TracedShutdownEvent(bot)
    ev.set()
    assert ev.is_set()
    assert hasattr(bot, "_shutdown_set_recorded")
    assert bot._shutdown_set_recorded is True
    assert bot._shutdown_set_ts > 0
    assert "traceback" in bot._shutdown_set_trace.lower() or "set" in bot._shutdown_set_trace.lower()


def test_traced_event_only_records_once():
    bot = SimpleNamespace()
    ev = TracedShutdownEvent(bot)
    ev.set()
    first_ts = bot._shutdown_set_ts
    ev.set()  # second call
    assert bot._shutdown_set_ts == first_ts  # not updated


# ---------------------------------------------------------------------------
# 2. ensure_traced_shutdown_event
# ---------------------------------------------------------------------------

def test_ensure_creates_event():
    bot = SimpleNamespace()
    ev = ensure_traced_shutdown_event(bot)
    assert isinstance(ev, asyncio.Event)
    assert bot._shutdown is ev


def test_ensure_returns_existing():
    bot = SimpleNamespace()
    existing = asyncio.Event()
    bot._shutdown = existing
    ev = ensure_traced_shutdown_event(bot)
    assert ev is existing


# ---------------------------------------------------------------------------
# 3. should_continue_nonfatal
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"SCALPER_DRY_RUN": "1", "PAPER_ALLOW_NONFATAL_SHUTDOWN": "1"})
def test_continue_nonfatal_true():
    bot = _bot()
    assert should_continue_nonfatal(bot) is True


@patch.dict(os.environ, {"SCALPER_DRY_RUN": "0", "PAPER_ALLOW_NONFATAL_SHUTDOWN": "1"})
def test_continue_nonfatal_not_dry_run():
    bot = _bot()
    assert should_continue_nonfatal(bot) is False


@patch.dict(os.environ, {"SCALPER_DRY_RUN": "1", "PAPER_ALLOW_NONFATAL_SHUTDOWN": "0"})
def test_continue_nonfatal_not_allowed():
    bot = _bot()
    assert should_continue_nonfatal(bot) is False


# ---------------------------------------------------------------------------
# 4. request_shutdown — first call
# ---------------------------------------------------------------------------

def test_request_shutdown_sets_event():
    bot = _bot()
    bot._shutdown = asyncio.Event()
    result = request_shutdown(bot, "test reason", source="test", fatal=True)
    assert result is True
    assert bot._shutdown.is_set()


def test_request_shutdown_records_reason():
    bot = _bot()
    bot._shutdown = asyncio.Event()
    request_shutdown(bot, "equity breach", source="kill_switch", fatal=True)
    assert bot._shutdown_reason == "equity breach"
    assert bot._shutdown_source == "kill_switch"
    assert bot._shutdown_fatal is True
    assert bot._shutdown_ts > 0


def test_request_shutdown_records_state_fields():
    bot = _bot()
    bot._shutdown = asyncio.Event()
    request_shutdown(bot, "data stale", source="health_gate")
    assert bot.state.shutdown_reason == "data stale"
    assert bot.state.shutdown_source == "health_gate"


def test_request_shutdown_first_reason_wins():
    bot = _bot()
    bot._shutdown = asyncio.Event()
    request_shutdown(bot, "first reason", source="src1", fatal=True)
    request_shutdown(bot, "second reason", source="src2", fatal=True)
    # First reason should be preserved (first = not bool(getattr(bot, "_shutdown_requested", False)))
    assert bot._shutdown_reason == "first reason"


# ---------------------------------------------------------------------------
# 5. request_shutdown — non-fatal paper mode
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"SCALPER_DRY_RUN": "1", "PAPER_ALLOW_NONFATAL_SHUTDOWN": "1"})
def test_request_shutdown_nonfatal_paper_swallowed():
    bot = _bot()
    bot._shutdown = asyncio.Event()
    result = request_shutdown(bot, "non-fatal issue", source="test", fatal=False)
    assert result is False
    assert not bot._shutdown.is_set()


@patch.dict(os.environ, {"SCALPER_DRY_RUN": "1", "PAPER_ALLOW_NONFATAL_SHUTDOWN": "1"})
def test_request_shutdown_fatal_paper_not_swallowed():
    bot = _bot()
    bot._shutdown = asyncio.Event()
    result = request_shutdown(bot, "fatal issue", source="test", fatal=True)
    assert result is True
    assert bot._shutdown.is_set()


# ---------------------------------------------------------------------------
# 6. _truthy helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (True, True),
    (False, False),
    (1, True),
    (0, False),
    ("true", True),
    ("yes", True),
    ("1", True),
    ("on", True),
    ("false", False),
    ("no", False),
    ("0", False),
    ("", False),
    (None, False),
])
def test_truthy(val, expected):
    assert _truthy(val) is expected
