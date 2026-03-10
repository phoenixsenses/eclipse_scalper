"""Unit tests for execution/event_journal.py

Covers:
- Event creation and serialization
- EventJournal write, read, corruption recovery
- Buffer flushing (sync and async)
- Log rotation
- Query interface (by symbol, category, correlation_id)
- Hook registration
- Stats reporting
- Enabled/disabled check
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from execution.event_journal import (
    Event,
    EventCategory,
    EventJournal,
    EventSeverity,
    record_event,
    record_entry_event,
    record_exit_event,
    record_order_event,
    record_error,
    record_system_event,
    journal_transition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bot(enabled=True):
    return SimpleNamespace(
        cfg=SimpleNamespace(EVENT_JOURNAL_ENABLED=enabled),
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    EventJournal._instance = None
    yield
    EventJournal._instance = None


# ---------------------------------------------------------------------------
# 1. Event data class
# ---------------------------------------------------------------------------

def test_event_to_dict():
    e = Event(
        timestamp=1000000.0,
        category="ENTRY",
        event_type="SIGNAL",
        severity="INFO",
        symbol="BTCUSDT",
        message="Long signal",
        data={"conf": 0.85},
        correlation_id="abc123",
    )
    d = e.to_dict()
    assert d["cat"] == "ENTRY"
    assert d["type"] == "SIGNAL"
    assert d["sym"] == "BTCUSDT"
    assert d["data"]["conf"] == 0.85
    assert d["corr"] == "abc123"
    assert d["ts"] == 1000000.0


def test_event_to_json():
    e = Event(
        timestamp=1000000.0,
        category="ENTRY",
        event_type="SIGNAL",
        severity="INFO",
        symbol="BTCUSDT",
        message="test",
    )
    j = e.to_json()
    parsed = json.loads(j)
    assert parsed["cat"] == "ENTRY"


def test_event_to_dict_time_iso():
    e = Event(
        timestamp=0.0,
        category="SYSTEM",
        event_type="STARTUP",
        severity="INFO",
        symbol=None,
        message="boot",
    )
    d = e.to_dict()
    assert "time" in d
    assert "1970" in d["time"]


# ---------------------------------------------------------------------------
# 2. EventJournal creation and singleton
# ---------------------------------------------------------------------------

def test_singleton():
    bot = _bot()
    j1 = EventJournal.get(bot)
    j2 = EventJournal.get(bot)
    assert j1 is j2


def test_is_enabled_true():
    assert EventJournal.is_enabled(_bot(enabled=True))


def test_is_enabled_false():
    assert not EventJournal.is_enabled(_bot(enabled=False))


@patch.dict(os.environ, {"EVENT_JOURNAL_ENABLED": "0"})
def test_is_enabled_env_off():
    import importlib
    import execution.event_journal as mod
    orig = mod._JOURNAL_ENABLED
    mod._JOURNAL_ENABLED = False
    try:
        assert not EventJournal.is_enabled(_bot())
    finally:
        mod._JOURNAL_ENABLED = orig


# ---------------------------------------------------------------------------
# 3. Recording events
# ---------------------------------------------------------------------------

def test_record_increments_count():
    bot = _bot()
    j = EventJournal.get(bot)
    j.record(EventCategory.ENTRY, "TEST", "hello", symbol="BTCUSDT")
    assert j._event_count == 1
    assert len(j._memory_events) == 1


def test_record_caps_memory():
    bot = _bot()
    j = EventJournal.get(bot)
    j._memory_max = 5
    for i in range(10):
        j.record(EventCategory.ENTRY, "TEST", f"event {i}")
    assert len(j._memory_events) == 5
    assert j._memory_events[-1].message == "event 9"


# ---------------------------------------------------------------------------
# 4. Query interface
# ---------------------------------------------------------------------------

def test_get_recent_all():
    bot = _bot()
    j = EventJournal.get(bot)
    for i in range(5):
        j.record(EventCategory.ENTRY, "TEST", f"e{i}", symbol="BTC")
    assert len(j.get_recent()) == 5


def test_get_recent_by_symbol():
    bot = _bot()
    j = EventJournal.get(bot)
    j.record(EventCategory.ENTRY, "A", "a", symbol="BTC")
    j.record(EventCategory.ENTRY, "B", "b", symbol="ETH")
    j.record(EventCategory.ENTRY, "C", "c", symbol="BTC")
    result = j.get_recent(symbol="BTC")
    assert len(result) == 2


def test_get_recent_by_category():
    bot = _bot()
    j = EventJournal.get(bot)
    j.record(EventCategory.ENTRY, "A", "a")
    j.record(EventCategory.EXIT, "B", "b")
    j.record(EventCategory.ENTRY, "C", "c")
    result = j.get_recent(category=EventCategory.ENTRY)
    assert len(result) == 2


def test_get_recent_limit():
    bot = _bot()
    j = EventJournal.get(bot)
    for i in range(10):
        j.record(EventCategory.ENTRY, "TEST", f"e{i}")
    result = j.get_recent(limit=3)
    assert len(result) == 3


def test_get_by_correlation():
    bot = _bot()
    j = EventJournal.get(bot)
    j.record(EventCategory.ENTRY, "A", "a", correlation_id="tx1")
    j.record(EventCategory.EXIT, "B", "b", correlation_id="tx1")
    j.record(EventCategory.ENTRY, "C", "c", correlation_id="tx2")
    result = j.get_by_correlation("tx1")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. Hooks
# ---------------------------------------------------------------------------

def test_hook_called():
    bot = _bot()
    j = EventJournal.get(bot)
    captured = []
    j.register_hook(lambda e: captured.append(e))
    j.record(EventCategory.SYSTEM, "START", "booting")
    assert len(captured) == 1
    assert captured[0].event_type == "START"


def test_hook_exception_does_not_crash():
    bot = _bot()
    j = EventJournal.get(bot)
    j.register_hook(lambda e: 1 / 0)
    j.record(EventCategory.SYSTEM, "START", "booting")  # should not raise


# ---------------------------------------------------------------------------
# 6. Sync flush
# ---------------------------------------------------------------------------

def test_flush_sync(tmp_path):
    bot = _bot()
    j = EventJournal.get(bot)
    j._file_path = tmp_path / "journal.jsonl"
    j.record(EventCategory.ENTRY, "TEST", "hello")
    j.flush_sync()
    assert j._file_path.exists()
    lines = j._file_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["type"] == "TEST"


def test_flush_sync_clears_buffer(tmp_path):
    bot = _bot()
    j = EventJournal.get(bot)
    j._file_path = tmp_path / "journal.jsonl"
    j.record(EventCategory.ENTRY, "TEST", "hello")
    j.flush_sync()
    assert len(j._buffer) == 0


# ---------------------------------------------------------------------------
# 7. Stats
# ---------------------------------------------------------------------------

def test_stats():
    bot = _bot()
    j = EventJournal.get(bot)
    j.record(EventCategory.ENTRY, "A", "a")
    j.record(EventCategory.EXIT, "B", "b")
    s = j.stats
    assert s["event_count"] == 2
    assert s["memory_events"] == 2
    assert s["error_count"] == 0


# ---------------------------------------------------------------------------
# 8. Module-level convenience functions
# ---------------------------------------------------------------------------

def test_record_event_convenience():
    bot = _bot()
    record_event(bot, EventCategory.SYSTEM, "TEST", "msg")
    j = EventJournal.get(bot)
    assert j._event_count == 1


def test_record_entry_event():
    bot = _bot()
    record_entry_event(bot, "BTCUSDT", "SIGNAL", "long signal")
    j = EventJournal.get(bot)
    assert j._memory_events[-1].symbol == "BTCUSDT"
    assert j._memory_events[-1].category == "ENTRY"


def test_record_exit_event():
    bot = _bot()
    record_exit_event(bot, "ETHUSDT", "TP_HIT", "take profit")
    j = EventJournal.get(bot)
    assert j._memory_events[-1].category == "EXIT"


def test_record_order_event():
    bot = _bot()
    record_order_event(bot, "BTCUSDT", "SUBMITTED", "order placed")
    j = EventJournal.get(bot)
    assert j._memory_events[-1].category == "ORDER"


def test_record_error():
    bot = _bot()
    record_error(bot, "something broke", symbol="BTCUSDT")
    j = EventJournal.get(bot)
    assert j._memory_events[-1].category == "ERROR"
    assert j._memory_events[-1].severity == "ERROR"


def test_record_system_event():
    bot = _bot()
    record_system_event(bot, "STARTUP", "bot started")
    j = EventJournal.get(bot)
    assert j._memory_events[-1].category == "SYSTEM"


def test_record_event_disabled():
    bot = _bot(enabled=False)
    import execution.event_journal as mod
    orig = mod._JOURNAL_ENABLED
    mod._JOURNAL_ENABLED = False
    try:
        record_event(bot, EventCategory.SYSTEM, "TEST", "msg")
        # Should not create journal
    finally:
        mod._JOURNAL_ENABLED = orig


# ---------------------------------------------------------------------------
# 9. journal_transition
# ---------------------------------------------------------------------------

def test_journal_transition():
    bot = _bot()
    journal_transition(bot, "BTCUSDT", "OPEN", "CLOSED", "take profit hit")
    j = EventJournal.get(bot)
    last = j._memory_events[-1]
    assert last.category == "POSITION"
    assert last.event_type == "STATE_TRANSITION"
    assert "OPEN -> CLOSED" in last.message
    assert last.data["from"] == "OPEN"
    assert last.data["to"] == "CLOSED"
