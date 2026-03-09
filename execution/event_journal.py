# execution/event_journal.py — SCALPER ETERNAL — EVENT JOURNAL — 2026 v1.0
# Records all significant events for audit, debugging, and replay.
#
# Features:
# - Append-only event log (JSONL format)
# - Event categories: entry, exit, order, position, system, error
# - Automatic rotation by size/time
# - Query interface for recent events
#
# Design principles:
# - Never block trading operations
# - Best-effort persistence (fire and forget)
# - Human-readable format

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from pathlib import Path

from utils.logging import log_core


# Configuration
_JOURNAL_ENABLED = os.getenv("EVENT_JOURNAL_ENABLED", "1").lower() in ("1", "true", "yes", "on")
_JOURNAL_PATH = os.getenv("EVENT_JOURNAL_PATH", "logs/event_journal.jsonl")
_JOURNAL_MAX_SIZE_MB = float(os.getenv("EVENT_JOURNAL_MAX_SIZE_MB", "50"))
_JOURNAL_ROTATE_COUNT = int(os.getenv("EVENT_JOURNAL_ROTATE_COUNT", "5"))
_JOURNAL_BUFFER_SIZE = int(os.getenv("EVENT_JOURNAL_BUFFER_SIZE", "100"))
_JOURNAL_FLUSH_INTERVAL_SEC = float(os.getenv("EVENT_JOURNAL_FLUSH_INTERVAL_SEC", "5"))


class EventCategory(Enum):
    """Event categories."""
    ENTRY = "ENTRY"           # Entry signals and orders
    EXIT = "EXIT"             # Exit signals and orders
    ORDER = "ORDER"           # Order lifecycle events
    POSITION = "POSITION"     # Position changes
    PROTECTION = "PROTECTION" # Stop/TP events
    RECONCILE = "RECONCILE"   # Reconciliation events
    SYSTEM = "SYSTEM"         # System events (startup, shutdown)
    RISK = "RISK"             # Risk management events
    ERROR = "ERROR"           # Errors and exceptions
    DEBUG = "DEBUG"           # Debug information


class EventSeverity(Enum):
    """Event severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Event:
    """A journaled event."""
    timestamp: float
    category: str
    event_type: str
    severity: str
    symbol: Optional[str]
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.timestamp,
            "time": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "cat": self.category,
            "type": self.event_type,
            "sev": self.severity,
            "sym": self.symbol,
            "msg": self.message,
            "data": self.data,
            "corr": self.correlation_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class EventJournal:
    """
    Append-only event journal for audit and debugging.

    Usage:
        journal = EventJournal.get(bot)

        # Record events
        journal.record(
            category=EventCategory.ENTRY,
            event_type="SIGNAL_GENERATED",
            symbol="BTCUSDT",
            message="Long signal with confidence 0.85",
            data={"confidence": 0.85, "side": "long"}
        )

        # Query recent events
        events = journal.get_recent(symbol="BTCUSDT", limit=50)
    """

    _instance: Optional['EventJournal'] = None

    def __init__(self, bot):
        self.bot = bot
        self._buffer: List[Event] = []
        self._buffer_lock = asyncio.Lock()
        self._file_path = Path(_JOURNAL_PATH)
        self._last_flush_ts = time.time()
        self._event_count = 0
        self._error_count = 0
        self._hooks: List[Callable[[Event], None]] = []
        self._memory_events: List[Event] = []  # In-memory for queries
        self._memory_max = 1000

        # Ensure directory exists
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_core.warning(f"[event_journal] Cannot create journal directory: {e}")

    @classmethod
    def get(cls, bot) -> 'EventJournal':
        """Get or create event journal (singleton)."""
        if cls._instance is None:
            cls._instance = cls(bot)
        return cls._instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if event journal is enabled."""
        if not _JOURNAL_ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "EVENT_JOURNAL_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _JOURNAL_ENABLED

    def record(
        self,
        category: EventCategory,
        event_type: str,
        message: str,
        symbol: Optional[str] = None,
        severity: EventSeverity = EventSeverity.INFO,
        data: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Record an event (non-blocking).

        Args:
            category: Event category
            event_type: Specific event type (e.g., "ORDER_SUBMITTED")
            message: Human-readable message
            symbol: Trading symbol (if applicable)
            severity: Event severity
            data: Additional structured data
            correlation_id: ID to correlate related events
        """
        event = Event(
            timestamp=time.time(),
            category=category.value,
            event_type=event_type,
            severity=severity.value,
            symbol=symbol,
            message=message,
            data=data or {},
            correlation_id=correlation_id,
        )

        self._buffer.append(event)
        self._event_count += 1

        # Keep in memory for queries
        self._memory_events.append(event)
        if len(self._memory_events) > self._memory_max:
            self._memory_events = self._memory_events[-self._memory_max:]

        # Call hooks
        for hook in self._hooks:
            try:
                hook(event)
            except Exception:
                pass

        # Auto-flush if buffer full or interval exceeded
        if len(self._buffer) >= _JOURNAL_BUFFER_SIZE:
            asyncio.create_task(self._flush_async())
        elif time.time() - self._last_flush_ts > _JOURNAL_FLUSH_INTERVAL_SEC:
            asyncio.create_task(self._flush_async())

    async def _flush_async(self) -> None:
        """Flush buffer to disk asynchronously."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            events_to_write = self._buffer.copy()
            self._buffer.clear()
            self._last_flush_ts = time.time()

        try:
            # Check rotation
            await self._maybe_rotate()

            # Write events
            with open(self._file_path, "a", encoding="utf-8") as f:
                for event in events_to_write:
                    f.write(event.to_json() + "\n")

        except Exception as e:
            self._error_count += 1
            if self._error_count <= 3:
                log_core.error(f"[event_journal] flush failed: {e}")

    async def _maybe_rotate(self) -> None:
        """Rotate journal if size exceeds limit."""
        try:
            if not self._file_path.exists():
                return

            size_mb = self._file_path.stat().st_size / (1024 * 1024)
            if size_mb < _JOURNAL_MAX_SIZE_MB:
                return

            # Rotate files
            for i in range(_JOURNAL_ROTATE_COUNT - 1, 0, -1):
                old_path = self._file_path.with_suffix(f".{i}.jsonl")
                new_path = self._file_path.with_suffix(f".{i+1}.jsonl")
                if old_path.exists():
                    old_path.rename(new_path)

            # Rotate current file
            self._file_path.rename(self._file_path.with_suffix(".1.jsonl"))

            log_core.info(f"[event_journal] rotated journal (was {size_mb:.1f}MB)")

        except Exception as e:
            log_core.warning(f"[event_journal] rotation failed: {e}")

    def flush_sync(self) -> None:
        """Synchronous flush (for shutdown)."""
        if not self._buffer:
            return

        try:
            with open(self._file_path, "a", encoding="utf-8") as f:
                for event in self._buffer:
                    f.write(event.to_json() + "\n")
            self._buffer.clear()
        except Exception as e:
            log_core.error(f"[event_journal] sync flush failed: {e}")

    def get_recent(
        self,
        symbol: Optional[str] = None,
        category: Optional[EventCategory] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get recent events from memory."""
        events = self._memory_events

        if symbol:
            events = [e for e in events if e.symbol == symbol]

        if category:
            events = [e for e in events if e.category == category.value]

        return events[-limit:]

    def get_by_correlation(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID."""
        return [e for e in self._memory_events if e.correlation_id == correlation_id]

    def register_hook(self, hook: Callable[[Event], None]) -> None:
        """Register a hook called on every event."""
        self._hooks.append(hook)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get journal statistics."""
        return {
            "event_count": self._event_count,
            "error_count": self._error_count,
            "buffer_size": len(self._buffer),
            "memory_events": len(self._memory_events),
            "file_path": str(self._file_path),
        }


# Module-level convenience functions
def record_event(
    bot,
    category: EventCategory,
    event_type: str,
    message: str,
    **kwargs
) -> None:
    """Record an event."""
    if not EventJournal.is_enabled(bot):
        return
    EventJournal.get(bot).record(category, event_type, message, **kwargs)


def record_entry_event(bot, symbol: str, event_type: str, message: str, **kwargs) -> None:
    """Record an entry-related event."""
    record_event(bot, EventCategory.ENTRY, event_type, message, symbol=symbol, **kwargs)


def record_exit_event(bot, symbol: str, event_type: str, message: str, **kwargs) -> None:
    """Record an exit-related event."""
    record_event(bot, EventCategory.EXIT, event_type, message, symbol=symbol, **kwargs)


def record_order_event(bot, symbol: str, event_type: str, message: str, **kwargs) -> None:
    """Record an order-related event."""
    record_event(bot, EventCategory.ORDER, event_type, message, symbol=symbol, **kwargs)


def record_position_event(bot, symbol: str, event_type: str, message: str, **kwargs) -> None:
    """Record a position-related event."""
    record_event(bot, EventCategory.POSITION, event_type, message, symbol=symbol, **kwargs)


def record_error(bot, message: str, symbol: Optional[str] = None, **kwargs) -> None:
    """Record an error event."""
    record_event(
        bot, EventCategory.ERROR, "ERROR", message,
        symbol=symbol, severity=EventSeverity.ERROR, **kwargs
    )


def record_system_event(bot, event_type: str, message: str, **kwargs) -> None:
    """Record a system event."""
    record_event(bot, EventCategory.SYSTEM, event_type, message, **kwargs)


def journal_transition(bot, symbol: str, from_state: str, to_state: str, reason: str = "") -> None:
    """
    Journal a state transition (compatibility function for reconcile.py).

    This bridges to the state_machine module.
    """
    record_event(
        bot,
        EventCategory.POSITION,
        "STATE_TRANSITION",
        f"{from_state} -> {to_state}: {reason}",
        symbol=symbol,
        data={"from": from_state, "to": to_state, "reason": reason}
    )

    # Also call state machine if available
    try:
        from execution.state_machine import journal_transition as sm_journal
        sm_journal(bot, symbol, from_state, to_state, reason)
    except ImportError:
        pass
