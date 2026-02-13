# utils/notify.py — SCALPER ETERNAL — NOTIFICATION SYSTEM — 2026 v1.0
# Unified notification system with Telegram support.
#
# Features:
# - Telegram bot notifications
# - Rate limiting to prevent spam
# - Priority-based message queuing
# - Retry with exponential backoff
# - Message batching for high-frequency events
#
# Design principles:
# - Non-blocking (fire and forget)
# - Graceful degradation on errors
# - Never block trading operations

from __future__ import annotations

import asyncio
import aiohttp
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from collections import deque

from utils.logging import log_core


# Configuration
_NOTIFY_ENABLED = os.getenv("NOTIFY_ENABLED", "1").lower() in ("1", "true", "yes", "on")
_TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_RATE_LIMIT_PER_MINUTE = int(os.getenv("NOTIFY_RATE_LIMIT_PER_MINUTE", "20"))
_BATCH_INTERVAL_SEC = float(os.getenv("NOTIFY_BATCH_INTERVAL_SEC", "5"))
_RETRY_MAX = int(os.getenv("NOTIFY_RETRY_MAX", "3"))
_RETRY_DELAY_SEC = float(os.getenv("NOTIFY_RETRY_DELAY_SEC", "2"))


class Priority(Enum):
    """Message priority levels."""
    LOW = 0       # Informational
    NORMAL = 1    # Standard notifications
    HIGH = 2      # Important events
    CRITICAL = 3  # Urgent alerts (always sent immediately)


@dataclass
class Message:
    """A notification message."""
    text: str
    priority: Priority
    timestamp: float = field(default_factory=time.time)
    symbol: Optional[str] = None
    category: str = "general"
    retry_count: int = 0


class RateLimiter:
    """Simple rate limiter for notifications."""

    def __init__(self, max_per_minute: int):
        self._max = max_per_minute
        self._timestamps: deque = deque()

    def can_send(self) -> bool:
        """Check if we can send a message."""
        now = time.time()
        # Remove old timestamps
        while self._timestamps and self._timestamps[0] < now - 60:
            self._timestamps.popleft()
        return len(self._timestamps) < self._max

    def record(self) -> None:
        """Record a sent message."""
        self._timestamps.append(time.time())

    @property
    def remaining(self) -> int:
        """Get remaining messages in current window."""
        now = time.time()
        while self._timestamps and self._timestamps[0] < now - 60:
            self._timestamps.popleft()
        return max(0, self._max - len(self._timestamps))


class TelegramSender:
    """Telegram message sender."""

    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram."""
        if not self._token or not self._chat_id:
            return False

        try:
            session = await self._get_session()
            url = f"{self._base_url}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    return True
                else:
                    body = await resp.text()
                    log_core.warning(f"[notify] Telegram error {resp.status}: {body[:200]}")
                    return False

        except asyncio.TimeoutError:
            log_core.warning("[notify] Telegram timeout")
            return False
        except Exception as e:
            log_core.warning(f"[notify] Telegram error: {e}")
            return False

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


class Notify:
    """
    Unified notification system.

    Usage:
        notify = Notify(bot)

        # Simple message
        await notify.send("Trade executed", priority="high")

        # With symbol
        await notify.send("Entry signal", symbol="BTCUSDT", priority="normal")

        # Speak method (alias)
        await notify.speak("Position closed", "critical")
    """

    _instance: Optional['Notify'] = None

    def __init__(self, bot=None):
        self.bot = bot
        self._telegram = TelegramSender(_TELEGRAM_TOKEN, _TELEGRAM_CHAT_ID)
        self._rate_limiter = RateLimiter(_RATE_LIMIT_PER_MINUTE)
        self._queue: List[Message] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._last_batch_ts = time.time()
        self._sent_count = 0
        self._dropped_count = 0
        self._error_count = 0

    @classmethod
    def get(cls, bot) -> 'Notify':
        """Get or create notify instance (singleton)."""
        if cls._instance is None:
            cls._instance = cls(bot)
        return cls._instance

    @classmethod
    def is_enabled(cls, bot=None) -> bool:
        """Check if notifications are enabled."""
        if not _NOTIFY_ENABLED:
            return False
        if not _TELEGRAM_TOKEN or not _TELEGRAM_CHAT_ID:
            return False
        return True

    def _priority_from_str(self, priority: str) -> Priority:
        """Convert string priority to enum."""
        mapping = {
            "low": Priority.LOW,
            "info": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "critical": Priority.CRITICAL,
            "urgent": Priority.CRITICAL,
        }
        return mapping.get(priority.lower(), Priority.NORMAL)

    async def send(
        self,
        text: str,
        priority: str = "normal",
        symbol: Optional[str] = None,
        category: str = "general",
    ) -> bool:
        """
        Send a notification.

        Args:
            text: Message text
            priority: low/normal/high/critical
            symbol: Trading symbol (for context)
            category: Message category

        Returns:
            True if sent or queued, False if dropped
        """
        if not self.is_enabled():
            return False

        prio = self._priority_from_str(priority)
        msg = Message(
            text=text,
            priority=prio,
            symbol=symbol,
            category=category,
        )

        # Critical messages sent immediately
        if prio == Priority.CRITICAL:
            return await self._send_immediate(msg)

        # Otherwise queue for batching
        self._queue.append(msg)
        self._ensure_batch_task()
        return True

    async def speak(self, message: str, priority: str = "normal") -> bool:
        """
        Alias for send() - compatibility with existing code.

        Args:
            message: Message text
            priority: Priority level
        """
        return await self.send(message, priority=priority)

    async def _send_immediate(self, msg: Message) -> bool:
        """Send a message immediately (for critical priority)."""
        if not self._rate_limiter.can_send():
            # Critical messages bypass rate limit but log warning
            log_core.warning("[notify] Rate limit exceeded for critical message")

        formatted = self._format_message(msg)
        success = await self._telegram.send(formatted)

        if success:
            self._rate_limiter.record()
            self._sent_count += 1
        else:
            self._error_count += 1
            # Retry for critical messages
            if msg.retry_count < _RETRY_MAX:
                msg.retry_count += 1
                await asyncio.sleep(_RETRY_DELAY_SEC * msg.retry_count)
                return await self._send_immediate(msg)

        return success

    def _ensure_batch_task(self) -> None:
        """Ensure batch processing task is running."""
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._batch_loop())

    async def _batch_loop(self) -> None:
        """Process queued messages in batches."""
        while True:
            await asyncio.sleep(_BATCH_INTERVAL_SEC)

            if not self._queue:
                continue

            # Take all queued messages
            messages = self._queue.copy()
            self._queue.clear()

            # Group by priority
            by_priority: Dict[Priority, List[Message]] = {}
            for msg in messages:
                if msg.priority not in by_priority:
                    by_priority[msg.priority] = []
                by_priority[msg.priority].append(msg)

            # Send highest priority first
            for prio in sorted(by_priority.keys(), key=lambda p: p.value, reverse=True):
                batch = by_priority[prio]

                if not self._rate_limiter.can_send():
                    self._dropped_count += len(batch)
                    log_core.warning(f"[notify] Dropped {len(batch)} messages (rate limit)")
                    continue

                # Combine messages into one
                if len(batch) == 1:
                    formatted = self._format_message(batch[0])
                else:
                    formatted = self._format_batch(batch)

                success = await self._telegram.send(formatted)
                if success:
                    self._rate_limiter.record()
                    self._sent_count += len(batch)
                else:
                    self._error_count += len(batch)

    def _format_message(self, msg: Message) -> str:
        """Format a single message."""
        parts = []

        # Priority emoji
        emoji_map = {
            Priority.LOW: "📝",
            Priority.NORMAL: "📊",
            Priority.HIGH: "⚠️",
            Priority.CRITICAL: "🚨",
        }
        parts.append(emoji_map.get(msg.priority, "📊"))

        # Symbol if present
        if msg.symbol:
            parts.append(f"<b>{msg.symbol}</b>")

        # Main text
        parts.append(msg.text)

        return " ".join(parts)

    def _format_batch(self, messages: List[Message]) -> str:
        """Format multiple messages into a batch."""
        lines = [f"📦 <b>Batch Update ({len(messages)} events)</b>", ""]

        for msg in messages[:10]:  # Limit to 10 messages
            line = self._format_message(msg)
            lines.append(f"• {line}")

        if len(messages) > 10:
            lines.append(f"... and {len(messages) - 10} more")

        return "\n".join(lines)

    async def close(self) -> None:
        """Close the notifier."""
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        await self._telegram.close()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            "sent": self._sent_count,
            "dropped": self._dropped_count,
            "errors": self._error_count,
            "queued": len(self._queue),
            "rate_limit_remaining": self._rate_limiter.remaining,
        }


# Module-level convenience functions
async def send_notification(bot, text: str, priority: str = "normal", **kwargs) -> bool:
    """Send a notification."""
    if not Notify.is_enabled(bot):
        return False
    return await Notify.get(bot).send(text, priority=priority, **kwargs)


async def notify_trade(bot, symbol: str, action: str, details: str = "") -> bool:
    """Send a trade notification."""
    text = f"{action}: {details}" if details else action
    return await send_notification(bot, text, symbol=symbol, priority="high", category="trade")


async def notify_error(bot, error: str, symbol: Optional[str] = None) -> bool:
    """Send an error notification."""
    return await send_notification(bot, f"ERROR: {error}", symbol=symbol, priority="critical", category="error")


async def notify_system(bot, message: str) -> bool:
    """Send a system notification."""
    return await send_notification(bot, message, priority="normal", category="system")
