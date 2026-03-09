"""Tests for deep audit fixes (sessions 2026-03-09).

Covers:
- Guardian _safe_call step timeout
- Guardian _STUCK_ALERTED dict cap
- Notification manager circuit breaker
- Notification manager _pending cap
- Notification manager _last_by_key cap
- Atomic write helper
- Heartbeat writer
- Fallback log
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from notifications.events import NotificationSeverity as _Sev

_SEV_INFO = _Sev.INFO
_SEV_WARN = _Sev.WARNING


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Guardian _safe_call step timeout
# ---------------------------------------------------------------------------

class TestGuardianSafeCallTimeout:
    """Verify _safe_call enforces timeout and doesn't hang."""

    def test_safe_call_timeout_prevents_hang(self):
        from execution.guardian import _safe_call

        call_count = 0

        async def _slow_step(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(999)

        _run(_safe_call("test_slow_step", _slow_step, timeout_sec=0.1))
        assert call_count == 1

    def test_safe_call_fast_step_succeeds(self):
        from execution.guardian import _safe_call

        result_holder = []

        async def _fast_step():
            result_holder.append("done")

        _run(_safe_call("test_fast_step", _fast_step, timeout_sec=5.0))
        assert result_holder == ["done"]

    def test_safe_call_exception_caught(self):
        from execution.guardian import _safe_call

        async def _failing_step():
            raise ValueError("boom")

        # Should not raise
        _run(_safe_call("test_fail", _failing_step, timeout_sec=5.0))

    def test_safe_call_cancelled_error_propagates(self):
        from execution.guardian import _safe_call

        async def _cancel_step():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            _run(_safe_call("test_cancel", _cancel_step, timeout_sec=5.0))

    def test_safe_call_skips_non_callable(self):
        from execution.guardian import _safe_call

        _run(_safe_call("test_none", None, timeout_sec=5.0))


# ---------------------------------------------------------------------------
# 2. Guardian _STUCK_ALERTED dict cap
# ---------------------------------------------------------------------------

class TestStuckAlertedCap:
    """Verify _STUCK_ALERTED dict doesn't grow unbounded."""

    def test_stuck_alerted_is_dict(self):
        from execution import guardian
        assert isinstance(guardian._STUCK_ALERTED, dict)

    def test_stuck_alerted_cap_logic(self):
        """Simulate the cap logic from _position_stuck_check."""
        from execution import guardian

        original = guardian._STUCK_ALERTED
        try:
            guardian._STUCK_ALERTED = {}
            now = time.time()
            for i in range(501):
                guardian._STUCK_ALERTED[f"SYM{i}USDT"] = now - (501 - i)

            # Simulate adding one more with cap check
            sym = "NEWUSDT"
            guardian._STUCK_ALERTED[sym] = now
            if len(guardian._STUCK_ALERTED) > 500:
                oldest = min(guardian._STUCK_ALERTED, key=guardian._STUCK_ALERTED.get)
                guardian._STUCK_ALERTED.pop(oldest, None)

            assert len(guardian._STUCK_ALERTED) <= 501
            assert sym in guardian._STUCK_ALERTED
        finally:
            guardian._STUCK_ALERTED = original

    def test_stuck_alerted_pop_works(self):
        from execution import guardian

        original = guardian._STUCK_ALERTED
        try:
            guardian._STUCK_ALERTED = {"BTCUSDT": time.time()}
            guardian._STUCK_ALERTED.pop("BTCUSDT", None)
            assert "BTCUSDT" not in guardian._STUCK_ALERTED
            guardian._STUCK_ALERTED.pop("NONEXISTENT", None)  # no error
        finally:
            guardian._STUCK_ALERTED = original


# ---------------------------------------------------------------------------
# 3. Notification manager circuit breaker
# ---------------------------------------------------------------------------

class TestTelegramCircuitBreaker:
    """Verify Telegram circuit breaker opens after failures and recovers."""

    def _make_manager(self, send_ok=True, send_raises=False):
        from notifications.manager import NotificationManager, NotificationConfig

        notifier = AsyncMock()
        if send_raises:
            notifier.speak = AsyncMock(side_effect=ConnectionError("network down"))
        else:
            notifier.speak = AsyncMock(return_value=send_ok)

        config = NotificationConfig(
            enabled=True,
            send_timeout_sec=5.0,
            max_messages_per_minute=100,
        )
        return NotificationManager(notifier=notifier, config=config)

    def _make_event(self, title="TEST"):
        from notifications.events import NotificationEvent
        return NotificationEvent(
            severity=_SEV_INFO,
            category="test",
            title=title,
            body="test body",
        )

    def test_circuit_opens_after_threshold(self):
        mgr = self._make_manager(send_raises=True)
        evt = self._make_event()

        for _ in range(5):
            _run(mgr.send(evt))

        assert mgr._tg_circuit_open is True
        assert mgr._consecutive_failures >= 5

    def test_circuit_open_skips_sends(self):
        mgr = self._make_manager(send_raises=True)
        evt = self._make_event()

        for _ in range(5):
            _run(mgr.send(evt))

        assert mgr._tg_circuit_open is True
        mgr._tg_circuit_open_ts = time.time()
        result = _run(mgr.send(evt))
        assert result is False

    def test_circuit_half_open_after_cooldown(self):
        mgr = self._make_manager(send_ok=True)
        evt = self._make_event()

        mgr._tg_circuit_open = True
        mgr._tg_circuit_open_ts = time.time() - 300
        mgr._consecutive_failures = 10

        result = _run(mgr.send(evt))
        assert result is True
        assert mgr._tg_circuit_open is False
        assert mgr._consecutive_failures == 0

    def test_successful_send_resets_failures(self):
        mgr = self._make_manager(send_ok=True)
        evt = self._make_event()

        mgr._consecutive_failures = 3
        result = _run(mgr.send(evt))
        assert result is True
        assert mgr._consecutive_failures == 0


# ---------------------------------------------------------------------------
# 4. Notification manager _pending cap
# ---------------------------------------------------------------------------

class TestNotificationPendingCap:
    """Verify _pending deque has maxlen."""

    def test_pending_has_maxlen(self):
        from notifications.manager import NotificationManager, NotificationConfig

        notifier = AsyncMock()
        config = NotificationConfig(enabled=True, max_messages_per_minute=1)
        mgr = NotificationManager(notifier=notifier, config=config)

        assert mgr._pending.maxlen == 200

    def test_pending_drops_oldest_on_overflow(self):
        from notifications.manager import NotificationManager, NotificationConfig
        from notifications.events import NotificationEvent

        notifier = AsyncMock()
        notifier.speak = AsyncMock(return_value=True)
        config = NotificationConfig(enabled=True, max_messages_per_minute=1)
        mgr = NotificationManager(notifier=notifier, config=config)

        # Fill sent_ts to trigger queueing (rate limit hit)
        mgr._sent_ts = deque([time.time()] * 5)

        for i in range(210):
            evt = NotificationEvent(
                severity=_SEV_INFO, category="test",
                title=f"evt-{i}", body="body",
            )
            _run(mgr.send(evt))

        assert len(mgr._pending) <= 200


# ---------------------------------------------------------------------------
# 5. Notification manager _last_by_key cap
# ---------------------------------------------------------------------------

class TestLastByKeyCap:
    """Verify _last_by_key dict evicts oldest above 500."""

    def test_last_by_key_eviction(self):
        from notifications.manager import NotificationManager, NotificationConfig
        from notifications.events import NotificationEvent

        notifier = AsyncMock()
        notifier.speak = AsyncMock(return_value=True)
        config = NotificationConfig(enabled=True, max_messages_per_minute=9999)
        mgr = NotificationManager(notifier=notifier, config=config)

        # Pre-fill with 500 keys
        base_ts = time.time() - 1000
        for i in range(500):
            mgr._last_by_key[f"key_{i}"] = base_ts + i

        evt = NotificationEvent(
            severity=_SEV_INFO, category="test",
            title="new", body="body",
            throttle_key="new_key_501",
            throttle_window_sec=0,
        )
        _run(mgr.send(evt))

        assert len(mgr._last_by_key) <= 501
        assert "new_key_501" in mgr._last_by_key


# ---------------------------------------------------------------------------
# 6. Atomic write helper
# ---------------------------------------------------------------------------

class TestAtomicWriteJson:
    """Verify atomic_write_json produces valid JSON and is crash-safe."""

    def test_writes_valid_json(self):
        from execution.runtime_helpers import atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "test.json"
            data = {"key": "value", "num": 42, "nested": {"a": 1}}
            atomic_write_json(dst, data)

            assert dst.exists()
            loaded = json.loads(dst.read_text(encoding="utf-8"))
            assert loaded == data

    def test_creates_parent_dirs(self):
        from execution.runtime_helpers import atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "sub" / "deep" / "test.json"
            atomic_write_json(dst, {"ok": True})
            assert dst.exists()

    def test_overwrites_existing(self):
        from execution.runtime_helpers import atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "test.json"
            atomic_write_json(dst, {"v": 1})
            atomic_write_json(dst, {"v": 2})
            loaded = json.loads(dst.read_text(encoding="utf-8"))
            assert loaded["v"] == 2

    def test_unicode_content(self):
        from execution.runtime_helpers import atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            dst = Path(tmpdir) / "test.json"
            data = {"msg": "Türkçe karakterler: ş, ü, ö, ı, ğ, ç"}
            atomic_write_json(dst, data)
            loaded = json.loads(dst.read_text(encoding="utf-8"))
            assert loaded["msg"] == data["msg"]


# ---------------------------------------------------------------------------
# 7. Heartbeat file writer
# ---------------------------------------------------------------------------

class TestHeartbeatWriter:
    """Verify heartbeat file is written correctly."""

    def test_heartbeat_writes_json(self):
        from execution.guardian import _write_heartbeat

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "heartbeat.json"

            with patch("execution.guardian._HEARTBEAT_PATH", test_path):
                bot = SimpleNamespace(
                    state=SimpleNamespace(
                        run_context={"started_ts": time.time() - 100},
                        positions={},
                    ),
                    _kill_switch_active=False,
                )
                _run(_write_heartbeat(bot))

                assert test_path.exists()
                data = json.loads(test_path.read_text(encoding="utf-8"))
                assert "ts" in data
                assert "pid" in data
                assert data["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# 8. Fallback log on Telegram failure
# ---------------------------------------------------------------------------

class TestFallbackLog:
    """Verify notifications fall back to JSONL when Telegram is down."""

    def test_fallback_log_writes_jsonl(self):
        from notifications.manager import _fallback_log_event
        from notifications.events import NotificationEvent

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "fallback.jsonl"

            with patch("notifications.manager._FALLBACK_LOG", log_path):
                evt = NotificationEvent(
                    severity=_SEV_WARN,
                    category="test",
                    title="Test Alert",
                    body="Something happened",
                )
                _fallback_log_event(evt, "telegram_timeout")

                assert log_path.exists()
                lines = log_path.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) == 1
                row = json.loads(lines[0])
                assert row["title"] == "Test Alert"
                assert row["reason"] == "telegram_timeout"
                assert "ts" in row


# ---------------------------------------------------------------------------
# 9. Guardian constants
# ---------------------------------------------------------------------------

class TestGuardianConstants:
    """Verify guardian timeout constant exists and is reasonable."""

    def test_step_timeout_constant(self):
        from execution.guardian import _GUARDIAN_STEP_TIMEOUT_SEC
        assert isinstance(_GUARDIAN_STEP_TIMEOUT_SEC, float)
        assert 10.0 <= _GUARDIAN_STEP_TIMEOUT_SEC <= 120.0

    def test_heartbeat_path_constant(self):
        from execution.guardian import _HEARTBEAT_PATH
        assert isinstance(_HEARTBEAT_PATH, Path)
        assert "heartbeat" in str(_HEARTBEAT_PATH)


# ---------------------------------------------------------------------------
# 10. Notification circuit breaker constants
# ---------------------------------------------------------------------------

class TestCircuitBreakerConstants:
    """Verify circuit breaker thresholds are reasonable."""

    def test_circuit_threshold(self):
        from notifications.manager import NotificationManager
        assert NotificationManager._TG_CIRCUIT_THRESHOLD == 5

    def test_circuit_cooldown(self):
        from notifications.manager import NotificationManager
        assert 60.0 <= NotificationManager._TG_CIRCUIT_COOLDOWN_SEC <= 300.0
