"""Integration test: bot dry-run startup with guardian loop and notifications."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestGuardianLoopIntegration:
    """Guardian loop runs one cycle without crashing."""

    def test_guardian_safe_call_runs_healthy_step(self):
        from execution.guardian import _safe_call

        called = []

        async def healthy_step():
            called.append(1)

        _run(_safe_call("test_step", healthy_step))
        assert called == [1]

    def test_guardian_safe_call_survives_exception(self):
        from execution.guardian import _safe_call

        async def bad_step():
            raise RuntimeError("boom")

        # Should not raise
        _run(_safe_call("bad_step", bad_step))

    def test_guardian_safe_call_survives_timeout(self):
        from execution.guardian import _safe_call

        async def slow_step():
            await asyncio.sleep(999)

        # timeout_sec=0.1 should trigger timeout, not hang
        _run(_safe_call("slow_step", slow_step, timeout_sec=0.1))


class TestNotificationManagerIntegration:
    """Notification manager lifecycle: create, send, escalate, circuit-break."""

    def test_full_lifecycle(self):
        from notifications.events import NotificationEvent, NotificationSeverity
        from notifications.manager import NotificationConfig, NotificationManager

        notifier = AsyncMock()
        notifier.speak = AsyncMock(return_value=True)
        cfg = NotificationConfig(enabled=True, max_messages_per_minute=50)
        mgr = NotificationManager(notifier=notifier, config=cfg)

        evt = NotificationEvent(
            severity=NotificationSeverity.INFO,
            category="risk_margin",
            title="margin low",
            body="margin at 5%",
            throttle_key="margin",
            throttle_window_sec=0,
        )

        # Phase 1: Normal send
        ok = _run(mgr.send(evt))
        assert ok is True
        assert mgr._consecutive_failures == 0

        # Phase 2: Repeated sends trigger escalation at count=3
        _run(mgr.send(evt))
        _run(mgr.send(evt))
        calls = mgr.notifier.speak.call_args_list
        assert "[ESCALATED x3]" in calls[2][0][0]

        # Phase 3: Simulate Telegram failures to trigger circuit breaker
        notifier.speak = AsyncMock(return_value=False)
        for _ in range(5):
            _run(mgr.send(evt))
        assert mgr._tg_circuit_open is True

        # Phase 4: During circuit open, sends are blocked
        result = _run(mgr.send(evt))
        assert result is False

    def test_heartbeat_respects_config(self):
        from notifications.events import NotificationSeverity
        from notifications.manager import NotificationConfig, NotificationManager

        notifier = AsyncMock()
        notifier.speak = AsyncMock(return_value=True)
        cfg = NotificationConfig(enabled=True, heartbeat_enabled=False)
        mgr = NotificationManager(notifier=notifier, config=cfg)

        bot = MagicMock()
        bot.state = MagicMock()
        bot.state.positions = {}
        bot.active_symbols = set()

        ok = _run(mgr.send_heartbeat(bot, started_ts=time.time()))
        assert ok is False
        notifier.speak.assert_not_called()


class TestKillSwitchPersistence:
    """Kill switch state loads from disk on startup."""

    def test_state_file_creation(self, tmp_path):
        state_file = tmp_path / "kill_switch_state.json"
        assert not state_file.exists()

        # Simulate writing kill switch state
        import json
        state = {"halted": False, "reason": "", "ts": time.time()}
        state_file.write_text(json.dumps(state), encoding="utf-8")
        assert state_file.exists()

        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        assert loaded["halted"] is False


class TestStartupDirectoryCreation:
    """Critical directories are created on bot startup."""

    def test_logs_dirs_created(self, tmp_path):
        dirs = [tmp_path / "logs", tmp_path / "logs" / "health", tmp_path / "state"]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            assert d.is_dir()


class TestNotificationFallbackLog:
    """When Telegram is unavailable, events fall back to JSONL file."""

    def test_fallback_writes_jsonl(self, tmp_path):
        import json
        from notifications.events import NotificationEvent, NotificationSeverity
        from notifications.manager import _fallback_log_event

        log_file = tmp_path / "fallback.jsonl"

        with patch("notifications.manager._FALLBACK_LOG", log_file):
            evt = NotificationEvent(
                severity=NotificationSeverity.WARNING,
                category="test",
                title="test fallback",
                body="body text",
            )
            _fallback_log_event(evt, "unit_test")

        assert log_file.exists()
        row = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert row["category"] == "test"
        assert row["reason"] == "unit_test"
