"""Comprehensive unit tests for the notifications/ package.

Covers: manager, telegram, events, health_alerts, daily_summary,
        trade_alerts, risk_alerts.

All external calls (Telegram, X/Twitter, DB) are mocked.
Target: 25-30 tests.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from notifications.events import NotificationEvent, NotificationSeverity
from notifications.manager import (
    NotificationConfig,
    NotificationManager,
    _fallback_log_event,
    load_config_from_env,
)
from notifications.health_alerts import (
    build_crash_event,
    build_data_stale_event,
    build_heartbeat_event,
    build_startup_event,
)
from notifications.trade_alerts import (
    build_entry_event,
    build_exit_event,
    build_scratch_event,
)
from notifications.risk_alerts import (
    build_drawdown_event,
    build_entry_blocked_event,
    build_regime_change_event,
    build_scratch_pause_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw: Any) -> NotificationConfig:
    defaults = dict(
        enabled=True,
        trade_alerts=True,
        risk_alerts=True,
        daily_summary=True,
        heartbeat_enabled=True,
        heartbeat_interval_sec=14400.0,
        max_messages_per_minute=60,
        silent_heartbeat=False,
        send_timeout_sec=2.0,
    )
    defaults.update(kw)
    return NotificationConfig(**defaults)


def _mock_notifier() -> Mock:
    n = Mock()
    n.speak = AsyncMock(return_value=True)
    return n


def _evt(
    sev: NotificationSeverity = NotificationSeverity.INFO,
    cat: str = "test",
    title: str = "T",
    body: str = "B",
    throttle_key: str | None = None,
    throttle_window_sec: float = 0.0,
    silent: bool = False,
) -> NotificationEvent:
    return NotificationEvent(
        severity=sev,
        category=cat,
        title=title,
        body=body,
        throttle_key=throttle_key,
        throttle_window_sec=throttle_window_sec,
        silent=silent,
    )


# ===================================================================
# 1. NotificationEvent (events.py)
# ===================================================================

class TestNotificationEvent:
    def test_render_title_only(self):
        e = _evt(body="")
        rendered = e.render()
        assert "T" in rendered

    def test_render_title_and_body(self):
        e = _evt(title="HEAD", body="details here")
        rendered = e.render()
        assert "HEAD" in rendered
        assert "details here" in rendered
        assert "\n" in rendered

    def test_severity_values(self):
        assert NotificationSeverity.INFO.value == "\u2139"
        assert NotificationSeverity.CRITICAL.value == "\U0001f6a8"
        assert NotificationSeverity.WARNING.value == "\u26a0"
        assert NotificationSeverity.SUCCESS.value == "\u2705"
        assert NotificationSeverity.DAILY.value == "\U0001f4ca"

    def test_throttle_defaults(self):
        e = _evt()
        assert e.throttle_key is None
        assert e.throttle_window_sec == 0.0
        assert e.silent is False

    def test_frozen_dataclass(self):
        e = _evt()
        with pytest.raises(AttributeError):
            e.title = "new"  # type: ignore[misc]


# ===================================================================
# 2. Manager — throttle / dedup
# ===================================================================

def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestManagerThrottle:
    def test_throttle_blocks_duplicate_within_window(self):
        mgr = NotificationManager(_mock_notifier(), _cfg())
        e = _evt(throttle_key="k1", throttle_window_sec=600.0)
        assert _run(mgr.send(e)) is True
        assert _run(mgr.send(e)) is False

    def test_throttle_allows_after_window(self):
        mgr = NotificationManager(_mock_notifier(), _cfg())
        e = _evt(throttle_key="k2", throttle_window_sec=1.0)
        assert _run(mgr.send(e)) is True
        mgr._last_by_key["k2"] = time.time() - 2.0
        assert _run(mgr.send(e)) is True

    def test_no_throttle_key_sends_always(self):
        mgr = NotificationManager(_mock_notifier(), _cfg())
        e = _evt()
        assert _run(mgr.send(e)) is True
        assert _run(mgr.send(e)) is True


# ===================================================================
# 3. Manager — rate limiting
# ===================================================================

class TestManagerRateLimit:
    def test_rate_limit_queues_excess(self):
        mgr = NotificationManager(_mock_notifier(), _cfg(max_messages_per_minute=2))
        for _ in range(2):
            assert _run(mgr.send(_evt())) is True
        assert _run(mgr.send(_evt())) is False

    def test_disabled_manager_returns_false(self):
        mgr = NotificationManager(_mock_notifier(), _cfg(enabled=False))
        assert _run(mgr.send(_evt())) is False

    def test_trade_alerts_disabled(self):
        mgr = NotificationManager(_mock_notifier(), _cfg(trade_alerts=False))
        e = _evt(cat="trade_entry")
        assert _run(mgr.send(e)) is False

    def test_risk_alerts_disabled(self):
        mgr = NotificationManager(_mock_notifier(), _cfg(risk_alerts=False))
        e = _evt(cat="risk_block")
        assert _run(mgr.send(e)) is False


# ===================================================================
# 4. Manager — escalation
# ===================================================================

class TestManagerEscalation:
    def test_escalation_after_threshold(self):
        mgr = NotificationManager(_mock_notifier(), _cfg())
        e = _evt(sev=NotificationSeverity.INFO, cat="test_esc", throttle_key="esc1")
        for _ in range(3):
            _run(mgr.send(e))
        esc_key = f"{e.category}:{e.throttle_key}"
        assert mgr._alert_repeat_count[esc_key] >= 3

    def test_escalation_warning_to_critical(self):
        mgr = NotificationManager(_mock_notifier(), _cfg())
        e = _evt(sev=NotificationSeverity.WARNING, cat="esc_w")
        for _ in range(3):
            result = mgr._maybe_escalate(e)
        # After 3 calls the result should be CRITICAL
        assert result.severity == NotificationSeverity.CRITICAL
        assert "[ESCALATED" in result.title

    def test_escalation_memory_cap(self):
        mgr = NotificationManager(_mock_notifier(), _cfg())
        # Fill beyond _ESCALATION_MAX_TRACKED
        for i in range(mgr._ESCALATION_MAX_TRACKED + 10):
            mgr._alert_repeat_count[f"key_{i}"] = 1
        before = len(mgr._alert_repeat_count)
        # Trigger escalation which prunes oldest when over cap
        mgr._maybe_escalate(_evt(cat="new_cat", title="new_title"))
        # After pruning, size should be <= before (at least one evicted)
        assert len(mgr._alert_repeat_count) <= before


# ===================================================================
# 5. Manager — circuit breaker
# ===================================================================

class TestManagerCircuitBreaker:
    def test_circuit_opens_after_threshold_failures(self):
        notifier = _mock_notifier()
        notifier.speak = AsyncMock(return_value=False)
        mgr = NotificationManager(notifier, _cfg())
        for _ in range(5):
            _run(mgr.send(_evt()))
        assert mgr._tg_circuit_open is True
        assert mgr._consecutive_failures >= 5

    def test_circuit_open_blocks_sends(self):
        notifier = _mock_notifier()
        notifier.speak = AsyncMock(return_value=False)
        mgr = NotificationManager(notifier, _cfg())
        mgr._tg_circuit_open = True
        mgr._tg_circuit_open_ts = time.time()
        result = _run(mgr.send(_evt()))
        assert result is False

    def test_circuit_half_open_after_cooldown(self):
        notifier = _mock_notifier()
        notifier.speak = AsyncMock(return_value=True)
        mgr = NotificationManager(notifier, _cfg())
        mgr._tg_circuit_open = True
        mgr._tg_circuit_open_ts = time.time() - 200.0
        result = _run(mgr.send(_evt()))
        assert result is True
        assert mgr._tg_circuit_open is False


# ===================================================================
# 6. Manager — fallback to JSONL
# ===================================================================

class TestManagerFallback:
    def test_fallback_on_no_notifier(self, tmp_path):
        log_file = tmp_path / "fallback.jsonl"
        with patch("notifications.manager._FALLBACK_LOG", log_file):
            mgr = NotificationManager(None, _cfg())  # type: ignore[arg-type]
            _run(mgr.send(_evt(title="FALLBACK_TEST")))
        assert log_file.exists()
        row = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert row["title"] == "FALLBACK_TEST"
        assert row["reason"] == "no_notifier"

    def test_fallback_log_event_writes_jsonl(self, tmp_path):
        log_file = tmp_path / "fb.jsonl"
        with patch("notifications.manager._FALLBACK_LOG", log_file):
            _fallback_log_event(_evt(title="FB"), "test_reason")
        assert log_file.exists()
        row = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert row["reason"] == "test_reason"

    def test_fallback_on_telegram_exception(self, tmp_path):
        log_file = tmp_path / "exc_fb.jsonl"
        notifier = _mock_notifier()
        notifier.speak = AsyncMock(side_effect=RuntimeError("boom"))
        mgr = NotificationManager(notifier, _cfg())
        with patch("notifications.manager._FALLBACK_LOG", log_file):
            result = _run(mgr.send(_evt(title="BOOM")))
        assert result is False
        assert log_file.exists()


# ===================================================================
# 7. Telegram (telegram.py) — HTML escape / failures
# ===================================================================

class TestTelegram:
    def test_html_escape_special_chars(self):
        with patch("notifications.telegram.Bot") as MockBot, \
             patch("notifications.telegram.XTweetPublisher"):
            bot_inst = MockBot.return_value
            bot_inst.send_message = AsyncMock(return_value=True)
            from notifications.telegram import Notifier
            n = Notifier(token="fake", chat_id="123")
            n.bot = bot_inst
            n._x = Mock()
            _run(n.speak("<script>alert('xss')&</script>", priority="critical"))
            call_args = bot_inst.send_message.call_args
            text_sent = call_args.kwargs.get("text", call_args[1].get("text", ""))
            assert "&lt;" in text_sent
            assert "&gt;" in text_sent
            assert "&amp;" in text_sent

    def test_send_failure_returns_false(self):
        with patch("notifications.telegram.Bot") as MockBot, \
             patch("notifications.telegram.XTweetPublisher"):
            bot_inst = MockBot.return_value
            bot_inst.send_message = AsyncMock(side_effect=Exception("network"))
            from notifications.telegram import Notifier
            n = Notifier(token="fake", chat_id="123")
            n.bot = bot_inst
            n._x = Mock()
            result = _run(n.speak("hello"))
            assert result is False

    def test_no_bot_returns_false(self):
        with patch("notifications.telegram.XTweetPublisher"):
            from notifications.telegram import Notifier
            n = Notifier(token=None, chat_id="123")
            result = _run(n.speak("hello"))
            assert result is False


# ===================================================================
# 8. Health Alerts (health_alerts.py)
# ===================================================================

class TestHealthAlerts:
    def test_startup_event(self):
        e = build_startup_event(
            symbols="BTCUSDT,ETHUSDT",
            horizon_sec=300,
            scratch_enabled=True,
            watchdog_active=True,
        )
        assert e.severity == NotificationSeverity.SUCCESS
        assert e.category == "startup"
        assert "BTCUSDT" in e.body
        assert "ENABLED" in e.body
        assert "ACTIVE" in e.body

    def test_heartbeat_event(self):
        bot = SimpleNamespace(
            state=SimpleNamespace(total_trades=10, total_wins=7, daily_pnl_bps=5.5),
            _notify_last_regime="UP",
            _notify_last_rolling_return=0.002,
        )
        e = build_heartbeat_event(bot, started_ts=time.time() - 3600)
        assert e.severity == NotificationSeverity.INFO
        assert e.category == "heartbeat"
        assert "HEARTBEAT" in e.body
        assert e.silent is True

    def test_crash_event(self):
        e = build_crash_event("KeyError: foo", uptime_sec=7200.0)
        assert e.severity == NotificationSeverity.CRITICAL
        assert "KeyError" in e.body

    def test_data_stale_event(self):
        e = build_data_stale_event(stale_sec=120, last_row_utc="2026-03-10 00:00", symbols="BTCUSDT")
        assert e.severity == NotificationSeverity.CRITICAL
        assert "120" in e.body


# ===================================================================
# 9. Daily Summary (daily_summary.py)
# ===================================================================

class TestDailySummary:
    @patch("notifications.daily_summary.analyze_collection_health", return_value={"gap_count": 2, "uptime_pct": 99.5})
    @patch("notifications.daily_summary.generate_summary")
    def test_compose_daily_summary(self, mock_summary, mock_health):
        mock_summary.return_value = {
            "total_trades": 10,
            "win_rate": 0.6,
            "total_pnl_bps": 12.5,
            "scratch_rate": 0.1,
            "max_drawdown_bps": 3.0,
            "regime_up_pct": 0.7,
            "regime_down_pct": 0.3,
        }
        with patch("notifications.daily_summary.sqlite3") as mock_sql:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = [500]
            mock_sql.connect.return_value = mock_conn
            from notifications.daily_summary import compose_daily_summary
            e = compose_daily_summary(trades_db="fake.db", micro_db="fake_micro.db")
        assert e.severity == NotificationSeverity.DAILY
        assert e.category == "daily_summary"
        assert "DAILY SUMMARY" in e.body
        assert "12.50" in e.body or "12.5" in e.body


# ===================================================================
# 10. Trade Alerts (trade_alerts.py)
# ===================================================================

class TestTradeAlerts:
    def test_entry_event(self):
        e = build_entry_event(
            symbol="BTCUSDT", side="long", entry_price=65000.0,
            regime="UP", regime_age_sec=120.0, rolling_return=0.005,
        )
        assert e.severity == NotificationSeverity.SUCCESS
        assert e.category == "trade_entry"
        assert "LONG" in e.body
        assert "BTCUSDT" in e.body

    def test_exit_positive_pnl(self):
        e = build_exit_event(
            symbol="ETHUSDT", side="short", pnl_bps=5.0,
            exit_type="tp", hold_sec=30.0, peak_bps=6.0, adverse_bps=1.0,
        )
        assert e.severity == NotificationSeverity.SUCCESS

    def test_exit_negative_pnl(self):
        e = build_exit_event(
            symbol="ETHUSDT", side="short", pnl_bps=-3.0,
            exit_type="sl", hold_sec=15.0, peak_bps=1.0, adverse_bps=4.0,
        )
        assert e.severity == NotificationSeverity.WARNING

    def test_scratch_event(self):
        e = build_scratch_event(
            symbol="BTCUSDT", side="long", pnl_bps=-0.5,
            adverse_limit_bps=2.0, elapsed_sec=10.0, scratches=3,
        )
        assert e.severity == NotificationSeverity.WARNING
        assert "SCRATCH" in e.body


# ===================================================================
# 11. Risk Alerts (risk_alerts.py)
# ===================================================================

class TestRiskAlerts:
    def test_entry_blocked(self):
        e = build_entry_blocked_event("daily_loss_limit", risk_state={"daily_pnl_bps": -40.0, "config": {"max_daily_loss_bps": 50.0}})
        assert e.severity == NotificationSeverity.WARNING
        assert e.throttle_key == "risk_block:daily_loss_limit"
        assert e.throttle_window_sec == 600.0

    def test_drawdown_event(self):
        e = build_drawdown_event(max_drawdown_bps=100.0, peak_pnl_bps=20.0, current_pnl_bps=-80.0)
        assert e.severity == NotificationSeverity.CRITICAL
        assert e.throttle_key == "drawdown_limit"

    def test_regime_change_event(self):
        e = build_regime_change_event("UP", "DOWN", cooldown_sec=60.0, open_positions=1)
        assert e.severity == NotificationSeverity.INFO
        assert "UP" in e.body and "DOWN" in e.body

    def test_scratch_pause_event(self):
        e = build_scratch_pause_event(consecutive=5, pause_sec=120.0, last_trades_bps=[-0.5, -0.3, -0.8])
        assert e.severity == NotificationSeverity.CRITICAL
        assert "5" in e.body
