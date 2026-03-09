"""Tests for graceful degradation mode: exchange down → entries blocked, exits allowed."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from execution.guardian import (
    _exchange_connectivity_tick,
    _set_exchange_degraded,
    is_exchange_degraded,
    _DEGRADED_THRESHOLD,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_bot(health_check_ok=True):
    bot = MagicMock()
    bot.state = MagicMock()
    bot.state.run_context = {}
    bot.notify = AsyncMock()
    bot.notify.speak = AsyncMock(return_value=True)
    ex = MagicMock()
    ex.last_health_check = 0.0  # force probe
    if health_check_ok:
        ex._health_check = AsyncMock()
    else:
        ex._health_check = AsyncMock(side_effect=ConnectionError("exchange down"))
    bot.ex = ex
    return bot


class TestDegradedModeFlag:
    def test_not_degraded_by_default(self):
        bot = _make_bot()
        assert is_exchange_degraded(bot) is False

    def test_set_degraded_true(self):
        bot = _make_bot()
        _set_exchange_degraded(bot, True)
        assert is_exchange_degraded(bot) is True
        assert bot.state.run_context["exchange_degraded_ts"] > 0

    def test_set_degraded_false(self):
        bot = _make_bot()
        _set_exchange_degraded(bot, True)
        _set_exchange_degraded(bot, False)
        assert is_exchange_degraded(bot) is False
        assert bot.state.run_context["exchange_degraded_ts"] == 0.0

    def test_is_degraded_with_missing_state(self):
        bot = MagicMock()
        bot.state = None
        assert is_exchange_degraded(bot) is False


class TestExchangeConnectivityDegradation:
    def setup_method(self):
        # Reset global state before each test
        import execution.guardian as g
        g._DEGRADED_CONSECUTIVE_FAILURES = 0
        g._DEGRADED_NOTIFIED = False

    def test_successful_probe_no_degradation(self):
        bot = _make_bot(health_check_ok=True)
        _run(_exchange_connectivity_tick(bot))
        assert is_exchange_degraded(bot) is False

    def test_single_failure_no_degradation(self):
        bot = _make_bot(health_check_ok=False)
        _run(_exchange_connectivity_tick(bot))
        assert is_exchange_degraded(bot) is False

    def test_threshold_failures_trigger_degradation(self):
        bot = _make_bot(health_check_ok=False)
        for _ in range(_DEGRADED_THRESHOLD):
            bot.ex.last_health_check = 0.0  # force probe each time
            _run(_exchange_connectivity_tick(bot))
        assert is_exchange_degraded(bot) is True

    def test_recovery_after_degradation(self):
        bot = _make_bot(health_check_ok=False)
        # Trigger degradation
        for _ in range(_DEGRADED_THRESHOLD):
            bot.ex.last_health_check = 0.0
            _run(_exchange_connectivity_tick(bot))
        assert is_exchange_degraded(bot) is True

        # Now fix the exchange
        bot.ex._health_check = AsyncMock()
        bot.ex.last_health_check = 0.0
        _run(_exchange_connectivity_tick(bot))
        assert is_exchange_degraded(bot) is False

    def test_recovery_sends_notification(self):
        bot = _make_bot(health_check_ok=False)
        for _ in range(_DEGRADED_THRESHOLD):
            bot.ex.last_health_check = 0.0
            _run(_exchange_connectivity_tick(bot))

        # Recover
        bot.ex._health_check = AsyncMock()
        bot.ex.last_health_check = 0.0
        bot.notify.speak.reset_mock()
        _run(_exchange_connectivity_tick(bot))
        # Should have notified about recovery
        bot.notify.speak.assert_called()
        call_text = bot.notify.speak.call_args[0][0]
        assert "RESTORED" in call_text

    def test_degradation_notifies_only_once(self):
        bot = _make_bot(health_check_ok=False)
        for _ in range(_DEGRADED_THRESHOLD + 3):
            bot.ex.last_health_check = 0.0
            _run(_exchange_connectivity_tick(bot))
        # notify.speak should have been called only once for degradation
        speak_calls = [c for c in bot.notify.speak.call_args_list if "DEGRADED" in str(c)]
        assert len(speak_calls) == 1
