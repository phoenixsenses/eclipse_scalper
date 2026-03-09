"""Tests for alert escalation: repeated alerts auto-escalate severity."""
from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import AsyncMock

from notifications.events import NotificationEvent, NotificationSeverity
from notifications.manager import NotificationConfig, NotificationManager

_Sev = NotificationSeverity


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mgr(max_msg=100) -> NotificationManager:
    notifier = AsyncMock()
    notifier.speak = AsyncMock(return_value=True)
    cfg = NotificationConfig(enabled=True, max_messages_per_minute=max_msg)
    return NotificationManager(notifier=notifier, config=cfg)


def _info_event(cat: str = "risk_margin", title: str = "margin low", key: str = "m1") -> NotificationEvent:
    return NotificationEvent(
        severity=_Sev.INFO,
        category=cat,
        title=title,
        body="test body",
        throttle_key=key,
        throttle_window_sec=0,
    )


class TestAlertEscalation:
    def test_no_escalation_below_threshold(self):
        mgr = _make_mgr()
        evt = _info_event()
        # Send 2 times — should NOT escalate (threshold is 3)
        for _ in range(2):
            _run(mgr.send(evt))
        # Check internal count
        esc_key = f"{evt.category}:{evt.throttle_key}"
        assert mgr._alert_repeat_count[esc_key] == 2

    def test_escalation_at_threshold(self):
        mgr = _make_mgr()
        evt = _info_event()
        # Send 3 times — third should escalate INFO -> WARNING
        for _ in range(3):
            _run(mgr.send(evt))
        # The notifier should have received an escalated message on the 3rd call
        calls = mgr.notifier.speak.call_args_list
        assert len(calls) == 3
        # 3rd call should contain "[ESCALATED x3]"
        rendered_text = calls[2][0][0]  # first positional arg
        assert "[ESCALATED x3]" in rendered_text

    def test_escalation_info_to_warning_to_critical(self):
        mgr = _make_mgr()
        evt = _info_event()
        # Send 3 → escalates INFO→WARNING
        for _ in range(3):
            _run(mgr.send(evt))
        third_text = mgr.notifier.speak.call_args_list[2][0][0]
        assert _Sev.WARNING.value in third_text  # ⚠ symbol

        # Now create a WARNING event with same key (simulating the escalated state)
        # Actually the manager tracks by category:key, so sending the same INFO event
        # more times should continue escalating. At count=3 it became WARNING.
        # But internally _maybe_escalate receives the original INFO event each time,
        # so at count >= 3 it maps INFO→WARNING. It doesn't chain to CRITICAL automatically
        # because the input severity is still INFO.
        # To test WARNING→CRITICAL, send a WARNING event:
        warn_evt = NotificationEvent(
            severity=_Sev.WARNING,
            category="risk_warn",
            title="warn alert",
            body="body",
            throttle_key="w1",
            throttle_window_sec=0,
        )
        for _ in range(3):
            _run(mgr.send(warn_evt))
        warn_calls = [c for c in mgr.notifier.speak.call_args_list if "warn alert" in c[0][0]]
        assert len(warn_calls) == 3
        assert "[ESCALATED x3]" in warn_calls[2][0][0]
        assert _Sev.CRITICAL.value in warn_calls[2][0][0]

    def test_critical_not_escalated_further(self):
        mgr = _make_mgr()
        evt = NotificationEvent(
            severity=_Sev.CRITICAL,
            category="crash",
            title="crash alert",
            body="body",
            throttle_key="c1",
            throttle_window_sec=0,
        )
        for _ in range(5):
            _run(mgr.send(evt))
        # CRITICAL has no escalation target — all calls should be unmodified
        for call in mgr.notifier.speak.call_args_list:
            assert "[ESCALATED" not in call[0][0]

    def test_different_keys_tracked_independently(self):
        mgr = _make_mgr()
        evt_a = _info_event(key="a")
        evt_b = _info_event(key="b")
        # Send A twice, B twice — neither should escalate
        _run(mgr.send(evt_a))
        _run(mgr.send(evt_a))
        _run(mgr.send(evt_b))
        _run(mgr.send(evt_b))
        for call in mgr.notifier.speak.call_args_list:
            assert "[ESCALATED" not in call[0][0]
        # Send A third time — only A escalates
        _run(mgr.send(evt_a))
        last_call = mgr.notifier.speak.call_args_list[-1][0][0]
        assert "[ESCALATED x3]" in last_call

    def test_escalation_tracker_capped(self):
        mgr = _make_mgr()
        # Fill tracker beyond cap
        cap = mgr._ESCALATION_MAX_TRACKED
        for i in range(cap + 50):
            evt = _info_event(key=f"k{i}")
            _run(mgr.send(evt))
        assert len(mgr._alert_repeat_count) <= cap + 1  # +1 for in-flight

    def test_success_severity_not_escalated(self):
        mgr = _make_mgr()
        evt = NotificationEvent(
            severity=_Sev.SUCCESS,
            category="trade_fill",
            title="filled",
            body="body",
            throttle_key="s1",
            throttle_window_sec=0,
        )
        for _ in range(5):
            _run(mgr.send(evt))
        for call in mgr.notifier.speak.call_args_list:
            assert "[ESCALATED" not in call[0][0]
