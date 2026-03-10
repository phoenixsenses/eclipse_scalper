"""Unit tests for execution/circuit_breaker.py

Covers:
- State transitions: CLOSED → OPEN → HALF_OPEN → CLOSED
- Rate limit detection and threshold behavior
- Drawdown trip and reset
- Cooldown timer (recovery timeout)
- Fail-open for exits (always allow)
- Force open / force close
- Module-level convenience functions
- Stats reporting
- Singleton pattern (get)
- Enabled/disabled via env and config
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from execution.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitState,
    circuit_allow_request,
    circuit_record_failure,
    circuit_record_success,
    get_circuit_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fresh_breaker(name: str = "test", **kwargs) -> CircuitBreaker:
    """Create a fresh CircuitBreaker (not cached in _instances)."""
    # Remove from cache first
    CircuitBreaker._instances.pop(name, None)
    defaults = dict(
        failure_threshold=3,
        recovery_timeout=5.0,
        half_open_max=2,
        success_threshold=2,
    )
    defaults.update(kwargs)
    cb = CircuitBreaker(name=name, **defaults)
    CircuitBreaker._instances[name] = cb
    return cb


def _bot(enabled=True):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            CIRCUIT_BREAKER_ENABLED=enabled,
            CIRCUIT_FAILURE_THRESHOLD=3,
            CIRCUIT_RECOVERY_TIMEOUT_SEC=5.0,
            CIRCUIT_HALF_OPEN_MAX_REQUESTS=2,
            CIRCUIT_SUCCESS_THRESHOLD=2,
        ),
    )


@pytest.fixture(autouse=True)
def _clear_instances():
    """Clear singleton cache between tests."""
    CircuitBreaker._instances.clear()
    yield
    CircuitBreaker._instances.clear()


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_closed():
    cb = _fresh_breaker()
    assert cb.is_closed
    assert not cb.is_open
    assert cb.current_state == CircuitState.CLOSED


def test_initial_stats():
    cb = _fresh_breaker()
    s = cb.stats
    assert s["state"] == "CLOSED"
    assert s["failure_count"] == 0
    assert s["total_failures"] == 0
    assert s["total_blocked"] == 0


# ---------------------------------------------------------------------------
# 2. CLOSED → OPEN transition
# ---------------------------------------------------------------------------

def test_failures_below_threshold_stay_closed():
    cb = _fresh_breaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_closed
    assert cb.allow_request()


def test_failures_at_threshold_trip_to_open():
    cb = _fresh_breaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open
    assert not cb.allow_request()


def test_success_resets_failure_count_in_closed():
    cb = _fresh_breaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    # failure count should be reset
    cb.record_failure()
    cb.record_failure()
    assert cb.is_closed  # still below threshold after reset


# ---------------------------------------------------------------------------
# 3. OPEN state behavior
# ---------------------------------------------------------------------------

def test_open_blocks_requests():
    cb = _fresh_breaker(failure_threshold=1)
    cb.record_failure()
    assert cb.is_open
    assert not cb.allow_request()
    assert cb.stats["total_blocked"] == 1


def test_open_allows_exits():
    cb = _fresh_breaker(failure_threshold=1)
    cb.record_failure()
    assert cb.is_open
    assert cb.allow_request(is_exit=True)


def test_open_blocked_count_increments():
    cb = _fresh_breaker(failure_threshold=1, recovery_timeout=9999)
    cb.record_failure()
    cb.allow_request()
    cb.allow_request()
    cb.allow_request()
    assert cb.stats["total_blocked"] == 3


# ---------------------------------------------------------------------------
# 4. OPEN → HALF_OPEN after recovery timeout
# ---------------------------------------------------------------------------

def test_half_open_after_timeout():
    cb = _fresh_breaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    assert cb.is_open
    time.sleep(0.02)
    assert cb.allow_request()
    assert cb.current_state == CircuitState.HALF_OPEN


# ---------------------------------------------------------------------------
# 5. HALF_OPEN behavior
# ---------------------------------------------------------------------------

def test_half_open_limits_requests():
    cb = _fresh_breaker(failure_threshold=1, recovery_timeout=0.01, half_open_max=2)
    cb.record_failure()
    time.sleep(0.02)
    # Transition OPEN→HALF_OPEN (does not count as half_open request)
    assert cb.allow_request()  # triggers transition
    assert cb.current_state == CircuitState.HALF_OPEN
    # Now use a long recovery timeout so it won't re-enter half-open
    cb.recovery_timeout = 9999
    assert cb.allow_request()  # 1st counted half_open request
    assert cb.allow_request()  # 2nd counted half_open request (max=2)
    assert not cb.allow_request()  # 3rd: blocked


def test_half_open_failure_goes_back_to_open():
    cb = _fresh_breaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.allow_request()
    cb.record_failure()
    assert cb.is_open


def test_half_open_successes_close_circuit():
    cb = _fresh_breaker(failure_threshold=1, recovery_timeout=0.01, success_threshold=2)
    cb.record_failure()
    time.sleep(0.02)
    cb.allow_request()
    cb.record_success()
    assert cb.current_state == CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.is_closed


# ---------------------------------------------------------------------------
# 6. Force open / force close
# ---------------------------------------------------------------------------

def test_force_open():
    cb = _fresh_breaker()
    cb.force_open()
    assert cb.is_open


def test_force_close():
    cb = _fresh_breaker(failure_threshold=1)
    cb.record_failure()
    assert cb.is_open
    cb.force_close()
    assert cb.is_closed


def test_force_close_resets_failure_count():
    cb = _fresh_breaker(failure_threshold=1)
    cb.record_failure()
    cb.force_close()
    assert cb.stats["failure_count"] == 0


# ---------------------------------------------------------------------------
# 7. Singleton pattern
# ---------------------------------------------------------------------------

def test_get_returns_same_instance():
    bot = _bot()
    cb1 = CircuitBreaker.get(bot, "singleton_test")
    cb2 = CircuitBreaker.get(bot, "singleton_test")
    assert cb1 is cb2


def test_get_different_names_different_instances():
    bot = _bot()
    cb1 = CircuitBreaker.get(bot, "a")
    cb2 = CircuitBreaker.get(bot, "b")
    assert cb1 is not cb2


# ---------------------------------------------------------------------------
# 8. Enabled check
# ---------------------------------------------------------------------------

@patch.dict("os.environ", {"CIRCUIT_BREAKER_ENABLED": "0"})
def test_is_enabled_env_off():
    # Need to reload module-level constant
    import importlib
    import execution.circuit_breaker as mod
    orig = mod._ENABLED
    mod._ENABLED = False
    try:
        assert not CircuitBreaker.is_enabled(_bot())
    finally:
        mod._ENABLED = orig


def test_is_enabled_cfg_false():
    bot = _bot(enabled=False)
    import execution.circuit_breaker as mod
    orig = mod._ENABLED
    mod._ENABLED = True
    try:
        assert not CircuitBreaker.is_enabled(bot)
    finally:
        mod._ENABLED = orig


# ---------------------------------------------------------------------------
# 9. Module-level convenience functions
# ---------------------------------------------------------------------------

def test_circuit_allow_request_disabled():
    bot = _bot(enabled=False)
    import execution.circuit_breaker as mod
    orig = mod._ENABLED
    mod._ENABLED = False
    try:
        assert circuit_allow_request(bot, "conv_test")
    finally:
        mod._ENABLED = orig


def test_circuit_record_success_no_error():
    bot = _bot()
    CircuitBreaker._instances["default"] = _fresh_breaker("default")
    circuit_record_success(bot, "default")


def test_circuit_record_failure_no_error():
    bot = _bot()
    CircuitBreaker._instances["default"] = _fresh_breaker("default")
    circuit_record_failure(bot, "default")


def test_get_circuit_stats_disabled():
    bot = _bot(enabled=False)
    import execution.circuit_breaker as mod
    orig = mod._ENABLED
    mod._ENABLED = False
    try:
        s = get_circuit_stats(bot, "default")
        assert s == {"enabled": False}
    finally:
        mod._ENABLED = orig


# ---------------------------------------------------------------------------
# 10. Callback on state change
# ---------------------------------------------------------------------------

def test_callback_called_on_transition():
    cb = _fresh_breaker(failure_threshold=1)
    calls = []
    cb.set_callback(lambda name, old, new: calls.append((name, old, new)))
    cb.record_failure()
    assert len(calls) == 1
    assert calls[0][1] == CircuitState.CLOSED
    assert calls[0][2] == CircuitState.OPEN


def test_callback_exception_does_not_crash():
    cb = _fresh_breaker(failure_threshold=1)
    cb.set_callback(lambda *a: 1 / 0)
    cb.record_failure()  # should not raise
    assert cb.is_open
