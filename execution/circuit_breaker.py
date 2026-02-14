# execution/circuit_breaker.py — SCALPER ETERNAL — CIRCUIT BREAKER — 2026 v1.0
# Prevents cascade failures by temporarily blocking requests after repeated failures.
#
# States:
# - CLOSED: Normal operation, requests pass through
# - OPEN: Blocking all requests (after failure threshold)
# - HALF_OPEN: Allowing limited requests to test recovery
#
# Design principles:
# - Per-exchange circuit breakers
# - Fail-open for exits (must be able to close positions)
# - Configurable thresholds and timeouts
# - Telemetry integration

from __future__ import annotations

import asyncio
import time
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from utils.logging import log_core


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# Configuration
_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5"))
_RECOVERY_TIMEOUT_SEC = float(os.getenv("CIRCUIT_RECOVERY_TIMEOUT_SEC", "30.0"))
_HALF_OPEN_MAX_REQUESTS = int(os.getenv("CIRCUIT_HALF_OPEN_MAX_REQUESTS", "3"))
_SUCCESS_THRESHOLD = int(os.getenv("CIRCUIT_SUCCESS_THRESHOLD", "2"))
_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "1").lower() in ("1", "true", "yes", "on")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        cfg = getattr(bot, "cfg", None)
        return getattr(cfg, name, default) if cfg is not None else default
    except Exception:
        return default


@dataclass
class CircuitBreakerState:
    """State for a single circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_ts: float = 0.0
    last_state_change_ts: float = 0.0
    half_open_requests: int = 0
    total_blocked: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for exchange operations.

    Usage:
        breaker = CircuitBreaker.get(bot, "binance")

        # Check before request
        if not breaker.allow_request(is_exit=False):
            return None  # Circuit is open

        try:
            result = await exchange_call()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure(e)
            raise
    """

    _instances: Dict[str, 'CircuitBreaker'] = {}

    def __init__(
        self,
        name: str,
        failure_threshold: int = _FAILURE_THRESHOLD,
        recovery_timeout: float = _RECOVERY_TIMEOUT_SEC,
        half_open_max: int = _HALF_OPEN_MAX_REQUESTS,
        success_threshold: int = _SUCCESS_THRESHOLD,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.success_threshold = success_threshold
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()

        # Callbacks
        self._on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None

    @classmethod
    def get(cls, bot, name: str = "default") -> 'CircuitBreaker':
        """Get or create circuit breaker by name (singleton per name)."""
        if name in cls._instances:
            return cls._instances[name]

        # Get config from bot
        failure_threshold = int(_safe_float(_cfg(bot, "CIRCUIT_FAILURE_THRESHOLD", _FAILURE_THRESHOLD), _FAILURE_THRESHOLD))
        recovery_timeout = float(_safe_float(_cfg(bot, "CIRCUIT_RECOVERY_TIMEOUT_SEC", _RECOVERY_TIMEOUT_SEC), _RECOVERY_TIMEOUT_SEC))
        half_open_max = int(_safe_float(_cfg(bot, "CIRCUIT_HALF_OPEN_MAX_REQUESTS", _HALF_OPEN_MAX_REQUESTS), _HALF_OPEN_MAX_REQUESTS))
        success_threshold = int(_safe_float(_cfg(bot, "CIRCUIT_SUCCESS_THRESHOLD", _SUCCESS_THRESHOLD), _SUCCESS_THRESHOLD))

        instance = cls(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max=half_open_max,
            success_threshold=success_threshold,
        )
        cls._instances[name] = instance
        return instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if circuit breaker is enabled."""
        if not _ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "CIRCUIT_BREAKER_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _ENABLED

    def set_callback(self, on_state_change: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """Set callback for state changes."""
        self._on_state_change = on_state_change

    def allow_request(self, *, is_exit: bool = False) -> bool:
        """
        Check if request should be allowed.

        Args:
            is_exit: If True, always allow (fail-open for exits)

        Returns:
            True if request should proceed, False if blocked
        """
        # Always allow exits (fail-open)
        if is_exit:
            return True

        state = self._state.state
        now = time.time()

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            elapsed = now - self._state.last_state_change_ts
            if elapsed >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True

            self._state.total_blocked += 1
            return False

        if state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            if self._state.half_open_requests < self.half_open_max:
                self._state.half_open_requests += 1
                return True

            self._state.total_blocked += 1
            return False

        return True

    def record_success(self) -> None:
        """Record a successful request."""
        self._state.total_successes += 1

        if self._state.state == CircuitState.HALF_OPEN:
            self._state.success_count += 1
            if self._state.success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state.state == CircuitState.CLOSED:
            # Reset failure count on success
            self._state.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed request."""
        self._state.total_failures += 1
        self._state.failure_count += 1
        self._state.last_failure_ts = time.time()

        if self._state.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)
        elif self._state.state == CircuitState.CLOSED:
            if self._state.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state.state
        if old_state == new_state:
            return

        self._state.state = new_state
        self._state.last_state_change_ts = time.time()

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._state.half_open_requests = 0
            self._state.success_count = 0
        elif new_state == CircuitState.OPEN:
            self._state.success_count = 0

        log_core.warning(
            f"[circuit_breaker] {self.name}: {old_state.value} -> {new_state.value} | "
            f"failures={self._state.total_failures} blocked={self._state.total_blocked}"
        )

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception:
                pass

    def force_open(self) -> None:
        """Force circuit to open state (manual intervention)."""
        self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Force circuit to closed state (manual intervention)."""
        self._transition_to(CircuitState.CLOSED)

    @property
    def is_open(self) -> bool:
        """True if circuit is open (blocking requests)."""
        return self._state.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """True if circuit is closed (allowing requests)."""
        return self._state.state == CircuitState.CLOSED

    @property
    def current_state(self) -> CircuitState:
        """Current circuit state."""
        return self._state.state

    @property
    def stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "total_failures": self._state.total_failures,
            "total_successes": self._state.total_successes,
            "total_blocked": self._state.total_blocked,
            "last_failure_ts": self._state.last_failure_ts,
            "last_state_change_ts": self._state.last_state_change_ts,
            "half_open_requests": self._state.half_open_requests,
        }


# Module-level convenience functions
def circuit_allow_request(bot, name: str = "default", *, is_exit: bool = False) -> bool:
    """Check if request should be allowed through circuit breaker."""
    if not CircuitBreaker.is_enabled(bot):
        return True
    return CircuitBreaker.get(bot, name).allow_request(is_exit=is_exit)


def circuit_record_success(bot, name: str = "default") -> None:
    """Record successful request."""
    if not CircuitBreaker.is_enabled(bot):
        return
    CircuitBreaker.get(bot, name).record_success()


def circuit_record_failure(bot, name: str = "default", error: Optional[Exception] = None) -> None:
    """Record failed request."""
    if not CircuitBreaker.is_enabled(bot):
        return
    CircuitBreaker.get(bot, name).record_failure(error)


def get_circuit_stats(bot, name: str = "default") -> Dict[str, Any]:
    """Get circuit breaker statistics."""
    if not CircuitBreaker.is_enabled(bot):
        return {"enabled": False}
    return CircuitBreaker.get(bot, name).stats
