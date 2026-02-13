# execution/rate_limiter.py — SCALPER ETERNAL — RATE LIMITER — 2026 v1.0
# Token bucket rate limiter for exchange API calls.
#
# Features:
# - Token bucket algorithm with burst capacity
# - Per-endpoint rate limits
# - Automatic backoff on 429 responses
# - Binance-specific weight tracking
#
# Design principles:
# - Never block exits (fail-open)
# - Respect exchange rate limits proactively
# - Track API weight consumption

from __future__ import annotations

import asyncio
import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from utils.logging import log_core


# Configuration
_DEFAULT_RATE_PER_SEC = float(os.getenv("RATE_LIMIT_PER_SEC", "10.0"))
_DEFAULT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))
_BACKOFF_INITIAL_SEC = float(os.getenv("RATE_LIMIT_BACKOFF_SEC", "1.0"))
_BACKOFF_MAX_SEC = float(os.getenv("RATE_LIMIT_BACKOFF_MAX_SEC", "60.0"))
_BACKOFF_MULTIPLIER = float(os.getenv("RATE_LIMIT_BACKOFF_MULT", "2.0"))
_ENABLED = os.getenv("RATE_LIMITER_ENABLED", "1").lower() in ("1", "true", "yes", "on")

# Binance specific
_BINANCE_WEIGHT_LIMIT = int(os.getenv("BINANCE_WEIGHT_LIMIT", "1200"))
_BINANCE_WEIGHT_WINDOW_SEC = int(os.getenv("BINANCE_WEIGHT_WINDOW_SEC", "60"))


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
class RateLimiterState:
    """State for token bucket rate limiter."""
    tokens: float = 0.0
    last_refill_ts: float = 0.0
    backoff_until: float = 0.0
    current_backoff_sec: float = 0.0
    total_waits: int = 0
    total_throttled: int = 0
    total_requests: int = 0

    # Binance weight tracking
    weight_used: int = 0
    weight_window_start: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter with backoff support.

    Usage:
        limiter = RateLimiter.get(bot, "binance")

        # Acquire token before request
        await limiter.acquire(weight=1, is_exit=False)

        # Record 429 response
        limiter.record_rate_limit_hit(retry_after=30.0)
    """

    _instances: Dict[str, 'RateLimiter'] = {}

    def __init__(
        self,
        name: str,
        rate_per_sec: float = _DEFAULT_RATE_PER_SEC,
        burst: int = _DEFAULT_BURST,
        backoff_initial: float = _BACKOFF_INITIAL_SEC,
        backoff_max: float = _BACKOFF_MAX_SEC,
        backoff_mult: float = _BACKOFF_MULTIPLIER,
        weight_limit: int = _BINANCE_WEIGHT_LIMIT,
        weight_window: int = _BINANCE_WEIGHT_WINDOW_SEC,
    ):
        self.name = name
        self.rate_per_sec = rate_per_sec
        self.burst = burst
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.backoff_mult = backoff_mult
        self.weight_limit = weight_limit
        self.weight_window = weight_window

        self._state = RateLimiterState(
            tokens=float(burst),
            last_refill_ts=time.time(),
            weight_window_start=time.time(),
        )
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls, bot, name: str = "default") -> 'RateLimiter':
        """Get or create rate limiter by name (singleton per name)."""
        if name in cls._instances:
            return cls._instances[name]

        rate_per_sec = float(_safe_float(_cfg(bot, "RATE_LIMIT_PER_SEC", _DEFAULT_RATE_PER_SEC), _DEFAULT_RATE_PER_SEC))
        burst = int(_safe_float(_cfg(bot, "RATE_LIMIT_BURST", _DEFAULT_BURST), _DEFAULT_BURST))
        backoff_initial = float(_safe_float(_cfg(bot, "RATE_LIMIT_BACKOFF_SEC", _BACKOFF_INITIAL_SEC), _BACKOFF_INITIAL_SEC))
        backoff_max = float(_safe_float(_cfg(bot, "RATE_LIMIT_BACKOFF_MAX_SEC", _BACKOFF_MAX_SEC), _BACKOFF_MAX_SEC))

        instance = cls(
            name=name,
            rate_per_sec=rate_per_sec,
            burst=burst,
            backoff_initial=backoff_initial,
            backoff_max=backoff_max,
        )
        cls._instances[name] = instance
        return instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if rate limiter is enabled."""
        if not _ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "RATE_LIMITER_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _ENABLED

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._state.last_refill_ts
        if elapsed > 0:
            new_tokens = elapsed * self.rate_per_sec
            self._state.tokens = min(self.burst, self._state.tokens + new_tokens)
            self._state.last_refill_ts = now

    def _reset_weight_window(self) -> None:
        """Reset weight window if expired."""
        now = time.time()
        if now - self._state.weight_window_start >= self.weight_window:
            self._state.weight_used = 0
            self._state.weight_window_start = now

    async def acquire(
        self,
        *,
        weight: int = 1,
        is_exit: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire rate limit token(s).

        Args:
            weight: API weight to consume (Binance-specific)
            is_exit: If True, skip waiting (fail-open for exits)
            timeout: Maximum time to wait for token

        Returns:
            True if acquired, False if timed out
        """
        # Always allow exits
        if is_exit:
            return True

        async with self._lock:
            self._state.total_requests += 1

            now = time.time()

            # Check backoff
            if self._state.backoff_until > now:
                wait_time = self._state.backoff_until - now
                if timeout is not None and wait_time > timeout:
                    self._state.total_throttled += 1
                    return False

                log_core.warning(
                    f"[rate_limiter] {self.name}: backoff wait {wait_time:.1f}s"
                )
                self._state.total_waits += 1
                await asyncio.sleep(wait_time)

            # Refill tokens
            self._refill_tokens()

            # Check weight limit
            self._reset_weight_window()
            if self._state.weight_used + weight > self.weight_limit:
                wait_time = self.weight_window - (now - self._state.weight_window_start)
                if wait_time > 0:
                    if timeout is not None and wait_time > timeout:
                        self._state.total_throttled += 1
                        return False

                    log_core.warning(
                        f"[rate_limiter] {self.name}: weight limit wait {wait_time:.1f}s"
                    )
                    self._state.total_waits += 1
                    await asyncio.sleep(wait_time)
                    self._reset_weight_window()

            # Check tokens
            if self._state.tokens < 1:
                wait_time = (1 - self._state.tokens) / self.rate_per_sec
                if timeout is not None and wait_time > timeout:
                    self._state.total_throttled += 1
                    return False

                self._state.total_waits += 1
                await asyncio.sleep(wait_time)
                self._refill_tokens()

            # Consume
            self._state.tokens -= 1
            self._state.weight_used += weight

            return True

    def record_rate_limit_hit(self, retry_after: Optional[float] = None) -> None:
        """
        Record a 429 rate limit response.

        Args:
            retry_after: Retry-After header value from response
        """
        now = time.time()

        if retry_after is not None and retry_after > 0:
            self._state.backoff_until = now + retry_after
            self._state.current_backoff_sec = retry_after
        else:
            # Exponential backoff
            if self._state.current_backoff_sec <= 0:
                self._state.current_backoff_sec = self.backoff_initial
            else:
                self._state.current_backoff_sec = min(
                    self.backoff_max,
                    self._state.current_backoff_sec * self.backoff_mult,
                )
            self._state.backoff_until = now + self._state.current_backoff_sec

        log_core.warning(
            f"[rate_limiter] {self.name}: 429 hit, backoff until +{self._state.current_backoff_sec:.1f}s"
        )

    def record_success(self) -> None:
        """Record successful request (reset backoff)."""
        self._state.current_backoff_sec = 0

    def update_weight_from_headers(self, headers: Dict[str, Any]) -> None:
        """
        Update weight tracking from response headers.

        Binance headers:
        - X-MBX-USED-WEIGHT-1M: Weight used in last minute
        """
        try:
            used = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
            if used is not None:
                self._state.weight_used = int(used)
        except Exception:
            pass

    @property
    def is_throttled(self) -> bool:
        """True if currently in backoff."""
        return time.time() < self._state.backoff_until

    @property
    def available_tokens(self) -> float:
        """Current available tokens."""
        self._refill_tokens()
        return self._state.tokens

    @property
    def weight_remaining(self) -> int:
        """Remaining weight in current window."""
        self._reset_weight_window()
        return max(0, self.weight_limit - self._state.weight_used)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        self._refill_tokens()
        self._reset_weight_window()
        return {
            "name": self.name,
            "tokens": self._state.tokens,
            "weight_used": self._state.weight_used,
            "weight_remaining": self.weight_remaining,
            "backoff_until": self._state.backoff_until,
            "current_backoff_sec": self._state.current_backoff_sec,
            "is_throttled": self.is_throttled,
            "total_waits": self._state.total_waits,
            "total_throttled": self._state.total_throttled,
            "total_requests": self._state.total_requests,
        }


# Module-level convenience functions
async def rate_limit_acquire(
    bot,
    name: str = "default",
    *,
    weight: int = 1,
    is_exit: bool = False,
    timeout: Optional[float] = None,
) -> bool:
    """Acquire rate limit token."""
    if not RateLimiter.is_enabled(bot):
        return True
    return await RateLimiter.get(bot, name).acquire(weight=weight, is_exit=is_exit, timeout=timeout)


def rate_limit_hit(bot, name: str = "default", retry_after: Optional[float] = None) -> None:
    """Record 429 rate limit hit."""
    if not RateLimiter.is_enabled(bot):
        return
    RateLimiter.get(bot, name).record_rate_limit_hit(retry_after)


def rate_limit_success(bot, name: str = "default") -> None:
    """Record successful request."""
    if not RateLimiter.is_enabled(bot):
        return
    RateLimiter.get(bot, name).record_success()


def get_rate_limiter_stats(bot, name: str = "default") -> Dict[str, Any]:
    """Get rate limiter statistics."""
    if not RateLimiter.is_enabled(bot):
        return {"enabled": False}
    return RateLimiter.get(bot, name).stats
