# execution/health_monitor.py — SCALPER ETERNAL — HEALTH MONITOR — 2026 v1.0
# Centralized health monitoring for all bot components.
#
# Features:
# - Component health checks (exchange, data, positions, etc.)
# - Degraded state detection
# - Health score calculation
# - Alert emission on health degradation
#
# Design principles:
# - Non-blocking health checks
# - Graceful degradation reporting
# - Integration with all Phase 1 & 2 modules

from __future__ import annotations

import asyncio
import time
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.logging import log_core


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


# Configuration
_HEALTH_CHECK_INTERVAL_SEC = float(os.getenv("HEALTH_CHECK_INTERVAL_SEC", "30.0"))
_HEALTH_ENABLED = os.getenv("HEALTH_MONITOR_ENABLED", "1").lower() in ("1", "true", "yes", "on")
_EXCHANGE_TIMEOUT_SEC = float(os.getenv("HEALTH_EXCHANGE_TIMEOUT_SEC", "10.0"))


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
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    last_check_ts: float = 0.0
    check_duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus = HealthStatus.UNKNOWN
    score: float = 0.0  # 0-100
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    last_check_ts: float = 0.0
    uptime_sec: float = 0.0
    degraded_components: List[str] = field(default_factory=list)
    unhealthy_components: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Centralized health monitoring for all bot components.

    Usage:
        monitor = HealthMonitor(bot)

        # Run health check
        health = await monitor.check_health()

        # Get current status
        status = monitor.get_status()

        # Register custom health check
        monitor.register_check("my_component", my_health_check_fn)
    """

    _instance: Optional['HealthMonitor'] = None

    def __init__(self, bot):
        self.bot = bot
        self._start_time = time.time()
        self._last_health: Optional[SystemHealth] = None
        self._custom_checks: Dict[str, Callable] = {}
        self._check_lock = asyncio.Lock()

        # Alert callback
        self._on_health_change: Optional[Callable[[SystemHealth, SystemHealth], None]] = None

    @classmethod
    def get(cls, bot) -> 'HealthMonitor':
        """Get or create health monitor (singleton)."""
        if cls._instance is None:
            cls._instance = cls(bot)
        return cls._instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if health monitor is enabled."""
        if not _HEALTH_ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "HEALTH_MONITOR_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _HEALTH_ENABLED

    def set_callback(self, on_health_change: Callable[[SystemHealth, SystemHealth], None]) -> None:
        """Set callback for health status changes."""
        self._on_health_change = on_health_change

    def register_check(self, name: str, check_fn: Callable) -> None:
        """Register a custom health check function."""
        self._custom_checks[name] = check_fn

    async def check_health(self) -> SystemHealth:
        """
        Run all health checks and return system health status.

        Returns:
            SystemHealth object with component statuses
        """
        async with self._check_lock:
            components: Dict[str, ComponentHealth] = {}

            # Core component checks
            components["exchange"] = await self._check_exchange()
            components["positions"] = await self._check_positions()
            components["data_cache"] = await self._check_data_cache()
            components["circuit_breaker"] = await self._check_circuit_breaker()
            components["rate_limiter"] = await self._check_rate_limiter()
            components["intent_ledger"] = await self._check_intent_ledger()
            components["position_lock"] = await self._check_position_lock()

            # Custom checks
            for name, check_fn in self._custom_checks.items():
                try:
                    if asyncio.iscoroutinefunction(check_fn):
                        components[name] = await check_fn(self.bot)
                    else:
                        components[name] = check_fn(self.bot)
                except Exception as e:
                    components[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Check failed: {e}",
                        last_check_ts=time.time(),
                    )

            # Calculate overall health
            health = self._calculate_system_health(components)

            # Check for status change
            if self._last_health is not None and self._on_health_change:
                if health.status != self._last_health.status:
                    try:
                        self._on_health_change(self._last_health, health)
                    except Exception:
                        pass

            self._last_health = health
            return health

    async def _check_exchange(self) -> ComponentHealth:
        """Check exchange connectivity."""
        start = time.time()
        try:
            ex = getattr(self.bot, "ex", None) or getattr(self.bot, "exchange", None)
            if ex is None:
                return ComponentHealth(
                    name="exchange",
                    status=HealthStatus.UNHEALTHY,
                    message="No exchange connection",
                    last_check_ts=time.time(),
                )

            # Try a lightweight API call
            timeout = float(_safe_float(_cfg(self.bot, "HEALTH_EXCHANGE_TIMEOUT_SEC", _EXCHANGE_TIMEOUT_SEC), _EXCHANGE_TIMEOUT_SEC))

            try:
                async with asyncio.timeout(timeout):
                    # fetch_time is lightweight
                    if hasattr(ex, "fetch_time"):
                        await ex.fetch_time()
                    elif hasattr(ex, "fetch_ticker"):
                        await ex.fetch_ticker("BTC/USDT:USDT")
                    else:
                        # Fallback - just check if exchange object exists
                        pass
            except asyncio.TimeoutError:
                return ComponentHealth(
                    name="exchange",
                    status=HealthStatus.DEGRADED,
                    message=f"Exchange timeout ({timeout}s)",
                    last_check_ts=time.time(),
                    check_duration_ms=(time.time() - start) * 1000,
                )

            return ComponentHealth(
                name="exchange",
                status=HealthStatus.HEALTHY,
                message="Connected",
                last_check_ts=time.time(),
                check_duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return ComponentHealth(
                name="exchange",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                last_check_ts=time.time(),
                check_duration_ms=(time.time() - start) * 1000,
            )

    async def _check_positions(self) -> ComponentHealth:
        """Check position tracking health."""
        try:
            state = getattr(self.bot, "state", None)
            if state is None:
                return ComponentHealth(
                    name="positions",
                    status=HealthStatus.DEGRADED,
                    message="No state object",
                    last_check_ts=time.time(),
                )

            positions = getattr(state, "positions", None)
            if not isinstance(positions, dict):
                return ComponentHealth(
                    name="positions",
                    status=HealthStatus.DEGRADED,
                    message="Invalid positions dict",
                    last_check_ts=time.time(),
                )

            return ComponentHealth(
                name="positions",
                status=HealthStatus.HEALTHY,
                message=f"{len(positions)} active positions",
                last_check_ts=time.time(),
                details={"count": len(positions)},
            )

        except Exception as e:
            return ComponentHealth(
                name="positions",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check_ts=time.time(),
            )

    async def _check_data_cache(self) -> ComponentHealth:
        """Check data cache health."""
        try:
            data = getattr(self.bot, "data", None)
            if data is None:
                return ComponentHealth(
                    name="data_cache",
                    status=HealthStatus.DEGRADED,
                    message="No data cache",
                    last_check_ts=time.time(),
                )

            return ComponentHealth(
                name="data_cache",
                status=HealthStatus.HEALTHY,
                message="Available",
                last_check_ts=time.time(),
            )

        except Exception as e:
            return ComponentHealth(
                name="data_cache",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check_ts=time.time(),
            )

    async def _check_circuit_breaker(self) -> ComponentHealth:
        """Check circuit breaker status."""
        try:
            from execution.circuit_breaker import CircuitBreaker, CircuitState

            if not CircuitBreaker.is_enabled(self.bot):
                return ComponentHealth(
                    name="circuit_breaker",
                    status=HealthStatus.HEALTHY,
                    message="Disabled",
                    last_check_ts=time.time(),
                )

            breaker = CircuitBreaker.get(self.bot, "exchange")
            stats = breaker.stats

            if breaker.current_state == CircuitState.OPEN:
                return ComponentHealth(
                    name="circuit_breaker",
                    status=HealthStatus.DEGRADED,
                    message=f"OPEN - blocking requests",
                    last_check_ts=time.time(),
                    details=stats,
                )
            elif breaker.current_state == CircuitState.HALF_OPEN:
                return ComponentHealth(
                    name="circuit_breaker",
                    status=HealthStatus.DEGRADED,
                    message="HALF_OPEN - testing recovery",
                    last_check_ts=time.time(),
                    details=stats,
                )

            return ComponentHealth(
                name="circuit_breaker",
                status=HealthStatus.HEALTHY,
                message="CLOSED",
                last_check_ts=time.time(),
                details=stats,
            )

        except ImportError:
            return ComponentHealth(
                name="circuit_breaker",
                status=HealthStatus.HEALTHY,
                message="Not installed",
                last_check_ts=time.time(),
            )
        except Exception as e:
            return ComponentHealth(
                name="circuit_breaker",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check_ts=time.time(),
            )

    async def _check_rate_limiter(self) -> ComponentHealth:
        """Check rate limiter status."""
        try:
            from execution.rate_limiter import RateLimiter

            if not RateLimiter.is_enabled(self.bot):
                return ComponentHealth(
                    name="rate_limiter",
                    status=HealthStatus.HEALTHY,
                    message="Disabled",
                    last_check_ts=time.time(),
                )

            limiter = RateLimiter.get(self.bot, "exchange")
            stats = limiter.stats

            if limiter.is_throttled:
                return ComponentHealth(
                    name="rate_limiter",
                    status=HealthStatus.DEGRADED,
                    message=f"Throttled - backoff active",
                    last_check_ts=time.time(),
                    details=stats,
                )

            # Check if weight is getting close to limit
            if limiter.weight_remaining < 200:
                return ComponentHealth(
                    name="rate_limiter",
                    status=HealthStatus.DEGRADED,
                    message=f"Low weight remaining: {limiter.weight_remaining}",
                    last_check_ts=time.time(),
                    details=stats,
                )

            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.HEALTHY,
                message=f"OK - {limiter.weight_remaining} weight remaining",
                last_check_ts=time.time(),
                details=stats,
            )

        except ImportError:
            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.HEALTHY,
                message="Not installed",
                last_check_ts=time.time(),
            )
        except Exception as e:
            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check_ts=time.time(),
            )

    async def _check_intent_ledger(self) -> ComponentHealth:
        """Check intent ledger persistence health."""
        try:
            from execution.intent_ledger import get_persistence_stats

            stats = get_persistence_stats()

            if not stats.get("available", True):
                return ComponentHealth(
                    name="intent_ledger",
                    status=HealthStatus.HEALTHY,
                    message="Persistence not available",
                    last_check_ts=time.time(),
                )

            if stats.get("degraded", False):
                return ComponentHealth(
                    name="intent_ledger",
                    status=HealthStatus.DEGRADED,
                    message="Persistence degraded",
                    last_check_ts=time.time(),
                    details=stats,
                )

            return ComponentHealth(
                name="intent_ledger",
                status=HealthStatus.HEALTHY,
                message=f"OK - {stats.get('total_writes', 0)} writes",
                last_check_ts=time.time(),
                details=stats,
            )

        except ImportError:
            return ComponentHealth(
                name="intent_ledger",
                status=HealthStatus.HEALTHY,
                message="Not installed",
                last_check_ts=time.time(),
            )
        except Exception as e:
            return ComponentHealth(
                name="intent_ledger",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check_ts=time.time(),
            )

    async def _check_position_lock(self) -> ComponentHealth:
        """Check position lock manager health."""
        try:
            from execution.position_lock import PositionLockManager

            if not PositionLockManager.is_enabled(self.bot):
                return ComponentHealth(
                    name="position_lock",
                    status=HealthStatus.HEALTHY,
                    message="Disabled",
                    last_check_ts=time.time(),
                )

            lock_mgr = PositionLockManager.get(self.bot)
            stats = lock_mgr.stats

            if lock_mgr.is_degraded:
                return ComponentHealth(
                    name="position_lock",
                    status=HealthStatus.DEGRADED,
                    message="Lock manager degraded (timeouts)",
                    last_check_ts=time.time(),
                    details=stats,
                )

            return ComponentHealth(
                name="position_lock",
                status=HealthStatus.HEALTHY,
                message=f"OK - v{stats.get('version', 0)}",
                last_check_ts=time.time(),
                details=stats,
            )

        except ImportError:
            return ComponentHealth(
                name="position_lock",
                status=HealthStatus.HEALTHY,
                message="Not installed",
                last_check_ts=time.time(),
            )
        except Exception as e:
            return ComponentHealth(
                name="position_lock",
                status=HealthStatus.UNKNOWN,
                message=str(e),
                last_check_ts=time.time(),
            )

    def _calculate_system_health(self, components: Dict[str, ComponentHealth]) -> SystemHealth:
        """Calculate overall system health from component statuses."""
        degraded = []
        unhealthy = []
        scores = []

        for name, comp in components.items():
            if comp.status == HealthStatus.HEALTHY:
                scores.append(100)
            elif comp.status == HealthStatus.DEGRADED:
                scores.append(50)
                degraded.append(name)
            elif comp.status == HealthStatus.UNHEALTHY:
                scores.append(0)
                unhealthy.append(name)
            else:
                scores.append(75)  # Unknown treated as potentially ok

        avg_score = sum(scores) / len(scores) if scores else 0

        if unhealthy:
            status = HealthStatus.UNHEALTHY
        elif degraded:
            status = HealthStatus.DEGRADED
        elif avg_score >= 90:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.DEGRADED

        return SystemHealth(
            status=status,
            score=avg_score,
            components=components,
            last_check_ts=time.time(),
            uptime_sec=time.time() - self._start_time,
            degraded_components=degraded,
            unhealthy_components=unhealthy,
        )

    def get_status(self) -> Optional[SystemHealth]:
        """Get last health check result."""
        return self._last_health

    def get_summary(self) -> Dict[str, Any]:
        """Get health summary as dict."""
        if self._last_health is None:
            return {"status": "UNKNOWN", "message": "No health check run yet"}

        h = self._last_health
        return {
            "status": h.status.value,
            "score": round(h.score, 1),
            "uptime_sec": round(h.uptime_sec, 0),
            "last_check_ts": h.last_check_ts,
            "degraded": h.degraded_components,
            "unhealthy": h.unhealthy_components,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                }
                for name, comp in h.components.items()
            },
        }


# Module-level convenience functions
async def check_health(bot) -> SystemHealth:
    """Run health check."""
    if not HealthMonitor.is_enabled(bot):
        return SystemHealth(status=HealthStatus.UNKNOWN, score=0)
    return await HealthMonitor.get(bot).check_health()


def get_health_summary(bot) -> Dict[str, Any]:
    """Get health summary."""
    if not HealthMonitor.is_enabled(bot):
        return {"status": "DISABLED"}
    return HealthMonitor.get(bot).get_summary()


async def health_check_tick(bot) -> None:
    """
    Periodic health check tick - call from guardian loop.
    """
    if not HealthMonitor.is_enabled(bot):
        return

    try:
        state = getattr(bot, "state", None)
        if state is None:
            return

        rc = getattr(state, "run_context", None)
        if not isinstance(rc, dict):
            try:
                state.run_context = {}
                rc = state.run_context
            except Exception:
                rc = {}

        now = time.time()
        last_check = _safe_float(rc.get("_last_health_check_ts"), 0.0)
        interval = float(_safe_float(_cfg(bot, "HEALTH_CHECK_INTERVAL_SEC", _HEALTH_CHECK_INTERVAL_SEC), _HEALTH_CHECK_INTERVAL_SEC))

        if now - last_check < interval:
            return

        rc["_last_health_check_ts"] = now

        health = await check_health(bot)

        if health.status == HealthStatus.UNHEALTHY:
            log_core.critical(
                f"[health_monitor] UNHEALTHY: score={health.score:.0f} | "
                f"unhealthy={health.unhealthy_components}"
            )
        elif health.status == HealthStatus.DEGRADED:
            log_core.warning(
                f"[health_monitor] DEGRADED: score={health.score:.0f} | "
                f"degraded={health.degraded_components}"
            )

    except Exception as e:
        log_core.error(f"[health_monitor] health_check_tick failed: {e}")
