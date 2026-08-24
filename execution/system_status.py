# execution/system_status.py — SCALPER ETERNAL — SYSTEM STATUS — 2026 v1.0
# Unified system status reporting for bot monitoring.
#
# Features:
# - Aggregates health, metrics, and state into single report
# - Periodic status logging
# - Telegram/webhook status notifications
# - JSON export for external monitoring
#
# Design principles:
# - Single source of truth for bot status
# - Human-readable and machine-parseable output
# - Non-blocking, fire-and-forget notifications

from __future__ import annotations

import asyncio
import time
import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from utils.logging import log_core


# Configuration
_STATUS_INTERVAL_SEC = float(os.getenv("STATUS_INTERVAL_SEC", "300.0"))  # 5 min
_STATUS_ENABLED = os.getenv("SYSTEM_STATUS_ENABLED", "1").lower() in ("1", "true", "yes", "on")


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
class SystemStatus:
    """Complete system status snapshot."""
    # Identification
    timestamp: str
    uptime_sec: float
    version: str

    # Health
    health_status: str
    health_score: float
    degraded_components: List[str]

    # Positions
    active_positions: int
    total_exposure_usdt: float

    # Performance
    equity: float
    session_pnl: float
    session_pnl_pct: float

    # Execution stats
    orders_created: int
    orders_failed: int
    circuit_breaker_state: str
    rate_limiter_throttled: bool

    # Warnings
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'='*50}",
            f"SYSTEM STATUS - {self.timestamp}",
            f"{'='*50}",
            f"Health: {self.health_status} ({self.health_score:.0f}/100)",
            f"Uptime: {self.uptime_sec/3600:.1f}h",
            f"",
            f"Positions: {self.active_positions}",
            f"Exposure: ${self.total_exposure_usdt:,.0f}",
            f"Equity: ${self.equity:,.0f}",
            f"Session PnL: ${self.session_pnl:,.0f} ({self.session_pnl_pct:+.2f}%)",
            f"",
            f"Orders: {self.orders_created} created, {self.orders_failed} failed",
            f"Circuit: {self.circuit_breaker_state}",
            f"Rate Limited: {'Yes' if self.rate_limiter_throttled else 'No'}",
        ]

        if self.degraded_components:
            lines.append(f"")
            lines.append(f"Degraded: {', '.join(self.degraded_components)}")

        if self.warnings:
            lines.append(f"")
            lines.append(f"Warnings:")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")

        lines.append(f"{'='*50}")
        return "\n".join(lines)


class SystemStatusReporter:
    """
    Unified system status reporting.

    Usage:
        reporter = SystemStatusReporter(bot)

        # Get current status
        status = await reporter.get_status()

        # Log status periodically
        await reporter.log_status()

        # Send status notification
        await reporter.notify_status()
    """

    _instance: Optional['SystemStatusReporter'] = None
    _start_time: float = time.time()

    def __init__(self, bot):
        self.bot = bot
        self._last_status: Optional[SystemStatus] = None
        self._session_start_equity: Optional[float] = None

    @classmethod
    def get(cls, bot) -> 'SystemStatusReporter':
        """Get or create reporter (singleton)."""
        if cls._instance is None:
            cls._instance = cls(bot)
        return cls._instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if status reporting is enabled."""
        if not _STATUS_ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "SYSTEM_STATUS_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _STATUS_ENABLED

    async def get_status(self) -> SystemStatus:
        """Get current system status."""
        warnings = []

        # Timestamp
        timestamp = datetime.now(timezone.utc).isoformat()
        uptime = time.time() - self._start_time

        # Version
        version = "unknown"
        try:
            state = getattr(self.bot, "state", None)
            if state:
                version = str(getattr(state, "version", "unknown"))
        except Exception:
            pass

        # Health
        health_status = "UNKNOWN"
        health_score = 0.0
        degraded_components = []
        try:
            from execution.health_monitor import HealthMonitor, HealthStatus
            if HealthMonitor.is_enabled(self.bot):
                health = HealthMonitor.get(self.bot).get_status()
                if health:
                    health_status = health.status.value
                    health_score = health.score
                    degraded_components = health.degraded_components + health.unhealthy_components
        except Exception as e:
            warnings.append(f"Health check failed: {e}")

        # Positions
        active_positions = 0
        total_exposure = 0.0
        try:
            state = getattr(self.bot, "state", None)
            if state:
                positions = getattr(state, "positions", None)
                if isinstance(positions, dict):
                    active_positions = len(positions)
                    for pos in positions.values():
                        size = abs(_safe_float(getattr(pos, "size", 0), 0))
                        entry = _safe_float(getattr(pos, "entry_price", 0), 0)
                        total_exposure += size * entry
        except Exception as e:
            warnings.append(f"Position check failed: {e}")

        # Equity & PnL
        equity = 0.0
        session_pnl = 0.0
        session_pnl_pct = 0.0
        try:
            state = getattr(self.bot, "state", None)
            if state:
                equity = _safe_float(getattr(state, "current_equity", 0), 0)

                if self._session_start_equity is None and equity > 0:
                    self._session_start_equity = equity

                if self._session_start_equity and self._session_start_equity > 0:
                    session_pnl = equity - self._session_start_equity
                    session_pnl_pct = (session_pnl / self._session_start_equity) * 100
        except Exception as e:
            warnings.append(f"Equity check failed: {e}")

        # Execution stats
        orders_created = 0
        orders_failed = 0
        try:
            from execution.metrics_collector import MetricsCollector
            if MetricsCollector.is_enabled():
                m = MetricsCollector.get()
                orders_created = int(m.counter("orders_created").get())
                orders_failed = int(m.counter("orders_failed").get())
        except Exception:
            pass

        # Circuit breaker
        circuit_state = "CLOSED"
        try:
            from execution.circuit_breaker import CircuitBreaker
            if CircuitBreaker.is_enabled(self.bot):
                breaker = CircuitBreaker.get(self.bot, "exchange")
                circuit_state = breaker.current_state.value
                if circuit_state != "CLOSED":
                    warnings.append(f"Circuit breaker is {circuit_state}")
        except Exception:
            pass

        # Rate limiter
        rate_limited = False
        try:
            from execution.rate_limiter import RateLimiter
            if RateLimiter.is_enabled(self.bot):
                limiter = RateLimiter.get(self.bot, "exchange")
                rate_limited = limiter.is_throttled
                if rate_limited:
                    warnings.append("Rate limiter is throttling requests")
        except Exception:
            pass

        # Build status
        status = SystemStatus(
            timestamp=timestamp,
            uptime_sec=uptime,
            version=version,
            health_status=health_status,
            health_score=health_score,
            degraded_components=degraded_components,
            active_positions=active_positions,
            total_exposure_usdt=total_exposure,
            equity=equity,
            session_pnl=session_pnl,
            session_pnl_pct=session_pnl_pct,
            orders_created=orders_created,
            orders_failed=orders_failed,
            circuit_breaker_state=circuit_state,
            rate_limiter_throttled=rate_limited,
            warnings=warnings,
        )

        self._last_status = status
        return status

    async def log_status(self) -> None:
        """Log current status."""
        status = await self.get_status()
        log_core.info(
            f"[status] health={status.health_status} score={status.health_score:.0f} | "
            f"pos={status.active_positions} exp=${status.total_exposure_usdt:,.0f} | "
            f"eq=${status.equity:,.0f} pnl=${status.session_pnl:,.0f} ({status.session_pnl_pct:+.2f}%) | "
            f"orders={status.orders_created}/{status.orders_failed} | "
            f"circuit={status.circuit_breaker_state}"
        )

    async def notify_status(self, force: bool = False) -> None:
        """Send status notification via Telegram."""
        try:
            notify = getattr(self.bot, "notify", None)
            if notify is None:
                return

            status = await self.get_status()

            # Only notify if there are warnings or forced
            if not force and not status.warnings and status.health_status == "HEALTHY":
                return

            message = status.to_summary()

            speak_fn = getattr(notify, "speak", None)
            if callable(speak_fn):
                await speak_fn(message, "info")

        except Exception as e:
            log_core.error(f"[status] notify failed: {e}")

    def get_last_status(self) -> Optional[SystemStatus]:
        """Get last computed status."""
        return self._last_status


# Module-level convenience functions
async def get_system_status(bot) -> SystemStatus:
    """Get current system status."""
    return await SystemStatusReporter.get(bot).get_status()


async def log_system_status(bot) -> None:
    """Log current system status."""
    if not SystemStatusReporter.is_enabled(bot):
        return
    await SystemStatusReporter.get(bot).log_status()


async def notify_system_status(bot, force: bool = False) -> None:
    """Send status notification."""
    if not SystemStatusReporter.is_enabled(bot):
        return
    await SystemStatusReporter.get(bot).notify_status(force)


def get_status_json(bot) -> str:
    """Get last status as JSON."""
    reporter = SystemStatusReporter.get(bot)
    status = reporter.get_last_status()
    if status:
        return status.to_json()
    return "{}"


async def status_tick(bot) -> None:
    """
    Periodic status tick - call from guardian loop.
    """
    if not SystemStatusReporter.is_enabled(bot):
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
        last_status = _safe_float(rc.get("_last_status_ts"), 0.0)
        interval = float(_safe_float(_cfg(bot, "STATUS_INTERVAL_SEC", _STATUS_INTERVAL_SEC), _STATUS_INTERVAL_SEC))

        if now - last_status < interval:
            return

        rc["_last_status_ts"] = now

        await log_system_status(bot)

    except Exception as e:
        log_core.error(f"[system_status] status_tick failed: {e}")
