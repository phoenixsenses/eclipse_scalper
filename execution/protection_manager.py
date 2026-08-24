# execution/protection_manager.py — SCALPER ETERNAL — PROTECTION MANAGER — 2026 v1.0
# Manages stop-loss and take-profit coverage for positions.
#
# Features:
# - Assess stop coverage ratio per position
# - Assess TP coverage ratio per position
# - Decide when to refresh protection orders
# - Track protection order health
#
# Design principles:
# - Never block exits
# - Fail-safe: assume uncovered if check fails
# - Throttle refresh to prevent spam

from __future__ import annotations

import asyncio
import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from utils.logging import log_core


# Configuration
_PROTECTION_ENABLED = os.getenv("PROTECTION_MANAGER_ENABLED", "1").lower() in ("1", "true", "yes", "on")
_STOP_MIN_COVERAGE_RATIO = float(os.getenv("RECONCILE_STOP_MIN_COVERAGE_RATIO", "0.98"))
_TP_MIN_COVERAGE_RATIO = float(os.getenv("RECONCILE_TP_MIN_COVERAGE_RATIO", "0.98"))
_REFRESH_MIN_DELTA_RATIO = float(os.getenv("RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO", "0.10"))
_REFRESH_MAX_INTERVAL_SEC = float(os.getenv("RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC", "45"))
_REFRESH_THROTTLE_SEC = float(os.getenv("PROTECTION_REFRESH_THROTTLE_SEC", "60"))


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


class CoverageStatus(Enum):
    FULL = "FULL"           # >= min coverage ratio
    PARTIAL = "PARTIAL"     # > 0 but < min coverage ratio
    NONE = "NONE"           # No protection order found
    UNKNOWN = "UNKNOWN"     # Check failed


@dataclass
class CoverageResult:
    """Result of coverage assessment."""
    status: CoverageStatus
    coverage_ratio: float
    position_size: float
    protected_size: float
    order_id: Optional[str] = None
    order_price: Optional[float] = None
    reason: str = ""


@dataclass
class RefreshDecision:
    """Decision on whether to refresh protection."""
    should_refresh: bool
    reason: str
    priority: str = "normal"  # normal, high, critical
    new_price: Optional[float] = None
    new_size: Optional[float] = None


@dataclass
class ProtectionState:
    """Per-symbol protection state."""
    symbol: str
    stop_coverage: CoverageResult = field(default_factory=lambda: CoverageResult(
        status=CoverageStatus.UNKNOWN, coverage_ratio=0.0,
        position_size=0.0, protected_size=0.0
    ))
    tp_coverage: CoverageResult = field(default_factory=lambda: CoverageResult(
        status=CoverageStatus.UNKNOWN, coverage_ratio=0.0,
        position_size=0.0, protected_size=0.0
    ))
    last_stop_refresh_ts: float = 0.0
    last_tp_refresh_ts: float = 0.0
    stop_refresh_count: int = 0
    tp_refresh_count: int = 0


class ProtectionManager:
    """
    Manages protection orders (stop-loss, take-profit) for positions.

    Usage:
        pm = ProtectionManager.get(bot)

        # Assess coverage
        stop_cov = await pm.assess_stop_coverage(symbol, position, open_orders)
        tp_cov = await pm.assess_tp_coverage(symbol, position, open_orders)

        # Decide if refresh needed
        decision = pm.should_refresh_protection(symbol, "stop", stop_cov, current_price)
    """

    _instance: Optional['ProtectionManager'] = None

    def __init__(self, bot):
        self.bot = bot
        self._states: Dict[str, ProtectionState] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls, bot) -> 'ProtectionManager':
        """Get or create protection manager (singleton)."""
        if cls._instance is None:
            cls._instance = cls(bot)
        return cls._instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if protection manager is enabled."""
        if not _PROTECTION_ENABLED:
            return False
        try:
            cfg = getattr(bot, "cfg", None)
            if cfg is not None:
                v = getattr(cfg, "PROTECTION_MANAGER_ENABLED", None)
                if v is not None:
                    if isinstance(v, bool):
                        return v
                    return str(v).lower() in ("1", "true", "yes", "on")
        except Exception:
            pass
        return _PROTECTION_ENABLED

    def _get_state(self, symbol: str) -> ProtectionState:
        """Get or create protection state for symbol."""
        if symbol not in self._states:
            self._states[symbol] = ProtectionState(symbol=symbol)
        return self._states[symbol]

    async def assess_stop_coverage(
        self,
        symbol: str,
        position: Any,
        open_orders: List[Dict[str, Any]],
    ) -> CoverageResult:
        """
        Assess stop-loss coverage for a position.

        Args:
            symbol: Trading symbol
            position: Position object with size, side, entry_price
            open_orders: List of open orders from exchange

        Returns:
            CoverageResult with coverage status and details
        """
        try:
            pos_size = abs(_safe_float(getattr(position, "size", 0), 0))
            pos_side = str(getattr(position, "side", "")).lower()

            if pos_size <= 0:
                return CoverageResult(
                    status=CoverageStatus.NONE,
                    coverage_ratio=0.0,
                    position_size=0.0,
                    protected_size=0.0,
                    reason="No position"
                )

            # Find stop orders
            stop_orders = []
            for order in open_orders:
                order_type = str(order.get("type", "")).upper()
                order_side = str(order.get("side", "")).lower()
                order_status = str(order.get("status", "")).lower()

                # Stop orders close the position (opposite side)
                is_stop = "STOP" in order_type
                is_closing = (
                    (pos_side == "long" and order_side == "sell") or
                    (pos_side == "short" and order_side == "buy")
                )
                is_open = order_status in ("open", "new", "partially_filled")

                if is_stop and is_closing and is_open:
                    stop_orders.append(order)

            if not stop_orders:
                result = CoverageResult(
                    status=CoverageStatus.NONE,
                    coverage_ratio=0.0,
                    position_size=pos_size,
                    protected_size=0.0,
                    reason="No stop orders found"
                )
            else:
                # Sum protected size from all stop orders
                protected_size = sum(
                    _safe_float(o.get("amount", 0), 0) or _safe_float(o.get("remaining", 0), 0)
                    for o in stop_orders
                )

                coverage_ratio = protected_size / pos_size if pos_size > 0 else 0.0
                min_ratio = _safe_float(_cfg(self.bot, "RECONCILE_STOP_MIN_COVERAGE_RATIO", _STOP_MIN_COVERAGE_RATIO), _STOP_MIN_COVERAGE_RATIO)

                if coverage_ratio >= min_ratio:
                    status = CoverageStatus.FULL
                elif coverage_ratio > 0:
                    status = CoverageStatus.PARTIAL
                else:
                    status = CoverageStatus.NONE

                # Get primary stop order details
                primary_order = stop_orders[0]

                result = CoverageResult(
                    status=status,
                    coverage_ratio=coverage_ratio,
                    position_size=pos_size,
                    protected_size=protected_size,
                    order_id=str(primary_order.get("id", "")),
                    order_price=_safe_float(primary_order.get("stopPrice", 0) or primary_order.get("price", 0), None),
                    reason=f"{len(stop_orders)} stop order(s), {coverage_ratio:.1%} coverage"
                )

            # Update state
            state = self._get_state(symbol)
            state.stop_coverage = result

            return result

        except Exception as e:
            log_core.error(f"[protection_manager] assess_stop_coverage failed: {e}")
            return CoverageResult(
                status=CoverageStatus.UNKNOWN,
                coverage_ratio=0.0,
                position_size=0.0,
                protected_size=0.0,
                reason=f"Error: {e}"
            )

    async def assess_tp_coverage(
        self,
        symbol: str,
        position: Any,
        open_orders: List[Dict[str, Any]],
    ) -> CoverageResult:
        """
        Assess take-profit coverage for a position.

        Args:
            symbol: Trading symbol
            position: Position object
            open_orders: List of open orders from exchange

        Returns:
            CoverageResult with coverage status and details
        """
        try:
            pos_size = abs(_safe_float(getattr(position, "size", 0), 0))
            pos_side = str(getattr(position, "side", "")).lower()

            if pos_size <= 0:
                return CoverageResult(
                    status=CoverageStatus.NONE,
                    coverage_ratio=0.0,
                    position_size=0.0,
                    protected_size=0.0,
                    reason="No position"
                )

            # Find TP orders (TAKE_PROFIT or LIMIT orders that close position)
            tp_orders = []
            for order in open_orders:
                order_type = str(order.get("type", "")).upper()
                order_side = str(order.get("side", "")).lower()
                order_status = str(order.get("status", "")).lower()
                reduce_only = bool(order.get("reduceOnly", False))

                is_tp = "TAKE_PROFIT" in order_type or (order_type == "LIMIT" and reduce_only)
                is_closing = (
                    (pos_side == "long" and order_side == "sell") or
                    (pos_side == "short" and order_side == "buy")
                )
                is_open = order_status in ("open", "new", "partially_filled")

                if is_tp and is_closing and is_open:
                    tp_orders.append(order)

            if not tp_orders:
                result = CoverageResult(
                    status=CoverageStatus.NONE,
                    coverage_ratio=0.0,
                    position_size=pos_size,
                    protected_size=0.0,
                    reason="No TP orders found"
                )
            else:
                protected_size = sum(
                    _safe_float(o.get("amount", 0), 0) or _safe_float(o.get("remaining", 0), 0)
                    for o in tp_orders
                )

                coverage_ratio = protected_size / pos_size if pos_size > 0 else 0.0
                min_ratio = _safe_float(_cfg(self.bot, "RECONCILE_TP_MIN_COVERAGE_RATIO", _TP_MIN_COVERAGE_RATIO), _TP_MIN_COVERAGE_RATIO)

                if coverage_ratio >= min_ratio:
                    status = CoverageStatus.FULL
                elif coverage_ratio > 0:
                    status = CoverageStatus.PARTIAL
                else:
                    status = CoverageStatus.NONE

                primary_order = tp_orders[0]

                result = CoverageResult(
                    status=status,
                    coverage_ratio=coverage_ratio,
                    position_size=pos_size,
                    protected_size=protected_size,
                    order_id=str(primary_order.get("id", "")),
                    order_price=_safe_float(primary_order.get("price", 0), None),
                    reason=f"{len(tp_orders)} TP order(s), {coverage_ratio:.1%} coverage"
                )

            state = self._get_state(symbol)
            state.tp_coverage = result

            return result

        except Exception as e:
            log_core.error(f"[protection_manager] assess_tp_coverage failed: {e}")
            return CoverageResult(
                status=CoverageStatus.UNKNOWN,
                coverage_ratio=0.0,
                position_size=0.0,
                protected_size=0.0,
                reason=f"Error: {e}"
            )

    def should_refresh_protection(
        self,
        symbol: str,
        protection_type: str,  # "stop" or "tp"
        coverage: CoverageResult,
        current_price: Optional[float] = None,
        target_price: Optional[float] = None,
    ) -> RefreshDecision:
        """
        Decide if protection order should be refreshed.

        Args:
            symbol: Trading symbol
            protection_type: "stop" or "tp"
            coverage: Current coverage result
            current_price: Current market price
            target_price: New target price for protection

        Returns:
            RefreshDecision with recommendation
        """
        now = time.time()
        state = self._get_state(symbol)

        # Get last refresh timestamp
        if protection_type == "stop":
            last_refresh = state.last_stop_refresh_ts
        else:
            last_refresh = state.last_tp_refresh_ts

        # Throttle check
        throttle_sec = _safe_float(
            _cfg(self.bot, "PROTECTION_REFRESH_THROTTLE_SEC", _REFRESH_THROTTLE_SEC),
            _REFRESH_THROTTLE_SEC
        )

        if now - last_refresh < throttle_sec:
            return RefreshDecision(
                should_refresh=False,
                reason=f"Throttled ({throttle_sec - (now - last_refresh):.0f}s remaining)"
            )

        # No coverage = definitely refresh
        if coverage.status == CoverageStatus.NONE:
            return RefreshDecision(
                should_refresh=True,
                reason="No protection order exists",
                priority="critical",
                new_size=coverage.position_size
            )

        # Partial coverage = refresh
        if coverage.status == CoverageStatus.PARTIAL:
            gap = coverage.position_size - coverage.protected_size
            return RefreshDecision(
                should_refresh=True,
                reason=f"Partial coverage ({coverage.coverage_ratio:.1%}), gap={gap:.4f}",
                priority="high",
                new_size=coverage.position_size
            )

        # Check price delta if target provided
        if target_price is not None and coverage.order_price is not None:
            delta_ratio = abs(target_price - coverage.order_price) / coverage.order_price if coverage.order_price > 0 else 0
            min_delta = _safe_float(
                _cfg(self.bot, "RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO", _REFRESH_MIN_DELTA_RATIO),
                _REFRESH_MIN_DELTA_RATIO
            )

            if delta_ratio >= min_delta:
                return RefreshDecision(
                    should_refresh=True,
                    reason=f"Price delta {delta_ratio:.1%} >= {min_delta:.1%}",
                    priority="normal",
                    new_price=target_price
                )

        # Max interval check
        max_interval = _safe_float(
            _cfg(self.bot, "RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC", _REFRESH_MAX_INTERVAL_SEC),
            _REFRESH_MAX_INTERVAL_SEC
        )

        if now - last_refresh > max_interval:
            return RefreshDecision(
                should_refresh=True,
                reason=f"Max interval exceeded ({max_interval}s)",
                priority="normal"
            )

        return RefreshDecision(
            should_refresh=False,
            reason="Coverage is adequate"
        )

    def mark_refreshed(self, symbol: str, protection_type: str) -> None:
        """Mark that protection was refreshed (for throttling)."""
        state = self._get_state(symbol)
        now = time.time()

        if protection_type == "stop":
            state.last_stop_refresh_ts = now
            state.stop_refresh_count += 1
        else:
            state.last_tp_refresh_ts = now
            state.tp_refresh_count += 1

    def get_state(self, symbol: str) -> Optional[ProtectionState]:
        """Get protection state for symbol."""
        return self._states.get(symbol)

    def get_all_states(self) -> Dict[str, ProtectionState]:
        """Get all protection states."""
        return dict(self._states)

    def clear_state(self, symbol: str) -> None:
        """Clear protection state for symbol (on position close)."""
        self._states.pop(symbol, None)


# Module-level convenience functions
async def assess_stop_coverage(bot, symbol: str, position: Any, open_orders: List[Dict]) -> CoverageResult:
    """Assess stop-loss coverage for a position."""
    if not ProtectionManager.is_enabled(bot):
        return CoverageResult(
            status=CoverageStatus.UNKNOWN,
            coverage_ratio=0.0,
            position_size=0.0,
            protected_size=0.0,
            reason="Protection manager disabled"
        )
    return await ProtectionManager.get(bot).assess_stop_coverage(symbol, position, open_orders)


async def assess_tp_coverage(bot, symbol: str, position: Any, open_orders: List[Dict]) -> CoverageResult:
    """Assess take-profit coverage for a position."""
    if not ProtectionManager.is_enabled(bot):
        return CoverageResult(
            status=CoverageStatus.UNKNOWN,
            coverage_ratio=0.0,
            position_size=0.0,
            protected_size=0.0,
            reason="Protection manager disabled"
        )
    return await ProtectionManager.get(bot).assess_tp_coverage(symbol, position, open_orders)


def should_refresh_protection(bot, symbol: str, protection_type: str, coverage: CoverageResult, **kwargs) -> RefreshDecision:
    """Decide if protection should be refreshed."""
    if not ProtectionManager.is_enabled(bot):
        return RefreshDecision(should_refresh=False, reason="Protection manager disabled")
    return ProtectionManager.get(bot).should_refresh_protection(symbol, protection_type, coverage, **kwargs)
