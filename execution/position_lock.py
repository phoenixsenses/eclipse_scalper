# execution/position_lock.py — SCALPER ETERNAL — POSITION LOCK MANAGER — 2026 v1.0
# Provides atomic position dictionary operations to prevent race conditions
# between entry/exit/reconcile/position_manager.
#
# Design principles:
# - Single global asyncio.Lock per bot (not per-symbol for simplicity)
# - Fail-closed for entries (safety first), fail-open for exits (must close)
# - Short critical sections only (no I/O under lock)
# - Timeout with graceful degradation

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Dict, Optional, TypeVar
from weakref import WeakValueDictionary

from utils.logging import log_core

T = TypeVar('T')

# Configuration
_POSITION_LOCK_TIMEOUT_SEC = float(os.getenv("POSITION_LOCK_TIMEOUT_SEC", "5.0"))
_POSITION_LOCK_ENABLED = os.getenv("POSITION_LOCK_ENABLED", "1").lower() in ("1", "true", "yes", "on")


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


def _truthy(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on", "t")
    return False


class PositionLockManager:
    """
    Thread-safe position dictionary manager.

    Provides atomic operations for position dict mutations to prevent race conditions
    between concurrent loops (entry, exit, reconcile, position_manager).

    Usage:
        lock_mgr = PositionLockManager.get(bot)

        # Atomic insert (entries)
        success = await lock_mgr.atomic_insert(bot.state, "BTCUSDT", position_obj)

        # Atomic remove (exits)
        removed = await lock_mgr.atomic_remove(bot.state, "BTCUSDT")

        # Atomic update (position_manager)
        async def update_stop(pos):
            pos.hard_stop_order_id = new_id
            return pos
        success = await lock_mgr.atomic_update(bot.state, "BTCUSDT", update_stop)

        # Generic locked operation
        result = await lock_mgr.with_positions_locked(
            bot.state,
            lambda positions: positions.get("BTCUSDT"),
            timeout=2.0,
        )
    """

    # Weak reference cache to prevent memory leaks
    _instances: WeakValueDictionary[int, 'PositionLockManager'] = WeakValueDictionary()

    def __init__(self, bot):
        self._lock = asyncio.Lock()
        self._version = 0
        self._bot_id = id(bot)
        self._degraded = False
        self._degraded_since = 0.0
        self._lock_acquisitions = 0
        self._lock_timeouts = 0
        self._last_contention_ts = 0.0

    @classmethod
    def get(cls, bot) -> 'PositionLockManager':
        """Get or create lock manager for bot (singleton per bot instance)."""
        bot_id = id(bot)
        if bot_id in cls._instances:
            return cls._instances[bot_id]

        instance = cls(bot)
        cls._instances[bot_id] = instance
        return instance

    @classmethod
    def is_enabled(cls, bot) -> bool:
        """Check if position locking is enabled."""
        if not _POSITION_LOCK_ENABLED:
            return False
        return _truthy(_cfg(bot, "POSITION_LOCK_ENABLED", True))

    async def with_positions_locked(
        self,
        state,
        operation: Callable[[Dict[str, Any]], T],
        *,
        timeout: Optional[float] = None,
        is_exit: bool = False,
        symbol: str = "",
    ) -> Optional[T]:
        """
        Execute operation with positions dict locked.

        Args:
            state: Bot state object with positions dict
            operation: Callable that receives positions dict and returns result
            timeout: Lock acquisition timeout (default from config)
            is_exit: If True, proceed even on lock timeout (fail-open for exits)
            symbol: Symbol for logging (optional)

        Returns:
            Result of operation, or None on failure

        Behavior:
            - Entries (is_exit=False): Fail-closed on timeout (returns None)
            - Exits (is_exit=True): Fail-open on timeout (proceeds unsafely)
        """
        if timeout is None:
            timeout = _POSITION_LOCK_TIMEOUT_SEC

        try:
            # Try to acquire lock with timeout
            acquired = False
            try:
                async with asyncio.timeout(timeout):
                    await self._lock.acquire()
                    acquired = True
            except asyncio.TimeoutError:
                self._lock_timeouts += 1
                self._last_contention_ts = time.time()

                if is_exit:
                    # Fail-open for exits - must be able to close positions
                    log_core.warning(
                        f"[position_lock] TIMEOUT (exit, {timeout:.1f}s) - proceeding unsafely | "
                        f"sym={symbol} timeouts={self._lock_timeouts}"
                    )
                    positions = getattr(state, "positions", None)
                    if not isinstance(positions, dict):
                        positions = {}
                    return operation(positions)
                else:
                    # Fail-closed for entries - safety first
                    log_core.error(
                        f"[position_lock] TIMEOUT (entry, {timeout:.1f}s) - BLOCKING | "
                        f"sym={symbol} timeouts={self._lock_timeouts}"
                    )
                    if not self._degraded:
                        self._degraded = True
                        self._degraded_since = time.time()
                    return None

            if acquired:
                try:
                    self._lock_acquisitions += 1

                    # Ensure positions dict exists
                    positions = getattr(state, "positions", None)
                    if not isinstance(positions, dict):
                        positions = {}
                        try:
                            state.positions = positions
                        except Exception:
                            pass

                    result = operation(positions)
                    self._version += 1

                    # Clear degraded state on successful operation
                    if self._degraded:
                        self._degraded = False
                        log_core.info("[position_lock] Recovered from degraded state")

                    return result

                finally:
                    self._lock.release()

        except Exception as e:
            log_core.error(f"[position_lock] Operation failed: {e} | sym={symbol}")
            return None

    async def atomic_insert(
        self,
        state,
        key: str,
        value: Any,
        *,
        timeout: Optional[float] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Atomically insert position if not exists (or overwrite if specified).

        Args:
            state: Bot state object
            key: Position key (symbol)
            value: Position object to insert
            timeout: Lock acquisition timeout
            overwrite: If True, overwrite existing position

        Returns:
            True if inserted, False if key exists (and overwrite=False) or lock failed
        """
        def _insert(positions: dict) -> bool:
            if key in positions and not overwrite:
                return False
            positions[key] = value
            return True

        result = await self.with_positions_locked(
            state, _insert, timeout=timeout, symbol=key
        )

        if result is True:
            log_core.debug(f"[position_lock] INSERT k={key}")

        return result is True

    async def atomic_remove(
        self,
        state,
        key: str,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Atomically remove and return position.

        Args:
            state: Bot state object
            key: Position key (symbol)
            timeout: Lock acquisition timeout

        Returns:
            Removed position object, or None if not found/lock failed

        Note: This is fail-open (is_exit=True) - exits must succeed.
        """
        def _remove(positions: dict) -> Any:
            return positions.pop(key, None)

        result = await self.with_positions_locked(
            state, _remove, timeout=timeout, is_exit=True, symbol=key
        )

        if result is not None:
            log_core.debug(f"[position_lock] REMOVE k={key}")

        return result

    async def atomic_update(
        self,
        state,
        key: str,
        updater: Callable[[Any], Any],
        *,
        timeout: Optional[float] = None,
        is_exit: bool = False,
    ) -> bool:
        """
        Atomically update position with updater function.

        Args:
            state: Bot state object
            key: Position key (symbol)
            updater: Function that takes current position and returns updated position
            timeout: Lock acquisition timeout
            is_exit: If True, fail-open on timeout

        Returns:
            True if updated, False if key not found or lock failed

        Example:
            async def update_stop(pos):
                pos.hard_stop_order_id = new_id
                return pos
            await lock_mgr.atomic_update(state, "BTCUSDT", update_stop)
        """
        def _update(positions: dict) -> bool:
            if key not in positions:
                return False
            positions[key] = updater(positions[key])
            return True

        result = await self.with_positions_locked(
            state, _update, timeout=timeout, is_exit=is_exit, symbol=key
        )
        return result is True

    async def atomic_get(
        self,
        state,
        key: str,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Atomically get position by key.

        Args:
            state: Bot state object
            key: Position key (symbol)
            timeout: Lock acquisition timeout

        Returns:
            Position object or None
        """
        def _get(positions: dict) -> Any:
            return positions.get(key)

        return await self.with_positions_locked(
            state, _get, timeout=timeout, is_exit=True, symbol=key
        )

    async def atomic_keys(
        self,
        state,
        *,
        timeout: Optional[float] = None,
    ) -> list:
        """
        Atomically get list of position keys.

        Args:
            state: Bot state object
            timeout: Lock acquisition timeout

        Returns:
            List of position keys
        """
        def _keys(positions: dict) -> list:
            return list(positions.keys())

        result = await self.with_positions_locked(
            state, _keys, timeout=timeout, is_exit=True
        )
        return result if result is not None else []

    async def atomic_items_snapshot(
        self,
        state,
        *,
        timeout: Optional[float] = None,
    ) -> list:
        """
        Atomically get snapshot of all (key, value) pairs.

        Args:
            state: Bot state object
            timeout: Lock acquisition timeout

        Returns:
            List of (key, position) tuples
        """
        def _items(positions: dict) -> list:
            return list(positions.items())

        result = await self.with_positions_locked(
            state, _items, timeout=timeout, is_exit=True
        )
        return result if result is not None else []

    @property
    def version(self) -> int:
        """Current version number (incremented on each mutation)."""
        return self._version

    @property
    def is_degraded(self) -> bool:
        """True if lock manager is in degraded state (recent timeout failures)."""
        return self._degraded

    @property
    def stats(self) -> dict:
        """Get lock statistics for monitoring."""
        return {
            "version": self._version,
            "acquisitions": self._lock_acquisitions,
            "timeouts": self._lock_timeouts,
            "degraded": self._degraded,
            "degraded_since": self._degraded_since,
            "last_contention_ts": self._last_contention_ts,
            "lock_held": self._lock.locked(),
        }


# Module-level convenience functions
async def atomic_position_insert(bot, key: str, value: Any, **kwargs) -> bool:
    """Insert position atomically."""
    if not PositionLockManager.is_enabled(bot):
        positions = getattr(bot.state, "positions", None) or {}
        if key in positions:
            return False
        positions[key] = value
        return True
    return await PositionLockManager.get(bot).atomic_insert(bot.state, key, value, **kwargs)


async def atomic_position_remove(bot, key: str, **kwargs) -> Any:
    """Remove position atomically."""
    if not PositionLockManager.is_enabled(bot):
        positions = getattr(bot.state, "positions", None) or {}
        return positions.pop(key, None)
    return await PositionLockManager.get(bot).atomic_remove(bot.state, key, **kwargs)


async def atomic_position_update(bot, key: str, updater: Callable, **kwargs) -> bool:
    """Update position atomically."""
    if not PositionLockManager.is_enabled(bot):
        positions = getattr(bot.state, "positions", None) or {}
        if key not in positions:
            return False
        positions[key] = updater(positions[key])
        return True
    return await PositionLockManager.get(bot).atomic_update(bot.state, key, updater, **kwargs)
