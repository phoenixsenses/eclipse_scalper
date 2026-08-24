# execution/shared_locks.py -- SCALPER ETERNAL -- SHARED LOCK COORDINATOR -- 2026 v1.0
# Per-symbol asyncio.Lock manager with LRU eviction.
# Thread-safe cache access; guardian-safe (never raises).

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_LOCKS = 500


# ---------------------------------------------------------------------------
# Lock cache (OrderedDict for LRU eviction)
# ---------------------------------------------------------------------------
_lock_cache: OrderedDict[str, asyncio.Lock] = OrderedDict()
_cache_mutex = threading.Lock()


def _normalize(symbol: str) -> str:
    """Minimal symbol normalization (consistent with symkey pattern)."""
    try:
        s = (symbol or "").upper().strip()
        s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
        s = s.replace(":USDT", "USDT").replace(":", "").replace("/", "")
        if s.endswith("USDTUSDT"):
            s = s[:-4]
        return s
    except Exception:
        return str(symbol or "").upper().strip()


def get_symbol_lock(symbol: str) -> asyncio.Lock:
    """Return the asyncio.Lock for *symbol*, creating one if needed.

    Same normalized symbol always returns the same lock instance.
    Cache is capped at MAX_LOCKS; least-recently-used entries are evicted.

    Thread-safe: the internal dict is guarded by a threading.Lock so
    callers from different threads (e.g. signal handlers) are safe.
    """
    key = _normalize(symbol)
    with _cache_mutex:
        if key in _lock_cache:
            # Move to end (most-recently used)
            _lock_cache.move_to_end(key)
            return _lock_cache[key]
        # Create new lock
        lock = asyncio.Lock()
        _lock_cache[key] = lock
        # Evict LRU if over cap
        while len(_lock_cache) > MAX_LOCKS:
            _lock_cache.popitem(last=False)
        return lock


def get_all_locked_symbols() -> List[str]:
    """Return list of symbols whose lock is currently acquired.

    For diagnostics only; the result is a snapshot and may be stale.
    """
    with _cache_mutex:
        return [sym for sym, lock in _lock_cache.items() if lock.locked()]


def lock_count() -> int:
    """Return number of cached locks."""
    with _cache_mutex:
        return len(_lock_cache)
