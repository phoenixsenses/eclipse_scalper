# execution/shared_locks.py — SCALPER ETERNAL — SHARED SYMBOL LOCKS — 2026 v1.0
#
# Process-wide per-symbol asyncio locks used by reconcile and position_manager
# to serialize protective-order placement (stop / TP / trailing) and prevent
# duplicate order races.
#
# Rules:
#   - Never hold more than one symbol lock at a time (deadlock-safe).
#   - Keep lock scope tight: wrap only order placement / cancel-replace sequences.

from __future__ import annotations

import asyncio

from execution.entry_primitives import symkey


_SYMBOL_LOCKS: dict[str, asyncio.Lock] = {}


def get_symbol_lock(k: str) -> asyncio.Lock:
    """
    Return the process-wide asyncio.Lock for *canonical* symbol key ``k``.

    The key is canonicalized through ``symkey()`` so that callers using
    different symbol formats (e.g. ``BTC/USDT:USDT`` vs ``BTCUSDT``) converge
    on the same lock instance.
    """
    canon = symkey(k)
    lk = _SYMBOL_LOCKS.get(canon)
    if lk is None:
        lk = asyncio.Lock()
        _SYMBOL_LOCKS[canon] = lk
    return lk
