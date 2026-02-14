#!/usr/bin/env python3
"""
Unit tests for execution/shared_locks.py — process-wide per-symbol asyncio locks
used to serialize protective-order placement between reconcile and position_manager.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "eclipse_scalper"
for p in (ROOT, PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eclipse_scalper.execution.shared_locks import get_symbol_lock  # noqa: E402


class TestSharedLocks(unittest.TestCase):
    """Validate the shared symbol lock serialization contract."""

    def test_same_key_returns_same_lock(self):
        lk1 = get_symbol_lock("BTCUSDT")
        lk2 = get_symbol_lock("BTCUSDT")
        self.assertIs(lk1, lk2)

    def test_different_key_returns_different_lock(self):
        lk1 = get_symbol_lock("BTCUSDT")
        lk2 = get_symbol_lock("ETHUSDT")
        self.assertIsNot(lk1, lk2)

    def test_canonical_variants_converge(self):
        """BTC/USDT:USDT, BTC/USDT, and BTCUSDT all map to the same lock."""
        lk1 = get_symbol_lock("BTC/USDT:USDT")
        lk2 = get_symbol_lock("BTC/USDT")
        lk3 = get_symbol_lock("BTCUSDT")
        self.assertIs(lk1, lk2)
        self.assertIs(lk2, lk3)

    def test_concurrent_tasks_are_serialized(self):
        """
        Two async tasks racing to enter the same symbol lock must execute the
        critical section one at a time.  We verify by checking that a shared
        counter never exceeds 1 inside the lock.
        """
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_serialization_test())
        finally:
            loop.close()

    @staticmethod
    async def _run_serialization_test():
        lock = get_symbol_lock("SERIALIZATION_TEST_SYM")
        inside_count = 0
        max_inside = 0
        barrier = asyncio.Event()

        async def worker(worker_id: int):
            nonlocal inside_count, max_inside
            # Both workers start at roughly the same time
            if worker_id == 0:
                barrier.set()
            else:
                await barrier.wait()

            async with lock:
                inside_count += 1
                current = inside_count
                if current > max_inside:
                    max_inside = current
                # Yield control so the other task has a chance to try acquiring
                await asyncio.sleep(0.01)
                inside_count -= 1

        t1 = asyncio.create_task(worker(0))
        t2 = asyncio.create_task(worker(1))
        await asyncio.gather(t1, t2)

        # If the lock works, max_inside must be exactly 1 (never 2).
        assert max_inside == 1, f"Expected max 1 task inside critical section, got {max_inside}"

    def test_lock_does_not_deadlock_sequential_acquire(self):
        """Acquiring different symbol locks sequentially must not deadlock."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_sequential_test())
        finally:
            loop.close()

    @staticmethod
    async def _run_sequential_test():
        async with get_symbol_lock("SYM_A"):
            pass  # released
        async with get_symbol_lock("SYM_B"):
            pass  # released
        # If we reach here, no deadlock occurred.


if __name__ == "__main__":
    unittest.main()
