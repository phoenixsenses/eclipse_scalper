"""
Multi-pocket scheduler.

Tracks per-pocket cooldowns and daily trade quotas so that multiple
micro-signal pockets (different imbalance/intensity thresholds) can
run concurrently without interfering with each other.

Usage
-----
    sched = PocketScheduler.from_env()
    ok, reason = sched.can_fire("imb0.50_int3500", time.time())
    if ok:
        sched.record_fire("imb0.50_int3500", time.time())
    # At UTC midnight:
    sched.reset_daily()

ENV
---
    POCKET_SCHEDULER_COOLDOWN_SEC=120    (per-pocket cooldown, default 120s)
    POCKET_SCHEDULER_MAX_DAILY=0         (0 = unlimited per pocket per day)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PocketScheduler:
    """Tracks per-pocket cooldowns and daily quotas."""

    cooldown_sec: float = 120.0
    max_daily_per_pocket: int = 0  # 0 = unlimited

    _last_fire: Dict[str, float] = field(default_factory=dict)
    _daily_count: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "PocketScheduler":
        """Build scheduler from ENV vars."""
        try:
            cooldown = float(os.getenv("POCKET_SCHEDULER_COOLDOWN_SEC", "120") or "120")
        except Exception:
            cooldown = 120.0
        try:
            max_daily = int(os.getenv("POCKET_SCHEDULER_MAX_DAILY", "0") or "0")
        except Exception:
            max_daily = 0
        return cls(cooldown_sec=cooldown, max_daily_per_pocket=max_daily)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def can_fire(self, pocket_name: str, now: float | None = None) -> Tuple[bool, str]:
        """
        Check whether pocket_name is allowed to fire right now.

        Returns (True, "ok") or (False, <reason>).
        """
        if now is None:
            now = time.time()

        pn = pocket_name or "default"

        # Cooldown check
        elapsed = now - self._last_fire.get(pn, 0.0)
        if elapsed < self.cooldown_sec:
            remaining = self.cooldown_sec - elapsed
            return False, f"cooldown {remaining:.0f}s remaining"

        # Daily quota check
        if self.max_daily_per_pocket > 0:
            if self._daily_count.get(pn, 0) >= self.max_daily_per_pocket:
                return False, f"daily quota exhausted ({self.max_daily_per_pocket})"

        return True, "ok"

    def record_fire(self, pocket_name: str, now: float | None = None) -> None:
        """Record that pocket_name fired at `now`."""
        if now is None:
            now = time.time()
        pn = pocket_name or "default"
        self._last_fire[pn] = now
        self._daily_count[pn] = self._daily_count.get(pn, 0) + 1

    def reset_daily(self) -> None:
        """Clear daily counters. Call at UTC midnight."""
        self._daily_count.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def state_dict(self, now: float | None = None) -> dict:
        """Return a snapshot of current scheduler state."""
        if now is None:
            now = time.time()
        pockets = {}
        all_names = set(self._last_fire) | set(self._daily_count)
        for pn in sorted(all_names):
            elapsed = now - self._last_fire.get(pn, 0.0)
            pockets[pn] = {
                "last_fire_ago_sec": round(elapsed, 1),
                "cooldown_remaining_sec": max(0.0, round(self.cooldown_sec - elapsed, 1)),
                "daily_count": self._daily_count.get(pn, 0),
                "can_fire": elapsed >= self.cooldown_sec,
            }
        return {
            "cooldown_sec": self.cooldown_sec,
            "max_daily_per_pocket": self.max_daily_per_pocket,
            "pockets": pockets,
        }
