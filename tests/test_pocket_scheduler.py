"""Tests for execution.pocket_scheduler."""
import time

import pytest

from execution.pocket_scheduler import PocketScheduler


class TestPocketScheduler:
    def _sched(self, cooldown=10.0, max_daily=0):
        return PocketScheduler(cooldown_sec=cooldown, max_daily_per_pocket=max_daily)

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    def test_can_fire_fresh_pocket(self):
        s = self._sched()
        ok, reason = s.can_fire("p1", now=1000.0)
        assert ok
        assert reason == "ok"

    def test_blocked_within_cooldown(self):
        s = self._sched(cooldown=10.0)
        s.record_fire("p1", now=1000.0)
        ok, reason = s.can_fire("p1", now=1005.0)
        assert not ok
        assert "cooldown" in reason

    def test_allowed_after_cooldown(self):
        s = self._sched(cooldown=10.0)
        s.record_fire("p1", now=1000.0)
        ok, reason = s.can_fire("p1", now=1010.1)
        assert ok

    def test_cooldown_does_not_affect_other_pocket(self):
        s = self._sched(cooldown=10.0)
        s.record_fire("p1", now=1000.0)
        ok, reason = s.can_fire("p2", now=1000.5)
        assert ok

    # ------------------------------------------------------------------
    # Daily quota
    # ------------------------------------------------------------------
    def test_daily_quota_allows_within_limit(self):
        s = self._sched(cooldown=0.0, max_daily=3)
        for i in range(3):
            ok, _ = s.can_fire("p1", now=float(i * 100))
            assert ok
            s.record_fire("p1", now=float(i * 100))

    def test_daily_quota_blocks_after_limit(self):
        s = self._sched(cooldown=0.0, max_daily=2)
        s.record_fire("p1", now=100.0)
        s.record_fire("p1", now=200.0)
        ok, reason = s.can_fire("p1", now=300.0)
        assert not ok
        assert "quota" in reason

    def test_daily_quota_zero_means_unlimited(self):
        s = self._sched(cooldown=0.0, max_daily=0)
        for i in range(100):
            s.record_fire("p1", now=float(i))
        ok, _ = s.can_fire("p1", now=200.0)
        assert ok

    # ------------------------------------------------------------------
    # reset_daily
    # ------------------------------------------------------------------
    def test_reset_daily_clears_counts(self):
        s = self._sched(cooldown=0.0, max_daily=1)
        s.record_fire("p1", now=100.0)
        ok, _ = s.can_fire("p1", now=200.0)
        assert not ok
        s.reset_daily()
        ok, _ = s.can_fire("p1", now=200.0)
        assert ok

    def test_reset_daily_does_not_clear_cooldown(self):
        s = self._sched(cooldown=10.0, max_daily=1)
        s.record_fire("p1", now=1000.0)
        s.reset_daily()
        ok, reason = s.can_fire("p1", now=1005.0)
        assert not ok
        assert "cooldown" in reason

    # ------------------------------------------------------------------
    # state_dict
    # ------------------------------------------------------------------
    def test_state_dict_keys(self):
        s = self._sched()
        s.record_fire("p1", now=1000.0)
        d = s.state_dict(now=1005.0)
        assert "cooldown_sec" in d
        assert "pockets" in d
        p1 = d["pockets"]["p1"]
        assert p1["last_fire_ago_sec"] == pytest.approx(5.0)
        assert not p1["can_fire"]

    # ------------------------------------------------------------------
    # from_env
    # ------------------------------------------------------------------
    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("POCKET_SCHEDULER_COOLDOWN_SEC", raising=False)
        monkeypatch.delenv("POCKET_SCHEDULER_MAX_DAILY", raising=False)
        s = PocketScheduler.from_env()
        assert s.cooldown_sec == 120.0
        assert s.max_daily_per_pocket == 0

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("POCKET_SCHEDULER_COOLDOWN_SEC", "60")
        monkeypatch.setenv("POCKET_SCHEDULER_MAX_DAILY", "5")
        s = PocketScheduler.from_env()
        assert s.cooldown_sec == 60.0
        assert s.max_daily_per_pocket == 5
