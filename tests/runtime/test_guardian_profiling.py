"""Tests for guardian loop step profiling."""
from __future__ import annotations

import asyncio

from execution.guardian import _safe_call, _STEP_TIMINGS, get_step_timings


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStepProfiling:
    def setup_method(self):
        _STEP_TIMINGS.clear()

    def test_safe_call_records_timing(self):
        async def fast_step():
            pass

        _run(_safe_call("test_fast", fast_step))
        assert "test_fast" in _STEP_TIMINGS
        assert len(_STEP_TIMINGS["test_fast"]) == 1
        assert _STEP_TIMINGS["test_fast"][0] >= 0

    def test_multiple_calls_accumulate(self):
        async def step():
            pass

        for _ in range(5):
            _run(_safe_call("multi_step", step))
        assert len(_STEP_TIMINGS["multi_step"]) == 5

    def test_history_capped(self):
        from execution.guardian import _STEP_TIMINGS_MAX_HISTORY

        async def step():
            pass

        for _ in range(_STEP_TIMINGS_MAX_HISTORY + 10):
            _run(_safe_call("capped_step", step))
        assert len(_STEP_TIMINGS["capped_step"]) <= _STEP_TIMINGS_MAX_HISTORY

    def test_failed_step_still_profiled(self):
        async def bad_step():
            raise RuntimeError("boom")

        _run(_safe_call("bad_step", bad_step))
        assert "bad_step" in _STEP_TIMINGS
        assert len(_STEP_TIMINGS["bad_step"]) == 1

    def test_timeout_step_profiled(self):
        async def slow_step():
            await asyncio.sleep(999)

        _run(_safe_call("slow_step", slow_step, timeout_sec=0.05))
        assert "slow_step" in _STEP_TIMINGS
        # Duration should be ~0.05s (the timeout), not 999s
        assert _STEP_TIMINGS["slow_step"][0] < 1.0

    def test_get_step_timings_summary(self):
        async def step():
            await asyncio.sleep(0.01)

        for _ in range(3):
            _run(_safe_call("summary_step", step))
        summary = get_step_timings()
        assert "summary_step" in summary
        s = summary["summary_step"]
        assert "avg_ms" in s
        assert "max_ms" in s
        assert "last_ms" in s
        assert "count" in s
        assert s["count"] == 3.0
        assert s["avg_ms"] > 0
        assert s["max_ms"] >= s["avg_ms"]

    def test_max_steps_capped(self):
        from execution.guardian import _STEP_TIMINGS_MAX_STEPS

        async def step():
            pass

        for i in range(_STEP_TIMINGS_MAX_STEPS + 10):
            _run(_safe_call(f"step_{i}", step))
        assert len(_STEP_TIMINGS) <= _STEP_TIMINGS_MAX_STEPS + 1

    def test_empty_timings_returns_empty(self):
        summary = get_step_timings()
        assert summary == {}
