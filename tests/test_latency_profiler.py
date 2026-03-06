from __future__ import annotations

import time

from core.latency_profiler import LatencyProfiler


def test_latency_profiler_collects_and_resets() -> None:
    p = LatencyProfiler()
    with p.timer("a_ms"):
        time.sleep(0.001)
    p.add("a_ms", 2.0)
    snap = p.snapshot()
    assert "a_ms" in snap
    assert snap["a_ms"]["count"] >= 2
    assert snap["a_ms"]["mean_ms"] > 0.0
    p.reset()
    assert p.snapshot() == {}

