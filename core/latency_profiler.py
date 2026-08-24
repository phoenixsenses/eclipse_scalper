from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator


@dataclass
class _Agg:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, value_ms: float) -> None:
        v = float(max(0.0, value_ms))
        self.count += 1
        self.total_ms += v
        if v > self.max_ms:
            self.max_ms = v

    @property
    def mean_ms(self) -> float:
        return float(self.total_ms / self.count) if self.count > 0 else 0.0


class LatencyProfiler:
    """Small aggregation helper for periodic latency summaries."""

    def __init__(self) -> None:
        self._agg: Dict[str, _Agg] = {}

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.add(str(key), dt_ms)

    def add(self, key: str, value_ms: float) -> None:
        k = str(key)
        a = self._agg.get(k)
        if a is None:
            a = _Agg()
            self._agg[k] = a
        a.add(float(value_ms))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, a in self._agg.items():
            out[k] = {
                "count": float(a.count),
                "mean_ms": float(a.mean_ms),
                "max_ms": float(a.max_ms),
                "total_ms": float(a.total_ms),
            }
        return out

    def reset(self) -> None:
        self._agg.clear()

