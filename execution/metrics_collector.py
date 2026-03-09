# execution/metrics_collector.py — SCALPER ETERNAL — METRICS COLLECTOR — 2026 v1.0
# Centralized metrics collection from all bot components.
#
# Features:
# - Counter, gauge, histogram metric types
# - Time-windowed aggregations
# - Export to various formats (dict, prometheus, json)
# - Integration with all execution modules
#
# Design principles:
# - Lock-free where possible
# - Minimal memory footprint
# - Easy integration with monitoring systems

from __future__ import annotations

import asyncio
import time
import os
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock

from utils.logging import log_core


# Configuration
_METRICS_ENABLED = os.getenv("METRICS_ENABLED", "1").lower() in ("1", "true", "yes", "on")
_METRICS_WINDOW_SEC = int(os.getenv("METRICS_WINDOW_SEC", "60"))
_METRICS_MAX_SAMPLES = int(os.getenv("METRICS_MAX_SAMPLES", "1000"))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    value: float
    ts: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Monotonically increasing counter metric."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        with self._lock:
            self._value += value

    def get(self) -> float:
        """Get current value."""
        return self._value

    def reset(self) -> float:
        """Reset and return value."""
        with self._lock:
            v = self._value
            self._value = 0.0
            return v


class Gauge:
    """Gauge metric that can go up and down."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get current value."""
        return self._value


class Histogram:
    """Histogram metric for distributions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
        max_samples: int = _METRICS_MAX_SAMPLES,
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.max_samples = max_samples
        self._samples: List[Tuple[float, float]] = []  # (value, timestamp)
        self._lock = Lock()

    def observe(self, value: float) -> None:
        """Record a value."""
        with self._lock:
            self._samples.append((value, time.time()))
            # Trim old samples
            if len(self._samples) > self.max_samples:
                self._samples = self._samples[-self.max_samples:]

    def get_samples(self, window_sec: Optional[float] = None) -> List[float]:
        """Get samples within window."""
        if window_sec is None:
            return [v for v, _ in self._samples]

        cutoff = time.time() - window_sec
        return [v for v, ts in self._samples if ts >= cutoff]

    def get_percentile(self, p: float, window_sec: Optional[float] = None) -> float:
        """Get percentile value."""
        samples = sorted(self.get_samples(window_sec))
        if not samples:
            return 0.0
        idx = int(len(samples) * p / 100)
        return samples[min(idx, len(samples) - 1)]

    def get_stats(self, window_sec: Optional[float] = None) -> Dict[str, float]:
        """Get summary statistics."""
        samples = self.get_samples(window_sec)
        if not samples:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return {
            "count": n,
            "sum": sum(samples),
            "avg": sum(samples) / n,
            "min": sorted_samples[0],
            "max": sorted_samples[-1],
            "p50": sorted_samples[int(n * 0.5)],
            "p95": sorted_samples[min(int(n * 0.95), n - 1)],
            "p99": sorted_samples[min(int(n * 0.99), n - 1)],
        }


class MetricsCollector:
    """
    Centralized metrics collection.

    Usage:
        metrics = MetricsCollector.get()

        # Counters
        metrics.counter("orders_created").inc()
        metrics.counter("orders_failed").inc()

        # Gauges
        metrics.gauge("active_positions").set(5)
        metrics.gauge("equity").set(10000.0)

        # Histograms
        metrics.histogram("order_latency_ms").observe(150.0)

        # Export
        all_metrics = metrics.export()
    """

    _instance: Optional['MetricsCollector'] = None

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = Lock()
        self._start_time = time.time()

    @classmethod
    def get(cls) -> 'MetricsCollector':
        """Get or create metrics collector (singleton)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if metrics collection is enabled."""
        return _METRICS_ENABLED

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create counter."""
        if name not in self._counters:
            with self._lock:
                if name not in self._counters:
                    self._counters[name] = Counter(name, description)
        return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create gauge."""
        if name not in self._gauges:
            with self._lock:
                if name not in self._gauges:
                    self._gauges[name] = Gauge(name, description)
        return self._gauges[name]

    def histogram(self, name: str, description: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        """Get or create histogram."""
        if name not in self._histograms:
            with self._lock:
                if name not in self._histograms:
                    self._histograms[name] = Histogram(name, description, buckets)
        return self._histograms[name]

    def export(self, window_sec: Optional[float] = None) -> Dict[str, Any]:
        """Export all metrics as dict."""
        result = {
            "timestamp": time.time(),
            "uptime_sec": time.time() - self._start_time,
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        for name, counter in self._counters.items():
            result["counters"][name] = counter.get()

        for name, gauge in self._gauges.items():
            result["gauges"][name] = gauge.get()

        for name, histogram in self._histograms.items():
            result["histograms"][name] = histogram.get_stats(window_sec)

        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, counter in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get()}")

        for name, gauge in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get()}")

        for name, histogram in self._histograms.items():
            stats = histogram.get_stats()
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"{name}_sum {stats['sum']}")
            for bucket in histogram.buckets:
                count = sum(1 for v in histogram.get_samples() if v <= bucket)
                lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {stats["count"]}')

        return "\n".join(lines)

    def reset_counters(self) -> Dict[str, float]:
        """Reset all counters and return their values."""
        result = {}
        for name, counter in self._counters.items():
            result[name] = counter.reset()
        return result


# Pre-defined metrics for execution modules
def _init_execution_metrics() -> None:
    """Initialize standard execution metrics."""
    m = MetricsCollector.get()

    # Order metrics
    m.counter("orders_created", "Total orders created")
    m.counter("orders_failed", "Total orders failed")
    m.counter("orders_canceled", "Total orders canceled")
    m.counter("orders_filled", "Total orders filled")

    # Position metrics
    m.gauge("active_positions", "Number of active positions")
    m.gauge("total_exposure_usdt", "Total position exposure in USDT")

    # Latency metrics
    m.histogram("order_latency_ms", "Order creation latency in ms")
    m.histogram("exchange_latency_ms", "Exchange API latency in ms")

    # Error metrics
    m.counter("circuit_breaker_trips", "Circuit breaker trip count")
    m.counter("rate_limit_hits", "Rate limit hit count")
    m.counter("position_lock_timeouts", "Position lock timeout count")

    # Health metrics
    m.gauge("health_score", "Overall health score 0-100")


# Initialize on import
if _METRICS_ENABLED:
    _init_execution_metrics()


# Module-level convenience functions
def inc_counter(name: str, value: float = 1.0) -> None:
    """Increment a counter."""
    if not _METRICS_ENABLED:
        return
    MetricsCollector.get().counter(name).inc(value)


def set_gauge(name: str, value: float) -> None:
    """Set a gauge value."""
    if not _METRICS_ENABLED:
        return
    MetricsCollector.get().gauge(name).set(value)


def observe_histogram(name: str, value: float) -> None:
    """Observe a histogram value."""
    if not _METRICS_ENABLED:
        return
    MetricsCollector.get().histogram(name).observe(value)


def get_metrics() -> Dict[str, Any]:
    """Get all metrics as dict."""
    if not _METRICS_ENABLED:
        return {"enabled": False}
    return MetricsCollector.get().export()


def get_metrics_prometheus() -> str:
    """Get metrics in Prometheus format."""
    if not _METRICS_ENABLED:
        return "# Metrics disabled"
    return MetricsCollector.get().export_prometheus()


async def collect_bot_metrics(bot) -> None:
    """
    Collect metrics from bot state.

    Call this periodically from guardian loop.
    """
    if not _METRICS_ENABLED:
        return

    try:
        m = MetricsCollector.get()

        # Position count
        state = getattr(bot, "state", None)
        if state:
            positions = getattr(state, "positions", None)
            if isinstance(positions, dict):
                m.gauge("active_positions").set(len(positions))

                # Calculate exposure
                total_exposure = 0.0
                for pos in positions.values():
                    size = abs(_safe_float(getattr(pos, "size", 0), 0))
                    entry = _safe_float(getattr(pos, "entry_price", 0), 0)
                    total_exposure += size * entry
                m.gauge("total_exposure_usdt").set(total_exposure)

            # Equity
            equity = _safe_float(getattr(state, "current_equity", 0), 0)
            if equity > 0:
                m.gauge("equity").set(equity)

        # Health score
        try:
            from execution.health_monitor import HealthMonitor
            if HealthMonitor.is_enabled(bot):
                health = HealthMonitor.get(bot).get_status()
                if health:
                    m.gauge("health_score").set(health.score)
        except Exception:
            pass

    except Exception as e:
        log_core.error(f"[metrics_collector] collect_bot_metrics failed: {e}")
