from __future__ import annotations

import math
import statistics
from typing import Callable

FEATURE_REGISTRY: dict[str, Callable[[list[dict]], float]] = {}


def register_feature(name: str):
    """Register a deterministic candle feature function."""

    def _decorator(fn: Callable[[list[dict]], float]):
        FEATURE_REGISTRY[str(name)] = fn
        return fn

    return _decorator


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _close_at(candles: list[dict], idx_from_end: int = 0) -> float:
    if not candles:
        return 0.0
    i = len(candles) - 1 - idx_from_end
    if i < 0 or i >= len(candles):
        return 0.0
    return _to_float((candles[i] or {}).get("close"), 0.0)


def _safe_return(curr: float, prev: float) -> float:
    if prev == 0.0:
        return 0.0
    return (curr / prev) - 1.0


def _recent_returns(candles: list[dict], periods: int) -> list[float]:
    if len(candles) < periods + 1:
        return []
    out: list[float] = []
    start = len(candles) - periods
    for i in range(start, len(candles)):
        c = _to_float((candles[i] or {}).get("close"), 0.0)
        p = _to_float((candles[i - 1] or {}).get("close"), 0.0)
        out.append(_safe_return(c, p))
    return out


@register_feature("returns_1")
def feat_returns_1(candles: list[dict]) -> float:
    if len(candles) < 2:
        return 0.0
    return _safe_return(_close_at(candles, 0), _close_at(candles, 1))


@register_feature("returns_5")
def feat_returns_5(candles: list[dict]) -> float:
    if len(candles) < 6:
        return 0.0
    return _safe_return(_close_at(candles, 0), _close_at(candles, 5))


@register_feature("returns_20")
def feat_returns_20(candles: list[dict]) -> float:
    if len(candles) < 21:
        return 0.0
    return _safe_return(_close_at(candles, 0), _close_at(candles, 20))


@register_feature("volatility_10")
def feat_volatility_10(candles: list[dict]) -> float:
    r = _recent_returns(candles, 10)
    if len(r) < 2:
        return 0.0
    try:
        return float(statistics.pstdev(r))
    except Exception:
        return 0.0


@register_feature("volatility_50")
def feat_volatility_50(candles: list[dict]) -> float:
    r = _recent_returns(candles, 50)
    if len(r) < 2:
        return 0.0
    try:
        return float(statistics.pstdev(r))
    except Exception:
        return 0.0


@register_feature("high_low_range")
def feat_high_low_range(candles: list[dict]) -> float:
    if not candles:
        return 0.0
    c = candles[-1] or {}
    hi = _to_float(c.get("high"), 0.0)
    lo = _to_float(c.get("low"), 0.0)
    close = _to_float(c.get("close"), 0.0)
    if close == 0.0:
        return 0.0
    return (hi - lo) / close


@register_feature("volume_zscore_20")
def feat_volume_zscore_20(candles: list[dict]) -> float:
    if len(candles) < 20:
        return 0.0
    vols = [_to_float((c or {}).get("volume"), 0.0) for c in candles[-20:]]
    try:
        mu = float(statistics.fmean(vols))
        sd = float(statistics.pstdev(vols))
    except Exception:
        return 0.0
    if sd == 0.0 or math.isnan(sd):
        return 0.0
    return (vols[-1] - mu) / sd


def compute_features(candles: list[dict]) -> dict[str, float]:
    """
    Compute all registered features.
    Never raises for insufficient/partial history; falls back to safe defaults.
    """
    out: dict[str, float] = {}
    for name in sorted(FEATURE_REGISTRY.keys()):
        fn = FEATURE_REGISTRY[name]
        try:
            out[name] = float(fn(candles))
        except Exception:
            out[name] = 0.0
    return out

