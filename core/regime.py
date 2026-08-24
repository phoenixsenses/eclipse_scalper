from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class _Point:
    ts: float
    price: float


class RegimeClassifier:
    """Rolling-return regime classifier with optional debounce."""

    def __init__(self, lookback_sec: int = 3600, debounce_sec: int = 60, source: str = "mark_price"):
        self.lookback_sec = max(1, int(lookback_sec))
        self.debounce_sec = max(0, int(debounce_sec))
        self.source = str(source)

        self._lock = threading.Lock()
        self._points: Deque[_Point] = deque()
        self._current_ts: Optional[float] = None

        self._raw_regime: str = "UNKNOWN"
        self._confirmed_regime: str = "UNKNOWN"
        self._pending_regime: Optional[str] = None
        self._pending_since: Optional[float] = None
        self._last_change_ts: Optional[float] = None
        self._rolling_return: float = float("nan")
        self._data_ready: bool = False

    def update(self, timestamp: float, price: float) -> None:
        ts = float(timestamp)
        px = float(price)
        if not math.isfinite(ts) or not math.isfinite(px) or px <= 0.0:
            return
        with self._lock:
            if self._current_ts is not None and ts < self._current_ts:
                # Ignore backward updates to keep monotonic state.
                return
            self._current_ts = ts
            self._points.append(_Point(ts=ts, price=px))
            self._prune_points_locked(ts)
            self._recompute_locked(ts)

    @property
    def current_regime(self) -> str:
        with self._lock:
            if not self._data_ready:
                return "UNKNOWN"
            if self.debounce_sec > 0 and self._pending_regime and self._pending_regime != self._confirmed_regime:
                return "TRANSITION"
            return self._confirmed_regime

    @property
    def regime_age_sec(self) -> float:
        with self._lock:
            if self._current_ts is None or self._last_change_ts is None:
                return 0.0
            return max(0.0, float(self._current_ts - self._last_change_ts))

    @property
    def rolling_return(self) -> float:
        with self._lock:
            return float(self._rolling_return)

    def state_dict(self) -> Dict[str, object]:
        with self._lock:
            if not self._data_ready:
                current_regime = "UNKNOWN"
            elif self.debounce_sec > 0 and self._pending_regime and self._pending_regime != self._confirmed_regime:
                current_regime = "TRANSITION"
            else:
                current_regime = self._confirmed_regime
            if self._current_ts is None or self._last_change_ts is None:
                age = 0.0
            else:
                age = max(0.0, float(self._current_ts - self._last_change_ts))
            return {
                "source": self.source,
                "lookback_sec": self.lookback_sec,
                "debounce_sec": self.debounce_sec,
                "current_regime": current_regime,
                "raw_regime": self._raw_regime,
                "confirmed_regime": self._confirmed_regime,
                "pending_regime": self._pending_regime,
                "regime_age_sec": age,
                "rolling_return": self._rolling_return,
                "buffer_size": len(self._points),
                "current_ts": self._current_ts,
                "last_change_ts": self._last_change_ts,
                "data_ready": self._data_ready,
            }

    def _prune_points_locked(self, now_ts: float) -> None:
        # Keep enough history for rolling return and interpolation around lookback boundary.
        target = now_ts - self.lookback_sec
        while len(self._points) >= 2 and self._points[1].ts <= target:
            self._points.popleft()

    def _recompute_locked(self, now_ts: float) -> None:
        if not self._points:
            self._set_unknown_locked()
            return
        span = now_ts - self._points[0].ts
        if span < (self.lookback_sec * 0.5):
            self._set_unknown_locked()
            return
        lookback_price = self._price_at_locked(now_ts - self.lookback_sec)
        if lookback_price is None or lookback_price <= 0.0:
            self._set_unknown_locked()
            return
        now_price = self._points[-1].price
        ret = math.log(now_price / lookback_price)
        self._rolling_return = ret
        self._data_ready = True
        raw = "UP" if ret >= 0.0 else "DOWN"
        self._raw_regime = raw
        self._apply_debounce_locked(raw, now_ts)

    def _set_unknown_locked(self) -> None:
        self._data_ready = False
        self._raw_regime = "UNKNOWN"
        self._rolling_return = float("nan")
        self._pending_regime = None
        self._pending_since = None
        self._confirmed_regime = "UNKNOWN"
        self._last_change_ts = None

    def _apply_debounce_locked(self, raw_regime: str, now_ts: float) -> None:
        if self.debounce_sec <= 0:
            if self._confirmed_regime != raw_regime:
                self._confirmed_regime = raw_regime
                self._last_change_ts = now_ts
            self._pending_regime = None
            self._pending_since = None
            return

        if self._confirmed_regime == "UNKNOWN":
            if self._pending_regime == raw_regime and self._pending_since is not None:
                if (now_ts - self._pending_since) >= self.debounce_sec:
                    self._confirmed_regime = raw_regime
                    self._last_change_ts = now_ts
                    self._pending_regime = None
                    self._pending_since = None
            else:
                self._pending_regime = raw_regime
                self._pending_since = now_ts
            return

        if raw_regime == self._confirmed_regime:
            self._pending_regime = None
            self._pending_since = None
            return

        if self._pending_regime != raw_regime:
            self._pending_regime = raw_regime
            self._pending_since = now_ts
            return

        if self._pending_since is None:
            self._pending_since = now_ts
            return

        if (now_ts - self._pending_since) >= self.debounce_sec:
            self._confirmed_regime = raw_regime
            self._last_change_ts = now_ts
            self._pending_regime = None
            self._pending_since = None

    def _price_at_locked(self, target_ts: float) -> Optional[float]:
        if not self._points:
            return None
        first = self._points[0]
        last = self._points[-1]
        if target_ts <= first.ts:
            return first.price
        if target_ts >= last.ts:
            return last.price
        prev = first
        for i in range(1, len(self._points)):
            cur = self._points[i]
            if cur.ts >= target_ts:
                if cur.ts == prev.ts:
                    return cur.price
                frac = (target_ts - prev.ts) / (cur.ts - prev.ts)
                return prev.price + frac * (cur.price - prev.price)
            prev = cur
        return last.price
