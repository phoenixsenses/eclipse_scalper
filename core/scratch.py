from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional


@dataclass(frozen=True)
class ScratchConfig:
    max_adverse_bps: float = 5.0
    scratch_cooldown_sec: float = 10.0
    trailing_stop_bps: float = 0.0
    take_profit_bps: float = 0.0
    hard_horizon_sec: float = 120.0


@dataclass(frozen=True)
class ExitDecision:
    action: str
    reason: str
    unrealized_bps: float
    elapsed_sec: float
    peak_favorable_bps: float
    max_adverse_bps: float


class ScratchEngine:
    """Stateful per-position scratch/exit evaluator."""

    def __init__(self, config: ScratchConfig):
        self.config = config
        self._lock = Lock()
        self._entry_price: Optional[float] = None
        self._entry_time: Optional[float] = None
        self._side: Optional[str] = None
        self._peak_fav_bps: float = 0.0
        self._max_adv_bps: float = 0.0
        self._best_price: Optional[float] = None

    def reset(self, entry_price: float, side: str, entry_time: float) -> None:
        ep = float(entry_price)
        if ep <= 0.0:
            raise ValueError("entry_price must be > 0")
        s = str(side or "").strip().lower()
        if s not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        with self._lock:
            self._entry_price = ep
            self._entry_time = float(entry_time)
            self._side = s
            self._peak_fav_bps = 0.0
            self._max_adv_bps = 0.0
            self._best_price = ep

    def evaluate(self, current_price: float, current_time: float) -> ExitDecision:
        with self._lock:
            if self._entry_price is None or self._entry_time is None or self._side is None:
                raise RuntimeError("ScratchEngine not initialized; call reset() on entry.")
            cp = float(current_price)
            if cp <= 0.0:
                raise ValueError("current_price must be > 0")
            elapsed = max(0.0, float(current_time) - self._entry_time)
            unrealized_bps = self._signed_bps(cp)
            fav_bps = max(0.0, unrealized_bps)
            adv_bps = max(0.0, -unrealized_bps)
            if fav_bps > self._peak_fav_bps:
                self._peak_fav_bps = fav_bps
            if adv_bps > self._max_adv_bps:
                self._max_adv_bps = adv_bps
            # Track best favorable price for trailing stop.
            if self._side == "buy":
                if self._best_price is None or cp > self._best_price:
                    self._best_price = cp
            else:
                if self._best_price is None or cp < self._best_price:
                    self._best_price = cp

            # 1) Hard horizon always wins.
            if float(self.config.hard_horizon_sec) > 0.0 and elapsed >= float(self.config.hard_horizon_sec):
                return self._decision("HORIZON_EXIT", "hard_horizon", unrealized_bps, elapsed)

            # 2) Take profit always active.
            if float(self.config.take_profit_bps) > 0.0 and fav_bps >= float(self.config.take_profit_bps):
                return self._decision("TAKE_PROFIT", "take_profit", unrealized_bps, elapsed)

            cooldown = max(0.0, float(self.config.scratch_cooldown_sec))
            if elapsed <= cooldown:
                return self._decision("HOLD", "cooldown", unrealized_bps, elapsed)

            # 3) Trailing stop.
            trail_bps = float(self.config.trailing_stop_bps)
            if trail_bps > 0.0 and self._best_price is not None:
                retrace_bps = self._retrace_from_best_bps(cp)
                if retrace_bps >= trail_bps:
                    return self._decision("SCRATCH", "trailing_stop", unrealized_bps, elapsed)

            # 4) Max adverse.
            max_adv = float(self.config.max_adverse_bps)
            if max_adv > 0.0 and adv_bps >= max_adv:
                return self._decision("SCRATCH", "max_adverse", unrealized_bps, elapsed)

            return self._decision("HOLD", "hold", unrealized_bps, elapsed)

    def _decision(self, action: str, reason: str, unrealized_bps: float, elapsed_sec: float) -> ExitDecision:
        return ExitDecision(
            action=str(action),
            reason=str(reason),
            unrealized_bps=float(unrealized_bps),
            elapsed_sec=float(elapsed_sec),
            peak_favorable_bps=float(self._peak_fav_bps),
            max_adverse_bps=float(self._max_adv_bps),
        )

    def _signed_bps(self, price: float) -> float:
        assert self._entry_price is not None and self._side is not None
        ep = float(self._entry_price)
        cp = float(price)
        if self._side == "buy":
            return ((cp / ep) - 1.0) * 10000.0
        return ((ep / cp) - 1.0) * 10000.0

    def _retrace_from_best_bps(self, price: float) -> float:
        assert self._best_price is not None and self._side is not None
        bp = float(self._best_price)
        cp = float(price)
        if self._side == "buy":
            return max(0.0, ((bp / cp) - 1.0) * 10000.0)
        return max(0.0, ((cp / bp) - 1.0) * 10000.0)

