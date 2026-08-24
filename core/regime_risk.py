from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RegimeRiskConfig:
    max_concurrent_positions: int = 1
    max_daily_loss_bps: float = 50.0
    max_daily_trades: int = 100
    regime_change_policy: str = "hold"  # hold | close | reduce
    cooldown_after_regime_change_sec: float = 300.0
    max_consecutive_scratches: int = 3
    scratch_pause_sec: float = 600.0
    max_drawdown_bps: float = 100.0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    risk_state: Dict[str, Any]


@dataclass(frozen=True)
class RiskAction:
    action: str  # close_position | reduce_position | pause_trading | resume_trading
    target_position_id: Optional[str]
    reason: str


@dataclass
class _State:
    daily_pnl_bps: float = 0.0
    daily_trades: int = 0
    consecutive_scratches: int = 0
    scratch_pause_until_ts: float = 0.0
    regime_cooldown_until_ts: float = 0.0
    peak_pnl_bps: float = 0.0
    current_pnl_bps: float = 0.0
    last_regime: str = ""
    last_regime_change_ts: float = 0.0
    total_entries: int = 0


class RegimeRiskManager:
    def __init__(self, config: RegimeRiskConfig):
        self.config = config
        self._lock = Lock()
        self._st = _State()

    def check_entry(self, side: str, regime: str, current_positions: list) -> RiskDecision:
        now_ts = time()
        with self._lock:
            s = self._st
            snap = self._state_dict_unlocked()
            if now_ts < float(s.scratch_pause_until_ts):
                return RiskDecision(False, "scratch_pause_active", snap)
            if now_ts < float(s.regime_cooldown_until_ts):
                return RiskDecision(False, "regime_cooldown_active", snap)
            if float(self.config.max_daily_loss_bps) > 0 and float(s.daily_pnl_bps) <= -abs(float(self.config.max_daily_loss_bps)):
                return RiskDecision(False, "daily_loss_limit", snap)
            if int(self.config.max_daily_trades) > 0 and int(s.daily_trades) >= int(self.config.max_daily_trades):
                return RiskDecision(False, "daily_trade_limit", snap)
            if float(self.config.max_drawdown_bps) > 0:
                dd = float(s.peak_pnl_bps) - float(s.current_pnl_bps)
                if dd > float(self.config.max_drawdown_bps):
                    return RiskDecision(False, "max_drawdown_limit", snap)
            if int(self.config.max_concurrent_positions) > 0 and len(list(current_positions or [])) >= int(self.config.max_concurrent_positions):
                return RiskDecision(False, "max_concurrent_positions", snap)
            return RiskDecision(True, "", snap)

    def on_entry_submitted(self) -> None:
        with self._lock:
            self._st.daily_trades += 1
            self._st.total_entries += 1

    def on_trade_exit(self, exit_decision: Any, pnl_bps: float) -> List[RiskAction]:
        actions: List[RiskAction] = []
        now_ts = time()
        with self._lock:
            s = self._st
            pb = float(pnl_bps or 0.0)
            s.daily_pnl_bps += pb
            s.current_pnl_bps += pb
            if s.current_pnl_bps > s.peak_pnl_bps:
                s.peak_pnl_bps = s.current_pnl_bps
            reason = ""
            if isinstance(exit_decision, dict):
                reason = str(exit_decision.get("reason") or exit_decision.get("action") or "").lower().strip()
            else:
                reason = str(getattr(exit_decision, "reason", "") or getattr(exit_decision, "action", "") or "").lower().strip()
            is_scratch = ("scratch" in reason)
            if is_scratch:
                s.consecutive_scratches += 1
            else:
                s.consecutive_scratches = 0
            if (
                int(self.config.max_consecutive_scratches) > 0
                and int(s.consecutive_scratches) >= int(self.config.max_consecutive_scratches)
            ):
                s.scratch_pause_until_ts = max(s.scratch_pause_until_ts, now_ts + max(0.0, float(self.config.scratch_pause_sec)))
                actions.append(RiskAction("pause_trading", None, "max_consecutive_scratches"))
        return actions

    def on_regime_change(self, old_regime: str, new_regime: str, open_positions: list) -> List[RiskAction]:
        actions: List[RiskAction] = []
        now_ts = time()
        policy = str(self.config.regime_change_policy or "hold").strip().lower()
        with self._lock:
            s = self._st
            s.last_regime = str(new_regime or "")
            s.last_regime_change_ts = now_ts
            cd = max(0.0, float(self.config.cooldown_after_regime_change_sec))
            if cd > 0:
                s.regime_cooldown_until_ts = max(s.regime_cooldown_until_ts, now_ts + cd)
            positions = list(open_positions or [])
            if policy == "close":
                for p in positions:
                    pid = str(getattr(p, "symbol", None) or (p.get("symbol") if isinstance(p, dict) else None) or "")
                    actions.append(RiskAction("close_position", pid or None, "regime_change_close"))
            elif policy == "reduce":
                for p in positions:
                    pid = str(getattr(p, "symbol", None) or (p.get("symbol") if isinstance(p, dict) else None) or "")
                    actions.append(RiskAction("reduce_position", pid or None, "regime_change_reduce"))
        return actions

    def reset_daily(self) -> None:
        with self._lock:
            self._st.daily_pnl_bps = 0.0
            self._st.daily_trades = 0
            self._st.consecutive_scratches = 0
            self._st.scratch_pause_until_ts = 0.0
            self._st.regime_cooldown_until_ts = 0.0
            self._st.peak_pnl_bps = 0.0
            self._st.current_pnl_bps = 0.0

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return self._state_dict_unlocked()

    def _state_dict_unlocked(self) -> Dict[str, Any]:
        s = self._st
        return {
            "daily_pnl_bps": float(s.daily_pnl_bps),
            "daily_trades": int(s.daily_trades),
            "consecutive_scratches": int(s.consecutive_scratches),
            "scratch_pause_until_ts": float(s.scratch_pause_until_ts),
            "regime_cooldown_until_ts": float(s.regime_cooldown_until_ts),
            "peak_pnl_bps": float(s.peak_pnl_bps),
            "current_pnl_bps": float(s.current_pnl_bps),
            "last_regime": str(s.last_regime),
            "last_regime_change_ts": float(s.last_regime_change_ts),
            "total_entries": int(s.total_entries),
            "config": {
                "max_concurrent_positions": int(self.config.max_concurrent_positions),
                "max_daily_loss_bps": float(self.config.max_daily_loss_bps),
                "max_daily_trades": int(self.config.max_daily_trades),
                "regime_change_policy": str(self.config.regime_change_policy),
                "cooldown_after_regime_change_sec": float(self.config.cooldown_after_regime_change_sec),
                "max_consecutive_scratches": int(self.config.max_consecutive_scratches),
                "scratch_pause_sec": float(self.config.scratch_pause_sec),
                "max_drawdown_bps": float(self.config.max_drawdown_bps),
            },
        }
