from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class GuardKnobs:
    allow_entries: bool = True
    max_notional_usdt: float = 0.0
    max_leverage: int = 0
    min_entry_conf: float = 0.0
    entry_cooldown_seconds: float = 0.0
    max_open_orders_per_symbol: int = 1
    mode: str = "GREEN"
    kill_switch_trip: bool = False
    reason: str = ""
    debt_score: float = 0.0
    debt_growth_per_min: float = 0.0
    runtime_gate_degraded: bool = False
    runtime_gate_reason: str = ""
    runtime_gate_degrade_score: float = 0.0
    reconcile_first_gate_degraded: bool = False
    reconcile_first_gate_count: int = 0
    reconcile_first_gate_max_severity: float = 0.0
    reconcile_first_gate_max_streak: int = 0
    recovery_stage: str = ""
    unlock_conditions: str = ""
    next_unlock_sec: float = 0.0
    protection_refresh_budget_blocked_level: float = 0.0
    protection_refresh_budget_force_level: float = 0.0
    per_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_obj(cls, value: Any) -> "GuardKnobs":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                allow_entries=bool(value.get("allow_entries", True)),
                max_notional_usdt=float(value.get("max_notional_usdt", 0.0) or 0.0),
                max_leverage=int(value.get("max_leverage", 0) or 0),
                min_entry_conf=float(value.get("min_entry_conf", 0.0) or 0.0),
                entry_cooldown_seconds=float(value.get("entry_cooldown_seconds", 0.0) or 0.0),
                max_open_orders_per_symbol=int(value.get("max_open_orders_per_symbol", 1) or 1),
                mode=str(value.get("mode", "GREEN") or "GREEN").upper(),
                kill_switch_trip=bool(value.get("kill_switch_trip", False)),
                reason=str(value.get("reason", "") or ""),
                debt_score=float(value.get("debt_score", 0.0) or 0.0),
                debt_growth_per_min=float(value.get("debt_growth_per_min", 0.0) or 0.0),
                runtime_gate_degraded=bool(value.get("runtime_gate_degraded", False)),
                runtime_gate_reason=str(value.get("runtime_gate_reason", "") or ""),
                runtime_gate_degrade_score=float(value.get("runtime_gate_degrade_score", 0.0) or 0.0),
                reconcile_first_gate_degraded=bool(value.get("reconcile_first_gate_degraded", False)),
                reconcile_first_gate_count=int(value.get("reconcile_first_gate_count", 0) or 0),
                reconcile_first_gate_max_severity=float(
                    value.get("reconcile_first_gate_max_severity", 0.0) or 0.0
                ),
                reconcile_first_gate_max_streak=int(value.get("reconcile_first_gate_max_streak", 0) or 0),
                recovery_stage=str(value.get("recovery_stage", "") or ""),
                unlock_conditions=str(value.get("unlock_conditions", "") or ""),
                next_unlock_sec=float(value.get("next_unlock_sec", 0.0) or 0.0),
                protection_refresh_budget_blocked_level=float(
                    value.get("protection_refresh_budget_blocked_level", 0.0) or 0.0
                ),
                protection_refresh_budget_force_level=float(
                    value.get("protection_refresh_budget_force_level", 0.0) or 0.0
                ),
                per_symbol=dict(value.get("per_symbol") or {}),
            )
        return cls()
