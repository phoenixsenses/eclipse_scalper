from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class RiskDecision:
    ts_ms: int
    symbol: str
    desired_side: str
    base_notional: float
    final_notional: float
    action: str  # TRADE | SKIP | KILL
    reason: str
    factors: Dict[str, float] = field(default_factory=dict)

