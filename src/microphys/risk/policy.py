from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class RiskPolicy(BaseModel):
    starting_equity: float = 10_000.0
    max_gross_exposure: float = 1.0
    max_position_per_symbol: float = 1.0
    base_risk_per_trade: float = 0.01
    min_trade_notional: float = 10.0
    max_trade_notional: float = 1_000.0
    confidence_curve: Dict[str, float] = Field(default_factory=lambda: {"score_mid": 0.20, "slope": 4.0, "floor": 0.0, "ceil": 1.0})
    regime_quality_floor: float = 0.0
    execution_fill_floor: float = 0.0
    spread_risk_scale: float = 0.5
    vol_risk_scale: float = 0.5
    liq_risk_scale: float = 0.5
    drawdown_kill_pct: float = 0.03
    kill_cooldown_minutes: int = 60
    health_skip_on_bad: bool = True
    max_missing_bar_ratio_1h: float = 0.05
    drift_skip_threshold: float = 1.0


def load_risk_policy(path: str, *, starting_equity_override: float | None = None) -> RiskPolicy:
    if str(path).strip():
        p = Path(str(path))
        payload = json.loads(p.read_text(encoding="utf-8"))
        pol = RiskPolicy(**dict(payload))
    else:
        pol = RiskPolicy()
    if starting_equity_override is not None and float(starting_equity_override) > 0:
        pol = pol.model_copy(update={"starting_equity": float(starting_equity_override)})
    return pol


def dump_risk_policy(policy: RiskPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(policy.model_dump_json(indent=2) + "\n", encoding="utf-8")

