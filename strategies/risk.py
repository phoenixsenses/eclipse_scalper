import numpy as np
from config.settings import Config

cfg = Config()

# Safe defaults for optional config keys.
_DEFAULT_BASE_EDGE = 0.55
_DEFAULT_EXPECTED_RR = 2.0
_DEFAULT_FRACTIONAL_KELLY = 0.5
_DEFAULT_TARGET_DAILY_VOL = 0.02

def optimal_risk_pct(atr_pct: float) -> float:
    base_edge = getattr(cfg, "BASE_EDGE", _DEFAULT_BASE_EDGE)
    expected_rr = getattr(cfg, "EXPECTED_RR", _DEFAULT_EXPECTED_RR)
    fractional_kelly = getattr(cfg, "FRACTIONAL_KELLY", _DEFAULT_FRACTIONAL_KELLY)
    target_daily_vol = getattr(cfg, "TARGET_DAILY_VOL", _DEFAULT_TARGET_DAILY_VOL)

    kelly = float(base_edge) / max(float(expected_rr), 1e-9)
    base = kelly * float(fractional_kelly)
    scaling = float(target_daily_vol) / max(float(atr_pct), 0.006)
    scaling = np.clip(scaling, 0.4, 2.4)
    return round(base * scaling, 5)

def portfolio_heat(positions: dict, equity: float) -> float:
    """
    Portfolio heat as notional exposure / equity.
    Position size is absolute quantity, so notional = size * entry_price.
    Do NOT multiply by leverage (that would double-count).
    """
    if equity <= 0:
        return 0.0
    total_notional = 0.0
    for p in positions.values():
        try:
            total_notional += abs(float(p.size) * float(p.entry_price))
        except Exception:
            continue
    return total_notional / equity
