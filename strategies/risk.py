import numpy as np
from config.settings import Config

cfg = Config()

def optimal_risk_pct(atr_pct: float) -> float:
    kelly = cfg.BASE_EDGE / cfg.EXPECTED_RR
    base = kelly * cfg.FRACTIONAL_KELLY
    scaling = cfg.TARGET_DAILY_VOL / max(atr_pct, 0.006)
    scaling = np.clip(scaling, 0.4, 2.4)
    return round(base * scaling, 5)

def portfolio_heat(positions: dict, equity: float) -> float:
    return sum(abs(p.size * p.entry_price * p.leverage) for p in positions.values()) / equity