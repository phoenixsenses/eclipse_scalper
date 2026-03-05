from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    fee_bps: float = 0.5
    latency_bars: int = 2
    mode: str = "taker"  # taker | maker
    fill_prob: float = 0.3


def _fee_frac(fee_bps: float) -> float:
    return float(fee_bps) / 10000.0


def evaluate_trade_net(
    ret: pd.Series,
    spread: pd.Series,
    side: pd.Series,
    r1: pd.Series,
    cfg: CostConfig,
) -> pd.Series:
    # ret is horizon return from entry bar, side in {+1,-1}
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    s = pd.to_numeric(spread, errors="coerce").fillna(0.0)
    sd = pd.to_numeric(side, errors="coerce").fillna(0.0)
    nxt = pd.to_numeric(r1, errors="coerce").fillna(0.0)

    fee = _fee_frac(cfg.fee_bps)
    gross = sd * r

    if cfg.mode == "maker":
        # maker: no spread crossing, but adverse selection weighted by fill probability
        adverse = np.where(sd > 0, np.maximum(0.0, -nxt), np.maximum(0.0, nxt))
        cost = fee + (1.0 - float(cfg.fill_prob)) * 0.0 + float(cfg.fill_prob) * adverse
    else:
        # taker: spread cross and fee, plus adverse selection proxy
        adverse = np.where(sd > 0, np.maximum(0.0, -nxt), np.maximum(0.0, nxt))
        cost = fee + 0.5 * s + adverse

    return gross - cost
