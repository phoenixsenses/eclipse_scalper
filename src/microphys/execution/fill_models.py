from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HazardParams:
    a: float = 1.0
    b: float = -0.7
    c: float = 0.5
    d: float = -0.5
    ttl_bars: int = 10
    fill_threshold: float = 0.5


def hazard_fill_prob(intensity_z: float, spread_z: float, imbalance: float, params: HazardParams) -> float:
    z = (params.a * float(intensity_z)) + (params.b * float(spread_z)) + (params.c * float(imbalance)) + params.d
    return float(1.0 / (1.0 + np.exp(-z)))


def simulate_maker_hazard_fill(
    frame: pd.DataFrame,
    *,
    entry_idx: int,
    side: Literal["buy", "sell"],
    params: HazardParams,
) -> dict:
    n = len(frame)
    if n == 0 or entry_idx < 0 or entry_idx >= n:
        return {"filled": False, "fill_idx": None, "fill_delay_bars": None, "ttl_expired": True}
    iz = pd.to_numeric(frame.get("F_intensity_z"), errors="coerce").fillna(0.0).to_numpy()
    sz = pd.to_numeric(frame.get("spread_z"), errors="coerce").fillna(0.0).to_numpy()
    if "quote_imbalance" in frame.columns:
        qi = pd.to_numeric(frame["quote_imbalance"], errors="coerce")
    else:
        qi = pd.Series([np.nan] * n)
    if qi.isna().all():
        bid = pd.to_numeric(frame["bid_qty"], errors="coerce").fillna(0.0) if "bid_qty" in frame.columns else pd.Series([0.0] * n)
        ask = pd.to_numeric(frame["ask_qty"], errors="coerce").fillna(0.0) if "ask_qty" in frame.columns else pd.Series([0.0] * n)
        qi = (bid - ask) / (bid + ask + 1e-12)
    q = qi.fillna(0.0).to_numpy()

    ttl = max(1, int(params.ttl_bars))
    stop = min(n - 1, entry_idx + ttl)
    p_fill = 0.0
    for i in range(entry_idx + 1, stop + 1):
        imb = float(q[i])
        # buy fills easier with negative imbalance, sell with positive.
        signed_imb = -imb if side == "buy" else imb
        p = hazard_fill_prob(float(iz[i]), float(sz[i]), signed_imb, params)
        p_fill = 1.0 - ((1.0 - p_fill) * (1.0 - p))
        if p_fill >= float(params.fill_threshold):
            return {
                "filled": True,
                "fill_idx": int(i),
                "fill_delay_bars": int(i - entry_idx),
                "ttl_expired": False,
                "cum_fill_prob": float(p_fill),
            }
    return {
        "filled": False,
        "fill_idx": None,
        "fill_delay_bars": None,
        "ttl_expired": True,
        "cum_fill_prob": float(p_fill),
    }
