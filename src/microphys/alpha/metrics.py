from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def composite_score(row: Dict[str, float]) -> float:
    return float(
        (0.45 * float(row.get("test_sharpe", 0.0)))
        + (0.25 * float(row.get("stability_score", 0.0)))
        + (0.20 * float(row.get("test_net_mean", 0.0)))
        - (0.10 * float(row.get("overfit_gap", 0.0)))
        - (0.05 * float(row.get("regime_concentration", 0.0)))
    )


def daily_sharpe(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(mu / sd)


def stability_score(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if abs(mu) < 1e-12:
        return 0.0
    cv = float(sd / (abs(mu) + 1e-12))
    return float(1.0 / (1.0 + cv))


def quantile_5(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return 0.0
    return float(np.quantile(x.to_numpy(dtype=float), 0.05))
