from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class ImpactModelStub:
    """Backward-compatible placeholder (kept for compatibility)."""

    note: str = "square-root impact model placeholder"

    def fit(self, _rows: Any) -> None:
        raise NotImplementedError("ImpactModelStub.fit is a placeholder")


@dataclass(frozen=True)
class PropagatorStub:
    """Backward-compatible placeholder (kept for compatibility)."""

    note: str = "propagator model placeholder"

    def fit(self, _rows: Any) -> None:
        raise NotImplementedError("PropagatorStub.fit is a placeholder")


@dataclass(frozen=True)
class ImpactFitResult:
    model: str
    alpha: float
    beta: float
    r2: float
    n: int


def _fit_ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    if len(y) == 0 or len(x) == 0:
        return 0.0, 0.0, 0.0
    X = np.column_stack([np.ones(len(x)), x])
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coeff[0])
    beta = float(coeff[1])
    yhat = alpha + beta * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + EPS))
    return alpha, beta, r2


def fit_impact_models(volume: pd.Series, abs_return: pd.Series) -> dict[str, ImpactFitResult]:
    v = pd.to_numeric(volume, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(abs_return, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(v) & np.isfinite(y) & (v > 0)
    v = v[m]
    y = y[m]
    if len(v) < 5:
        z = ImpactFitResult(model="linear", alpha=0.0, beta=0.0, r2=0.0, n=int(len(v)))
        return {"linear": z, "sqrt": ImpactFitResult(model="sqrt", alpha=0.0, beta=0.0, r2=0.0, n=int(len(v)))}

    a_lin, b_lin, r2_lin = _fit_ols(y, v)
    a_sqrt, b_sqrt, r2_sqrt = _fit_ols(y, np.sqrt(v))
    return {
        "linear": ImpactFitResult(model="linear", alpha=a_lin, beta=b_lin, r2=r2_lin, n=int(len(v))),
        "sqrt": ImpactFitResult(model="sqrt", alpha=a_sqrt, beta=b_sqrt, r2=r2_sqrt, n=int(len(v))),
    }


def bucket_impact(volume: pd.Series, abs_return: pd.Series, q: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"volume": pd.to_numeric(volume, errors="coerce"), "abs_return": pd.to_numeric(abs_return, errors="coerce")})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["volume"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["bucket", "count", "volume_mean", "abs_return_mean"]) 
    try:
        df["bucket"] = pd.qcut(df["volume"], q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["bucket", "count", "volume_mean", "abs_return_mean"]) 
    g = (
        df.groupby("bucket", as_index=False)
        .agg(count=("abs_return", "size"), volume_mean=("volume", "mean"), abs_return_mean=("abs_return", "mean"))
        .sort_values("bucket")
    )
    return g
