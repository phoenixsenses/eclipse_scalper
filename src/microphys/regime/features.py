from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    min_p = max(10, min(int(window), max(20, int(window) // 10)))
    mu = series.rolling(window, min_periods=min_p).mean()
    sd = series.rolling(window, min_periods=min_p).std(ddof=0)
    return (series - mu) / (sd + EPS)


def rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    w = int(max(5, window))
    for i in range(len(vals)):
        s = i - w + 1
        if s < 0:
            continue
        seg = vals[s : i + 1]
        if len(seg) <= lag:
            continue
        a = seg[:-lag]
        b = seg[lag:]
        if np.nanstd(a) == 0.0 or np.nanstd(b) == 0.0:
            out[i] = 0.0
            continue
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            continue
        out[i] = float(np.corrcoef(a[m], b[m])[0, 1])
    return pd.Series(out, index=series.index)


def build_regime_features(df: pd.DataFrame, rolling: int = 2000) -> pd.DataFrame:
    out = df.copy().sort_values("ts_ms").reset_index(drop=True)

    out["rv_z"] = rolling_z(pd.to_numeric(out.get("rv_short"), errors="coerce").fillna(0.0), rolling)
    out["spread_z"] = pd.to_numeric(out.get("spread_z"), errors="coerce").fillna(0.0)
    out["intensity_z"] = pd.to_numeric(out.get("F_intensity_z"), errors="coerce").fillna(0.0)
    out["liq_rate_z"] = pd.to_numeric(out.get("liq_rate_z"), errors="coerce").fillna(0.0)

    q = pd.to_numeric(out.get("volume_proxy"), errors="coerce").fillna(0.0)
    abs_r1 = pd.to_numeric(out.get("r_1"), errors="coerce").abs().fillna(0.0)
    out["impact_proxy"] = abs_r1 / (np.sqrt(q) + EPS)

    out["micro_trend"] = pd.to_numeric(out.get("r_1"), errors="coerce").fillna(0.0).rolling(rolling, min_periods=max(10, rolling // 20)).mean()
    out["of_flow_persistence"] = rolling_autocorr(pd.to_numeric(out.get("F_ofi"), errors="coerce").fillna(0.0), window=max(20, rolling // 4), lag=1)

    cols = [
        "ts_ms",
        "ts_utc",
        "symbol",
        "rv_z",
        "spread_z",
        "intensity_z",
        "liq_rate_z",
        "impact_proxy",
        "micro_trend",
        "of_flow_persistence",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].copy()
