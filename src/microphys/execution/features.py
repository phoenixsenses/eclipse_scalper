from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _as_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _z(s: pd.Series, window: int = 300) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mu = x.rolling(window, min_periods=max(5, window // 10)).mean()
    sd = x.rolling(window, min_periods=max(5, window // 10)).std(ddof=0)
    return (x - mu) / (sd + EPS)


def build_execution_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for c in ("ts_ms", "ts_utc", "symbol"):
        if c not in df.columns:
            df[c] = 0 if c == "ts_ms" else ""
    bid_qty = _as_series(df, "bid_qty", 0.0).fillna(0.0)
    ask_qty = _as_series(df, "ask_qty", 0.0).fillna(0.0)
    spread = _as_series(df, "spread", 0.0).fillna(0.0)
    mid = _as_series(df, "mid", 0.0).replace(0.0, np.nan).ffill().fillna(0.0)
    intensity = _as_series(df, "trade_intensity", np.nan)
    if pd.to_numeric(intensity, errors="coerce").isna().all():
        intensity = _as_series(df, "trade_intensity_qty_per_sec", 0.0).fillna(0.0)
    else:
        intensity = pd.to_numeric(intensity, errors="coerce").fillna(0.0)

    qimb = (bid_qty - ask_qty) / (bid_qty + ask_qty + EPS)
    queue_pressure_bid = bid_qty / (ask_qty + EPS)
    queue_pressure_ask = ask_qty / (bid_qty + EPS)

    tick = pd.to_numeric(mid.diff().abs().replace(0.0, np.nan), errors="coerce").median()
    if not np.isfinite(float(tick)) or float(tick) <= 0:
        tick = float((mid.abs() * 1e-6).median() or 1e-6)
    spread_ticks = spread / max(EPS, float(tick / (mid.replace(0.0, np.nan).median() + EPS)))

    zi = _z(intensity, 300)
    zs = _z(spread, 300)
    zq = _z(qimb.abs(), 300)
    trade_through_prob = 1.0 / (1.0 + np.exp(-(1.2 * zi - 0.8 * zs + 0.6 * zq)))

    out = pd.DataFrame(
        {
            "ts_ms": pd.to_numeric(df.get("ts_ms"), errors="coerce").fillna(0).astype("int64"),
            "ts_utc": df.get("ts_utc").astype(str),
            "symbol": df.get("symbol").astype(str),
            "queue_pressure_bid": queue_pressure_bid,
            "queue_pressure_ask": queue_pressure_ask,
            "quote_imbalance": qimb,
            "spread_ticks": spread_ticks.fillna(0.0),
            "trade_through_prob": pd.to_numeric(trade_through_prob, errors="coerce").fillna(0.0).clip(0.0, 1.0),
        }
    )
    return out.sort_values("ts_ms").reset_index(drop=True)
