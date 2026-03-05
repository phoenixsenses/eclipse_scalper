from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _to_day(ts_utc: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_utc, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")


def _histogram_probs(x: pd.Series, bins: int = 10) -> np.ndarray:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return np.full(bins, 1.0 / bins)
    lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.full(bins, 1.0 / bins)
    counts, _ = np.histogram(s.to_numpy(), bins=bins, range=(lo, hi))
    probs = counts.astype(float) + 1e-9
    probs /= probs.sum()
    return probs


def _psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    pa = _histogram_probs(a, bins=bins)
    pb = _histogram_probs(b, bins=bins)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def compute_live_metrics(
    *,
    physics_recent: pd.DataFrame,
    live_trades: pd.DataFrame,
    baseline: Dict[str, Any],
    db_last_event_ts: float | None,
    interval_ms: int,
) -> Dict[str, Any]:
    now = _now_ts()
    out: Dict[str, Any] = {"ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if physics_recent.empty:
        out.update(
            {
                "state": "degraded",
                "reason": "no_physics_recent",
                "data_freshness_sec": float("inf"),
                "missing_bars_pct_1h": 100.0,
                "spread_median": 0.0,
                "spread_p95": 0.0,
                "ofi_shift": 0.0,
                "regime_shift": 0.0,
                "signal_rate_per_hour": 0.0,
                "pnl_net_mean": 0.0,
                "pnl_net_sum": 0.0,
                "adverse_proxy_rate": 0.0,
            }
        )
        return out

    ts_ms = pd.to_numeric(physics_recent.get("ts_ms"), errors="coerce").dropna()
    last_ts = float(ts_ms.max() / 1000.0) if not ts_ms.empty else 0.0
    data_freshness = float(max(0.0, now - last_ts)) if last_ts > 0 else float("inf")
    if db_last_event_ts is not None and db_last_event_ts > 0:
        data_freshness = float(max(0.0, now - float(db_last_event_ts)))

    # Missing bars over recent 1h
    horizon_sec = 3600.0
    expected = int(max(1.0, horizon_sec / max(0.001, float(interval_ms) / 1000.0)))
    ts1h = ts_ms[ts_ms >= (ts_ms.max() - horizon_sec * 1000.0)] if not ts_ms.empty else pd.Series([], dtype=float)
    got = int(ts1h.nunique()) if not ts1h.empty else 0
    missing_pct = float(max(0.0, (expected - got) / max(1, expected) * 100.0))

    spread = pd.to_numeric(physics_recent.get("spread"), errors="coerce")
    spread_median = float(spread.median()) if not spread.dropna().empty else 0.0
    spread_p95 = float(spread.quantile(0.95)) if not spread.dropna().empty else 0.0
    base_spread_median = float(baseline.get("spread_median", spread_median) or spread_median or 1e-9)
    spread_jump_frac = float((spread_median - base_spread_median) / max(1e-9, base_spread_median))

    ofi = pd.to_numeric(physics_recent.get("F_ofi_z"), errors="coerce")
    base_ofi_ref = pd.to_numeric(pd.Series(baseline.get("ofi_ref", [])), errors="coerce")
    if base_ofi_ref.dropna().empty:
        base_ofi_ref = ofi
    ofi_shift = float(abs(ofi.median() - base_ofi_ref.median())) if not ofi.dropna().empty else 0.0

    cur_regime = pd.to_numeric((ofi.fillna(0.0) >= 0).astype(int), errors="coerce")
    base_regime = pd.to_numeric(pd.Series(baseline.get("regime_ref", [])), errors="coerce")
    if base_regime.dropna().empty:
        base_regime = cur_regime
    regime_shift = float(_psi(cur_regime, base_regime, bins=2))

    trades = live_trades.copy()
    signal_rate = 0.0
    pnl_mean = 0.0
    pnl_sum = 0.0
    adverse_rate = 0.0
    if not trades.empty:
        t = pd.to_datetime(trades.get("entry_ts_utc"), utc=True, errors="coerce")
        t = t.dropna()
        if not t.empty:
            span_h = max(1e-9, (t.max().timestamp() - t.min().timestamp()) / 3600.0)
            signal_rate = float(len(t) / span_h)
        pnl = pd.to_numeric(trades.get("pnl_net"), errors="coerce").dropna()
        if not pnl.empty:
            pnl_mean = float(pnl.mean())
            pnl_sum = float(pnl.sum())
        gross = pd.to_numeric(trades.get("pnl_gross"), errors="coerce")
        if not gross.dropna().empty:
            adverse_rate = float((gross < 0).mean())

    out.update(
        {
            "state": "ok",
            "reason": "ok",
            "data_freshness_sec": data_freshness,
            "missing_bars_pct_1h": missing_pct,
            "spread_median": spread_median,
            "spread_p95": spread_p95,
            "spread_jump_frac": spread_jump_frac,
            "ofi_shift": ofi_shift,
            "regime_shift": regime_shift,
            "signal_rate_per_hour": signal_rate,
            "pnl_net_mean": pnl_mean,
            "pnl_net_sum": pnl_sum,
            "adverse_proxy_rate": adverse_rate,
            "last_ts_ms": int(ts_ms.max()) if not ts_ms.empty else 0,
        }
    )
    return out


def write_status(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
