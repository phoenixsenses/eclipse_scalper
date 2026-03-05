from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .cost_models import CostConfig, evaluate_trade_net


@dataclass(frozen=True)
class SignalCondition:
    name: str
    side: int  # +1 buy, -1 sell
    predicate: Callable[[pd.DataFrame], pd.Series]


def builtin_conditions() -> list[SignalCondition]:
    return [
        SignalCondition("ofi_top_decile_buy", +1, lambda d: d["F_ofi_z"] >= d["F_ofi_z"].quantile(0.9)),
        SignalCondition("ofi_bottom_decile_sell", -1, lambda d: d["F_ofi_z"] <= d["F_ofi_z"].quantile(0.1)),
        SignalCondition("compression_plus_ofi_buy", +1, lambda d: d["compression_flag"].fillna(False) & (d["F_ofi_z"] >= d["F_ofi_z"].quantile(0.9))),
        SignalCondition("vacuum_contrarian_buy", +1, lambda d: d["vacuum_flag"].fillna(False) & (d["F_ofi_z"] <= d["F_ofi_z"].quantile(0.1))),
        SignalCondition("vacuum_trend_sell", -1, lambda d: d["vacuum_flag"].fillna(False) & (d["F_ofi_z"] <= d["F_ofi_z"].quantile(0.1))),
        SignalCondition("liq_burst_intensity_shock_sell", -1, lambda d: d["liq_burst_flag"].fillna(False) & (d["F_intensity_z"] > 1.5)),
        SignalCondition("high_rv_wide_spread_ofi_buy", +1, lambda d: (d["rv_z"] > 1.0) & (d["spread_z"] > 1.0) & (d["F_ofi_z"] > 1.0)),
    ]


def evaluate_conditions(df: pd.DataFrame, horizon: int, cfg: CostConfig) -> pd.DataFrame:
    out_rows: list[dict[str, float | int | str]] = []

    # latency shift: execute after N bars
    lat = int(max(0, cfg.latency_bars))
    mid = pd.to_numeric(df.get("mid"), errors="coerce").replace(0.0, np.nan)
    r_h = np.log(mid.shift(-(horizon + lat)) / mid.shift(lat))
    r1_lat = np.log(mid.shift(-(1 + lat)) / mid.shift(lat))

    base = df.copy().reset_index(drop=True)
    # Backward-compatible defaults when inputs come directly from physics parquet.
    if "rv_z" not in base.columns:
        base["rv_z"] = 0.0
    if "spread_z" not in base.columns:
        base["spread_z"] = 0.0
    if "F_intensity_z" not in base.columns:
        base["F_intensity_z"] = 0.0
    if "compression_flag" not in base.columns:
        base["compression_flag"] = False
    if "vacuum_flag" not in base.columns:
        base["vacuum_flag"] = False
    if "liq_burst_flag" not in base.columns:
        base["liq_burst_flag"] = False
    if "F_ofi_z" not in base.columns:
        base["F_ofi_z"] = 0.0
    base["ret_h"] = r_h
    base["r1_lat"] = r1_lat
    base["spread_eff"] = pd.to_numeric(base.get("spread"), errors="coerce").fillna(0.0)

    conds = builtin_conditions()
    for c in conds:
        mask = c.predicate(base).fillna(False)
        s = base[mask].copy()
        if s.empty:
            out_rows.append({
                "condition": c.name,
                "side": "buy" if c.side > 0 else "sell",
                "count": 0,
                "gross_mean": 0.0,
                "net_mean": 0.0,
                "net_median": 0.0,
                "worst_5pct_day": 0.0,
                "t_stat": 0.0,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.0,
            })
            continue

        side = pd.Series(np.full(len(s), c.side, dtype=float), index=s.index)
        gross = side * pd.to_numeric(s["ret_h"], errors="coerce").fillna(0.0)
        net = evaluate_trade_net(
            ret=s["ret_h"],
            spread=s["spread_eff"],
            side=side,
            r1=s["r1_lat"],
            cfg=cfg,
        )

        # daily stability using ts_utc day
        day = pd.to_datetime(s["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        daily = pd.DataFrame({"day": day, "net": net}).groupby("day", as_index=False).agg(net_mean=("net", "mean"))
        worst5 = float(daily["net_mean"].quantile(0.05)) if len(daily) else 0.0

        x = net.to_numpy(dtype=float)
        m = float(np.nanmean(x)) if len(x) else 0.0
        sd = float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
        t = m / (sd / np.sqrt(max(1, len(x)))) if sd > 0 else 0.0

        # deterministic bootstrap (fixed seed)
        rs = np.random.RandomState(42)
        bs = []
        if len(x) > 1:
            for _ in range(200):
                idx = rs.randint(0, len(x), size=len(x))
                bs.append(float(np.mean(x[idx])))
        ci_lo = float(np.quantile(bs, 0.05)) if bs else m
        ci_hi = float(np.quantile(bs, 0.95)) if bs else m

        out_rows.append({
            "condition": c.name,
            "side": "buy" if c.side > 0 else "sell",
            "count": int(len(s)),
            "gross_mean": float(gross.mean()),
            "net_mean": float(net.mean()),
            "net_median": float(net.median()),
            "worst_5pct_day": worst5,
            "t_stat": float(t),
            "bootstrap_ci_low": ci_lo,
            "bootstrap_ci_high": ci_hi,
        })

    return pd.DataFrame(out_rows).sort_values("condition").reset_index(drop=True)
