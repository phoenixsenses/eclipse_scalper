from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.microphys.impact.models import fit_impact_models
from src.microphys.propagator.kernel import compute_response_kernel


def ofi_decile_lift(df: pd.DataFrame, return_col: str = "r_1") -> dict[str, float]:
    d = df[["F_ofi_z", return_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return {"count": 0.0, "top_mean": 0.0, "bot_mean": 0.0, "lift": 0.0, "direction_acc": 0.0}
    try:
        d["decile"] = pd.qcut(d["F_ofi_z"], q=10, labels=False, duplicates="drop")
    except Exception:
        return {"count": float(len(d)), "top_mean": 0.0, "bot_mean": 0.0, "lift": 0.0, "direction_acc": 0.0}
    top = d[d["decile"] == d["decile"].max()][return_col]
    bot = d[d["decile"] == d["decile"].min()][return_col]
    acc = float((np.sign(d["F_ofi_z"]) == np.sign(d[return_col])).mean()) if len(d) else 0.0
    top_m = float(top.mean()) if len(top) else 0.0
    bot_m = float(bot.mean()) if len(bot) else 0.0
    return {"count": float(len(d)), "top_mean": top_m, "bot_mean": bot_m, "lift": top_m - bot_m, "direction_acc": acc}


def regime_kernel(df: pd.DataFrame, tau_max: int = 200) -> pd.DataFrame:
    return compute_response_kernel(df["mid"], df["ofi"], max_lag=tau_max)


def impact_stats(df: pd.DataFrame) -> dict[str, float]:
    fits = fit_impact_models(pd.to_numeric(df.get("volume_proxy"), errors="coerce"), pd.to_numeric(df.get("r_1"), errors="coerce").abs())
    return {
        "linear_r2": float(fits["linear"].r2),
        "sqrt_r2": float(fits["sqrt"].r2),
        "linear_beta": float(fits["linear"].beta),
        "sqrt_beta": float(fits["sqrt"].beta),
    }


def conditional_flag_stats(df: pd.DataFrame, flag: str, ret_col: str = "r_1") -> dict[str, float]:
    if flag not in df.columns:
        return {"n": 0.0, "mean": 0.0, "median": 0.0}
    s = df[df[flag].fillna(False)]
    r = pd.to_numeric(s.get(ret_col), errors="coerce").dropna()
    return {"n": float(len(r)), "mean": float(r.mean()) if len(r) else 0.0, "median": float(r.median()) if len(r) else 0.0}


def compute_regime_metrics(merged: pd.DataFrame, tau_max: int = 200) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    kernels: dict[int, pd.DataFrame] = {}

    for rid, g in merged.groupby("regime_id", sort=True):
        rg = g.sort_values("ts_ms")
        k = regime_kernel(rg, tau_max=tau_max)
        kernels[int(rid)] = k
        lift1 = ofi_decile_lift(rg, return_col="r_1")
        lift5 = ofi_decile_lift(rg, return_col="r_5") if "r_5" in rg.columns else ofi_decile_lift(rg, return_col="r_1")
        imp = impact_stats(rg)
        comp = conditional_flag_stats(rg, "compression_flag", ret_col="r_1")
        vac = conditional_flag_stats(rg, "vacuum_flag", ret_col="r_1")
        liq = conditional_flag_stats(rg, "liq_burst_flag", ret_col="r_1")

        rows.append(
            {
                "regime_id": int(rid),
                "count": int(len(rg)),
                "ofi_lift_r1": float(lift1["lift"]),
                "ofi_lift_r5": float(lift5["lift"]),
                "ofi_dir_acc_r1": float(lift1["direction_acc"]),
                "compression_mean_r1": float(comp["mean"]),
                "vacuum_mean_r1": float(vac["mean"]),
                "liq_burst_mean_r1": float(liq["mean"]),
                "linear_r2": float(imp["linear_r2"]),
                "sqrt_r2": float(imp["sqrt_r2"]),
                "linear_beta": float(imp["linear_beta"]),
                "sqrt_beta": float(imp["sqrt_beta"]),
                "kernel_lag1": float(k["response"].iloc[0]) if len(k) else 0.0,
                "kernel_auc": float(k["response"].sum()) if len(k) else 0.0,
                "kernel_smooth_score": float(1.0 - (k["abs_response"].diff().fillna(0.0).gt(0).mean() if len(k) else 0.0)),
            }
        )

    return pd.DataFrame(rows).sort_values("regime_id").reset_index(drop=True), kernels
