from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.microphys.alpha.generalization import infer_family
from src.microphys.alpha.transfer_regime import attach_aligned_regime


@dataclass(frozen=True)
class ExpertBuildConfig:
    top_k_per_regime: int = 5
    min_trades_per_signal: int = 10
    min_regime_rows: int = 50


def _safe_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(0.0)


def compute_transfer_penalties(transfer_by_regime: pd.DataFrame) -> Tuple[Dict[Tuple[str, int], float], Dict[int, float]]:
    fam_pen: Dict[Tuple[str, int], float] = {}
    reg_pen: Dict[int, float] = {}
    if transfer_by_regime.empty:
        return fam_pen, reg_pen
    df = transfer_by_regime.copy()
    df["aligned_regime_id"] = pd.to_numeric(df.get("aligned_regime_id"), errors="coerce").fillna(-1).astype(int)
    net = _safe_num(df.get("mean_net_ret", pd.Series([], dtype=float)))
    df["mean_net_ret"] = net
    for rid, g in df.groupby("aligned_regime_id", sort=True):
        m = float(_safe_num(g["mean_net_ret"]).mean()) if not g.empty else 0.0
        # deterministic, conservative scaling
        if m < -0.001:
            reg_pen[int(rid)] = 0.25
        elif m < 0.0:
            reg_pen[int(rid)] = 0.50
        else:
            reg_pen[int(rid)] = 1.0
        if "family" in g.columns:
            for fam, fg in g.groupby("family", sort=True):
                mm = float(_safe_num(fg["mean_net_ret"]).mean()) if not fg.empty else 0.0
                if mm < -0.001:
                    fam_pen[(str(fam), int(rid))] = 0.25
                elif mm < 0.0:
                    fam_pen[(str(fam), int(rid))] = 0.50
                else:
                    fam_pen[(str(fam), int(rid))] = 1.0
    return fam_pen, reg_pen


def build_regime_experts(
    *,
    eval_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    aligned_regimes_df: pd.DataFrame,
    symbol: str,
    transfer_by_regime_df: pd.DataFrame | None = None,
    cfg: ExpertBuildConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ExpertBuildConfig()
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "aligned_regime_id",
                "signal",
                "family",
                "trade_count",
                "mean_net_ret",
                "win_rate",
                "fill_rate",
                "base_weight",
                "penalty",
                "weight",
                "expected_trigger_rate",
                "expected_fill_rate",
                "expert_quality",
                "regime_rows",
            ]
        )

    tr = trades_df.copy()
    tr = attach_aligned_regime(tr, aligned_regimes_df, symbol=symbol)
    tr["signal"] = tr.get("signal", pd.Series([], dtype=str)).astype(str)
    tr["net_ret"] = _safe_num(tr.get("net_ret", pd.Series([], dtype=float)))
    tr["filled"] = _safe_num(tr.get("filled", pd.Series(np.ones(len(tr), dtype=float), index=tr.index)))
    tr["aligned_regime_id"] = pd.to_numeric(tr.get("aligned_regime_id"), errors="coerce").fillna(-1).astype(int)

    ev = eval_df.copy() if not eval_df.empty else pd.DataFrame(columns=["signal", "stability_score"])
    ev["signal"] = ev.get("signal", pd.Series([], dtype=str)).astype(str)
    stab = ev.groupby("signal", as_index=False).agg(stability_score=("stability_score", "mean")) if not ev.empty else pd.DataFrame(columns=["signal", "stability_score"])

    regime_rows_map = (
        aligned_regimes_df[aligned_regimes_df["symbol"] == symbol]["aligned_regime_id"].value_counts().to_dict()
        if not aligned_regimes_df.empty and "aligned_regime_id" in aligned_regimes_df.columns
        else {}
    )
    fam_pen, reg_pen = compute_transfer_penalties(transfer_by_regime_df if transfer_by_regime_df is not None else pd.DataFrame())

    agg = (
        tr.groupby(["aligned_regime_id", "signal"], as_index=False)
        .agg(
            trade_count=("signal", "count"),
            mean_net_ret=("net_ret", "mean"),
            win_rate=("net_ret", lambda s: float((_safe_num(s) > 0).mean())),
            fill_rate=("filled", "mean"),
        )
        .sort_values(["aligned_regime_id", "signal"])
        .reset_index(drop=True)
    )
    if agg.empty:
        return agg
    agg = agg.merge(stab, on="signal", how="left")
    agg["stability_score"] = _safe_num(agg.get("stability_score", pd.Series([], dtype=float)))
    agg["family"] = agg["signal"].map(infer_family)
    agg["regime_rows"] = agg["aligned_regime_id"].map(lambda r: int(regime_rows_map.get(int(r), 0)))
    agg = agg[agg["trade_count"] >= int(cfg.min_trades_per_signal)].copy()
    agg = agg[agg["regime_rows"] >= int(cfg.min_regime_rows)].copy()
    if agg.empty:
        return agg

    agg["base_weight"] = _safe_num(agg["mean_net_ret"]) * (1.0 + _safe_num(agg["stability_score"]).clip(lower=0.0))
    agg["penalty"] = 1.0
    agg["penalty"] = agg.apply(
        lambda r: float(
            fam_pen.get((str(r["family"]), int(r["aligned_regime_id"])), reg_pen.get(int(r["aligned_regime_id"]), 1.0))
        ),
        axis=1,
    )
    agg["weight"] = _safe_num(agg["base_weight"]) * _safe_num(agg["penalty"])
    # fallback to tiny positive ranking if all <=0 in a regime
    out_rows: List[pd.DataFrame] = []
    for rid, g in agg.groupby("aligned_regime_id", sort=True):
        gg = g.sort_values(["weight", "mean_net_ret", "signal"], ascending=[False, False, True]).head(max(1, int(cfg.top_k_per_regime))).copy()
        if float(_safe_num(gg["weight"]).sum()) <= 0:
            gg["weight"] = (_safe_num(gg["mean_net_ret"]).rank(method="first", ascending=False) / max(1, len(gg))).astype(float)
        sw = float(_safe_num(gg["weight"]).sum())
        gg["weight"] = _safe_num(gg["weight"]) / max(1e-9, sw)
        quality = float((_safe_num(gg["mean_net_ret"]) * _safe_num(gg["weight"])).sum())
        gg["expert_quality"] = quality
        gg["expected_trigger_rate"] = _safe_num(gg["trade_count"]).sum() / max(1, int(gg["regime_rows"].iloc[0]))
        gg["expected_fill_rate"] = float(_safe_num(gg["fill_rate"]).mean())
        out_rows.append(gg)
    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    out["symbol"] = str(symbol)
    keep = [
        "symbol",
        "aligned_regime_id",
        "signal",
        "family",
        "trade_count",
        "mean_net_ret",
        "win_rate",
        "fill_rate",
        "base_weight",
        "penalty",
        "weight",
        "expected_trigger_rate",
        "expected_fill_rate",
        "expert_quality",
        "regime_rows",
    ]
    return out[keep].sort_values(["aligned_regime_id", "weight", "signal"], ascending=[True, False, True]).reset_index(drop=True)

