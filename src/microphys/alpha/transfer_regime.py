from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def attach_aligned_regime(
    trades: pd.DataFrame,
    aligned: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    if trades.empty:
        return trades
    t = trades.copy()
    a = aligned[aligned["symbol"] == symbol].copy() if "symbol" in aligned.columns else aligned.copy()
    if a.empty:
        t["aligned_regime_id"] = -1
        return t
    if "ts_ms" in t.columns and "ts_ms" in a.columns:
        amap = a[["ts_ms", "aligned_regime_id"]].drop_duplicates(subset=["ts_ms"], keep="last")
        out = t.merge(amap, on="ts_ms", how="left")
    else:
        t["ts_utc"] = pd.to_datetime(t.get("ts_utc"), utc=True, errors="coerce")
        a["ts_utc"] = pd.to_datetime(a.get("ts_utc"), utc=True, errors="coerce")
        amap = a[["ts_utc", "aligned_regime_id"]].dropna().drop_duplicates(subset=["ts_utc"], keep="last")
        out = t.merge(amap, on="ts_utc", how="left")
    out["aligned_regime_id"] = pd.to_numeric(out.get("aligned_regime_id"), errors="coerce").fillna(-1).astype(int)
    return out


def summarize_transfer_by_regime(
    trades: pd.DataFrame,
    aligned_target: pd.DataFrame,
    *,
    target_symbol: str,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "aligned_regime_id",
                "trade_count",
                "trigger_rate",
                "mean_net_ret",
                "win_rate",
                "fill_rate",
            ]
        )
    at = aligned_target[aligned_target["symbol"] == target_symbol].copy() if "symbol" in aligned_target.columns else aligned_target.copy()
    at_counts = at["aligned_regime_id"].value_counts().to_dict() if "aligned_regime_id" in at.columns else {}
    tt = attach_aligned_regime(trades, aligned_target, symbol=target_symbol)
    if "filled" in tt.columns:
        fill_col = pd.to_numeric(tt.get("filled"), errors="coerce").fillna(0.0)
    else:
        fill_col = pd.Series(np.ones(len(tt), dtype=float), index=tt.index)
    grp = tt.groupby("aligned_regime_id", as_index=False).agg(
        trade_count=("signal", "count"),
        mean_net_ret=("net_ret", "mean"),
        win_rate=("net_ret", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
        fill_rate=("signal", lambda x: float(fill_col.loc[x.index].mean()) if len(x.index) else 0.0),
    )
    grp["target_bars"] = grp["aligned_regime_id"].map(lambda rid: int(at_counts.get(int(rid), 0)))
    grp["trigger_rate"] = pd.to_numeric(grp["trade_count"], errors="coerce") / pd.to_numeric(grp["target_bars"], errors="coerce").replace(0.0, np.nan)
    grp["trigger_rate"] = grp["trigger_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return grp.sort_values("aligned_regime_id").reset_index(drop=True)


def mismatch_diagnostic(source_by_regime: pd.DataFrame, target_presence: pd.DataFrame) -> Dict[str, Any]:
    if source_by_regime.empty or target_presence.empty:
        return {"has_mismatch": False, "source_focus_regime": -1, "target_presence": 0.0}
    src = source_by_regime.copy()
    src["w"] = pd.to_numeric(src.get("trade_count"), errors="coerce").fillna(0.0)
    if float(src["w"].sum()) <= 0:
        return {"has_mismatch": False, "source_focus_regime": -1, "target_presence": 0.0}
    top = src.sort_values(["w", "aligned_regime_id"], ascending=[False, True]).iloc[0]
    rid = int(top["aligned_regime_id"])
    tmap = dict(zip(target_presence["aligned_regime_id"].astype(int).tolist(), target_presence["presence_frac"].astype(float).tolist()))
    pres = float(tmap.get(rid, 0.0))
    return {
        "has_mismatch": bool(pres < 0.10),
        "source_focus_regime": rid,
        "target_presence": pres,
    }

