from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _safe_series(df: pd.DataFrame, col: str, default: Any) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def build_risk_attribution(
    trades_df: pd.DataFrame,
    *,
    gating_df: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    if trades_df is None or trades_df.empty:
        empty = pd.DataFrame(columns=["group", "count", "net_sum", "net_mean", "win_rate"])
        return {"by_side": empty.copy(), "by_fill": empty.copy(), "by_reason": empty.copy(), "by_expert": empty.copy()}

    t = trades_df.copy()
    t["side"] = _safe_series(t, "side", "unknown").astype(str)
    t["filled"] = pd.to_numeric(_safe_series(t, "filled", 1), errors="coerce").fillna(0).astype(int)
    t["risk_reason"] = _safe_series(t, "risk_reason", "unknown").astype(str)
    net = pd.to_numeric(_safe_series(t, "pnl_net_notional", 0.0), errors="coerce")
    if net.abs().sum() <= 0:
        net = pd.to_numeric(_safe_series(t, "pnl_net", 0.0), errors="coerce").fillna(0.0)
    t["_net"] = net.fillna(0.0)
    t["_win"] = (t["_net"] > 0.0).astype(float)

    def _agg(group_col: str, rename: str = "group") -> pd.DataFrame:
        g = (
            t.groupby(group_col, dropna=False)
            .agg(
                count=("_net", "size"),
                net_sum=("_net", "sum"),
                net_mean=("_net", "mean"),
                win_rate=("_win", "mean"),
            )
            .reset_index()
            .rename(columns={group_col: rename})
            .sort_values(["net_sum", "count"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return g

    by_side = _agg("side")
    by_fill = _agg("filled")
    by_reason = _agg("risk_reason")

    by_expert = pd.DataFrame(columns=["group", "count", "net_sum", "net_mean", "win_rate"])
    if gating_df is not None and not gating_df.empty and "ts_ms" in gating_df.columns and "_entry_ts_ms" in t.columns:
        g = gating_df.copy()
        g["ts_ms"] = pd.to_numeric(g["ts_ms"], errors="coerce").fillna(-1).astype("int64")
        g = g.drop_duplicates(subset=["ts_ms"], keep="last")
        g["active_expert_id"] = pd.to_numeric(_safe_series(g, "active_expert_id", -1), errors="coerce").fillna(-1).astype(int)
        t["_entry_ts_ms"] = pd.to_numeric(t["_entry_ts_ms"], errors="coerce").fillna(-1).astype("int64")
        m = t.merge(g[["ts_ms", "active_expert_id"]], left_on="_entry_ts_ms", right_on="ts_ms", how="left")
        m["active_expert_id"] = pd.to_numeric(_safe_series(m, "active_expert_id", -1), errors="coerce").fillna(-1).astype(int)
        m["_win"] = (pd.to_numeric(_safe_series(m, "_net", 0.0), errors="coerce").fillna(0.0) > 0.0).astype(float)
        by_expert = (
            m.groupby("active_expert_id", dropna=False)
            .agg(
                count=("_net", "size"),
                net_sum=("_net", "sum"),
                net_mean=("_net", "mean"),
                win_rate=("_win", "mean"),
            )
            .reset_index()
            .rename(columns={"active_expert_id": "group"})
            .sort_values(["net_sum", "count"], ascending=[False, False])
            .reset_index(drop=True)
        )
        by_expert["group"] = by_expert["group"].astype(int)

    return {
        "by_side": by_side,
        "by_fill": by_fill,
        "by_reason": by_reason,
        "by_expert": by_expert,
    }

