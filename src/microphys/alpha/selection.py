from __future__ import annotations

from typing import Dict

import pandas as pd

from .metrics import composite_score


def summarize_signals(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "splits",
                "test_trade_count",
                "test_net_mean",
                "test_sharpe",
                "stability_score",
                "overfit_gap",
                "regime_concentration",
                "composite_score",
            ]
        )
    g = (
        eval_df.groupby("signal", as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            test_trade_count=("test_trade_count", "sum"),
            test_net_mean=("test_net_mean", "mean"),
            test_sharpe=("test_sharpe", "mean"),
            stability_score=("stability_score", "mean"),
            overfit_gap=("overfit_gap", "mean"),
            regime_concentration=("regime_concentration", "mean"),
            positive_test_folds=("test_net_mean", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .sort_values("signal")
    )
    g["composite_score"] = g.apply(lambda r: composite_score(r.to_dict()), axis=1)
    return g.sort_values(["composite_score", "test_sharpe", "signal"], ascending=[False, False, True]).reset_index(drop=True)


def select_robust_signals(
    summary_df: pd.DataFrame,
    *,
    min_trades_per_split: int = 10,
    min_stability: float = 0.2,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    df = summary_df.copy()
    req_trades = df["splits"].clip(lower=1) * int(min_trades_per_split)
    mask = (
        (pd.to_numeric(df["test_trade_count"], errors="coerce") >= req_trades)
        & (pd.to_numeric(df["positive_test_folds"], errors="coerce") >= pd.to_numeric(df["splits"], errors="coerce"))
        & (pd.to_numeric(df["stability_score"], errors="coerce") >= float(min_stability))
    )
    out = df[mask].copy()
    return out.sort_values(["composite_score", "test_sharpe", "signal"], ascending=[False, False, True]).reset_index(drop=True)
