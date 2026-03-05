from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List

import pandas as pd

from .selection import summarize_signals


@dataclass(frozen=True)
class FilterSetting:
    min_trades_per_split: int
    require_positive_all_folds: bool
    stability_max_cv: float
    allow_one_fold_negative: int


def run_filter_sweep(eval_df: pd.DataFrame, settings: Iterable[FilterSetting]) -> pd.DataFrame:
    summary = summarize_signals(eval_df)
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "min_trades_per_split",
                "require_positive_all_folds",
                "stability_max_cv",
                "allow_one_fold_negative",
                "selected_count",
                "top_signal",
                "top_score",
            ]
        )
    rows: List[dict] = []
    for s in settings:
        df = summary.copy()
        req_trades = df["splits"].clip(lower=1) * int(s.min_trades_per_split)
        neg_folds = pd.to_numeric(df["splits"], errors="coerce") - pd.to_numeric(df["positive_test_folds"], errors="coerce")
        mask = pd.to_numeric(df["test_trade_count"], errors="coerce") >= req_trades
        if s.require_positive_all_folds:
            mask = mask & (neg_folds <= int(s.allow_one_fold_negative))
        cv = 1.0 - pd.to_numeric(df["stability_score"], errors="coerce")
        mask = mask & (cv <= float(s.stability_max_cv))
        selected = df[mask].sort_values(["composite_score", "signal"], ascending=[False, True]).reset_index(drop=True)
        rows.append(
            {
                "min_trades_per_split": int(s.min_trades_per_split),
                "require_positive_all_folds": int(bool(s.require_positive_all_folds)),
                "stability_max_cv": float(s.stability_max_cv),
                "allow_one_fold_negative": int(s.allow_one_fold_negative),
                "selected_count": int(len(selected)),
                "top_signal": str(selected.iloc[0]["signal"]) if not selected.empty else "",
                "top_score": float(selected.iloc[0]["composite_score"]) if not selected.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selected_count", "top_score", "min_trades_per_split", "stability_max_cv", "allow_one_fold_negative"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def default_settings() -> List[FilterSetting]:
    out: List[FilterSetting] = []
    for min_tr, req_all, cv, allow_neg in product((10, 25, 50), (True, False), (0.5, 1.0, 2.0), (0, 1)):
        out.append(
            FilterSetting(
                min_trades_per_split=int(min_tr),
                require_positive_all_folds=bool(req_all),
                stability_max_cv=float(cv),
                allow_one_fold_negative=int(allow_neg),
            )
        )
    return out
