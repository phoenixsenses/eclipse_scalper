from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .calibration import CalibrationContext

_CMP = {"gt", "gte", "lt", "lte", "eq", "ne"}
_BOOL = {"and", "or", "not"}
_FUN = {"z_gt", "pct_lt", "q_gt", "q_lt", "abs_q_gt", "between_q"}


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _comparison(df: pd.DataFrame, expr: Dict[str, Any]) -> pd.Series:
    op = str(expr.get("op", "")).lower()
    if op not in _CMP:
        raise ValueError(f"invalid op: {op}")
    left = _series(df, str(expr.get("left", "")))
    right_raw = expr.get("right", 0.0)
    if isinstance(right_raw, dict):
        right = evaluate_expr(df, right_raw)
    else:
        right = float(right_raw)
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    if op == "eq":
        return left == right
    return left != right


def _cal_q(df: pd.DataFrame, col: str, q: float, calibration: CalibrationContext | None, abs_mode: bool = False) -> float:
    if calibration is not None:
        key = f"abs({col})" if abs_mode else col
        return float(calibration.q(key, q, default=0.0))
    x = _series(df, col).abs() if abs_mode else _series(df, col)
    return float(x.quantile(max(0.0, min(1.0, float(q)))))


def _function(df: pd.DataFrame, expr: Dict[str, Any], calibration: CalibrationContext | None) -> pd.Series:
    fn = str(expr.get("fn", "")).lower()
    if fn not in _FUN:
        raise ValueError(f"invalid fn: {fn}")
    col = str(expr.get("col", ""))
    x = _series(df, col)
    if fn == "z_gt":
        thr = float(expr.get("thr", 0.0))
        return x > thr
    if fn == "pct_lt":
        p = float(expr.get("p", 0.2))
        q = _cal_q(df, col, p, calibration, abs_mode=False)
        return x < q
    if fn == "q_gt":
        q = float(expr.get("q", 0.9))
        thr = _cal_q(df, col, q, calibration, abs_mode=False)
        return x > thr
    if fn == "q_lt":
        q = float(expr.get("q", 0.1))
        thr = _cal_q(df, col, q, calibration, abs_mode=False)
        return x < thr
    if fn == "abs_q_gt":
        q = float(expr.get("q", 0.9))
        thr = _cal_q(df, col, q, calibration, abs_mode=True)
        return x.abs() > thr
    q_lo = float(expr.get("q_lo", 0.2))
    q_hi = float(expr.get("q_hi", 0.8))
    lo = _cal_q(df, col, q_lo, calibration, abs_mode=False)
    hi = _cal_q(df, col, q_hi, calibration, abs_mode=False)
    return (x >= lo) & (x <= hi)


def evaluate_expr(df: pd.DataFrame, expr: Dict[str, Any], calibration: CalibrationContext | None = None) -> pd.Series:
    kind = str(expr.get("type", "")).lower()
    if kind in _CMP:
        return _comparison(df, expr).fillna(False)
    if kind == "in":
        col = str(expr.get("col", ""))
        vals = set(expr.get("values", []) or [])
        return df.get(col, pd.Series(index=df.index, dtype=object)).isin(vals).fillna(False)
    if kind in _BOOL:
        args = expr.get("args", []) or []
        if kind == "not":
            if len(args) != 1:
                raise ValueError("not requires one arg")
            return ~evaluate_expr(df, dict(args[0]), calibration=calibration).fillna(False)
        if not args:
            return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
        out = evaluate_expr(df, dict(args[0]), calibration=calibration).fillna(False)
        for raw in args[1:]:
            nxt = evaluate_expr(df, dict(raw), calibration=calibration).fillna(False)
            out = out & nxt if kind == "and" else out | nxt
        return out
    if kind == "fn":
        return _function(df, expr, calibration=calibration).fillna(False)
    raise ValueError(f"invalid expression type: {kind}")
