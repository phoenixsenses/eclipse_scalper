from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def calibrate_execution_models(physics: pd.DataFrame) -> Dict[str, Any]:
    df = physics.copy()
    if df.empty:
        return {
            "maker_queue": {"queue_frac": 0.25, "ttl_bars": 10, "min_depth": 1.0},
            "maker_queue_v2": calibrate_queue_position_params(df),
            "maker_hazard": {"a": 1.0, "b": -0.7, "c": 0.5, "d": -0.5, "ttl_bars": 10, "fill_threshold": 0.5},
            "adverse": {"buy_mean": 0.0, "sell_mean": 0.0},
        }

    r1 = _series(df, "r_1", 0.0).fillna(0.0)
    ofi = _series(df, "F_ofi_z", 0.0).fillna(0.0)
    spread = _series(df, "spread_z", 0.0).fillna(0.0)
    intensity = _series(df, "F_intensity_z", 0.0).fillna(0.0)

    # Heuristic calibration from empirical medians/quantiles.
    buy_adv = float(np.maximum(0.0, -r1[ofi > 0]).mean()) if bool((ofi > 0).any()) else 0.0
    sell_adv = float(np.maximum(0.0, r1[ofi < 0]).mean()) if bool((ofi < 0).any()) else 0.0
    ttl = int(max(3, min(20, np.ceil((spread.abs().median() + 1.0) * 5))))
    queue_frac = float(np.clip(0.15 + 0.1 * spread.abs().median(), 0.05, 0.50))

    params = {
        "maker_queue": {"queue_frac": queue_frac, "ttl_bars": ttl, "min_depth": 1.0},
        "maker_queue_v2": calibrate_queue_position_params(df),
        "maker_hazard": {
            "a": float(np.clip(1.0 + 0.2 * intensity.median(), 0.2, 3.0)),
            "b": float(np.clip(-0.7 - 0.1 * spread.median(), -2.0, -0.1)),
            "c": float(np.clip(0.5 + 0.1 * ofi.abs().median(), 0.1, 2.0)),
            "d": -0.5,
            "ttl_bars": ttl,
            "fill_threshold": 0.5,
        },
        "adverse": {"buy_mean": buy_adv, "sell_mean": sell_adv},
    }
    return params


def calibrate_queue_position_params(
    physics: pd.DataFrame,
    *,
    symbol: str | None = None,
    regime_id: int | None = None,
) -> Dict[str, Any]:
    df = physics.copy()
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()]
    if regime_id is not None and "regime_id" in df.columns:
        try:
            df = df[pd.to_numeric(df["regime_id"], errors="coerce").fillna(-9999).astype(int) == int(regime_id)]
        except Exception:
            pass

    if df.empty:
        return {
            "initial_queue_frac": 0.25,
            "same_side_join_rate": 0.10,
            "same_side_cancel_rate": 0.08,
            "opposite_flow_scale": 1.0,
            "pressure_floor": 0.10,
            "ttl_bars": 10,
            "min_depth": 1.0,
        }

    spread = _series(df, "spread_z", 0.0).abs().fillna(0.0)
    intensity = _series(df, "F_intensity_z", 0.0).fillna(0.0)
    tt = _series(df, "trade_through_prob", 0.5).fillna(0.5).clip(0.0, 1.0)

    queue_frac = float(np.clip(0.20 + 0.05 * float(spread.median()), 0.05, 0.60))
    join_rate = float(np.clip(0.06 + 0.03 * float(spread.quantile(0.75)), 0.01, 0.30))
    cancel_rate = float(np.clip(0.05 + 0.02 * float(tt.median()), 0.01, 0.40))
    flow_scale = float(np.clip(0.8 + 0.2 * float(intensity.median()), 0.20, 2.0))
    ttl = int(np.clip(np.ceil(6 + 2 * float(spread.median())), 3, 30))
    pressure_floor = float(np.clip(0.08 + 0.02 * float(tt.quantile(0.50)), 0.05, 0.40))
    return {
        "initial_queue_frac": queue_frac,
        "same_side_join_rate": join_rate,
        "same_side_cancel_rate": cancel_rate,
        "opposite_flow_scale": flow_scale,
        "pressure_floor": pressure_floor,
        "ttl_bars": ttl,
        "min_depth": 1.0,
    }


def save_execution_params(path: Path, params: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_execution_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
