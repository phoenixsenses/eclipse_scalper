from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .calibration import CalibrationContext
from .dsl import evaluate_expr
from .spec import SignalSpec


def pick_topk_by_regime(
    summary_df: pd.DataFrame,
    signal_specs: Dict[str, SignalSpec],
    *,
    top_k: int = 3,
) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    if summary_df.empty:
        return out
    candidates = summary_df.sort_values(["composite_score", "signal"], ascending=[False, True]).reset_index(drop=True)
    for _, row in candidates.iterrows():
        name = str(row["signal"])
        spec = signal_specs.get(name)
        if spec is None:
            continue
        regimes = list(spec.regime_filter or [])
        if not regimes:
            continue
        for rid in regimes:
            cur = out.setdefault(int(rid), [])
            if len(cur) < int(top_k) and name not in cur:
                cur.append(name)
    return out


def build_ensemble_scores(
    df: pd.DataFrame,
    selected_specs: Iterable[SignalSpec],
    weights: Dict[str, float] | None = None,
    calibration: CalibrationContext | None = None,
) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    out = frame[["ts_ms", "ts_utc", "symbol", "regime_id"]].copy()
    score = np.zeros(len(frame), dtype=float)
    fire_count = np.zeros(len(frame), dtype=float)
    for spec in selected_specs:
        w = float((weights or {}).get(spec.name, 1.0))
        mask = evaluate_expr(frame, spec.condition, calibration=calibration).fillna(False)
        if spec.regime_filter:
            mask = mask & frame.get("regime_id", pd.Series(index=frame.index, dtype=float)).isin(spec.regime_filter).fillna(False)
        side = 1.0 if spec.side == "buy" else (-1.0 if spec.side == "sell" else 0.0)
        sig = mask.astype(float).to_numpy(dtype=float) * side
        score += w * sig
        fire_count += mask.astype(float).to_numpy(dtype=float)
    out["ensemble_score"] = score
    out["signal_count"] = fire_count.astype(int)
    out["ensemble_side"] = np.sign(score)
    return out
