from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.microphys.alpha.calibration import CalibrationContext
from src.microphys.alpha.dsl import evaluate_expr
from src.microphys.alpha.spec import SignalSpec


def build_gating_decisions(
    frame: pd.DataFrame,
    experts_df: pd.DataFrame,
    *,
    regime_col: str = "aligned_regime_id",
    data_quality_ok: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts_ms", "active_expert_id", "confidence_score", "reason", "fallback_used"])
    out = frame[["ts_ms"]].copy() if "ts_ms" in frame.columns else pd.DataFrame({"ts_ms": np.arange(len(frame), dtype=int)})
    rid = pd.to_numeric(frame.get(regime_col, pd.Series(np.full(len(frame), -1))), errors="coerce").fillna(-1).astype(int)
    regime_quality = (
        experts_df.groupby("aligned_regime_id", as_index=False).agg(expert_quality=("expert_quality", "mean"))
        if not experts_df.empty
        else pd.DataFrame(columns=["aligned_regime_id", "expert_quality"])
    )
    qmap = {int(r["aligned_regime_id"]): float(r["expert_quality"]) for _, r in regime_quality.iterrows()}
    has_reg = set(int(x) for x in experts_df.get("aligned_regime_id", pd.Series([], dtype=int)).tolist())

    active = []
    conf = []
    reason = []
    fallback = []
    for x in rid.tolist():
        if (x in has_reg) and data_quality_ok:
            active.append(int(x))
            c = qmap.get(int(x), 0.0)
            conf.append(float(max(0.0, min(1.0, 0.5 + c * 50.0))))
            reason.append("expert_active")
            fallback.append(False)
        elif x in has_reg:
            active.append(int(x))
            conf.append(0.0)
            reason.append("data_quality_low")
            fallback.append(True)
        else:
            active.append(-1)
            conf.append(0.0)
            reason.append("fallback_global")
            fallback.append(True)
    out["active_expert_id"] = pd.Series(active, dtype=int)
    out["confidence_score"] = pd.Series(conf, dtype=float)
    out["reason"] = pd.Series(reason, dtype=str)
    out["fallback_used"] = pd.Series(fallback, dtype=bool)
    return out


def build_gated_ensemble_scores(
    frame: pd.DataFrame,
    specs: Iterable[SignalSpec],
    experts_df: pd.DataFrame,
    *,
    calibration: CalibrationContext | None = None,
    regime_col: str = "aligned_regime_id",
    global_ensemble: pd.DataFrame | None = None,
    data_quality_ok: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    specs_map: Dict[str, SignalSpec] = {s.name: s for s in specs}
    needed = sorted(set(experts_df.get("signal", pd.Series([], dtype=str)).astype(str).tolist()))
    masks: Dict[str, np.ndarray] = {}
    sides: Dict[str, float] = {}
    for name in needed:
        s = specs_map.get(name)
        if s is None:
            continue
        m = evaluate_expr(frame, s.condition, calibration=calibration).fillna(False)
        if s.regime_filter and "regime_id" in frame.columns:
            m = m & frame["regime_id"].isin(s.regime_filter).fillna(False)
        masks[name] = m.to_numpy(dtype=float)
        sides[name] = 1.0 if s.side == "buy" else (-1.0 if s.side == "sell" else 0.0)

    gd = build_gating_decisions(frame, experts_df, regime_col=regime_col, data_quality_ok=data_quality_ok)
    score = np.zeros(len(frame), dtype=float)
    fire = np.zeros(len(frame), dtype=float)
    for rid, g in experts_df.groupby("aligned_regime_id", sort=True):
        idx = np.flatnonzero(gd["active_expert_id"].to_numpy(dtype=int) == int(rid))
        if idx.size == 0:
            continue
        s = np.zeros(len(idx), dtype=float)
        f = np.zeros(len(idx), dtype=float)
        for _, r in g.iterrows():
            nm = str(r["signal"])
            if nm not in masks:
                continue
            w = float(r.get("weight", 0.0) or 0.0)
            side = float(sides.get(nm, 0.0))
            mv = masks[nm][idx]
            s += w * side * mv
            f += mv
        score[idx] = s
        fire[idx] = f

    # fallback to global ensemble where no expert active or explicit fallback
    fb_idx = np.flatnonzero(gd["fallback_used"].to_numpy(dtype=bool))
    if fb_idx.size > 0 and global_ensemble is not None and not global_ensemble.empty:
        gscore = pd.to_numeric(global_ensemble.get("ensemble_score"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        gfire = pd.to_numeric(global_ensemble.get("signal_count"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        lim = min(len(gscore), len(score))
        take = fb_idx[fb_idx < lim]
        score[take] = gscore[take]
        fire[take] = gfire[take]

    out = frame[["ts_ms", "ts_utc", "symbol"]].copy()
    if "regime_id" in frame.columns:
        out["regime_id"] = pd.to_numeric(frame.get("regime_id"), errors="coerce").fillna(-1).astype(int)
    out["ensemble_score"] = score
    out["signal_count"] = fire.astype(int)
    out["ensemble_side"] = np.sign(score)
    gating = gd.copy()
    gating["used_signal_count"] = fire.astype(int)
    return out, gating

