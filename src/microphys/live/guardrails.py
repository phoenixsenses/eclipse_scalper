from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.microphys.alpha.calibration import CalibrationContext
from src.microphys.alpha.dsl import evaluate_expr
from src.microphys.live.probes import DirectionalProbe, Probe, default_directional_probes, default_probes


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def validate_calibration_payload(
    payload: Dict[str, Any],
    *,
    required_columns: Iterable[str] = ("F_ofi_z", "F_intensity_z", "spread_z"),
    max_nan_ratio: float = 0.8,
) -> tuple[bool, List[str]]:
    errs: List[str] = []
    q = dict(payload.get("quantiles", {}) or {})
    n = dict(payload.get("nan_ratio", {}) or {})
    for col in required_columns:
        if col not in q:
            errs.append(f"missing_quantiles:{col}")
            continue
        qv = dict(q.get(col, {}) or {})
        if not qv:
            errs.append(f"empty_quantiles:{col}")
            continue
        keys = sorted(float(k) for k in qv.keys())
        vals = [float(qv[f"{k:.4f}"] if f"{k:.4f}" in qv else qv[str(k)]) for k in keys]
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                errs.append(f"non_monotone_quantiles:{col}")
                break
        nr_raw = n.get(col, 1.0)
        nr = float(1.0 if nr_raw is None else nr_raw)
        if nr > float(max_nan_ratio):
            errs.append(f"nan_ratio_too_high:{col}:{nr:.4f}")
    try:
        sc = int(payload.get("sample_count", 0) or 0)
        if sc <= 0:
            errs.append("sample_count_non_positive")
    except Exception:
        errs.append("sample_count_invalid")
    return (len(errs) == 0), errs


def validate_execution_params_payload(
    payload: Dict[str, Any],
    *,
    min_fill_threshold: float = 0.0,
    max_fill_threshold: float = 1.0,
    min_ttl_bars: int = 1,
    max_ttl_bars: int = 10_000,
) -> tuple[bool, List[str]]:
    errs: List[str] = []
    mh = dict(payload.get("maker_hazard", {}) or {})
    mq = dict(payload.get("maker_queue", {}) or {})
    adv = dict(payload.get("adverse", {}) or {})
    for key in ("a", "b", "c", "d", "fill_threshold"):
        if key not in mh or not _is_finite(mh.get(key)):
            errs.append(f"maker_hazard_invalid:{key}")
    if "fill_threshold" in mh:
        ft = float(mh.get("fill_threshold", 0.0) or 0.0)
        if ft < float(min_fill_threshold) or ft > float(max_fill_threshold):
            errs.append(f"fill_threshold_out_of_bounds:{ft:.4f}")
    ttl = int(mh.get("ttl_bars", 0) or 0)
    if ttl < int(min_ttl_bars) or ttl > int(max_ttl_bars):
        errs.append(f"maker_hazard_ttl_out_of_bounds:{ttl}")

    if "queue_frac" not in mq or not _is_finite(mq.get("queue_frac")):
        errs.append("maker_queue_invalid:queue_frac")
    else:
        qf = float(mq.get("queue_frac", 0.0) or 0.0)
        if qf < 0.0 or qf > 1.0:
            errs.append(f"maker_queue_frac_out_of_bounds:{qf:.4f}")
    qttl = int(mq.get("ttl_bars", 0) or 0)
    if qttl < int(min_ttl_bars) or qttl > int(max_ttl_bars):
        errs.append(f"maker_queue_ttl_out_of_bounds:{qttl}")

    for k in ("buy_mean", "sell_mean"):
        if k not in adv or not _is_finite(adv.get(k)):
            errs.append(f"adverse_invalid:{k}")

    return (len(errs) == 0), errs


def validate_calibration_file(path: Path) -> tuple[bool, List[str], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ok, errs = validate_calibration_payload(payload)
    return ok, errs, payload


def validate_execution_file(path: Path) -> tuple[bool, List[str], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ok, errs = validate_execution_params_payload(payload)
    return ok, errs, payload


def _expr_columns(expr: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    kind = str(expr.get("type", "")).lower()
    if kind in {"gt", "gte", "lt", "lte", "eq", "ne"}:
        left = str(expr.get("left", "")).strip()
        if left:
            out.append(left)
        right = expr.get("right")
        if isinstance(right, dict):
            out.extend(_expr_columns(right))
        return out
    if kind == "fn":
        col = str(expr.get("col", "")).strip()
        if col:
            out.append(col)
        return out
    if kind == "in":
        col = str(expr.get("col", "")).strip()
        if col:
            out.append(col)
        return out
    if kind in {"and", "or", "not"}:
        for a in list(expr.get("args", []) or []):
            if isinstance(a, dict):
                out.extend(_expr_columns(a))
    return out


def evaluate_probe_trigger_sanity(
    frame: pd.DataFrame,
    calibration_payload: Dict[str, Any],
    *,
    probes: List[Probe] | None = None,
    probe_min_triggers: int = 10,
    probe_max_density: float = 0.95,
    total_density_min: float = 0.001,
    total_density_max: float = 0.60,
) -> tuple[bool, List[str], Dict[str, Any]]:
    errs: List[str] = []
    if frame.empty:
        return False, ["probe_eval_empty_frame"], {"probe_stats": [], "total_density": 0.0, "days": 0.0}
    ctx = CalibrationContext.from_dict(calibration_payload)
    items = list(probes or default_probes())
    n = int(len(frame))
    if "ts_utc" in frame.columns:
        t = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        days = float(max(1.0 / 24.0, (t.max() - t.min()).total_seconds() / 86400.0)) if t.notna().any() else 1.0
    elif "ts_ms" in frame.columns:
        x = pd.to_numeric(frame["ts_ms"], errors="coerce")
        days = float(max(1.0 / 24.0, (float(x.max()) - float(x.min())) / 1000.0 / 86400.0)) if x.notna().any() else 1.0
    else:
        days = 1.0
    probe_stats: List[Dict[str, Any]] = []
    union_mask = pd.Series([False] * n, index=frame.index)
    for p in items:
        mask = evaluate_expr(frame, p.condition, calibration=ctx).fillna(False)
        union_mask = union_mask | mask
        trig = int(mask.sum())
        density = float(trig / max(1, n))
        tpd = float(trig / max(1e-9, days))
        cols = sorted(set(_expr_columns(p.condition)))
        col_missing = [c for c in cols if c not in frame.columns]
        na_block = 0
        if cols:
            for c in cols:
                if c in frame.columns:
                    na_block += int(pd.to_numeric(frame[c], errors="coerce").isna().sum())
        na_block = int(min(n, na_block))
        min_tpd = int(max(0, p.min_triggers_per_day, probe_min_triggers))
        if tpd < float(min_tpd):
            errs.append(f"probe_low_trigger:{p.name}:{tpd:.3f}<min:{min_tpd}")
        if density > float(probe_max_density):
            errs.append(f"probe_high_density:{p.name}:{density:.6f}>max:{float(probe_max_density):.6f}")
        if col_missing:
            errs.append(f"probe_missing_columns:{p.name}:{','.join(col_missing)}")
        probe_stats.append(
            {
                "probe": p.name,
                "triggers": trig,
                "triggers_per_day": tpd,
                "density": density,
                "required_cols": cols,
                "na_blocked_count": na_block,
            }
        )
    total_density = float(union_mask.sum() / max(1, n))
    if total_density < float(total_density_min):
        errs.append(f"total_density_too_low:{total_density:.6f}<min:{float(total_density_min):.6f}")
    if total_density > float(total_density_max):
        errs.append(f"total_density_too_high:{total_density:.6f}>max:{float(total_density_max):.6f}")
    summary = {
        "probe_stats": probe_stats,
        "total_density": total_density,
        "total_rows": n,
        "days": days,
        "bands": {
            "probe_min_triggers": int(probe_min_triggers),
            "probe_max_density": float(probe_max_density),
            "total_density_min": float(total_density_min),
            "total_density_max": float(total_density_max),
        },
    }
    return (len(errs) == 0), errs, summary


def _forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    col = f"r_{int(horizon)}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if "mid" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    mid = pd.to_numeric(df["mid"], errors="coerce")
    if mid.dropna().empty:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    return (mid.shift(-int(horizon)) - mid) / (mid.replace(0.0, pd.NA).astype(float))


def evaluate_probe_directional_sanity(
    frame: pd.DataFrame,
    calibration_payload: Dict[str, Any],
    *,
    probes: List[DirectionalProbe] | None = None,
    horizons: Iterable[int] = (1, 5),
    min_dir_triggers: int = 50,
    max_fail_probes: int = 2,
    mean_eps: float = 0.0,
    min_win_rate: float = 0.40,
) -> tuple[bool, List[str], Dict[str, Any]]:
    errs: List[str] = []
    if frame.empty:
        return False, ["directional_eval_empty_frame"], {"directional_probe_stats": [], "failed_count": 0}
    ctx = CalibrationContext.from_dict(calibration_payload)
    hs = [max(1, int(h)) for h in horizons]
    items = list(probes or default_directional_probes())
    stats: List[Dict[str, Any]] = []
    fail_count = 0
    for p in items:
        h = int(p.horizon_bars)
        if h not in hs:
            h = hs[0]
        mask = evaluate_expr(frame, p.condition, calibration=ctx).fillna(False)
        fwd = pd.to_numeric(_forward_return(frame, h), errors="coerce")
        signed = fwd if str(p.side).lower() == "buy" else (-1.0 * fwd)
        vals = signed[mask].replace([float("inf"), float("-inf")], pd.NA).dropna()
        n = int(len(vals))
        mean_v = float(vals.mean()) if n > 0 else 0.0
        med_v = float(vals.median()) if n > 0 else 0.0
        win = float((vals > 0).mean()) if n > 0 else 0.0
        failed = False
        reason = ""
        if n >= int(min_dir_triggers):
            if mean_v < (-1.0 * float(mean_eps)):
                failed = True
                reason = f"negative_mean:{mean_v:.8f}"
            elif win < float(min_win_rate):
                failed = True
                reason = f"low_win_rate:{win:.6f}"
        if failed:
            fail_count += 1
            errs.append(f"directional_probe_failed:{p.name}:{reason}")
        stats.append(
            {
                "probe": p.name,
                "side": str(p.side).lower(),
                "horizon_bars": h,
                "n_triggers": n,
                "mean_signed_return": mean_v,
                "median_signed_return": med_v,
                "win_rate": win,
                "failed": bool(failed),
                "failed_reason": reason,
            }
        )
    if int(fail_count) > int(max_fail_probes):
        errs.append(f"directional_failed_count_exceeded:{int(fail_count)}>{int(max_fail_probes)}")
    summary = {
        "directional_probe_stats": stats,
        "failed_count": int(fail_count),
        "max_fail_probes": int(max_fail_probes),
        "min_dir_triggers": int(min_dir_triggers),
        "mean_eps": float(mean_eps),
        "min_win_rate": float(min_win_rate),
        "horizons": [int(x) for x in hs],
    }
    return (len(errs) == 0), errs, summary
