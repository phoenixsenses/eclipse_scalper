from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .calibration import CalibrationContext
from .column_guard import validate_signal_columns
from .eval import apply_signal_entries, make_walkforward_splits
from .spec import SignalSpec


def _hash_name(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _flatten_and_args(expr: dict) -> list[dict]:
    if str(expr.get("type", "")).lower() == "and":
        return [dict(x) for x in (expr.get("args") or [])]
    return [dict(expr)]


def _compose_and(args: list[dict]) -> dict:
    if len(args) == 1:
        return args[0]
    return {"type": "and", "args": args}


def _build_expr(
    side: str,
    ofi_q: float,
    *,
    with_compression: bool,
    with_vacuum: bool,
    trend_mode: str,
    spread_gate_q: float | None,
    intensity_q: float | None,
) -> dict:
    args: list[dict] = []
    if with_compression:
        args.append({"type": "gte", "op": "gte", "left": "compression_flag", "right": 1})
    if with_vacuum:
        args.append({"type": "gte", "op": "gte", "left": "vacuum_flag", "right": 1})
    if spread_gate_q is not None:
        args.append({"type": "fn", "fn": "q_lt", "col": "spread_z", "q": float(spread_gate_q)})
    if intensity_q is not None:
        args.append({"type": "fn", "fn": "q_gt", "col": "F_intensity_z", "q": float(intensity_q)})

    if side == "buy":
        if trend_mode == "trend":
            args.append({"type": "fn", "fn": "q_gt", "col": "F_ofi_z", "q": float(ofi_q)})
        else:
            args.append({"type": "fn", "fn": "q_lt", "col": "F_ofi_z", "q": float(1.0 - ofi_q)})
    else:
        if trend_mode == "trend":
            args.append({"type": "fn", "fn": "q_lt", "col": "F_ofi_z", "q": float(1.0 - ofi_q)})
        else:
            args.append({"type": "fn", "fn": "q_gt", "col": "F_ofi_z", "q": float(ofi_q)})
    return _compose_and(args)


def _relax_expr(expr: dict, *, step: int) -> dict:
    out = deepcopy(expr)
    args = _flatten_and_args(out)
    # Step 1..: relax quantiles
    for node in args:
        if str(node.get("type", "")).lower() != "fn":
            continue
        fn = str(node.get("fn", "")).lower()
        if fn in {"q_gt", "abs_q_gt"} and "q" in node:
            base = float(node["q"])
            node["q"] = max(0.55, base - (0.05 * step))
        elif fn == "q_lt" and "q" in node:
            base = float(node["q"])
            node["q"] = min(0.45, base + (0.05 * step))
    # Later: drop hard gates
    if step >= 3:
        args = [n for n in args if str(n.get("left", "")) not in {"compression_flag", "vacuum_flag"}]
    if step >= 2:
        args = [n for n in args if not (str(n.get("type", "")).lower() == "fn" and str(n.get("col", "")) == "F_intensity_z")]
    if step >= 5:
        args = [n for n in args if not (str(n.get("type", "")).lower() == "fn" and str(n.get("fn", "")).lower() == "q_lt" and str(n.get("col", "")) == "spread_z")]
    return _compose_and(args if args else [{"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.8}])


def _tighten_expr(expr: dict, *, step: int) -> dict:
    out = deepcopy(expr)
    args = _flatten_and_args(out)
    for node in args:
        if str(node.get("type", "")).lower() != "fn":
            continue
        fn = str(node.get("fn", "")).lower()
        if fn in {"q_gt", "abs_q_gt"} and "q" in node:
            base = float(node["q"])
            node["q"] = min(0.999, base + (0.03 * step))
        elif fn == "q_lt" and "q" in node:
            base = float(node["q"])
            node["q"] = max(0.001, base - (0.03 * step))
    return _compose_and(args if args else [{"type": "fn", "fn": "abs_q_gt", "col": "F_ofi_z", "q": 0.99}])


def _triggered_count(frame: pd.DataFrame, spec: SignalSpec, calibration: CalibrationContext) -> int:
    stride = max(1, int(len(frame) // 5000))
    view = frame.iloc[::stride].reset_index(drop=True) if stride > 1 else frame
    m = apply_signal_entries(view, spec, calibration=calibration)
    wins = make_walkforward_splits(view["ts_ms"].astype(int).tolist(), 3)
    if not wins:
        return int(m.sum()) * stride
    total = 0
    ts = pd.to_numeric(view["ts_ms"], errors="coerce").fillna(0).astype(int)
    for w in wins:
        in_test = (ts >= int(w.test_start)) & (ts <= int(w.test_end))
        total += int((m & in_test).sum())
    return int(total) * stride


def _trigger_rate_per_day(frame: pd.DataFrame, spec: SignalSpec, calibration: CalibrationContext) -> float:
    stride = max(1, int(len(frame) // 5000))
    view = frame.iloc[::stride].reset_index(drop=True) if stride > 1 else frame
    m = apply_signal_entries(view, spec, calibration=calibration)
    n = int(m.sum())
    if n <= 0:
        return 0.0
    if "ts_utc" in view.columns:
        d = pd.to_datetime(view["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        days = int(max(1, d.dropna().nunique()))
    else:
        ts = pd.to_numeric(view.get("ts_ms"), errors="coerce").dropna()
        if ts.empty:
            days = 1
        else:
            span_ms = float(max(1.0, ts.max() - ts.min()))
            days = int(max(1.0, span_ms / 86_400_000.0))
    return float((n * stride) / max(1, days))


def _calibrate_selectivity(
    *,
    frame: pd.DataFrame,
    calibration: CalibrationContext,
    spec: SignalSpec,
    min_triggered: int,
    target_triggers_per_day: float,
    min_triggers_per_day: float,
    max_triggers_per_day: float,
    max_tries: int,
) -> tuple[SignalSpec, int, float, int, int]:
    current = spec
    trigger = _triggered_count(frame, current, calibration)
    tpd = _trigger_rate_per_day(frame, current, calibration)
    relax_steps = 0
    tighten_steps = 0

    for i in range(max(1, int(max_tries))):
        if trigger < int(min_triggered):
            relax_steps += 1
            current = SignalSpec(
                name=current.name,
                side=current.side,
                condition=_relax_expr(current.condition, step=i + 1),
                entry=current.entry,
                horizon_bars=current.horizon_bars,
                cooldown_bars=current.cooldown_bars,
                regime_filter=current.regime_filter,
                entry_mode_preference=current.entry_mode_preference,
                meta=current.meta,
            )
        elif tpd > float(max_triggers_per_day):
            tighten_steps += 1
            current = SignalSpec(
                name=current.name,
                side=current.side,
                condition=_tighten_expr(current.condition, step=i + 1),
                entry=current.entry,
                horizon_bars=current.horizon_bars,
                cooldown_bars=current.cooldown_bars,
                regime_filter=current.regime_filter,
                entry_mode_preference=current.entry_mode_preference,
                meta=current.meta,
            )
        elif tpd < float(min_triggers_per_day):
            relax_steps += 1
            current = SignalSpec(
                name=current.name,
                side=current.side,
                condition=_relax_expr(current.condition, step=i + 1),
                entry=current.entry,
                horizon_bars=current.horizon_bars,
                cooldown_bars=current.cooldown_bars,
                regime_filter=current.regime_filter,
                entry_mode_preference=current.entry_mode_preference,
                meta=current.meta,
            )
        else:
            break
        trigger = _triggered_count(frame, current, calibration)
        tpd = _trigger_rate_per_day(frame, current, calibration)

    band_lo = max(float(min_triggers_per_day), float(target_triggers_per_day) * 0.5)
    band_hi = min(float(max_triggers_per_day), float(target_triggers_per_day) * 1.5)
    if (tpd < band_lo or tpd > band_hi) and trigger > 0:
        # One final corrective step toward target band.
        current = SignalSpec(
            name=current.name,
            side=current.side,
            condition=(
                _tighten_expr(current.condition, step=1) if tpd > band_hi else _relax_expr(current.condition, step=1)
            ),
            entry=current.entry,
            horizon_bars=current.horizon_bars,
            cooldown_bars=current.cooldown_bars,
            regime_filter=current.regime_filter,
            entry_mode_preference=current.entry_mode_preference,
            meta=current.meta,
        )
        trigger = _triggered_count(frame, current, calibration)
        tpd = _trigger_rate_per_day(frame, current, calibration)
    return current, int(trigger), float(tpd), int(relax_steps), int(tighten_steps)


def generate_candidates(
    *,
    horizons: Iterable[int],
    compression_options: Iterable[bool],
    vacuum_options: Iterable[bool],
    regime_ids: Iterable[int] | None = None,
    limit: int = 500,
    calibration: CalibrationContext | None = None,
    frame: pd.DataFrame | None = None,
    coverage_guarantee: bool = False,
    min_triggered: int = 50,
    max_tries: int = 30,
    target_triggers_per_day: float = 200.0,
    min_triggers_per_day: float = 50.0,
    max_triggers_per_day: float = 500.0,
    available_columns: Iterable[str] | None = None,
    max_nan_ratio: float = 0.98,
) -> List[SignalSpec]:
    regimes = list(regime_ids or [])
    out: List[SignalSpec] = []
    ofi_qs = [0.99, 0.95, 0.90, 0.85, 0.80]
    spread_qs = [0.20, 0.30, None]
    intensity_qs: list[float | None] = [None, 0.90, 0.80]
    for hz in horizons:
        for use_comp in compression_options:
            for use_vac in vacuum_options:
                for side in ("buy", "sell"):
                    for trend_mode in ("trend", "contrarian"):
                        for spread_q in spread_qs:
                            for intensity_q in intensity_qs:
                                for ofi_q in ofi_qs:
                                    for entry_pref in ("taker", "maker", "both"):
                                        cond = _build_expr(
                                            side=side,
                                            ofi_q=ofi_q,
                                            with_compression=bool(use_comp),
                                            with_vacuum=bool(use_vac),
                                            trend_mode=trend_mode,
                                            spread_gate_q=spread_q,
                                            intensity_q=(float(intensity_q) if intensity_q is not None else None),
                                        )
                                        core = {
                                            "side": side,
                                            "horizon_bars": int(hz),
                                            "comp": bool(use_comp),
                                            "vac": bool(use_vac),
                                            "trend_mode": trend_mode,
                                            "spread_q": spread_q,
                                            "intensity_q": intensity_q,
                                            "ofi_q": ofi_q,
                                            "entry_pref": entry_pref,
                                            "regimes": regimes,
                                        }
                                        name = f"cand_{side}_{_hash_name(core)}"
                                        spec = SignalSpec(
                                            name=name,
                                            side=side,
                                            condition=cond,
                                            entry="market",
                                            horizon_bars=int(hz),
                                            cooldown_bars=max(1, int(hz // 2)),
                                            regime_filter=[int(x) for x in regimes],
                                            entry_mode_preference=entry_pref,
                                            meta={"tags": ["candidate", trend_mode, f"ofi_q{ofi_q:.2f}"]},
                                        )
                                        if available_columns is not None:
                                            vr = validate_signal_columns(
                                                spec,
                                                available_columns=available_columns,
                                                nan_ratio=(calibration.nan_ratio if calibration else {}),
                                                max_nan_ratio=float(max_nan_ratio),
                                            )
                                            if not vr.ok:
                                                continue
                                        if frame is not None and calibration is not None:
                                            if coverage_guarantee:
                                                spec, trigger, tpd, relax_steps, tighten_steps = _calibrate_selectivity(
                                                    frame=frame,
                                                    calibration=calibration,
                                                    spec=spec,
                                                    min_triggered=int(min_triggered),
                                                    target_triggers_per_day=float(target_triggers_per_day),
                                                    min_triggers_per_day=float(min_triggers_per_day),
                                                    max_triggers_per_day=float(max_triggers_per_day),
                                                    max_tries=int(max_tries),
                                                )
                                                if trigger <= 0:
                                                    continue
                                            else:
                                                trigger = _triggered_count(frame, spec, calibration)
                                                tpd = _trigger_rate_per_day(frame, spec, calibration)
                                                relax_steps = 0
                                                tighten_steps = 0
                                            spec = SignalSpec(
                                                name=spec.name,
                                                side=spec.side,
                                                condition=spec.condition,
                                                entry=spec.entry,
                                                horizon_bars=spec.horizon_bars,
                                                cooldown_bars=spec.cooldown_bars,
                                                regime_filter=spec.regime_filter,
                                                entry_mode_preference=spec.entry_mode_preference,
                                                meta={
                                                    **spec.meta,
                                                    "calibration_triggered": int(trigger),
                                                    "trigger_rate_per_day": float(tpd),
                                                    "relax_steps": int(relax_steps),
                                                    "tighten_steps": int(tighten_steps),
                                                    "target_triggers_per_day": float(target_triggers_per_day),
                                                    "final_quantile_mode": (
                                                        "tightened"
                                                        if int(tighten_steps) > 0
                                                        else ("relaxed" if int(relax_steps) > 0 else "base")
                                                    ),
                                                },
                                            )
                                        out.append(spec)
                                        if len(out) >= int(limit):
                                            return out
    if coverage_guarantee and frame is not None and calibration is not None and len(out) < min(int(limit), 120):
        hz_list = [max(1, int(h)) for h in list(horizons) or [5]]
        q_bands = [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
        for hz in hz_list:
            for side in ("buy", "sell"):
                for q_lo, q_hi in q_bands:
                    cond = {"type": "fn", "fn": "between_q", "col": "F_ofi_z", "q_lo": float(q_lo), "q_hi": float(q_hi)}
                    base = SignalSpec(
                        name=f"cand_fallback_{side}_{_hash_name({'hz': hz, 'q': (q_lo, q_hi), 's': side})}",
                        side=side,
                        condition=cond,
                        entry="market",
                        horizon_bars=hz,
                        cooldown_bars=max(1, hz // 2),
                        regime_filter=[int(x) for x in regimes],
                        entry_mode_preference="both",
                        meta={"tags": ["fallback"], "relax_steps": 99},
                    )
                    trig = _triggered_count(frame, base, calibration)
                    if trig <= 0:
                        continue
                    tpd = _trigger_rate_per_day(frame, base, calibration)
                    out.append(
                        SignalSpec(
                            name=base.name,
                            side=base.side,
                            condition=base.condition,
                            entry=base.entry,
                            horizon_bars=base.horizon_bars,
                            cooldown_bars=base.cooldown_bars,
                            regime_filter=base.regime_filter,
                            entry_mode_preference=base.entry_mode_preference,
                            meta={
                                **base.meta,
                                "calibration_triggered": int(trig),
                                "trigger_rate_per_day": float(tpd),
                                "relax_steps": 99,
                                "tighten_steps": 0,
                                "final_quantile_mode": "fallback",
                            },
                        )
                    )
                    if len(out) >= int(limit):
                        return out
    return out
