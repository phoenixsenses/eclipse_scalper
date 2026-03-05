from __future__ import annotations

import math
from typing import Any, Dict

from src.microphys.risk.policy import RiskPolicy
from src.microphys.risk.schemas import RiskDecision


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return v
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _confidence_factor(score_abs: float, curve: Dict[str, float]) -> float:
    mid = _safe_float(curve.get("score_mid"), 0.20)
    slope = _safe_float(curve.get("slope"), 4.0)
    floor = _safe_float(curve.get("floor"), 0.0)
    ceil = _safe_float(curve.get("ceil"), 1.0)
    raw = _sigmoid((score_abs - mid) * slope)
    return float(max(0.0, min(1.0, floor + (ceil - floor) * raw)))


def _risk_downscale(z: float, scale: float) -> float:
    return float(1.0 / (1.0 + max(0.0, abs(_safe_float(z))) * max(0.0, _safe_float(scale))))


def compute_risk_decision(
    *,
    ts_ms: int,
    symbol: str,
    desired_side: str,
    signal_row: Dict[str, Any],
    gating_row: Dict[str, Any],
    live_status: Dict[str, Any],
    policy: RiskPolicy,
    mtm: Dict[str, Any],
) -> RiskDecision:
    equity = max(0.0, _safe_float(mtm.get("equity"), policy.starting_equity))
    base = equity * float(policy.base_risk_per_trade)
    score_abs = abs(_safe_float(signal_row.get("ensemble_score"), 0.0))
    confidence_factor = _confidence_factor(score_abs, policy.confidence_curve)
    quality = _safe_float(gating_row.get("confidence_score"), 0.0)
    quality_factor = float(max(0.0, min(1.0, quality)))
    spread_factor = _risk_downscale(_safe_float(signal_row.get("spread_z"), signal_row.get("spread", 0.0)), policy.spread_risk_scale)
    vol_factor = _risk_downscale(_safe_float(signal_row.get("rv_z"), signal_row.get("rv_short", 0.0)), policy.vol_risk_scale)
    liq_factor = _risk_downscale(_safe_float(signal_row.get("liq_rate_z"), signal_row.get("liq_rate", 0.0)), policy.liq_risk_scale)
    fill_est = _safe_float(signal_row.get("expected_fill_rate"), 1.0)
    execution_factor = float(max(0.0, min(1.0, fill_est)))
    health_factor = 1.0
    missing_ratio = _safe_float(live_status.get("missing_bars_pct_1h"), 0.0) / 100.0
    if bool(policy.health_skip_on_bad) and (missing_ratio > float(policy.max_missing_bar_ratio_1h)):
        health_factor = 0.0
    if _safe_float(live_status.get("regime_shift"), 0.0) > float(policy.drift_skip_threshold):
        health_factor = min(health_factor, 0.25)

    factors = {
        "confidence_factor": confidence_factor,
        "quality_factor": quality_factor,
        "spread_factor": spread_factor,
        "vol_factor": vol_factor,
        "liq_factor": liq_factor,
        "health_factor": health_factor,
        "execution_factor": execution_factor,
    }
    n = base
    for k in ("confidence_factor", "quality_factor", "spread_factor", "vol_factor", "liq_factor", "health_factor", "execution_factor"):
        n *= max(0.0, _safe_float(factors[k], 0.0))

    if quality_factor < float(policy.regime_quality_floor):
        return RiskDecision(ts_ms=ts_ms, symbol=symbol, desired_side=desired_side, base_notional=base, final_notional=0.0, action="SKIP", reason="RISK_SKIP_BAD_REGIME", factors=factors)
    if execution_factor < float(policy.execution_fill_floor):
        return RiskDecision(ts_ms=ts_ms, symbol=symbol, desired_side=desired_side, base_notional=base, final_notional=0.0, action="SKIP", reason="RISK_SKIP_LOW_FILLRATE", factors=factors)
    if health_factor <= 0.0:
        return RiskDecision(ts_ms=ts_ms, symbol=symbol, desired_side=desired_side, base_notional=base, final_notional=0.0, action="SKIP", reason="RISK_SKIP_BAD_HEALTH", factors=factors)

    n = float(max(float(policy.min_trade_notional), min(float(policy.max_trade_notional), n)))
    if n < float(policy.min_trade_notional):
        return RiskDecision(ts_ms=ts_ms, symbol=symbol, desired_side=desired_side, base_notional=base, final_notional=0.0, action="SKIP", reason="RISK_SKIP_LOW_CONFIDENCE", factors=factors)

    # exposure caps
    gross_now = _safe_float(mtm.get("gross_notional"), 0.0)
    allowed_gross = equity * float(policy.max_gross_exposure)
    max_add = max(0.0, allowed_gross - gross_now)
    if n > max_add:
        n = max_add
    by_symbol = dict(mtm.get("by_symbol", {}) or {})
    sym_now = _safe_float(dict(by_symbol.get(symbol, {}) or {}).get("notional"), 0.0)
    allowed_sym = equity * float(policy.max_position_per_symbol)
    max_add_sym = max(0.0, allowed_sym - sym_now)
    if n > max_add_sym:
        n = max_add_sym
    if n < float(policy.min_trade_notional):
        return RiskDecision(ts_ms=ts_ms, symbol=symbol, desired_side=desired_side, base_notional=base, final_notional=0.0, action="SKIP", reason="RISK_CAP_EXPOSURE", factors=factors)

    return RiskDecision(
        ts_ms=int(ts_ms),
        symbol=str(symbol),
        desired_side=str(desired_side),
        base_notional=float(base),
        final_notional=float(n),
        action="TRADE",
        reason="OK",
        factors=factors,
    )

