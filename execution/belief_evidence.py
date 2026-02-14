from __future__ import annotations

import time
from typing import Any, Dict, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _cfg(cfg: Any, name: str, default: float) -> float:
    try:
        return _safe_float(getattr(cfg, name, default), default)
    except Exception:
        return default


def _freshness_score(ts: float, *, now: float, warn_sec: float, crit_sec: float) -> float:
    if ts <= 0:
        # Missing evidence is neutral so legacy behavior is preserved.
        return 1.0
    age = max(0.0, now - ts)
    w = max(1.0, float(warn_sec))
    c = max(w + 1.0, float(crit_sec))
    if age <= w:
        return 1.0
    if age >= c:
        return 0.0
    span = c - w
    return max(0.0, min(1.0, 1.0 - ((age - w) / span)))


def _max_ts(values: list[Any]) -> float:
    best = 0.0
    for v in values:
        f = _safe_float(v, 0.0)
        if f > best:
            best = f
    return best


def _run_context(bot) -> dict:
    st = getattr(bot, "state", None)
    if st is None:
        return {}
    rc = getattr(st, "run_context", None)
    return rc if isinstance(rc, dict) else {}


def _age(ts: float, *, now: float) -> float:
    if ts <= 0:
        return 0.0
    return max(0.0, now - ts)


def _coverage_score(
    ts: float,
    *,
    now: float,
    expected_interval_sec: float,
    miss_window_mult: float,
    gap_rate: float,
    error_rate: float,
    gap_weight: float,
    error_weight: float,
) -> float:
    # Preserve legacy neutral behavior when a source has never been seen.
    if ts <= 0:
        return 1.0
    age = max(0.0, now - ts)
    expected = max(1.0, float(expected_interval_sec))
    miss_window = expected * max(1.0, float(miss_window_mult))
    age_cov = _clamp(1.0 - (age / miss_window), 0.0, 1.0)
    return _clamp(age_cov - (max(0.0, float(gap_rate)) * gap_weight) - (max(0.0, float(error_rate)) * error_weight), 0.0, 1.0)


def _data_last_poll_ts(bot) -> float:
    data = getattr(bot, "data", None)
    dct = getattr(data, "last_poll_ts", None) if data is not None else None
    if not isinstance(dct, dict):
        return 0.0
    return _max_ts(list(dct.values()))


def compute_belief_evidence(bot, cfg: Any = None, *, now: Optional[float] = None) -> Dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    rc = _run_context(bot)
    st = getattr(bot, "state", None)
    km = getattr(st, "kill_metrics", None) if st is not None else None
    ex = getattr(bot, "ex", None)

    ws_warn = _cfg(cfg, "BELIEF_WS_WARN_SEC", 20.0)
    ws_crit = _cfg(cfg, "BELIEF_WS_CRIT_SEC", 90.0)
    rest_warn = _cfg(cfg, "BELIEF_REST_WARN_SEC", 60.0)
    rest_crit = _cfg(cfg, "BELIEF_REST_CRIT_SEC", 240.0)
    fill_warn = _cfg(cfg, "BELIEF_FILL_WARN_SEC", 90.0)
    fill_crit = _cfg(cfg, "BELIEF_FILL_CRIT_SEC", 480.0)

    ws_weight = max(0.0, _cfg(cfg, "BELIEF_WS_WEIGHT", 0.35))
    rest_weight = max(0.0, _cfg(cfg, "BELIEF_REST_WEIGHT", 0.45))
    fill_weight = max(0.0, _cfg(cfg, "BELIEF_FILL_WEIGHT", 0.20))
    contradiction_weight = max(0.0, _cfg(cfg, "BELIEF_CONTRADICTION_WEIGHT", 0.45))
    contradiction_streak_weight = max(0.0, _cfg(cfg, "BELIEF_CONTRADICTION_STREAK_WEIGHT", 0.12))
    contradiction_burn_weight = max(0.0, _cfg(cfg, "BELIEF_CONTRADICTION_BURN_WEIGHT", 0.08))
    contradiction_severe_delta = max(0.0, _cfg(cfg, "BELIEF_CONTRADICTION_SEVERE_DELTA", 0.6))
    contradiction_burn_ref = max(1.0, _cfg(cfg, "BELIEF_CONTRADICTION_BURN_REF", 3.0))
    source_gap_weight = _clamp(_cfg(cfg, "BELIEF_SOURCE_GAP_WEIGHT", 0.5), 0.0, 3.0)
    source_error_weight = _clamp(_cfg(cfg, "BELIEF_SOURCE_ERROR_WEIGHT", 0.7), 0.0, 3.0)
    source_coverage_weight = _clamp(_cfg(cfg, "BELIEF_SOURCE_COVERAGE_WEIGHT", 0.5), 0.0, 1.0)
    coverage_gap_weight = _clamp(_cfg(cfg, "BELIEF_COVERAGE_GAP_WEIGHT", 0.6), 0.0, 3.0)
    coverage_error_weight = _clamp(_cfg(cfg, "BELIEF_COVERAGE_ERROR_WEIGHT", 0.4), 0.0, 3.0)
    coverage_window_mult = max(1.0, _cfg(cfg, "BELIEF_COVERAGE_WINDOW_MULT", 3.0))
    ws_expected_sec = max(1.0, _cfg(cfg, "BELIEF_WS_EXPECTED_INTERVAL_SEC", max(1.0, ws_warn / 2.0)))
    rest_expected_sec = max(1.0, _cfg(cfg, "BELIEF_REST_EXPECTED_INTERVAL_SEC", max(1.0, rest_warn / 2.0)))
    fill_expected_sec = max(1.0, _cfg(cfg, "BELIEF_FILL_EXPECTED_INTERVAL_SEC", max(1.0, fill_warn / 2.0)))
    ws_rest_delta_t = _clamp(_cfg(cfg, "BELIEF_CONTRADICTION_WS_REST_DELTA", 0.35), 0.0, 1.0)
    rest_fill_delta_t = _clamp(_cfg(cfg, "BELIEF_CONTRADICTION_REST_FILL_DELTA", 0.35), 0.0, 1.0)
    ws_fill_delta_t = _clamp(_cfg(cfg, "BELIEF_CONTRADICTION_WS_FILL_DELTA", 0.35), 0.0, 1.0)

    ws_ts = _max_ts(
        [
            rc.get("ws_last_event_ts"),
            rc.get("ws_order_last_ts"),
            rc.get("ws_account_last_ts"),
            rc.get("ws_heartbeat_ts"),
            getattr(st, "ws_last_event_ts", 0.0) if st is not None else 0.0,
            (km.get("ws_last_event_ts") if isinstance(km, dict) else 0.0),
        ]
    )
    rest_ts = _max_ts(
        [
            rc.get("rest_last_ok_ts"),
            getattr(ex, "last_health_check", 0.0) if ex is not None else 0.0,
            _data_last_poll_ts(bot),
            (km.get("rest_last_ok_ts") if isinstance(km, dict) else 0.0),
        ]
    )
    fill_ts = _max_ts(
        [
            rc.get("fills_last_ts"),
            rc.get("last_fill_ts"),
            getattr(st, "last_fill_ts", 0.0) if st is not None else 0.0,
            (km.get("fills_last_ts") if isinstance(km, dict) else 0.0),
        ]
    )

    ws_score = _freshness_score(ws_ts, now=now_ts, warn_sec=ws_warn, crit_sec=ws_crit)
    rest_score = _freshness_score(rest_ts, now=now_ts, warn_sec=rest_warn, crit_sec=rest_crit)
    fill_score = _freshness_score(fill_ts, now=now_ts, warn_sec=fill_warn, crit_sec=fill_crit)
    ws_age = _age(ws_ts, now=now_ts)
    rest_age = _age(rest_ts, now=now_ts)
    fill_age = _age(fill_ts, now=now_ts)

    ws_gap = float(max(0.0, _safe_float(rc.get("ws_gap_rate"), 0.0)))
    rest_gap = float(max(0.0, _safe_float(rc.get("rest_gap_rate"), 0.0)))
    fill_gap = float(max(0.0, _safe_float(rc.get("fill_gap_rate"), 0.0)))
    ws_err = float(max(0.0, _safe_float(rc.get("ws_error_rate"), 0.0)))
    rest_err = float(max(0.0, _safe_float(rc.get("rest_error_rate"), 0.0)))
    fill_err = float(max(0.0, _safe_float(rc.get("fill_error_rate"), 0.0)))

    ws_cov = _coverage_score(
        ws_ts,
        now=now_ts,
        expected_interval_sec=ws_expected_sec,
        miss_window_mult=coverage_window_mult,
        gap_rate=ws_gap,
        error_rate=ws_err,
        gap_weight=coverage_gap_weight,
        error_weight=coverage_error_weight,
    )
    rest_cov = _coverage_score(
        rest_ts,
        now=now_ts,
        expected_interval_sec=rest_expected_sec,
        miss_window_mult=coverage_window_mult,
        gap_rate=rest_gap,
        error_rate=rest_err,
        gap_weight=coverage_gap_weight,
        error_weight=coverage_error_weight,
    )
    fill_cov = _coverage_score(
        fill_ts,
        now=now_ts,
        expected_interval_sec=fill_expected_sec,
        miss_window_mult=coverage_window_mult,
        gap_rate=fill_gap,
        error_rate=fill_err,
        gap_weight=coverage_gap_weight,
        error_weight=coverage_error_weight,
    )

    ws_base_conf = _clamp(float(ws_score) - (ws_gap * source_gap_weight) - (ws_err * source_error_weight), 0.0, 1.0)
    rest_base_conf = _clamp(float(rest_score) - (rest_gap * source_gap_weight) - (rest_err * source_error_weight), 0.0, 1.0)
    fill_base_conf = _clamp(float(fill_score) - (fill_gap * source_gap_weight) - (fill_err * source_error_weight), 0.0, 1.0)
    ws_conf = _clamp((ws_base_conf * (1.0 - source_coverage_weight)) + (ws_cov * source_coverage_weight), 0.0, 1.0)
    rest_conf = _clamp((rest_base_conf * (1.0 - source_coverage_weight)) + (rest_cov * source_coverage_weight), 0.0, 1.0)
    fill_conf = _clamp((fill_base_conf * (1.0 - source_coverage_weight)) + (fill_cov * source_coverage_weight), 0.0, 1.0)

    weighted_sum = (ws_conf * ws_weight) + (rest_conf * rest_weight) + (fill_conf * fill_weight)
    total_weight = ws_weight + rest_weight + fill_weight
    confidence = (weighted_sum / total_weight) if total_weight > 0 else 1.0
    coverage_ratio = ((ws_cov * ws_weight) + (rest_cov * rest_weight) + (fill_cov * fill_weight)) / total_weight if total_weight > 0 else 1.0
    fresh_gate = max(0.0, min(1.0, _cfg(cfg, "BELIEF_CONTRADICTION_FRESH_GATE", 0.55)))
    scored = [("ws", ws_conf), ("rest", rest_conf), ("fill", fill_conf)]
    fresh_scores = [float(v) for _, v in scored if float(v) >= fresh_gate]
    contradiction_tags: list[str] = []
    # Contradiction should represent disagreement among healthy evidence sources, not plain staleness.
    base_contradiction = 0.0
    if len(fresh_scores) >= 2:
        lo = min(fresh_scores)
        hi = max(fresh_scores)
        base_contradiction = max(0.0, hi - lo)
    if ws_conf >= fresh_gate and rest_conf >= fresh_gate and abs(ws_conf - rest_conf) >= ws_rest_delta_t:
        contradiction_tags.append("ws_vs_rest")
        base_contradiction = max(base_contradiction, abs(ws_conf - rest_conf))
    if rest_conf >= fresh_gate and fill_conf >= fresh_gate and abs(rest_conf - fill_conf) >= rest_fill_delta_t:
        contradiction_tags.append("rest_vs_fill")
        base_contradiction = max(base_contradiction, abs(rest_conf - fill_conf))
    if ws_conf >= fresh_gate and fill_conf >= fresh_gate and abs(ws_conf - fill_conf) >= ws_fill_delta_t:
        contradiction_tags.append("ws_vs_fill")
        base_contradiction = max(base_contradiction, abs(ws_conf - fill_conf))
    contradiction_score = _safe_float(
        max(base_contradiction, _safe_float(rc.get("evidence_contradiction_score"), 0.0)),
        0.0,
    )
    contradiction_streak = max(0, int(_safe_float(rc.get("evidence_contradiction_streak"), 0.0)))
    contradiction_count = max(0, int(_safe_float(rc.get("evidence_contradiction_count"), 0.0)))
    contradiction_last_ts = _safe_float(rc.get("evidence_contradiction_last_ts"), 0.0)
    contradiction_prev_ts = _safe_float(rc.get("evidence_contradiction_prev_ts"), 0.0)
    contradiction_burn_rate = _safe_float(rc.get("evidence_contradiction_burn_rate"), 0.0)
    if contradiction_score >= contradiction_severe_delta:
        contradiction_streak += 1
        contradiction_count += 1
        contradiction_prev_ts = contradiction_last_ts
        contradiction_last_ts = now_ts
        if contradiction_prev_ts > 0.0 and contradiction_last_ts > contradiction_prev_ts:
            dt = max(1e-6, contradiction_last_ts - contradiction_prev_ts)
            contradiction_burn_rate = (60.0 / dt)
        else:
            contradiction_burn_rate = max(contradiction_burn_rate, 1.0)
    else:
        contradiction_streak = max(0, contradiction_streak - 1)
        contradiction_burn_rate = max(0.0, contradiction_burn_rate * 0.6)
    try:
        rc["evidence_contradiction_score"] = float(contradiction_score)
        rc["evidence_contradiction_streak"] = int(contradiction_streak)
        rc["evidence_contradiction_count"] = int(contradiction_count)
        rc["evidence_contradiction_last_ts"] = float(contradiction_last_ts)
        rc["evidence_contradiction_prev_ts"] = float(contradiction_prev_ts)
        rc["evidence_contradiction_burn_rate"] = float(contradiction_burn_rate)
        rc["evidence_last_compute_ts"] = float(now_ts)
    except Exception:
        pass
    confidence -= (
        (contradiction_score * contradiction_weight)
        + (float(contradiction_streak) * contradiction_streak_weight)
        + ((min(1.0, contradiction_burn_rate / contradiction_burn_ref)) * contradiction_burn_weight)
    )
    confidence = max(0.0, min(1.0, confidence))

    degraded = 0
    if ws_conf < 0.7:
        degraded += 1
    if rest_conf < 0.7:
        degraded += 1
    if fill_conf < 0.7:
        degraded += 1
    if contradiction_score >= contradiction_severe_delta:
        degraded += 1

    return {
        "evidence_confidence": float(confidence),
        "evidence_ws_last_seen_ts": float(ws_ts),
        "evidence_rest_last_seen_ts": float(rest_ts),
        "evidence_fill_last_seen_ts": float(fill_ts),
        "evidence_ws_score": float(ws_score),
        "evidence_rest_score": float(rest_score),
        "evidence_fill_score": float(fill_score),
        "evidence_ws_confidence": float(ws_conf),
        "evidence_rest_confidence": float(rest_conf),
        "evidence_fill_confidence": float(fill_conf),
        "evidence_ws_coverage_ratio": float(ws_cov),
        "evidence_rest_coverage_ratio": float(rest_cov),
        "evidence_fill_coverage_ratio": float(fill_cov),
        "evidence_coverage_ratio": float(coverage_ratio),
        "evidence_ws_age_sec": float(ws_age),
        "evidence_rest_age_sec": float(rest_age),
        "evidence_fill_age_sec": float(fill_age),
        "evidence_ws_gap_rate": float(ws_gap),
        "evidence_rest_gap_rate": float(rest_gap),
        "evidence_fill_gap_rate": float(fill_gap),
        "evidence_ws_error_rate": float(ws_err),
        "evidence_rest_error_rate": float(rest_err),
        "evidence_fill_error_rate": float(fill_err),
        "evidence_contradiction_score": float(contradiction_score),
        "evidence_contradiction_count": int(contradiction_count),
        "evidence_contradiction_streak": int(contradiction_streak),
        "evidence_contradiction_burn_rate": float(contradiction_burn_rate),
        "evidence_contradiction_tags": ",".join(contradiction_tags),
        "evidence_degraded_sources": int(degraded),
    }
