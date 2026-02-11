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

    weighted_sum = (ws_score * ws_weight) + (rest_score * rest_weight) + (fill_score * fill_weight)
    total_weight = ws_weight + rest_weight + fill_weight
    confidence = (weighted_sum / total_weight) if total_weight > 0 else 1.0
    fresh_gate = max(0.0, min(1.0, _cfg(cfg, "BELIEF_CONTRADICTION_FRESH_GATE", 0.55)))
    scored = [("ws", ws_score), ("rest", rest_score), ("fill", fill_score)]
    fresh_scores = [float(v) for _, v in scored if float(v) >= fresh_gate]
    # Contradiction should represent disagreement among healthy evidence sources, not plain staleness.
    base_contradiction = 0.0
    if len(fresh_scores) >= 2:
        lo = min(fresh_scores)
        hi = max(fresh_scores)
        base_contradiction = max(0.0, hi - lo)
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
    if ws_score < 0.7:
        degraded += 1
    if rest_score < 0.7:
        degraded += 1
    if fill_score < 0.7:
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
        "evidence_ws_age_sec": float(ws_age),
        "evidence_rest_age_sec": float(rest_age),
        "evidence_fill_age_sec": float(fill_age),
        "evidence_ws_gap_rate": float(max(0.0, _safe_float(rc.get("ws_gap_rate"), 0.0))),
        "evidence_rest_gap_rate": float(max(0.0, _safe_float(rc.get("rest_gap_rate"), 0.0))),
        "evidence_fill_gap_rate": float(max(0.0, _safe_float(rc.get("fill_gap_rate"), 0.0))),
        "evidence_ws_error_rate": float(max(0.0, _safe_float(rc.get("ws_error_rate"), 0.0))),
        "evidence_rest_error_rate": float(max(0.0, _safe_float(rc.get("rest_error_rate"), 0.0))),
        "evidence_fill_error_rate": float(max(0.0, _safe_float(rc.get("fill_error_rate"), 0.0))),
        "evidence_contradiction_score": float(contradiction_score),
        "evidence_contradiction_count": int(contradiction_count),
        "evidence_contradiction_streak": int(contradiction_streak),
        "evidence_contradiction_burn_rate": float(contradiction_burn_rate),
        "evidence_degraded_sources": int(degraded),
    }
