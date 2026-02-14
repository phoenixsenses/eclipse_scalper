from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Iterable, List, Tuple


def _now() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    try:
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    except Exception:
        pass
    try:
        raw = os.getenv(name, "")
        if raw != "":
            if isinstance(default, bool):
                return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")
            if isinstance(default, int):
                return int(float(raw))
            if isinstance(default, float):
                return float(raw)
            return raw
    except Exception:
        pass
    return default


def _symkey(sym: str) -> str:
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _parse_corr_groups(raw: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for token in str(raw or "").split(";"):
        t = token.strip()
        if not t or ":" not in t:
            continue
        g, syms = t.split(":", 1)
        gk = str(g or "").strip().upper()
        vals = [_symkey(x) for x in syms.split(",") if _symkey(x)]
        if gk and vals:
            out[gk] = vals
    return out


def _resolve_groups(bot, cfg: Any) -> Dict[str, List[str]]:
    try:
        g = getattr(cfg, "CORRELATION_GROUPS", None)
        if isinstance(g, dict) and g:
            return {str(k).upper(): [_symkey(s) for s in list(v or []) if _symkey(s)] for k, v in g.items()}
    except Exception:
        pass
    try:
        g2 = getattr(bot, "CORRELATION_GROUPS", None)
        if isinstance(g2, dict) and g2:
            return {str(k).upper(): [_symkey(s) for s in list(v or []) if _symkey(s)] for k, v in g2.items()}
    except Exception:
        pass
    parsed = _parse_corr_groups(str(_cfg(cfg, "CORR_GROUPS", "") or ""))
    if parsed:
        return parsed
    return {}


def _pearson(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    xa = a[-n:]
    xb = b[-n:]
    ma = sum(xa) / n
    mb = sum(xb) / n
    va = sum((x - ma) ** 2 for x in xa)
    vb = sum((x - mb) ** 2 for x in xb)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    cov = sum((xa[i] - ma) * (xb[i] - mb) for i in range(n))
    return _clamp(cov / math.sqrt(va * vb), -1.0, 1.0)


def _downside_corr(a: List[float], b: List[float]) -> float:
    paired = [(x, y) for (x, y) in zip(a, b) if x < 0.0 and y < 0.0]
    if len(paired) < 3:
        return 0.0
    xa = [p[0] for p in paired]
    xb = [p[1] for p in paired]
    return _pearson(xa, xb)


def _tail_coupling(a: List[float], b: List[float], threshold: float) -> float:
    n = min(len(a), len(b))
    if n < 5:
        return 0.0
    xa = a[-n:]
    xb = b[-n:]
    joint = 0
    down_a = 0
    down_b = 0
    thr = abs(float(threshold))
    for i in range(n):
        ia = float(xa[i]) <= -thr
        ib = float(xb[i]) <= -thr
        if ia:
            down_a += 1
        if ib:
            down_b += 1
        if ia and ib:
            joint += 1
    denom = max(1, min(down_a, down_b))
    return _clamp(float(joint) / float(denom), 0.0, 1.0)


def _pairs(items: Iterable[str]) -> Iterable[Tuple[str, str]]:
    vals = [str(x) for x in items]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            yield vals[i], vals[j]


def _corr_from_returns(returns_by_symbol: Dict[str, List[float]], symbols: List[str], tail_threshold: float) -> Dict[str, float]:
    if len(symbols) < 2:
        return {"rho_roll": 0.0, "rho_downside": 0.0, "tail_coupling": 0.0, "pairs": 0.0}
    roll_vals: List[float] = []
    down_vals: List[float] = []
    tail_vals: List[float] = []
    pair_count = 0
    for a, b in _pairs(symbols):
        ra = list(returns_by_symbol.get(a, []) or [])
        rb = list(returns_by_symbol.get(b, []) or [])
        if len(ra) < 3 or len(rb) < 3:
            continue
        pair_count += 1
        roll_vals.append(max(0.0, _pearson(ra, rb)))
        down_vals.append(max(0.0, _downside_corr(ra, rb)))
        tail_vals.append(_tail_coupling(ra, rb, tail_threshold))
    if pair_count <= 0:
        return {"rho_roll": 0.0, "rho_downside": 0.0, "tail_coupling": 0.0, "pairs": 0.0}
    return {
        "rho_roll": float(sum(roll_vals) / max(1, len(roll_vals))),
        "rho_downside": float(sum(down_vals) / max(1, len(down_vals))),
        "tail_coupling": float(sum(tail_vals) / max(1, len(tail_vals))),
        "pairs": float(pair_count),
    }


def _occupancy_proxy(open_count: int) -> float:
    return _clamp((max(0, int(open_count)) - 1.0) / 3.0, 0.0, 1.0)


def _group_hash(groups: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    for g in sorted(groups.keys()):
        vals = ",".join(sorted([_symkey(s) for s in list(groups.get(g, []) or []) if _symkey(s)]))
        parts.append(f"{g}:{vals}")
    return "|".join(parts)


def _extract_returns(bot, cfg: Any) -> Dict[str, List[float]]:
    run_context = getattr(getattr(bot, "state", None), "run_context", None)
    if not isinstance(run_context, dict):
        return {}
    key = str(_cfg(cfg, "CORR_RETURNS_KEY", "corr_returns") or "corr_returns")
    raw = run_context.get(key)
    out: Dict[str, List[float]] = {}
    if not isinstance(raw, dict):
        return out
    max_len = max(5, int(_safe_float(_cfg(cfg, "CORR_RETURNS_MAX_LEN", 240), 240.0)))
    for k, vals in raw.items():
        kk = _symkey(str(k))
        if not kk:
            continue
        try:
            seq = [_safe_float(x, 0.0) for x in list(vals or [])][-max_len:]
        except Exception:
            seq = []
        if seq:
            out[kk] = seq
    return out


def compute_correlation_risk(bot, metrics: Dict[str, Any], cfg: Any = None) -> Dict[str, Any]:
    groups = _resolve_groups(bot, cfg)
    returns_by_symbol = _extract_returns(bot, cfg)
    tail_threshold = abs(_safe_float(_cfg(cfg, "CORR_TAIL_THRESHOLD", 0.015), 0.015))
    debt_ref = max(1.0, _safe_float(_cfg(cfg, "CORR_DEBT_REF_SEC", 300.0), 300.0))
    uplift_max = _clamp(_safe_float(_cfg(cfg, "CORR_UNCERTAINTY_UPLIFT_MAX", 0.25), 0.25), 0.0, 1.0)
    debt_w = _clamp(_safe_float(_cfg(cfg, "CORR_DEBT_UPLIFT_WEIGHT", 0.15), 0.15), 0.0, 1.0)
    hidden_w = _clamp(_safe_float(_cfg(cfg, "CORR_HIDDEN_EXPOSURE_UPLIFT_WEIGHT", 0.15), 0.15), 0.0, 1.0)
    contrad_w = _clamp(_safe_float(_cfg(cfg, "CORR_CONTRADICTION_UPLIFT_WEIGHT", 0.10), 0.10), 0.0, 1.0)

    run_context = getattr(getattr(bot, "state", None), "run_context", None)
    if not isinstance(run_context, dict):
        run_context = {}
        try:
            getattr(bot, "state", None).run_context = run_context
        except Exception:
            pass
    state = run_context.setdefault("correlation_risk", {})
    if not isinstance(state, dict):
        state = {}
        run_context["correlation_risk"] = state

    current_hash = _group_hash(groups)
    baseline_hash = str(state.get("baseline_hash") or "")
    if not baseline_hash:
        baseline_hash = current_hash
        state["baseline_hash"] = baseline_hash
    drift_level = float(_safe_float(state.get("drift_level", 0.0), 0.0))
    if current_hash != baseline_hash:
        drift_level += 1.0
    else:
        drift_level = max(0.0, drift_level * 0.80)
    state["drift_level"] = float(drift_level)
    drift_ref = max(1.0, _safe_float(_cfg(cfg, "CORR_GROUP_DRIFT_REF", 3.0), 3.0))
    group_drift_debt = _clamp(drift_level / drift_ref, 0.0, 1.0)

    # Build open symbol footprint.
    open_symbols: List[str] = []
    positions = getattr(getattr(bot, "state", None), "positions", None)
    if isinstance(positions, dict):
        for k, pos in positions.items():
            try:
                size = abs(_safe_float(getattr(pos, "size", 0.0), 0.0))
            except Exception:
                size = 0.0
            if size > 0.0:
                open_symbols.append(_symkey(str(k)))
    open_symbols = sorted([s for s in open_symbols if s])

    # Group-level scores.
    group_scores: Dict[str, Dict[str, float]] = {}
    max_roll = 0.0
    max_down = 0.0
    max_tail = 0.0
    for group, syms in groups.items():
        members = [_symkey(s) for s in list(syms or []) if _symkey(s)]
        if not members:
            continue
        with_returns = [s for s in members if s in returns_by_symbol]
        if len(with_returns) >= 2:
            corr = _corr_from_returns(returns_by_symbol, with_returns, tail_threshold)
            rho_roll = _clamp(corr["rho_roll"], 0.0, 1.0)
            rho_down = _clamp(corr["rho_downside"], 0.0, 1.0)
            tail = _clamp(corr["tail_coupling"], 0.0, 1.0)
            pair_count = int(corr["pairs"])
        else:
            occ = sum(1 for s in members if s in open_symbols)
            proxy = _occupancy_proxy(occ)
            rho_roll = proxy
            rho_down = proxy
            tail = proxy
            pair_count = 0
        group_scores[group] = {
            "rho_roll": float(rho_roll),
            "rho_downside": float(rho_down),
            "tail_coupling": float(tail),
            "members": float(len(members)),
            "pairs": float(pair_count),
        }
        max_roll = max(max_roll, rho_roll)
        max_down = max(max_down, rho_down)
        max_tail = max(max_tail, tail)

    if not group_scores:
        # Fallback when no explicit groups: use current open-symbol occupancy.
        proxy = _occupancy_proxy(len(open_symbols))
        max_roll = proxy
        max_down = proxy
        max_tail = proxy

    debt_sec = _safe_float(metrics.get("belief_debt_sec", 0.0), 0.0)
    debt_norm = _clamp(debt_sec / debt_ref, 0.0, 1.0)
    hidden = (
        _safe_float(metrics.get("runtime_gate_cat_position", 0), 0.0)
        + _safe_float(metrics.get("runtime_gate_cat_orphan", 0), 0.0)
        + _safe_float(metrics.get("intent_unknown_count", 0), 0.0)
    )
    hidden_norm = _clamp(hidden / max(1.0, _safe_float(_cfg(cfg, "CORR_HIDDEN_REF", 3.0), 3.0)), 0.0, 1.0)
    contradiction_norm = _clamp(
        _safe_float(metrics.get("evidence_contradiction_score", 0.0), 0.0), 0.0, 1.0
    )
    uncertainty_uplift = _clamp(
        (debt_norm * debt_w) + (hidden_norm * hidden_w) + (contradiction_norm * contrad_w),
        0.0,
        uplift_max,
    )

    eff_roll = _clamp(max_roll + uncertainty_uplift, 0.0, 1.0)
    eff_down = _clamp(max_down + uncertainty_uplift, 0.0, 1.0)
    eff_tail = _clamp(max_tail + uncertainty_uplift, 0.0, 1.0)

    w_roll = _clamp(_safe_float(_cfg(cfg, "CORR_WEIGHT_ROLL", 0.35), 0.35), 0.0, 1.0)
    w_down = _clamp(_safe_float(_cfg(cfg, "CORR_WEIGHT_DOWNSIDE", 0.35), 0.35), 0.0, 1.0)
    w_tail = _clamp(_safe_float(_cfg(cfg, "CORR_WEIGHT_TAIL", 0.30), 0.30), 0.0, 1.0)
    w_sum = max(1e-9, w_roll + w_down + w_tail)
    corr_pressure_raw = ((eff_roll * w_roll) + (eff_down * w_down) + (eff_tail * w_tail)) / w_sum
    corr_pressure = _clamp(corr_pressure_raw + (group_drift_debt * _safe_float(_cfg(cfg, "CORR_DRIFT_WEIGHT", 0.15), 0.15)), 0.0, 1.0)

    # Confidence decays when little data is present and when sources disagree.
    sample_pairs = sum(int(v.get("pairs", 0.0) or 0.0) for v in group_scores.values())
    sample_norm = _clamp(float(sample_pairs) / max(1.0, _safe_float(_cfg(cfg, "CORR_SAMPLE_REF_PAIRS", 6.0), 6.0)), 0.0, 1.0)
    evidence_conf = _clamp(_safe_float(metrics.get("evidence_confidence", 1.0), 1.0), 0.0, 1.0)
    corr_confidence = _clamp((sample_norm * 0.6) + (evidence_conf * 0.4), 0.0, 1.0)

    # Regime hysteresis.
    stress_enter = _clamp(_safe_float(_cfg(cfg, "CORR_STRESS_ENTER", 0.72), 0.72), 0.0, 1.0)
    stress_exit = _clamp(_safe_float(_cfg(cfg, "CORR_STRESS_EXIT", 0.58), 0.58), 0.0, 1.0)
    tighten_enter = _clamp(_safe_float(_cfg(cfg, "CORR_TIGHTEN_ENTER", 0.48), 0.48), 0.0, 1.0)
    tighten_exit = _clamp(_safe_float(_cfg(cfg, "CORR_TIGHTEN_EXIT", 0.38), 0.38), 0.0, 1.0)
    persist_up = max(0.0, _safe_float(_cfg(cfg, "CORR_HYST_UP_SEC", 10.0), 10.0))
    persist_down = max(0.0, _safe_float(_cfg(cfg, "CORR_HYST_DOWN_SEC", 30.0), 30.0))
    regime = str(state.get("regime") or "NORMAL").upper()
    up_since = _safe_float(state.get("up_since", 0.0), 0.0)
    down_since = _safe_float(state.get("down_since", 0.0), 0.0)
    now = _now()

    if regime == "STRESS":
        if corr_pressure <= stress_exit:
            if down_since <= 0.0:
                down_since = now
            if (now - down_since) >= persist_down:
                regime = "TIGHTENING" if corr_pressure >= tighten_enter else "NORMAL"
                down_since = 0.0
        else:
            down_since = 0.0
    elif regime == "TIGHTENING":
        if corr_pressure >= stress_enter:
            if up_since <= 0.0:
                up_since = now
            if (now - up_since) >= persist_up:
                regime = "STRESS"
                up_since = 0.0
        elif corr_pressure <= tighten_exit:
            if down_since <= 0.0:
                down_since = now
            if (now - down_since) >= persist_down:
                regime = "NORMAL"
                down_since = 0.0
        else:
            up_since = 0.0
            down_since = 0.0
    else:  # NORMAL
        if corr_pressure >= stress_enter:
            if up_since <= 0.0:
                up_since = now
            if (now - up_since) >= persist_up:
                regime = "STRESS"
                up_since = 0.0
        elif corr_pressure >= tighten_enter:
            if up_since <= 0.0:
                up_since = now
            if (now - up_since) >= persist_up:
                regime = "TIGHTENING"
                up_since = 0.0
        else:
            up_since = 0.0
    state["regime"] = regime
    state["up_since"] = up_since
    state["down_since"] = down_since

    top_groups = sorted(
        [
            (
                g,
                max(
                    _safe_float(v.get("rho_roll", 0.0), 0.0),
                    _safe_float(v.get("rho_downside", 0.0), 0.0),
                    _safe_float(v.get("tail_coupling", 0.0), 0.0),
                ),
            )
            for g, v in group_scores.items()
        ],
        key=lambda kv: float(kv[1]),
        reverse=True,
    )[:3]
    worst_group = top_groups[0][0] if top_groups else ""

    tags: List[str] = []
    if eff_tail >= 0.6:
        tags.append("tail_coupling")
    if eff_down >= 0.6:
        tags.append("downside_corr")
    if uncertainty_uplift > 0.0:
        tags.append("belief_uplift")
    if group_drift_debt >= 0.5:
        tags.append("group_drift")
    if hidden_norm > 0.0:
        tags.append("hidden_exposure")
    if not tags:
        tags.append("stable")

    return {
        "corr_pressure": float(corr_pressure),
        "corr_regime": str(regime),
        "corr_confidence": float(corr_confidence),
        "corr_roll": float(max_roll),
        "corr_downside": float(max_down),
        "corr_tail_coupling": float(max_tail),
        "corr_eff_roll": float(eff_roll),
        "corr_eff_downside": float(eff_down),
        "corr_eff_tail_coupling": float(eff_tail),
        "corr_uncertainty_uplift": float(uncertainty_uplift),
        "corr_group_drift_debt": float(group_drift_debt),
        "corr_hidden_exposure_risk": float(hidden_norm),
        "corr_worst_group": str(worst_group),
        "corr_top_groups": [(str(g), float(v)) for (g, v) in top_groups],
        "corr_reason_tags": ",".join(tags),
        "corr_group_scores": {
            str(g): {
                "rho_roll": float(v.get("rho_roll", 0.0) or 0.0),
                "rho_downside": float(v.get("rho_downside", 0.0) or 0.0),
                "tail_coupling": float(v.get("tail_coupling", 0.0) or 0.0),
                "pairs": int(_safe_float(v.get("pairs", 0.0), 0.0)),
            }
            for g, v in group_scores.items()
        },
    }

