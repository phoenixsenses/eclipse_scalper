from __future__ import annotations

"""
Deterministic passive execution simulator (research prototype).

This module is intentionally standalone and side-effect free so it can be used
from tools without touching live execution paths.
"""

import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _quantile(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    if q <= 0.0:
        return xs[0]
    if q >= 1.0:
        return xs[-1]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _u01_from_key(seed: int, key: str) -> float:
    msg = f"{int(seed)}|{str(key)}".encode("utf-8", errors="ignore")
    h = hashlib.sha256(msg).digest()
    n = int.from_bytes(h[:8], byteorder="big", signed=False)
    return n / float(2**64 - 1)


def _bin_idx(x: Optional[float], edges: Sequence[Optional[float]]) -> Optional[int]:
    if x is None:
        return None
    e = [v for v in edges if v is not None]
    if len(e) != len(edges):
        return None
    xv = float(x)
    if len(e) == 2:
        if xv <= float(e[0]):
            return 0
        if xv <= float(e[1]):
            return 1
        return 2
    if len(e) == 3:
        if xv <= float(e[0]):
            return 0
        if xv <= float(e[1]):
            return 1
        if xv <= float(e[2]):
            return 2
        return 3
    return None


def _mean(vals: Iterable[float], default: float) -> float:
    xs = [float(v) for v in vals]
    if not xs:
        return float(default)
    return sum(xs) / len(xs)


def _feature_names() -> List[str]:
    return ["spread", "trade_intensity", "vol_proxy", "imbalance_for_fill"]


def _event_touch_and_depth(
    *,
    side: str,
    entry_price: float,
    spread_ratio: float,
    future_mids: Sequence[float],
) -> Tuple[bool, float, int]:
    s = str(side).upper()
    ep = float(entry_price)
    sp = max(1e-9, float(spread_ratio))
    if ep <= 0.0 or not future_mids:
        return False, 0.0, -1

    if s == "SHORT":
        limit = ep * (1.0 + 0.5 * sp)
        for i, m in enumerate(future_mids):
            px = float(m)
            if px >= limit:
                depth = (px - limit) / (ep * sp)
                return True, max(0.0, depth), i
        return False, 0.0, -1

    limit = ep * (1.0 - 0.5 * sp)
    for i, m in enumerate(future_mids):
        px = float(m)
        if px <= limit:
            depth = (limit - px) / (ep * sp)
            return True, max(0.0, depth), i
    return False, 0.0, -1


def calibrate_passive_model(samples: List[Dict[str, Any]], *, maker_fee_bps: float, seed: int) -> Dict[str, Any]:
    """
    Calibrate a passive fill proxy from empirical event samples.

    Required sample keys:
      spread, trade_intensity, vol_proxy, imbalance_for_fill, touched(bool),
      full_proxy(bool), adverse_bps(float)
    """
    if not samples:
        return {
            "seed": int(seed),
            "maker_fee_bps": float(maker_fee_bps),
            "base_touch": 0.0,
            "base_full_cond_touch": 0.0,
            "base_adverse_bps": 0.0,
            "edges": {},
            "touch_rates": {},
            "full_rates": {},
            "adverse_means": {},
            "depth_full_threshold": 0.5,
            "touch_without_cross_factor": 0.05,
            "partial_fill_fraction": 0.5,
        }

    touched = [1.0 if bool(s.get("touched")) else 0.0 for s in samples]
    touched_samples = [s for s in samples if bool(s.get("touched"))]
    full_cond_touch = [1.0 if bool(s.get("full_proxy")) else 0.0 for s in touched_samples]
    adverse = [_to_float(s.get("adverse_bps")) for s in samples]
    adverse = [float(v) for v in adverse if v is not None]

    edges: Dict[str, List[Optional[float]]] = {}
    touch_rates: Dict[str, List[float]] = {}
    full_rates: Dict[str, List[float]] = {}
    adverse_means: Dict[str, List[float]] = {}

    for fn in _feature_names():
        vals = [_to_float(s.get(fn)) for s in samples]
        vals = [float(v) for v in vals if v is not None]
        e = [_quantile(vals, 1.0 / 3.0), _quantile(vals, 2.0 / 3.0)]
        edges[fn] = e
        tbin = [[], [], []]
        fbin = [[], [], []]
        abin = [[], [], []]
        for s in samples:
            bi = _bin_idx(_to_float(s.get(fn)), e)
            if bi is None:
                continue
            tbin[bi].append(1.0 if bool(s.get("touched")) else 0.0)
            abin[bi].append(float(_to_float(s.get("adverse_bps")) or 0.0))
            if bool(s.get("touched")):
                fbin[bi].append(1.0 if bool(s.get("full_proxy")) else 0.0)
        touch_rates[fn] = [_mean(x, _mean(touched, 0.0)) for x in tbin]
        full_rates[fn] = [_mean(x, _mean(full_cond_touch, 0.0)) for x in fbin]
        adverse_means[fn] = [_mean(x, _mean(adverse, 0.0)) for x in abin]

    depth_vals = [_to_float(s.get("depth")) for s in touched_samples]
    depth_vals = [float(v) for v in depth_vals if v is not None]
    depth_thr = _quantile(depth_vals, 0.50)

    return {
        "seed": int(seed),
        "maker_fee_bps": float(maker_fee_bps),
        "base_touch": _mean(touched, 0.0),
        "base_full_cond_touch": _mean(full_cond_touch, 0.0),
        "base_adverse_bps": _mean(adverse, 0.0),
        "edges": edges,
        "touch_rates": touch_rates,
        "full_rates": full_rates,
        "adverse_means": adverse_means,
        "depth_full_threshold": float(depth_thr or 0.5),
        "touch_without_cross_factor": 0.05,
        "partial_fill_fraction": 0.5,
    }


def _blend_metric(
    *,
    features: Dict[str, Any],
    base: float,
    table: Dict[str, List[float]],
    edges: Dict[str, List[Optional[float]]],
) -> float:
    vals = [float(base)]
    for fn in _feature_names():
        bi = _bin_idx(_to_float(features.get(fn)), edges.get(fn, []))
        rates = table.get(fn)
        if bi is None or not isinstance(rates, list) or bi >= len(rates):
            continue
        vals.append(float(rates[bi]))
    return _clip01(_mean(vals, float(base)))


def _blend_bps_metric(
    *,
    features: Dict[str, Any],
    base_bps: float,
    table_bps: Dict[str, List[float]],
    edges: Dict[str, List[Optional[float]]],
) -> float:
    vals = [max(0.0, float(base_bps))]
    for fn in _feature_names():
        bi = _bin_idx(_to_float(features.get(fn)), edges.get(fn, []))
        rates = table_bps.get(fn)
        if bi is None or not isinstance(rates, list) or bi >= len(rates):
            continue
        vals.append(max(0.0, float(rates[bi])))
    return max(0.0, _mean(vals, max(0.0, float(base_bps))))


def _tertile_score(features: Dict[str, Any], edges: Dict[str, List[Optional[float]]], key: str) -> float:
    bi = _bin_idx(_to_float(features.get(key)), edges.get(key, []))
    if bi is None:
        return 0.5
    return float(bi) / 2.0


def simulate_passive_fill(event: Dict[str, Any], horizon_sec: int, features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic passive fill simulator.
    """
    side = str(event.get("side", "LONG")).upper()
    entry_price = float(_to_float(event.get("entry_price")) or 0.0)
    spread_ratio = _to_float(features.get("spread"))
    future_mids_raw = event.get("future_mids") or []
    future_mids = [float(x) for x in future_mids_raw if _to_float(x) is not None]
    event_id = str(event.get("event_id") or f"{event.get('symbol','')}|{event.get('signal_idx','')}")
    seed = int(params.get("seed", 42))

    touched_by_path, depth, touch_idx = _event_touch_and_depth(
        side=side,
        entry_price=entry_price,
        spread_ratio=float(spread_ratio or 0.0),
        future_mids=future_mids,
    )

    p_touch = _blend_metric(
        features=features,
        base=float(params.get("base_touch", 0.0)),
        table=params.get("touch_rates", {}) if isinstance(params.get("touch_rates"), dict) else {},
        edges=params.get("edges", {}) if isinstance(params.get("edges"), dict) else {},
    )
    edges = params.get("edges", {}) if isinstance(params.get("edges"), dict) else {}
    intensity_score = _tertile_score(features, edges, "trade_intensity")
    spread_score = _tertile_score(features, edges, "spread")
    vol_score = _tertile_score(features, edges, "vol_proxy")
    qc_strength = float(params.get("queue_competition_strength", 0.30))
    competition = max(0.0, (0.6 * intensity_score) + (0.4 * spread_score))
    p_touch *= max(0.05, 1.0 - (qc_strength * competition))
    if not touched_by_path:
        p_touch *= float(params.get("touch_without_cross_factor", 0.05))

    u_touch = _u01_from_key(seed, event_id + "|touch")
    filled = bool(u_touch < p_touch)
    if not filled:
        return {
            "filled": False,
            "fill_fraction": 0.0,
            "effective_cost_bps": 0.0,
            "adverse_selection_bps": 0.0,
            "execution_price_adjustment": 0.0,
            "fill_index_offset": -1,
            "p_touch": p_touch,
            "p_full_cond_touch": 0.0,
            "queue_competition_score": float(competition),
            "toxicity_score": float((0.5 * intensity_score) + (0.5 * vol_score)),
        }

    p_full = _blend_metric(
        features=features,
        base=float(params.get("base_full_cond_touch", 0.0)),
        table=params.get("full_rates", {}) if isinstance(params.get("full_rates"), dict) else {},
        edges=params.get("edges", {}) if isinstance(params.get("edges"), dict) else {},
    )

    # Depth helps distinguish partial vs full when touched.
    depth_thr = float(params.get("depth_full_threshold", 0.5))
    if depth <= depth_thr:
        p_full *= 0.7
    else:
        p_full = min(1.0, p_full * 1.1)
    u_full = _u01_from_key(seed, event_id + "|full")
    full_fill = bool(u_full < p_full)
    fill_fraction = 1.0 if full_fill else float(params.get("partial_fill_fraction", 0.5))

    adverse_bps = _blend_bps_metric(
        features=features,
        base_bps=float(params.get("base_adverse_bps", 0.0)),
        table_bps=params.get("adverse_means", {}) if isinstance(params.get("adverse_means"), dict) else {},
        edges=params.get("edges", {}) if isinstance(params.get("edges"), dict) else {},
    )
    adverse_toxicity_strength = float(params.get("adverse_toxicity_strength", 0.60))
    toxicity = max(0.0, (0.5 * intensity_score) + (0.5 * vol_score))
    adverse_bps *= max(0.5, 1.0 + (adverse_toxicity_strength * toxicity))
    adverse_mult = max(0.0, float(params.get("passive_adverse_mult", 1.0)))
    adverse_bps *= adverse_mult

    maker_fee_bps = float(params.get("maker_fee_bps", 0.5))
    fee_bps_total = 2.0 * maker_fee_bps
    effective_cost_bps = max(0.0, fee_bps_total + max(0.0, adverse_bps))

    sp = float(spread_ratio or 0.0)
    half_sp = 0.5 * sp * fill_fraction
    if side == "SHORT":
        execution_price_adjustment = +half_sp
    else:
        execution_price_adjustment = -half_sp

    return {
        "filled": True,
        "fill_fraction": float(fill_fraction),
        "effective_cost_bps": float(effective_cost_bps),
        "adverse_selection_bps": float(max(0.0, adverse_bps)),
        "execution_price_adjustment": float(execution_price_adjustment),
        "fill_index_offset": int(touch_idx),
        "p_touch": float(p_touch),
        "p_full_cond_touch": float(p_full),
        "queue_competition_score": float(competition),
        "toxicity_score": float(toxicity),
    }
