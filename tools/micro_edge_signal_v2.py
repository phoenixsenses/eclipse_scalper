from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


_V2_CACHE: Dict[Tuple[str, str, int, int, str], List[Dict[str, Optional[float]]]] = {}


def _sigmoid(x: float) -> float:
    z = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-z))


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        out = float(v)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _copy_rows(rows: Sequence[Dict[str, Optional[float]]]) -> List[Dict[str, Optional[float]]]:
    return [dict(r) for r in rows]


def _ts_hint(rows: Sequence[Dict[str, Optional[float]]]) -> Tuple[int, float, float]:
    if not rows:
        return (0, 0.0, 0.0)
    first = _safe(rows[0].get("ts_ms"), 0.0)
    last = _safe(rows[-1].get("ts_ms"), 0.0)
    return (len(rows), float(first), float(last))


def enrich_rows_with_v2(
    rows: Sequence[Dict[str, Optional[float]]],
    *,
    bucket_sec: int,
    cache_key: Optional[Tuple[str, str, int, int, str]] = None,
) -> List[Dict[str, Optional[float]]]:
    """
    Adds v2 microstructure features to rows in O(n) time.
    Cache key convention: (db_path, symbol, lookback_min, bucket_sec, rule_name)
    """
    if cache_key is not None and cache_key in _V2_CACHE:
        cached = _V2_CACHE[cache_key]
        if _ts_hint(cached) == _ts_hint(rows):
            return _copy_rows(cached)

    out = _copy_rows(rows)
    if not out:
        if cache_key is not None:
            _V2_CACHE[cache_key] = _copy_rows(out)
        return out

    alpha_ema = 0.2
    short_w = max(3, int(round(20 / max(1, bucket_sec))))
    long_w = max(short_w + 1, int(round(60 / max(1, bucket_sec))))
    sign_hist: List[int] = []
    intensity_hist: List[float] = []
    spread_delta_hist: List[float] = []
    ret_abs_hist: List[float] = []

    imb_ema = 0.0
    ret_ema = 0.0
    prev_spread = None
    prev_intensity = None
    prev_intensity_slope = 0.0
    prev_vol = None
    prev_sign = 0
    streak = 0

    for i, r in enumerate(out):
        spread = r.get("spread")
        intensity = r.get("trade_intensity")
        imb = r.get("imbalance")
        vol = r.get("micro_volatility")
        ret_1 = r.get("ret_1")
        liq_imb = r.get("liq_imbalance")
        liq_rate = r.get("liq_rate_per_sec")

        spread_f = _safe(spread, 0.0)
        intensity_f = _safe(intensity, 0.0)
        imb_f = _safe(imb, 0.0)
        vol_f = _safe(vol, 0.0)
        ret_f = _safe(ret_1, 0.0)
        liq_imb_f = _safe(liq_imb, 0.0)
        liq_rate_f = _safe(liq_rate, 0.0)

        sign = 1 if ret_f > 0 else (-1 if ret_f < 0 else 0)
        if sign != 0:
            if sign == prev_sign:
                streak += sign
            else:
                streak = sign
            prev_sign = sign

        # spread compression / speed
        delta_spread = 0.0 if prev_spread is None else (spread_f - prev_spread)
        spread_recompress = 0.0 if prev_spread is None or prev_spread <= 0 else ((prev_spread - spread_f) / prev_spread)
        prev_spread = spread_f
        spread_delta_hist.append(delta_spread)

        # intensity acceleration / decay
        d_intensity = 0.0 if prev_intensity is None else (intensity_f - prev_intensity)
        prev_intensity = intensity_f
        prev_int_win = intensity_hist[-short_w:]
        int_prev_mean = (sum(prev_int_win) / len(prev_int_win)) if prev_int_win else 0.0
        intensity_slope = 0.0 if int_prev_mean <= 0.0 else ((intensity_f - int_prev_mean) / int_prev_mean)
        intensity_accel = intensity_slope - prev_intensity_slope
        prev_intensity_slope = intensity_slope
        intensity_hist.append(intensity_f)
        recent_max_int = max(intensity_hist[-short_w:]) if intensity_hist else intensity_f
        intensity_decay = 0.0 if recent_max_int <= 0 else ((recent_max_int - intensity_f) / recent_max_int)

        # volatility burst vs stabilization
        vol_change = 0.0 if prev_vol is None else (vol_f - prev_vol)
        prev_vol = vol_f
        vol_stabilize = -vol_change

        # imbalance persistence
        imb_ema = (alpha_ema * imb_f) + ((1.0 - alpha_ema) * imb_ema)
        imb_streak_norm = max(-1.0, min(1.0, streak / float(long_w)))
        imb_persist = 0.5 * imb_ema + 0.5 * imb_streak_norm
        ret_ema = (alpha_ema * ret_f) + ((1.0 - alpha_ema) * ret_ema)

        # reversal risk proxy
        sign_hist.append(sign)
        flips = 0
        window_sign = [s for s in sign_hist[-short_w:] if s != 0]
        for a, b in zip(window_sign[:-1], window_sign[1:]):
            if a != b:
                flips += 1
        flip_rate = (flips / max(1, len(window_sign) - 1)) if window_sign else 0.0
        ret_abs_hist.append(abs(ret_f))
        ret_abs_mean = sum(ret_abs_hist[-short_w:]) / max(1, min(len(ret_abs_hist), short_w))
        meanrev_prob = min(1.0, 0.5 * flip_rate + 0.5 * (1.0 if abs(ret_f) > ret_abs_mean else 0.0))

        # liquidation reversal proxy: sell-liquidation spikes support long fade, buy-liquidation spikes support short fade
        liq_spike = min(1.0, liq_rate_f / max(1.0, intensity_f))
        liq_spike_active = max(0.0, liq_spike - 0.35)
        liq_gate_strength = 0.0
        if liq_spike_active > 0.0 and abs(liq_imb_f) >= 0.60 and spread_recompress > 0.0:
            liq_gate_strength = liq_spike_active * (0.5 + 0.5 * meanrev_prob)
        liq_reversal_signal = liq_imb_f * liq_gate_strength

        # score for extractable passive alpha
        side_signal = (0.96 * imb_persist) + (0.04 * liq_reversal_signal)
        fill_quality = (0.45 * spread_recompress) + (0.35 * intensity_decay) + (0.20 * vol_stabilize)
        fill_quality += 0.01 * liq_gate_strength * max(0.0, spread_recompress)
        toxicity = max(0.0, d_intensity) + max(0.0, vol_change) + (0.25 * meanrev_prob)
        core = (abs(side_signal) * fill_quality) - (0.50 * toxicity)
        v2_score_signed = (1.0 if side_signal >= 0 else -1.0) * core
        v2_conf = _sigmoid(abs(core) * 40.0)

        # v3: passive-extractable quality (no lookahead, past-only windows)
        neg_spread = sum(1 for d in spread_delta_hist[-short_w:] if d < 0.0)
        spread_shrink_ratio = (neg_spread / max(1, min(len(spread_delta_hist), short_w)))
        imb_dir = 1.0 if imb_persist > 0.0 else (-1.0 if imb_persist < 0.0 else 0.0)
        ret_dir = 1.0 if ret_ema > 0.0 else (-1.0 if ret_ema < 0.0 else 0.0)
        follow_agree = 1.0 if (imb_dir != 0.0 and imb_dir == ret_dir) else (0.0 if ret_dir == 0.0 else -1.0)
        follow_mult = 1.0 if follow_agree > 0.0 else (0.35 if follow_agree < 0.0 else 0.60)
        liq_follow = 1.0 if (liq_reversal_signal != 0.0 and (1.0 if liq_reversal_signal > 0.0 else -1.0) == imb_dir) else 0.0
        v3_persist = (0.97 * imb_persist) + (0.03 * liq_reversal_signal)
        v3_quality = (
            abs(imb_persist)
            * max(0.0, intensity_slope)
            * (0.5 + 0.5 * spread_shrink_ratio)
            * follow_mult
            + 0.20 * max(0.0, spread_recompress)
            + 0.10 * max(0.0, intensity_decay)
            + 0.01 * liq_gate_strength * (0.5 + 0.5 * liq_follow)
        )
        v3_toxic = (0.45 * max(0.0, vol_change)) + (0.35 * max(0.0, intensity_accel)) + (0.20 * meanrev_prob)
        v3_core = v3_quality - v3_toxic
        v3_side_signal = v3_persist if follow_agree >= 0.0 else (0.5 * imb_persist)
        v3_score_signed = (1.0 if v3_side_signal >= 0.0 else -1.0) * max(0.0, v3_core)
        v3_conf = _sigmoid(max(0.0, v3_core) * 45.0)

        r["v2_delta_spread"] = float(delta_spread)
        r["v2_spread_recompress"] = float(spread_recompress)
        r["v2_d_intensity"] = float(d_intensity)
        r["v2_intensity_decay"] = float(intensity_decay)
        r["v2_vol_change"] = float(vol_change)
        r["v2_vol_stabilize"] = float(vol_stabilize)
        r["v2_imbalance_ema"] = float(imb_ema)
        r["v2_imbalance_persist"] = float(imb_persist)
        r["v2_streak_norm"] = float(imb_streak_norm)
        r["v2_flip_rate"] = float(flip_rate)
        r["v2_meanrev_prob"] = float(meanrev_prob)
        r["v2_liq_spike"] = float(liq_spike)
        r["v2_liq_spike_active"] = float(liq_spike_active)
        r["v2_liq_gate_strength"] = float(liq_gate_strength)
        r["v2_liq_reversal_signal"] = float(liq_reversal_signal)
        r["v2_side_signal"] = float(side_signal)
        r["v2_score"] = float(v2_score_signed)
        r["v2_confidence"] = float(v2_conf)
        r["v3_spread_shrink_ratio"] = float(spread_shrink_ratio)
        r["v3_intensity_slope"] = float(intensity_slope)
        r["v3_intensity_accel"] = float(intensity_accel)
        r["v3_follow_agree"] = float(follow_agree)
        r["v3_liq_follow_agree"] = float(liq_follow)
        r["v3_imbalance_persist"] = float(v3_persist)
        r["v3_toxicity"] = float(v3_toxic)
        r["v3_side_signal"] = float(v3_side_signal)
        r["v3_score"] = float(v3_score_signed)
        r["v3_confidence"] = float(v3_conf)

    if cache_key is not None:
        _V2_CACHE[cache_key] = _copy_rows(out)
    return out
