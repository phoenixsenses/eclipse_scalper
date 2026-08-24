from __future__ import annotations

import os

from execution.schemas import FeatureSnapshot, validate_feature_snapshot
from data.features.registry import compute_features
from data.quality import journal_data_quality, validate_candles
from features.volatility_expansion import compute_volatility_state
from strategies.regime import detect_regime
from strategies.regime_effective import RegimeState, compute_effective_regime, journal_regime_transition

_REQ_KEYS = ("timestamp", "open", "high", "low", "close", "volume")
_REGIME_STATE: dict[tuple[str, str], RegimeState] = {}


def _parse_tf_minutes(tf: str) -> int:
    s = str(tf or "").strip().lower()
    if not s:
        return 1
    if s.endswith("m"):
        return max(1, int(float(s[:-1] or 1)))
    if s.endswith("h"):
        return max(1, int(float(s[:-1] or 1) * 60))
    return max(1, int(float(s)))


def _mtf_timeframes() -> list[str]:
    raw = str(os.getenv("MTF_TIMEFRAMES", "5m,15m") or "5m,15m")
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out or ["5m"]


def _median_step_ms(candles: list[dict]) -> int:
    if len(candles) < 3:
        return 60_000
    diffs: list[int] = []
    for i in range(1, min(len(candles), 120)):
        try:
            a = int(candles[i - 1]["timestamp"])
            b = int(candles[i]["timestamp"])
            d = b - a
            if d > 0:
                diffs.append(d)
        except Exception:
            continue
    if not diffs:
        return 60_000
    diffs.sort()
    return int(diffs[len(diffs) // 2])


def _aggregate_candles(candles: list[dict], bars_per: int) -> list[dict]:
    if bars_per <= 1:
        return list(candles)
    out: list[dict] = []
    for i in range(0, len(candles), bars_per):
        chunk = candles[i : i + bars_per]
        if len(chunk) < bars_per:
            continue
        try:
            out.append(
                {
                    "timestamp": int(chunk[-1]["timestamp"]),
                    "open": float(chunk[0]["open"]),
                    "high": max(float(c["high"]) for c in chunk),
                    "low": min(float(c["low"]) for c in chunk),
                    "close": float(chunk[-1]["close"]),
                    "volume": sum(float(c["volume"]) for c in chunk),
                }
            )
        except Exception:
            continue
    return out


def _compute_htf_descriptor(candles: list[dict], base_timeframe: str) -> dict:
    tfs = _mtf_timeframes()
    tf = tfs[0]
    target_ms = _parse_tf_minutes(tf) * 60_000
    base_ms = max(1, _median_step_ms(candles))
    bars_per = max(1, int(round(target_ms / float(base_ms))))
    agg = _aggregate_candles(candles, bars_per)
    out = {
        "htf_data_ready": 0.0,
        "htf_trend_dir": 0.0,
        "htf_trend_strength": 0.0,
        "htf_regime": "unknown",
        "htf_timeframe": str(tf),
        "htf_source_timeframe": str(base_timeframe),
    }
    if len(agg) < 30:
        return out
    closes = [float(c["close"]) for c in agg[-30:]]
    sma_fast = sum(closes[-10:]) / 10.0
    sma_slow = sum(closes[-30:]) / 30.0
    if sma_slow <= 0:
        return out
    gap = (sma_fast / sma_slow) - 1.0
    slope = (closes[-1] - closes[0]) / max(1e-12, abs(closes[0]) * float(len(closes) - 1))
    direction = 0
    if gap > 0 and slope > 0:
        direction = 1
    elif gap < 0 and slope < 0:
        direction = -1
    strength = max(abs(gap) / 0.002, abs(slope) / 0.0002)
    strength = max(0.0, min(1.0, strength))
    regime = "ranging"
    if direction > 0 and strength >= 0.2:
        regime = "trending_up"
    elif direction < 0 and strength >= 0.2:
        regime = "trending_down"
    out.update(
        {
            "htf_data_ready": 1.0,
            "htf_trend_dir": float(direction),
            "htf_trend_strength": float(strength),
            "htf_regime": str(regime),
        }
    )
    return out


def _require_candle_keys(c: dict, idx: int) -> None:
    missing = [k for k in _REQ_KEYS if k not in c]
    if missing:
        raise ValueError(f"candle at index {idx} missing keys: {missing}")


def build_feature_snapshot(
    symbol: str,
    timeframe: str,
    candles: list[dict],
) -> FeatureSnapshot:
    if not candles:
        raise ValueError("candles must be non-empty")
    quality = validate_candles(candles)
    journal_data_quality(symbol=str(symbol), timeframe=str(timeframe), report=quality)
    if str(quality.get("severity", "block")) == "block":
        raise ValueError("Data quality block")
    for i, c in enumerate(candles):
        if not isinstance(c, dict):
            raise ValueError(f"candle at index {i} must be dict")
        _require_candle_keys(c, i)

    c = candles[-1]
    feats = compute_features(candles)
    regime, conf = detect_regime(candles)
    raw_regime = str(regime) if float(conf) > 0.0 else "unknown"
    sk = (str(symbol), str(timeframe))
    prev_state = _REGIME_STATE.get(sk)
    curr_state = compute_effective_regime(
        raw_regime=raw_regime,
        confidence=float(conf),
        ts=int(c["timestamp"]),
        prev_state=prev_state,
    )
    _REGIME_STATE[sk] = curr_state
    journal_regime_transition(str(symbol), str(timeframe), prev_state, curr_state)
    feats["regime_raw"] = raw_regime
    feats["regime_effective"] = curr_state.effective_regime
    feats["regime_streak"] = int(curr_state.streak)
    feats["regime_confidence"] = float(conf)
    feats.update(_compute_htf_descriptor(candles, timeframe))
    vol_state = compute_volatility_state(candles)
    feats["vol_state"] = str(vol_state.get("state", "unknown") or "unknown")
    feats["compression_score"] = float(vol_state.get("compression_score", 0.0) or 0.0)
    feats["expansion_score"] = float(vol_state.get("expansion_score", 0.0) or 0.0)
    feats["atr_pct"] = float(vol_state.get("atr_pct", 0.0) or 0.0)
    feats["atr_pct_z"] = float(vol_state.get("atr_pct_z", 0.0) or 0.0)
    feats["range_pct"] = float(vol_state.get("range_pct", 0.0) or 0.0)
    snap = FeatureSnapshot(
        symbol=str(symbol),
        timeframe=str(timeframe),
        timestamp=int(c["timestamp"]),
        open=float(c["open"]),
        high=float(c["high"]),
        low=float(c["low"]),
        close=float(c["close"]),
        volume=float(c["volume"]),
        features=feats,
        regime=str(curr_state.effective_regime),
        regime_confidence=float(conf),
    )
    validate_feature_snapshot(snap)
    return snap


def build_snapshots_rolling(
    symbol: str,
    timeframe: str,
    candles: list[dict],
    start_index: int = 0,
) -> list[FeatureSnapshot]:
    out: list[FeatureSnapshot] = []
    start = max(0, int(start_index))
    for i in range(start, len(candles)):
        out.append(build_feature_snapshot(symbol, timeframe, candles[: i + 1]))
    return out
