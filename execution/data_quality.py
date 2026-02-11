"""
Unified staleness + data-quality scoring helpers.
Never raises; safe to call from loops/strategies.
"""

from __future__ import annotations

import time
import asyncio
import os
from typing import Any, Optional, Tuple

from execution.error_codes import ERR_STALE_DATA, ERR_DATA_QUALITY

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit_throttled, emit  # type: ignore
except Exception:
    emit_throttled = None
    emit = None


def _now() -> float:
    return time.time()


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _tf_minutes(tf: str) -> int:
    t = str(tf or "1m").strip().lower()
    if t.endswith("m"):
        try:
            return int(t[:-1])
        except Exception:
            return 1
    if t in ("1h", "60m", "60"):
        return 60
    return 1


def _env_float(*names: str, default: float = 0.0) -> float:
    for n in names:
        try:
            v = os.getenv(n, "")
            if v is None:
                continue
            s = str(v).strip()
            if s != "":
                return float(s)
        except Exception:
            pass
    return float(default)


def _spawn(fn, *args, **kwargs) -> None:
    try:
        loop = asyncio.get_running_loop()
    except Exception:
        return
    try:
        loop.create_task(fn(*args, **kwargs))
    except Exception:
        return


def _get_df(bot, k: str, tf: str):
    try:
        data = getattr(bot, "data", None)
        if data is None:
            return None
        fn = getattr(data, "get_df", None)
        if callable(fn):
            df = fn(k, tf)
            if df is not None:
                return df
        return None
    except Exception:
        return None


def _df_last_ts(df) -> float:
    try:
        if df is None or len(df) <= 0:
            return 0.0
        if "ts" in df.columns:
            ts = float(df["ts"].iloc[-1])
            if ts > 1e12:
                return ts / 1000.0
            if ts > 1e9:
                return ts
            return ts
        idx = getattr(df, "index", None)
        if idx is not None and len(idx) > 0:
            last = idx[-1]
            try:
                return float(getattr(last, "timestamp")())
            except Exception:
                pass
    except Exception:
        return 0.0
    return 0.0


def _ohlcv_last_ts(bot, k: str) -> int:
    try:
        data = getattr(bot, "data", None)
        ohlcv = getattr(data, "ohlcv", None) if data is not None else None
        if not isinstance(ohlcv, dict):
            return 0
        rows = ohlcv.get(k) or ohlcv.get(_symkey(k)) or []
        if not rows and data is not None:
            raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
            raw = None
            try:
                raw = raw_map.get(k) or raw_map.get(_symkey(k))
            except Exception:
                raw = None
            if raw:
                rows = ohlcv.get(raw) or []
        if not rows:
            return 0
        last = rows[-1]
        if isinstance(last, (list, tuple)) and len(last) > 0:
            return int(last[0])
        return 0
    except Exception:
        return 0


def staleness_check(
    bot,
    k: str,
    *,
    tf: str = "1m",
    max_sec: float = 180.0,
    emit_event: bool = True,
) -> Tuple[bool, float, str]:
    """
    Returns (ok, age_sec, source).
    """
    try:
        if max_sec <= 0:
            return True, 0.0, ""
        data = getattr(bot, "data", None)
        age = None
        source = ""
        if data is not None and hasattr(data, "get_cache_age"):
            try:
                age = _safe_float(data.get_cache_age(k, tf), None)
                if age == float("inf"):
                    age = None
                source = "cache_age"
            except Exception:
                age = None
        if age is None:
            ts = _ohlcv_last_ts(bot, k)
            if ts > 0:
                age = max(0.0, _now() - (ts / 1000.0))
                source = "ohlcv"
        if age is None:
            df = _get_df(bot, k, tf)
            ts_sec = _df_last_ts(df)
            if ts_sec > 0:
                age = max(0.0, _now() - ts_sec)
                source = "df"
        if age is None:
            return False, float("inf"), "missing"
        ok = age <= float(max_sec)
        if (not ok) and emit_event and callable(emit_throttled):
            _spawn(
                emit_throttled,
                bot,
                "data.stale",
                key=f"{_symkey(k)}:{tf}:stale",
                cooldown_sec=max(10.0, float(max_sec) * 0.25),
                data={"symbol": _symkey(k), "tf": tf, "age_sec": age, "max_sec": max_sec, "source": source, "code": ERR_STALE_DATA},
                symbol=k,
                level="warning",
            )
        return ok, float(age), source
    except Exception:
        return False, float("inf"), "error"


def quality_score(
    bot,
    k: str,
    *,
    tf: str = "1m",
    max_sec: float = 180.0,
    window: int = 120,
    gap_mult: float = 2.5,
) -> Tuple[float, dict]:
    """
    Returns (score 0-100, details).
    """
    score = 100.0
    details = {"stale_pen": 0.0, "gap_pen": 0.0, "atr_pen": 0.0}
    ok, age, _src = staleness_check(bot, k, tf=tf, max_sec=max_sec, emit_event=False)
    if not ok:
        pen = min(50.0, 50.0 * (float(age) / max(1.0, float(max_sec))))
        score -= pen
        details["stale_pen"] = pen

    df = _get_df(bot, k, tf)
    if df is None or len(df) < 5:
        score = max(0.0, score - 40.0)
        details["gap_pen"] = 40.0
        return max(0.0, min(100.0, score)), details

    try:
        df2 = df.tail(max(10, int(window)))
        if "ts" in df2.columns:
            ts = df2["ts"].astype(float).values
            if len(ts) > 2:
                diffs = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
                expected = _tf_minutes(tf) * 60_000
                gaps = sum(1 for d in diffs if d > (expected * gap_mult))
                pen = min(30.0, gaps * 5.0)
                score -= pen
                details["gap_pen"] = pen
    except Exception:
        pass

    try:
        if all(c in df.columns for c in ("h", "l", "c")):
            import ta  # lazy import
            atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"], window=14).average_true_range().iloc[-1]
            atr_pct = _safe_float(atr, 0.0) / _safe_float(df["c"].iloc[-1], 1.0)
            if atr_pct <= 0:
                score -= 10.0
                details["atr_pen"] = 10.0
    except Exception:
        pass

    return max(0.0, min(100.0, score)), details


def update_quality_state(
    bot,
    k: str,
    *,
    tf: str = "1m",
    max_sec: float = 180.0,
    window: int = 120,
    emit_sec: float = 60.0,
) -> float:
    """
    Compute + store score in bot.state.data_quality, emit telemetry occasionally.
    """
    score, details = quality_score(bot, k, tf=tf, max_sec=max_sec, window=window)
    history_max = int(_env_float("ENTRY_DATA_QUALITY_HISTORY_MAX", "DATA_QUALITY_HISTORY_MAX", default=50) or 50)
    roll_sec = float(_env_float("ENTRY_DATA_QUALITY_ROLL_SEC", "DATA_QUALITY_ROLL_SEC", default=900) or 900.0)
    now = _now()
    roll_score = score
    history_n = 0
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return score
        dq = getattr(st, "data_quality", None)
        if not isinstance(dq, dict):
            st.data_quality = {}
            dq = st.data_quality
        h = getattr(st, "data_quality_history", None)
        if not isinstance(h, dict):
            st.data_quality_history = {}
            h = st.data_quality_history
        key = _symkey(k)
        series = h.get(key)
        if not isinstance(series, list):
            series = []
            h[key] = series
        series.append({"ts": now, "score": float(score)})
        if history_max > 0 and len(series) > history_max:
            series[:] = series[-history_max:]
        if roll_sec > 0:
            cutoff = now - float(roll_sec)
            keep = [x for x in series if float(x.get("ts") or 0.0) >= cutoff]
            if keep:
                series[:] = keep
        history_n = len(series)
        if history_n > 0:
            roll_score = sum(float(x.get("score") or 0.0) for x in series) / float(history_n)
        dq[key] = {"score": score, "roll": roll_score, "ts": now, "tf": tf, "n": history_n}
    except Exception:
        pass

    if callable(emit_throttled) and emit_sec > 0:
        _spawn(
            emit_throttled,
            bot,
            "data.quality",
            key=f"{_symkey(k)}:{tf}:quality",
            cooldown_sec=max(10.0, float(emit_sec)),
            data={"symbol": _symkey(k), "tf": tf, "score": score, "roll": roll_score, "history_n": history_n, "details": details, "code": ERR_DATA_QUALITY},
            symbol=k,
            level="info",
        )
    return score
