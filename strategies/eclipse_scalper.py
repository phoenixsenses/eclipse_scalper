# strategies/eclipse_scalper.py — SCALPER ETERNAL — COSMIC SIGNAL ASCENDANT PRODUCTION — 2026 v3.0
# Patch vs v2.9:
# - ✅ NEW: SCALPER_DEBUG_LOOSE=1  → relax gates (controlled) to see entries sooner
# - ✅ NEW: SCALPER_FORCE_ENTRY_TEST=1 → plumbing validation (max N per symbol; dry-run by default)
# - ✅ NEW: Force-mode safety: blocked on live unless SCALPER_FORCE_ENTRY_ALLOW_LIVE=1
# - ✅ Keeps: multi-key df lookup, adaptive MIN_BARS, DIAG, near-miss logs, SciPy optional peaks
# - ✅ Guardian-safe: never raises

from __future__ import annotations

import os
import time
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator

from utils.logging import log
from types import SimpleNamespace

# Optional unified staleness check
try:
    from execution.data_quality import staleness_check  # type: ignore
except Exception:
    staleness_check = None

# SciPy optional
try:
    from scipy.signal import find_peaks  # type: ignore
except Exception:
    find_peaks = None


# ----------------------------
# Throttled diagnostics
# ----------------------------

_DIAG_LAST: dict[str, float] = {}

# Force-entry session memory (per process)
_FORCE_FIRED: set[str] = set()

# Strategy audit (CSV)
_AUDIT_LAST: dict[str, float] = {}
_AUDIT_HEADER_WRITTEN: bool = False


def _diag_on() -> bool:
    try:
        return os.getenv("SCALPER_SIGNAL_DIAG", "").strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _diag_throttled(key: str, msg: str, every_sec: float = 5.0) -> None:
    """Print a diag line at most once per key per every_sec (guarded by SCALPER_SIGNAL_DIAG)."""
    try:
        if not _diag_on():
            return
        now = time.time()
        last = float(_DIAG_LAST.get(key, 0.0) or 0.0)
        if (now - last) < max(0.2, float(every_sec)):
            return
        _DIAG_LAST[key] = now
        log.info(msg)
    except Exception:
        pass


def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name, "")
        if v is None or str(v).strip() == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name, "")
        if v is None or str(v).strip() == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _env_float_sym(name: str, default: float, sym_suffix: str) -> float:
    try:
        v = os.getenv(f"{name}_{sym_suffix}", "")
        if v is None or str(v).strip() == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _env_bool_sym(name: str, default: bool, sym_suffix: str) -> bool:
    try:
        v = os.getenv(f"{name}_{sym_suffix}", "")
        if v is None or str(v).strip() == "":
            return bool(default)
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return bool(default)


def _env_str(name: str, default: str = "") -> str:
    try:
        v = os.getenv(name, "")
        return str(v) if v is not None else str(default)
    except Exception:
        return str(default)


def _parse_hour_ranges(raw: str) -> list[tuple[int, int]]:
    """
    Parse "13-17,19-23" into [(13,17),(19,23)].
    Hours are inclusive bounds in UTC.
    """
    out: list[tuple[int, int]] = []
    s = str(raw or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                lo = int(float(a))
                hi = int(float(b))
                lo = max(0, min(23, lo))
                hi = max(0, min(23, hi))
                out.append((lo, hi))
            except Exception:
                continue
        else:
            try:
                h = int(float(p))
                h = max(0, min(23, h))
                out.append((h, h))
            except Exception:
                continue
    return out


def _hour_in_ranges(hour: int, ranges: list[tuple[int, int]]) -> bool:
    for lo, hi in ranges:
        if lo <= hi:
            if lo <= hour <= hi:
                return True
        else:
            # wrap-around (e.g., 22-2)
            if hour >= lo or hour <= hi:
                return True
    return False


def _safe_find_peaks(arr: np.ndarray, prominence: float):
    """SciPy optional peaks: returns empty arrays if unavailable or errors."""
    if find_peaks is None:
        return np.array([], dtype=int), {}
    try:
        peaks, props = find_peaks(arr, prominence=prominence)
        return peaks, props
    except Exception:
        return np.array([], dtype=int), {}


def _audit_on() -> bool:
    try:
        return os.getenv("SCALPER_AUDIT", "").strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _audit_path() -> str:
    try:
        p = os.getenv("SCALPER_AUDIT_PATH", "").strip()
        return p if p else os.path.join("logs", "strategy_audit.csv")
    except Exception:
        return os.path.join("logs", "strategy_audit.csv")


def _audit_emit(k: str, outcome: str, confidence: float, data: dict, blockers: list[str]) -> None:
    global _AUDIT_HEADER_WRITTEN
    try:
        if not _audit_on():
            return
        cooldown = float(os.getenv("SCALPER_AUDIT_COOLDOWN_SEC", "5") or 5.0)
        now = time.time()
        last = float(_AUDIT_LAST.get(k, 0.0) or 0.0)
        if cooldown > 0 and (now - last) < cooldown:
            return
        _AUDIT_LAST[k] = now

        path = _audit_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        row = {
            "ts": int(now),
            "symbol": k,
            "outcome": outcome,
            "confidence": round(float(confidence), 3),
            "blockers": "|".join(blockers or []),
        }
        row.update(data)

        write_header = not _AUDIT_HEADER_WRITTEN and (not os.path.exists(path) or os.path.getsize(path) == 0)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                _AUDIT_HEADER_WRITTEN = True
            w.writerow(row)
    except Exception:
        return


def _symkey(sym: str) -> str:
    """
    Canonicalize symbols for cache dict keys.
    Examples:
      'BTC/USDT:USDT' -> 'BTCUSDT'
      'BTC/USDT'      -> 'BTCUSDT'
      'BTCUSDT'       -> 'BTCUSDT'
    """
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _base_from_usdt(k: str) -> str | None:
    kk = _symkey(k)
    if kk.endswith("USDT") and len(kk) > 4:
        return kk[:-4]
    return None


def _candidate_keys(data_obj, sym: str) -> list[str]:
    """
    Candidate lookup keys in the order we prefer.
    Covers your cache reality:
      - canonical (BTCUSDT)
      - raw (BTC/USDT)
      - perp raw (BTC/USDT:USDT)
    """
    out: list[str] = []
    try:
        s = str(sym or "").strip()
    except Exception:
        s = ""
    k = _symkey(s)

    def add(x: str | None):
        if not x:
            return
        xx = str(x).strip()
        if not xx:
            return
        if xx not in out:
            out.append(xx)

    add(s)
    add(k)

    # raw_symbol map from data_loop (canonical -> raw)
    try:
        raw = getattr(data_obj, "raw_symbol", None)
        if isinstance(raw, dict):
            add(raw.get(k))
    except Exception:
        pass

    base = _base_from_usdt(k)
    if base:
        add(f"{base}/USDT")
        add(f"{base}/USDT:USDT")
        add(f"{base}USDT")  # redundant but harmless

    return out


def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling VWAP over `window` bars using typical price."""
    try:
        tp = (df["h"] + df["l"] + df["c"]) / 3.0
        pv = tp * df["v"]
        vwap = pv.rolling(window, min_periods=max(10, window // 5)).sum() / df["v"].rolling(
            window, min_periods=max(10, window // 5)
        ).sum()
        return vwap
    except Exception:
        return df["c"].copy()


def _pair_last_two_pivots_by_proximity(
    price_idx: np.ndarray,
    osc_idx: np.ndarray,
    max_sep: int = 12,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    try:
        if len(price_idx) < 2 or len(osc_idx) < 2:
            return None, None

        pairs = []
        for p in price_idx[::-1]:
            diffs = np.abs(osc_idx - p)
            j = int(np.argmin(diffs))
            if diffs[j] <= max_sep:
                pairs.append((int(p), int(osc_idx[j])))
            if len(pairs) >= 2:
                break

        if len(pairs) < 2:
            return None, None

        (p2, o2), (p1, o1) = pairs[0], pairs[1]
        if p1 > p2:
            p1, p2 = p2, p1
            o1, o2 = o2, o1

        return (p1, p2), (o1, o2)
    except Exception:
        return None, None


# ----------------------------
# Data adapters (MiniDataCache friendly)
# ----------------------------

def _tf_minutes(tf: str) -> int:
    t = str(tf or "").strip().lower()
    if t.endswith("m"):
        try:
            return int(t[:-1])
        except Exception:
            return 1
    if t in ("1h", "60m", "60"):
        return 60
    return 1


def _df_from_ohlcv_rows(rows: list) -> pd.DataFrame | None:
    """rows: [[ts_ms, o, h, l, c, v], ...] -> df(ts,o,h,l,c,v)"""
    try:
        if not isinstance(rows, list) or len(rows) < 10:
            return None
        data = []
        for r in rows:
            if isinstance(r, (list, tuple)) and len(r) >= 6:
                try:
                    ts = int(r[0])
                    o = float(r[1])
                    h = float(r[2])
                    l = float(r[3])
                    c = float(r[4])
                    v = float(r[5])
                    if ts > 0 and c > 0:
                        data.append((ts, o, h, l, c, v))
                except Exception:
                    continue
        if len(data) < 10:
            return None

        df = pd.DataFrame(data, columns=["ts", "o", "h", "l", "c", "v"]).sort_values("ts")
        df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        return df
    except Exception:
        return None


def _aggregate_from_1m(df1: pd.DataFrame, tf: str) -> pd.DataFrame | None:
    """Build higher TF candles from 1m data (requires df1['ts'] ms)."""
    try:
        m = _tf_minutes(tf)
        if m <= 1:
            return df1

        if df1 is None or len(df1) < (m * 20) or "ts" not in df1.columns:
            return None

        bucket = (df1["ts"] // (m * 60_000)) * (m * 60_000)
        g = df1.groupby(bucket, sort=True)

        out = pd.DataFrame(
            {
                "ts": g["ts"].last().astype(int),
                "o": g["o"].first().astype(float),
                "h": g["h"].max().astype(float),
                "l": g["l"].min().astype(float),
                "c": g["c"].last().astype(float),
                "v": g["v"].sum().astype(float),
            }
        ).reset_index(drop=True)

        return out if len(out) >= 50 else None
    except Exception:
        return None


def _normalize_df_for_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    """Ensure numeric o/h/l/c/v columns exist + drop NaNs."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 10:
            return None
        for col in ("o", "h", "l", "c", "v"):
            if col not in df.columns:
                return None

        out = df.copy()
        for col in ("o", "h", "l", "c", "v"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["o", "h", "l", "c", "v"])
        if out.empty or len(out) < 10:
            return None
        return out
    except Exception:
        return None


def _get_df_flexible(data_obj, sym: str, tf: str) -> tuple[pd.DataFrame | None, str]:
    """
    Returns (df, used_key).
    Tries:
      1) data_obj.get_df(key, tf) for multiple keys
      2) build from data_obj.ohlcv using multiple keys
      3) for higher TF: aggregate from 1m rows (needs ts)
    """
    tfs = str(tf or "1m").strip().lower()
    keys = _candidate_keys(data_obj, sym)

    # 1) Native get_df with multiple keys
    try:
        fn = getattr(data_obj, "get_df", None)
        if callable(fn):
            for key in keys:
                try:
                    df0 = fn(key, tfs)
                    df0 = _normalize_df_for_indicators(df0) if isinstance(df0, pd.DataFrame) else None
                    if df0 is not None:
                        return df0, key
                except Exception:
                    continue
    except Exception:
        pass

    # 2) From ohlcv map
    ohlcv_map = None
    try:
        ohlcv_map = getattr(data_obj, "ohlcv", None)
    except Exception:
        ohlcv_map = None

    if not isinstance(ohlcv_map, dict):
        return None, ""

    # find rows for 1m (always start from 1m rows so aggregation is possible)
    rows_1m = None
    used = ""
    for key in keys:
        try:
            rows = ohlcv_map.get(key)
            if isinstance(rows, list) and len(rows) >= 10:
                rows_1m = rows
                used = key
                break
        except Exception:
            continue

    if rows_1m is None:
        return None, ""

    df_rows = _df_from_ohlcv_rows(rows_1m)
    if df_rows is None:
        return None, ""

    if tfs == "1m":
        return _normalize_df_for_indicators(df_rows), used

    if tfs in ("5m", "15m", "30m", "1h"):
        agg = _aggregate_from_1m(df_rows, tfs)
        return (_normalize_df_for_indicators(agg) if agg is not None else None), used

    return _normalize_df_for_indicators(df_rows), used


# ----------------------------
# Main signal
# ----------------------------

def scalper_signal(sym: str, data=None, cfg=None, *args, **kwargs) -> tuple[bool, bool, float]:
    """
    Returns: (long_signal, short_signal, confidence)

    Accepts both:
      - scalper_signal("BTCUSDT", data=bot.data, cfg=bot.cfg)
      - scalper_signal(bot, "BTCUSDT")   # legacy callers
    """
    # ---- Caller adapter (legacy support) ----
    bot = None
    try:
        if not isinstance(sym, str):
            bot = sym
            sym2 = data if isinstance(data, str) else None
            if sym2 is None:
                sym2 = kwargs.get("sym") or kwargs.get("symbol")
            sym = str(sym2 or "")
            data = getattr(bot, "data", None)
            cfg = getattr(bot, "cfg", cfg)
        else:
            bot_kw = kwargs.get("bot", None)
            if bot_kw is not None and data is None:
                bot = bot_kw
                data = getattr(bot_kw, "data", None)
                if cfg is None:
                    cfg = getattr(bot_kw, "cfg", None)
    except Exception:
        pass

    sym = str(sym or "").strip()
    if not sym:
        return False, False, 0.0

    profile = os.getenv("SCALPER_SIGNAL_PROFILE", "").lower().strip()
    micro_profile = (profile == "micro") or (getattr(cfg, "CONFIG_VERSION", "").startswith("micro"))

    k = _symkey(sym)

    # ----------------------------
    # Freshness guard (1m data staleness)
    # ----------------------------
    STALE_MAX_SEC = _env_int("SCALPER_DATA_MAX_STALE_SEC", 180)

    # ----------------------------
    # Debug knobs (new)
    # ----------------------------
    DEBUG_LOOSE = os.getenv("SCALPER_DEBUG_LOOSE", "").strip().lower() in ("1", "true", "yes", "y", "on")

    FORCE_TEST = os.getenv("SCALPER_FORCE_ENTRY_TEST", "").strip().lower() in ("1", "true", "yes", "y", "on")
    FORCE_MAX_PER_SYM = _env_int("SCALPER_FORCE_ENTRY_MAX_PER_SYMBOL", 1)
    FORCE_MIN_CONF = _env_float("SCALPER_FORCE_ENTRY_MIN_CONF", 0.55 if micro_profile else 0.65)

    # Default safety: force-mode only in dry-run unless explicitly allowed
    ALLOW_FORCE_LIVE = os.getenv("SCALPER_FORCE_ENTRY_ALLOW_LIVE", "").strip().lower() in ("1", "true", "yes", "y", "on")
    IS_DRY = os.getenv("SCALPER_DRY_RUN", "").strip().lower() in ("1", "true", "yes", "y", "on")

    # ----------------------------
    # Tunables (env-overridable)
    # ----------------------------
    MIN_BARS_DEFAULT = _env_int("SCALPER_MIN_BARS", 260 if micro_profile else 360)

    # Volume Z-score gates (volume spike, not price volatility)
    VOL_Z_TH = _env_float("SCALPER_VOL_Z_TH", 1.3 if micro_profile else 1.8)
    VOL_Z_SOFT_TH = _env_float("SCALPER_VOL_Z_SOFT_TH", 0.6 if micro_profile else 0.9)
    PREV_VOL_Z_TH = _env_float("SCALPER_PREV_VOL_Z_TH", 1.0 if micro_profile else 1.3)
    VOL_RATIO_WEIGHT = _env_float("SCALPER_VOL_RATIO_WEIGHT", 0.4 if micro_profile else 0.35)
    VOL_RATIO_MAX_BOOST = _env_float("SCALPER_VOL_RATIO_MAX_BOOST", 0.4 if micro_profile else 0.5)
    VOL_RATIO_MAX_DROP = _env_float("SCALPER_VOL_RATIO_MAX_DROP", 0.2 if micro_profile else 0.25)

    # Trend / chop
    ADX_TH = _env_float("SCALPER_ADX_TH", 16.0 if micro_profile else 23.0)
    BB_CHOP_TH = _env_float("SCALPER_BB_CHOP_TH", 0.030 if micro_profile else 0.038)

    # VWAP distance gate
    VWAP_BASE_DIST = _env_float("SCALPER_VWAP_BASE_DIST", 0.006 if micro_profile else 0.010)
    VWAP_ATR_MULT = _env_float("SCALPER_VWAP_ATR_MULT", 1.25 if micro_profile else 1.60)

    # Dynamic momentum threshold
    DYN_MOM_MULT = _env_float("SCALPER_DYN_MOM_MULT", 1.15 if micro_profile else 1.45)
    DYN_MOM_FLOOR = _env_float("SCALPER_DYN_MOM_FLOOR", 0.0009 if micro_profile else 0.0012)

    # Soft “impulse” alternative when volume isn’t spiking but price is moving
    ATR_PCT_SOFT_TH = _env_float("SCALPER_ATR_PCT_SOFT_TH", 0.0018 if micro_profile else 0.0024)

    VWAP_WIN = _env_int("SCALPER_VWAP_WINDOW", 240)
    VWAP_WIN = max(60, min(VWAP_WIN, 2000))

    DIV_MAX_SEP = _env_int("SCALPER_DIV_MAX_SEP", 12)
    DIV_MAX_SEP = max(4, min(DIV_MAX_SEP, 30))

    NEAR_MISS_CONF = _env_float("SCALPER_NEAR_MISS_CONF", 0.62 if micro_profile else 0.70)

    # Optional: use confidence as a final “okay trade” threshold in micro
    MICRO_CONF_TH = _env_float("SCALPER_MICRO_CONF_TH", 0.0)  # 0 disables

    # Confidence calibration (power curve + clamp)
    CONF_POWER = _env_float("SCALPER_CONFIDENCE_POWER", 1.0)
    CONF_MIN = _env_float("SCALPER_CONFIDENCE_MIN", 0.0)
    CONF_MAX = _env_float("SCALPER_CONFIDENCE_MAX", 1.0)
    if CONF_MAX < CONF_MIN:
        CONF_MIN, CONF_MAX = CONF_MAX, CONF_MIN

    # Enhanced gates (on by default, env-disable if needed)
    ENHANCED = os.getenv("SCALPER_ENHANCED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    RANGE_ATR_MIN = _env_float("SCALPER_RANGE_ATR_MIN", 0.60 if micro_profile else 0.80)
    VOL_MIN_MULT = _env_float("SCALPER_VOL_MIN_MULT", 0.70 if micro_profile else 0.90)
    ENH_TREND_1H = os.getenv("SCALPER_ENH_TREND_1H", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    ATR_PCT_MIN = _env_float("SCALPER_ATR_PCT_MIN", 0.0)
    FAST_BACKTEST = os.getenv("SCALPER_FAST_BACKTEST", "").strip().lower() in ("1", "true", "yes", "y", "on")

    # Volatility regime guard + adaptive thresholds
    VOL_REGIME_ENABLED = os.getenv("SCALPER_VOL_REGIME_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    VOL_REGIME_GUARD = os.getenv("SCALPER_VOL_REGIME_GUARD", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    VOL_REGIME_LOW_ATR_PCT = _env_float("SCALPER_VOL_REGIME_LOW_ATR_PCT", 0.0010 if micro_profile else 0.0014)
    VOL_REGIME_HIGH_ATR_PCT = _env_float("SCALPER_VOL_REGIME_HIGH_ATR_PCT", 0.0030 if micro_profile else 0.0040)
    VOL_REGIME_LOW_BB_PCT = _env_float("SCALPER_VOL_REGIME_LOW_BB_PCT", 0.012 if micro_profile else 0.015)
    VOL_REGIME_HIGH_BB_PCT = _env_float("SCALPER_VOL_REGIME_HIGH_BB_PCT", 0.035 if micro_profile else 0.045)
    VOL_REGIME_LOW_MULT = _env_float("SCALPER_VOL_REGIME_LOW_MULT", 1.15 if micro_profile else 1.10)
    VOL_REGIME_HIGH_MULT = _env_float("SCALPER_VOL_REGIME_HIGH_MULT", 0.90 if micro_profile else 0.95)

    # Session filter (UTC hours). Empty => no filter.
    SESSION_UTC = _env_str("SCALPER_SESSION_UTC", "")
    SESSION_RANGES = _parse_hour_ranges(SESSION_UTC)

    # Session momentum guard (UTC hours). Empty => no filter.
    SESSION_MOM_UTC = _env_str("SCALPER_SESSION_MOM_UTC", "")
    SESSION_MOM_RANGES = _parse_hour_ranges(SESSION_MOM_UTC)
    SESSION_MOM_MIN = _env_float("SCALPER_SESSION_MOM_MIN", 0.0015)

    # Trend confirmation (higher TF EMA alignment)
    TREND_CONFIRM = os.getenv("SCALPER_TREND_CONFIRM", "1").strip().lower() in ("1", "true", "yes", "y", "on")
    TREND_CONFIRM_MODE = _env_str("SCALPER_TREND_CONFIRM_MODE", "gate").strip().lower()
    TREND_CONFIRM_TF = _env_str("SCALPER_TREND_CONFIRM_TF", "1h").strip().lower()
    TREND_CONFIRM_FAST = _env_int("SCALPER_TREND_CONFIRM_FAST", 50)
    TREND_CONFIRM_SLOW = _env_int("SCALPER_TREND_CONFIRM_SLOW", 200)

    # Cooldown after losses (requires bot.state)
    COOLDOWN_LOSSES = _env_int("SCALPER_COOLDOWN_LOSSES", 0)
    COOLDOWN_MINUTES = _env_int("SCALPER_COOLDOWN_MINUTES", 0)

    # Symbol-specific base overrides (e.g., SCALPER_ADX_TH_BTC=12)
    sym_base = _base_from_usdt(k) or k
    sym_suffix = sym_base.upper()
    VOL_Z_TH = _env_float_sym("SCALPER_VOL_Z_TH", VOL_Z_TH, sym_suffix)
    VOL_Z_SOFT_TH = _env_float_sym("SCALPER_VOL_Z_SOFT_TH", VOL_Z_SOFT_TH, sym_suffix)
    PREV_VOL_Z_TH = _env_float_sym("SCALPER_PREV_VOL_Z_TH", PREV_VOL_Z_TH, sym_suffix)
    ADX_TH = _env_float_sym("SCALPER_ADX_TH", ADX_TH, sym_suffix)
    BB_CHOP_TH = _env_float_sym("SCALPER_BB_CHOP_TH", BB_CHOP_TH, sym_suffix)
    VWAP_BASE_DIST = _env_float_sym("SCALPER_VWAP_BASE_DIST", VWAP_BASE_DIST, sym_suffix)
    VWAP_ATR_MULT = _env_float_sym("SCALPER_VWAP_ATR_MULT", VWAP_ATR_MULT, sym_suffix)
    DYN_MOM_MULT = _env_float_sym("SCALPER_DYN_MOM_MULT", DYN_MOM_MULT, sym_suffix)
    DYN_MOM_FLOOR = _env_float_sym("SCALPER_DYN_MOM_FLOOR", DYN_MOM_FLOOR, sym_suffix)
    ATR_PCT_SOFT_TH = _env_float_sym("SCALPER_ATR_PCT_SOFT_TH", ATR_PCT_SOFT_TH, sym_suffix)
    # Default DOGE volatility floor unless overridden
    if sym_suffix == "DOGE" and ATR_PCT_MIN <= 0:
        ATR_PCT_MIN = 0.003
    ATR_PCT_MIN = _env_float_sym("SCALPER_ATR_PCT_MIN", ATR_PCT_MIN, sym_suffix)
    RANGE_ATR_MIN = _env_float_sym("SCALPER_RANGE_ATR_MIN", RANGE_ATR_MIN, sym_suffix)
    VOL_MIN_MULT = _env_float_sym("SCALPER_VOL_MIN_MULT", VOL_MIN_MULT, sym_suffix)

    # Quality mode (opt-in): tighter filters for better trade quality
    QUALITY_MODE = os.getenv("SCALPER_QUALITY_MODE", "").strip().lower() in ("1", "true", "yes", "y", "on")
    QUALITY_CONF_MIN = _env_float("SCALPER_QUALITY_CONF_MIN", 0.62 if micro_profile else 0.72)
    QUALITY_DIST_MULT = _env_float("SCALPER_QUALITY_DIST_MULT", 1.10)
    QUALITY_MOM_MULT = _env_float("SCALPER_QUALITY_MOM_MULT", 1.10)
    QUALITY_ADX_BONUS = _env_float("SCALPER_QUALITY_ADX_BONUS", 3.0)

    # Symbol-specific override profile (e.g., SCALPER_QUALITY_MODE_BTC=1)
    QUALITY_MODE = _env_bool_sym("SCALPER_QUALITY_MODE", QUALITY_MODE, sym_suffix)
    QUALITY_CONF_MIN = _env_float_sym("SCALPER_QUALITY_CONF_MIN", QUALITY_CONF_MIN, sym_suffix)
    QUALITY_DIST_MULT = _env_float_sym("SCALPER_QUALITY_DIST_MULT", QUALITY_DIST_MULT, sym_suffix)
    QUALITY_MOM_MULT = _env_float_sym("SCALPER_QUALITY_MOM_MULT", QUALITY_MOM_MULT, sym_suffix)
    QUALITY_ADX_BONUS = _env_float_sym("SCALPER_QUALITY_ADX_BONUS", QUALITY_ADX_BONUS, sym_suffix)

    # ----------------------------
    # DEBUG_LOOSE relax (new)
    # ----------------------------
    if DEBUG_LOOSE:
        VOL_Z_TH = min(VOL_Z_TH, 1.0 if micro_profile else 1.4)
        VOL_Z_SOFT_TH = min(VOL_Z_SOFT_TH, 0.3 if micro_profile else 0.6)
        ATR_PCT_SOFT_TH = min(ATR_PCT_SOFT_TH, 0.0012 if micro_profile else 0.0018)

        VWAP_BASE_DIST = min(VWAP_BASE_DIST, 0.004 if micro_profile else 0.007)
        VWAP_ATR_MULT = min(VWAP_ATR_MULT, 0.95 if micro_profile else 1.25)

        DYN_MOM_MULT = min(DYN_MOM_MULT, 1.0 if micro_profile else 1.25)
        DYN_MOM_FLOOR = min(DYN_MOM_FLOOR, 0.0006 if micro_profile else 0.0009)

        _diag_throttled(
            f"{k}:debugloose",
            f"SIGNAL DIAG {k}: DEBUG_LOOSE=1 applied (softer gates).",
            every_sec=20.0,
        )

    # ---- Early validation ----
    if data is None:
        _diag_throttled(f"{k}:nodata", f"SIGNAL DIAG {k}: data=None (data loop not attached yet?)", every_sec=3.0)
        return False, False, 0.0

    # Session filter (UTC)
    if SESSION_RANGES:
        now_hour = datetime.now(timezone.utc).hour
        if not _hour_in_ranges(int(now_hour), SESSION_RANGES):
            _diag_throttled(
                f"{k}:session",
                f"SIGNAL DIAG {k}: outside session hours UTC={now_hour} ranges={SESSION_UTC}",
                every_sec=10.0,
            )
            return False, False, 0.0

    # Cooldown after consecutive losses (bot.state required)
    if bot is not None and COOLDOWN_LOSSES > 0 and COOLDOWN_MINUTES > 0:
        try:
            st = getattr(bot, "state", None)
            losses = 0
            last_exit = 0.0
            if st is not None:
                losses = int(getattr(st, "consecutive_losses", {}).get(k, 0) or 0)
                last_exit = float(getattr(st, "last_exit_time", {}).get(k, 0.0) or 0.0)
            if losses >= COOLDOWN_LOSSES and last_exit > 0:
                if (time.time() - last_exit) < (COOLDOWN_MINUTES * 60):
                    _diag_throttled(
                        f"{k}:cooldown",
                        f"SIGNAL DIAG {k}: cooldown active losses={losses} mins={COOLDOWN_MINUTES}",
                        every_sec=8.0,
                    )
                    return False, False, 0.0
        except Exception:
            pass

    # Adaptive MIN_BARS based on cache length across all candidate keys
    bars_available = 0
    try:
        ohlcv_map = getattr(data, "ohlcv", None)
        if isinstance(ohlcv_map, dict):
            for key in _candidate_keys(data, sym):
                rows = ohlcv_map.get(key)
                if isinstance(rows, list):
                    bars_available = max(bars_available, len(rows))
    except Exception:
        bars_available = 0

    # Stale data guard (1m) — unified
    if STALE_MAX_SEC > 0:
        if callable(staleness_check):
            tmp_bot = bot if bot is not None else SimpleNamespace(data=data)
            ok, age_sec, _src = staleness_check(tmp_bot, k, tf="1m", max_sec=STALE_MAX_SEC)
            if not ok:
                _diag_throttled(
                    f"{k}:stale",
                    f"SIGNAL DIAG {k}: stale data age={age_sec:.1f}s > {STALE_MAX_SEC}s",
                    every_sec=6.0,
                )
                return False, False, 0.0
        elif isinstance(getattr(data, "ohlcv", None), dict):
            try:
                latest_ts = 0
                for key in _candidate_keys(data, sym):
                    rows = getattr(data, "ohlcv", {}).get(key)
                    if isinstance(rows, list) and rows:
                        ts = int(rows[-1][0]) if isinstance(rows[-1], (list, tuple)) and len(rows[-1]) > 0 else 0
                        latest_ts = max(latest_ts, ts)
                if latest_ts > 0:
                    age_sec = max(0.0, (time.time() - (latest_ts / 1000.0)))
                    if age_sec > STALE_MAX_SEC:
                        _diag_throttled(
                            f"{k}:stale",
                            f"SIGNAL DIAG {k}: stale data age={age_sec:.1f}s > {STALE_MAX_SEC}s",
                            every_sec=6.0,
                        )
                        return False, False, 0.0
            except Exception:
                pass

    MIN_BARS = int(MIN_BARS_DEFAULT)
    if bars_available > 0:
        MIN_BARS = int(min(MIN_BARS_DEFAULT, max(120, bars_available - 5)))

    try:
        df, used_key = _get_df_flexible(data, sym, "1m")
        if df is None:
            hint = ""
            try:
                ok = isinstance(getattr(data, "ohlcv", None), dict)
                if ok:
                    keys = list(getattr(data, "ohlcv", {}).keys())
                    hint = f"ohlcv_keys_sample={keys[:10]}"
            except Exception:
                pass

            _diag_throttled(
                f"{k}:nodf",
                (
                    f"SIGNAL DIAG {k}: df=None (symbol mismatch or no rows yet). "
                    f"cand={_candidate_keys(data, sym)[:6]} bars_avail={bars_available} {hint}"
                ),
                every_sec=3.0,
            )
            return False, False, 0.0

        if _diag_on() and used_key:
            _diag_throttled(
                f"{k}:dfok",
                f"SIGNAL DIAG {k}: df OK via '{used_key}' len={len(df)} MIN_BARS={MIN_BARS} micro={micro_profile}",
                every_sec=12.0,
            )

        if len(df) < MIN_BARS:
            _diag_throttled(
                f"{k}:fewbars",
                f"SIGNAL DIAG {k}: not enough bars len={len(df)} need>={MIN_BARS} (micro={micro_profile})",
                every_sec=3.0,
            )
            return False, False, 0.0

        for col in ("o", "h", "l", "c", "v"):
            if col not in df.columns:
                _diag_throttled(
                    f"{k}:badcols",
                    f"SIGNAL DIAG {k}: missing col {col} in df.columns={list(df.columns)[:10]}",
                    every_sec=10.0,
                )
                return False, False, 0.0

        o = float(df["o"].iloc[-1])
        h = float(df["h"].iloc[-1])
        l = float(df["l"].iloc[-1])
        c = float(df["c"].iloc[-1])
        v = float(df["v"].iloc[-1])

        if not np.isfinite(c) or c <= 0:
            _diag_throttled(f"{k}:badc", f"SIGNAL DIAG {k}: bad close c={c}", every_sec=10.0)
            return False, False, 0.0

        atr14_s = AverageTrueRange(df["h"], df["l"], df["c"], window=14).average_true_range()
        atr50_s = AverageTrueRange(df["h"], df["l"], df["c"], window=50).average_true_range()

        atr = float(atr14_s.iloc[-1]) if np.isfinite(atr14_s.iloc[-1]) else 0.0
        if atr <= 0:
            _diag_throttled(f"{k}:atr0", f"SIGNAL DIAG {k}: atr<=0 atr={atr}", every_sec=10.0)
            return False, False, 0.0

        atr_pct_s = atr14_s / df["c"].replace(0, np.nan)
        atr_pct = float(atr_pct_s.iloc[-1]) if np.isfinite(atr_pct_s.iloc[-1]) else 0.0
        if atr_pct <= 0:
            _diag_throttled(f"{k}:atrpct0", f"SIGNAL DIAG {k}: atr_pct<=0 atr_pct={atr_pct}", every_sec=10.0)
            return False, False, 0.0

        if ATR_PCT_MIN > 0 and atr_pct < ATR_PCT_MIN:
            _diag_throttled(
                f"{k}:atrpctmin",
                f"SIGNAL DIAG {k}: atr_pct {atr_pct:.3%} < ATR_PCT_MIN {ATR_PCT_MIN:.3%}",
                every_sec=6.0,
            )
            return False, False, 0.0

        atr_ma = float(atr50_s.iloc[-1]) if np.isfinite(atr50_s.iloc[-1]) else atr

        # Enhanced gates: range + volume floor
        range_atr = (h - l) / atr if atr > 0 else 0.0

        vol_regime = (
            "HIGH" if atr > atr_ma * 1.5 else
            "LOW" if atr < atr_ma * 0.7 else
            "MEDIUM"
        )

        bb_width_pct = 0.0
        adx = 0.0
        trending = True
        choppy = False
        if not FAST_BACKTEST:
            bb = BollingerBands(close=df["c"], window=20, window_dev=2)
            mavg_s = bb.bollinger_mavg()
            hband_s = bb.bollinger_hband()
            lband_s = bb.bollinger_lband()
            mid = float(mavg_s.iloc[-1]) if np.isfinite(mavg_s.iloc[-1]) else c
            bw = float(hband_s.iloc[-1] - lband_s.iloc[-1]) if (np.isfinite(hband_s.iloc[-1]) and np.isfinite(lband_s.iloc[-1])) else 0.0
            bb_width_pct = (bw / mid) if (mid and mid > 0) else 0.0

            adx_ind = ADXIndicator(high=df["h"], low=df["l"], close=df["c"], window=14)
            adx_s = adx_ind.adx()
            adx = float(adx_s.iloc[-1]) if np.isfinite(adx_s.iloc[-1]) else 0.0
            trending = adx > ADX_TH

        # Volatility regime (ATR% + BB width) + adaptive thresholds
        if VOL_REGIME_ENABLED:
            low_regime = (atr_pct > 0 and atr_pct < VOL_REGIME_LOW_ATR_PCT) or (
                bb_width_pct > 0 and bb_width_pct < VOL_REGIME_LOW_BB_PCT
            )
            high_regime = (atr_pct > VOL_REGIME_HIGH_ATR_PCT) or (
                bb_width_pct > 0 and bb_width_pct > VOL_REGIME_HIGH_BB_PCT
            )
            if low_regime:
                vol_regime = "LOW"
            elif high_regime:
                vol_regime = "HIGH"
            else:
                vol_regime = "MEDIUM"

            if vol_regime == "LOW":
                DYN_MOM_MULT *= max(0.5, float(VOL_REGIME_LOW_MULT))
                VOL_Z_TH *= max(0.5, float(VOL_REGIME_LOW_MULT))
                VOL_Z_SOFT_TH *= max(0.5, float(VOL_REGIME_LOW_MULT))
                VWAP_BASE_DIST *= max(0.5, float(VOL_REGIME_LOW_MULT))
            elif vol_regime == "HIGH":
                DYN_MOM_MULT *= max(0.3, float(VOL_REGIME_HIGH_MULT))
                VOL_Z_TH *= max(0.3, float(VOL_REGIME_HIGH_MULT))
                VOL_Z_SOFT_TH *= max(0.3, float(VOL_REGIME_HIGH_MULT))
                VWAP_BASE_DIST *= max(0.3, float(VOL_REGIME_HIGH_MULT))

            if VOL_REGIME_GUARD and vol_regime == "LOW":
                _diag_throttled(
                    f"{k}:volreglow",
                    (
                        f"SIGNAL DIAG {k}: vol regime LOW "
                        f"atr_pct={atr_pct:.3%} bbW={bb_width_pct:.3%}"
                    ),
                    every_sec=8.0,
                )
                return False, False, 0.0

        if not FAST_BACKTEST:
            choppy = (bb_width_pct < BB_CHOP_TH and adx < (ADX_TH * 0.90)) and vol_regime != "HIGH"

        # Heikin-ashi-ish close momentum
        ha_c_curr = (o + h + l + c) / 4.0
        if len(df) > 2:
            ha_c_prev = (
                float(df["o"].iloc[-3]) +
                float(df["h"].iloc[-3]) +
                float(df["l"].iloc[-3]) +
                float(df["c"].iloc[-3])
            ) / 4.0
        else:
            ha_c_prev = ha_c_curr

        mom = (ha_c_curr - ha_c_prev) / ha_c_prev if ha_c_prev > 0 else 0.0

        dynamic_mom = max(float(DYN_MOM_FLOOR), atr_pct * float(DYN_MOM_MULT))
        long_mom = mom > dynamic_mom
        short_mom = mom < -dynamic_mom

        # Volume Z-score (volume, not volatility)
        vol_mean_s = df["v"].rolling(20).mean()
        vol_std_s = df["v"].rolling(20).std()

        vol_mean = float(vol_mean_s.iloc[-1]) if np.isfinite(vol_mean_s.iloc[-1]) else 0.0
        vol_std = float(vol_std_s.iloc[-1]) if np.isfinite(vol_std_s.iloc[-1]) else 0.0
        vol_z = (v - vol_mean) / vol_std if vol_std > 0 else 0.0
        _vol_base = vol_mean if vol_mean > 0 else max(v, 1e-9)
        vol_ratio = v / _vol_base if _vol_base > 0 else 1.0
        vol_ratio_delta = vol_ratio - 1.0
        vol_ratio_clamped = max(-VOL_RATIO_MAX_DROP, min(vol_ratio_delta, VOL_RATIO_MAX_BOOST))

        vol_floor_ok = True
        if vol_mean > 0:
            vol_floor_ok = v >= (vol_mean * VOL_MIN_MULT)

        vol_spike_hard = vol_z > VOL_Z_TH
        vol_spike_soft = vol_z > VOL_Z_SOFT_TH
        impulse_soft = atr_pct > ATR_PCT_SOFT_TH

        # VWAP + distance
        vwap_s = _rolling_vwap(df, VWAP_WIN)
        vwap_last = float(vwap_s.iloc[-1]) if np.isfinite(vwap_s.iloc[-1]) else c
        above_vwap = c > vwap_last
        vwap_distance = abs((c - vwap_last) / vwap_last) if vwap_last > 0 else 0.0

        dist_th = max(VWAP_BASE_DIST, atr_pct * VWAP_ATR_MULT)
        distance_ok = vwap_distance > dist_th

        # Persistence
        persistence_long = persistence_short = False
        if len(df) > 25:
            prev_v = float(df["v"].iloc[-2])
            prev_mean = float(vol_mean_s.iloc[-2]) if np.isfinite(vol_mean_s.iloc[-2]) else 0.0
            prev_std = float(vol_std_s.iloc[-2]) if np.isfinite(vol_std_s.iloc[-2]) else 0.0
            prev_vol_z = (prev_v - prev_mean) / prev_std if prev_std > 0 else 0.0
            prev_vol_spike = prev_vol_z > PREV_VOL_Z_TH

            prev_c = float(df["c"].iloc[-2])
            prev_vwap = float(vwap_s.iloc[-2]) if np.isfinite(vwap_s.iloc[-2]) else prev_c
            prev_above_vwap = prev_c > prev_vwap

            ha_c_prev2 = (
                float(df["o"].iloc[-2]) +
                float(df["h"].iloc[-2]) +
                float(df["l"].iloc[-2]) +
                float(df["c"].iloc[-2])
            ) / 4.0
            ha_c_prev3 = (
                float(df["o"].iloc[-3]) +
                float(df["h"].iloc[-3]) +
                float(df["l"].iloc[-3]) +
                float(df["c"].iloc[-3])
            ) / 4.0
            prev_mom = (ha_c_prev2 - ha_c_prev3) / ha_c_prev3 if ha_c_prev3 > 0 else 0.0

            prev_atr_pct = float(atr_pct_s.iloc[-2]) if np.isfinite(atr_pct_s.iloc[-2]) else atr_pct
            dynamic_mom_prev = max(float(DYN_MOM_FLOOR), prev_atr_pct * float(DYN_MOM_MULT))

            persistence_long = prev_mom > dynamic_mom_prev and prev_vol_spike and prev_above_vwap
            persistence_short = prev_mom < -dynamic_mom_prev and prev_vol_spike and (not prev_above_vwap)

        # Multi-timeframe momentum (5m)
        df_5m = None
        df_15m = None
        df_1h = None
        ha_mom_5m_ok = True
        mom_5m = 0.0
        if not FAST_BACKTEST:
            df_5m, _used5 = _get_df_flexible(data, sym, "5m")
            if df_5m is not None and len(df_5m) >= 50 and all(x in df_5m.columns for x in ("o", "h", "l", "c")):
                ha_c_5m = (df_5m["o"] + df_5m["h"] + df_5m["l"] + df_5m["c"]) / 4.0
                mom_5m = float(ha_c_5m.pct_change(2).iloc[-1]) if np.isfinite(ha_c_5m.pct_change(2).iloc[-1]) else 0.0
                mom_5m_min = max(0.0012 if micro_profile else 0.0016, atr_pct * (1.15 if micro_profile else 1.35))
                if long_mom:
                    ha_mom_5m_ok = (mom_5m > mom_5m_min)
                elif short_mom:
                    ha_mom_5m_ok = (mom_5m < -mom_5m_min)

        # Trend (EMA200) — 5m + 15m if available
        trend_long = trend_short = True
        if not FAST_BACKTEST:
            if df_5m is not None and len(df_5m) >= 200 and "c" in df_5m.columns:
                ema_200_5m = float(df_5m["c"].ewm(span=200, adjust=False).mean().iloc[-1])
                trend_long = c > ema_200_5m
                trend_short = c < ema_200_5m

            df_15m, _used15 = _get_df_flexible(data, sym, "15m")
            if df_15m is not None and len(df_15m) >= 200 and "c" in df_15m.columns:
                ema_200_15m = float(df_15m["c"].ewm(span=200, adjust=False).mean().iloc[-1])
                trend_long = trend_long and (c > ema_200_15m)
                trend_short = trend_short and (c < ema_200_15m)

            if ENHANCED and ENH_TREND_1H and (not DEBUG_LOOSE):
                df_1h, _used1h = _get_df_flexible(data, sym, "1h")
                if df_1h is not None and len(df_1h) >= 200 and "c" in df_1h.columns:
                    ema_200_1h = float(df_1h["c"].ewm(span=200, adjust=False).mean().iloc[-1])
                    trend_long = trend_long and (c > ema_200_1h)
                    trend_short = trend_short and (c < ema_200_1h)

        # Trend confirmation (EMA fast vs slow on higher TF)
        trend_confirm_long = True
        trend_confirm_short = True
        if TREND_CONFIRM and TREND_CONFIRM_FAST > 0 and TREND_CONFIRM_SLOW > 0:
            df_tc = None
            if TREND_CONFIRM_TF == "1h":
                df_tc = df_1h
            elif TREND_CONFIRM_TF == "15m":
                df_tc = df_15m
            elif TREND_CONFIRM_TF == "5m":
                df_tc = df_5m
            else:
                df_tc, _used_tc = _get_df_flexible(data, sym, TREND_CONFIRM_TF)

            if df_tc is not None and len(df_tc) >= max(TREND_CONFIRM_FAST, TREND_CONFIRM_SLOW) and "c" in df_tc.columns:
                ema_fast = float(df_tc["c"].ewm(span=TREND_CONFIRM_FAST, adjust=False).mean().iloc[-1])
                ema_slow = float(df_tc["c"].ewm(span=TREND_CONFIRM_SLOW, adjust=False).mean().iloc[-1])
                trend_confirm_long = ema_fast > ema_slow
                trend_confirm_short = ema_fast < ema_slow

        # Session momentum guard (5m + 15m)
        if SESSION_MOM_RANGES and (not FAST_BACKTEST):
            now_hour = datetime.now(timezone.utc).hour
            if _hour_in_ranges(int(now_hour), SESSION_MOM_RANGES):
                mom_15m = 0.0
                mom_15m_ok = True
                if df_15m is not None and len(df_15m) >= 50 and all(x in df_15m.columns for x in ("o", "h", "l", "c")):
                    ha_c_15m = (df_15m["o"] + df_15m["h"] + df_15m["l"] + df_15m["c"]) / 4.0
                    mom_15m = float(ha_c_15m.pct_change(2).iloc[-1]) if np.isfinite(ha_c_15m.pct_change(2).iloc[-1]) else 0.0
                    if long_mom:
                        mom_15m_ok = (mom_15m >= SESSION_MOM_MIN)
                    elif short_mom:
                        mom_15m_ok = (mom_15m <= -SESSION_MOM_MIN)

                mom_5m_ok_guard = True
                if long_mom:
                    mom_5m_ok_guard = (mom_5m >= SESSION_MOM_MIN)
                elif short_mom:
                    mom_5m_ok_guard = (mom_5m <= -SESSION_MOM_MIN)

                if (not mom_5m_ok_guard) or (not mom_15m_ok):
                    _diag_throttled(
                        f"{k}:sessmom",
                        (
                            f"SIGNAL DIAG {k}: session momentum block "
                            f"mom5m={mom_5m:+.4f} mom15m={mom_15m:+.4f} "
                            f"min={SESSION_MOM_MIN:.4f} UTC={now_hour} ranges={SESSION_MOM_UTC}"
                        ),
                        every_sec=8.0,
                    )
                    return False, False, 0.0

        # Quality gate: demand stronger trend/move/structure
        quality_ok = True
        if QUALITY_MODE:
            if not trending or (adx < (ADX_TH + QUALITY_ADX_BONUS)):
                quality_ok = False
            if vwap_distance <= (dist_th * QUALITY_DIST_MULT):
                quality_ok = False
            if long_mom and (mom < (dynamic_mom * QUALITY_MOM_MULT)):
                quality_ok = False
            if short_mom and (mom > (-dynamic_mom * QUALITY_MOM_MULT)):
                quality_ok = False
            if range_atr and range_atr < RANGE_ATR_MIN:
                quality_ok = False
            if not vol_floor_ok:
                quality_ok = False

        rsi = 50.0
        rsi_oversold = False
        rsi_overbought = False
        stoch_k = 50.0
        stoch_long = False
        stoch_short = False
        rsi_series = None
        if not FAST_BACKTEST:
            rsi_series = ta.momentum.RSIIndicator(df["c"], window=14).rsi()
            rsi = float(rsi_series.iloc[-1]) if np.isfinite(rsi_series.iloc[-1]) else 50.0
            rsi_oversold = rsi < (34 if micro_profile else 31)
            rsi_overbought = rsi > (66 if micro_profile else 69)

            stoch = StochasticOscillator(high=df["h"], low=df["l"], close=df["c"], window=14, smooth_window=3)
            stoch_k_s = stoch.stoch()
            stoch_k = float(stoch_k_s.iloc[-1]) if np.isfinite(stoch_k_s.iloc[-1]) else 50.0
            stoch_long = stoch_k < (26 if micro_profile else 21)
            stoch_short = stoch_k > (74 if micro_profile else 79)

        # Candles
        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        body_eff = max(body, 1e-12)

        if micro_profile:
            pinbar_bull = (lower_wick > body_eff * 2.0) and (upper_wick < body_eff * 0.8) and (c > o)
            pinbar_bear = (upper_wick > body_eff * 2.0) and (lower_wick < body_eff * 0.8) and (c < o)
        else:
            pinbar_bull = (lower_wick > body_eff * 2.4) and (upper_wick < body_eff * 0.5) and (c > o)
            pinbar_bear = (upper_wick > body_eff * 2.4) and (lower_wick < body_eff * 0.5) and (c < o)

        o_prev = float(df["o"].iloc[-2])
        c_prev = float(df["c"].iloc[-2])
        engulf_bull = (c > o) and (c_prev < o_prev) and (c > o_prev) and (o < c_prev)
        engulf_bear = (c < o) and (c_prev > o_prev) and (c < o_prev) and (o > c_prev)

        pattern_long = pinbar_bull or engulf_bull
        pattern_short = pinbar_bear or engulf_bear

        funding = float(getattr(data, "funding", {}).get(k, 0.0) or 0.0)
        funding_bias = abs(funding) > 0.0001 and ((funding > 0 and short_mom) or (funding < 0 and long_mom))

        # Divergence (RSI-based)
        bullish_div = bearish_div = hidden_bullish = hidden_bearish = False
        if not FAST_BACKTEST:
            try:
                lookback = min(120, len(df) - 40)
                lookback = max(60, lookback)
                price_slice = df["c"].iloc[-lookback:].values.astype(float)
                rsi_slice = rsi_series.iloc[-lookback:].values.astype(float) if rsi_series is not None else None
                if rsi_slice is None:
                    raise RuntimeError("rsi_series missing")

                price_prominence = atr * 1.0
                rsi_prominence = (price_prominence / c * 100.0) if c > 0 else 12.0

                price_peaks, _ = _safe_find_peaks(price_slice, prominence=price_prominence)
                price_troughs, _ = _safe_find_peaks(-price_slice, prominence=price_prominence)
                rsi_peaks, _ = _safe_find_peaks(rsi_slice, prominence=rsi_prominence)
                rsi_troughs, _ = _safe_find_peaks(-rsi_slice, prominence=rsi_prominence)

                p_pair, r_pair = _pair_last_two_pivots_by_proximity(price_troughs, rsi_troughs, max_sep=DIV_MAX_SEP)
                if p_pair and r_pair:
                    p1, p2 = p_pair
                    r1, r2 = r_pair
                    if price_slice[p2] < price_slice[p1] and rsi_slice[r2] > rsi_slice[r1]:
                        bullish_div = True
                    if price_slice[p2] > price_slice[p1] and rsi_slice[r2] < rsi_slice[r1]:
                        hidden_bullish = True

                p_pair, r_pair = _pair_last_two_pivots_by_proximity(price_peaks, rsi_peaks, max_sep=DIV_MAX_SEP)
                if p_pair and r_pair:
                    p1, p2 = p_pair
                    r1, r2 = r_pair
                    if price_slice[p2] > price_slice[p1] and rsi_slice[r2] < rsi_slice[r1]:
                        bearish_div = True
                    if price_slice[p2] < price_slice[p1] and rsi_slice[r2] > rsi_slice[r1]:
                        hidden_bearish = True
            except Exception:
                pass

        # Session boost
        current_hour = datetime.now(timezone.utc).hour
        high_volume_session = 13 <= current_hour <= 17

        # ----------------------------
        # Entry logic (tuned)
        # ----------------------------
        trig_long = (pattern_long or bullish_div or hidden_bullish or (rsi_oversold and stoch_long))
        trig_short = (pattern_short or bearish_div or hidden_bearish or (rsi_overbought and stoch_short))

        confirm_move = (vol_spike_hard or vol_spike_soft or impulse_soft)

        enhanced_ok = True
        if ENHANCED and (not DEBUG_LOOSE):
            if range_atr and range_atr < RANGE_ATR_MIN:
                enhanced_ok = False
            if not vol_floor_ok:
                enhanced_ok = False

        if micro_profile:
            base_long = long_mom and above_vwap and ha_mom_5m_ok and (trend_long or trending) and (not choppy) and enhanced_ok and quality_ok
            base_short = short_mom and (not above_vwap) and ha_mom_5m_ok and (trend_short or trending) and (not choppy) and enhanced_ok and quality_ok

            long_signal = base_long and distance_ok and confirm_move and trig_long
            short_signal = base_short and distance_ok and confirm_move and trig_short
        else:
            base_long = long_mom and above_vwap and distance_ok and trend_long and ha_mom_5m_ok and (not choppy) and trending and enhanced_ok and quality_ok
            base_short = short_mom and (not above_vwap) and distance_ok and trend_short and ha_mom_5m_ok and (not choppy) and trending and enhanced_ok and quality_ok

            confirm_long = persistence_long or vol_spike_hard or (impulse_soft and trending)
            confirm_short = persistence_short or vol_spike_hard or (impulse_soft and trending)

            long_signal = base_long and confirm_long and trig_long
            short_signal = base_short and confirm_short and trig_short

        # ----------------------------
        # Confidence scoring
        # ----------------------------
        votes = 0.0
        max_votes = 0.0
        include_persistence = not micro_profile

        if long_mom or short_mom:
            votes += 2.5
            bonus = max(0.0, abs(mom) / max(dynamic_mom, 1e-12) - 1.0) * 1.0
            votes += min(1.0, bonus)
        max_votes += 3.5

        if include_persistence:
            if persistence_long or persistence_short:
                votes += 2.2
            max_votes += 2.2

        if vol_spike_hard:
            votes += 2.8
            bonus = max(0.0, vol_z - VOL_Z_TH) * 0.6
            votes += min(1.2, bonus)
            max_votes += 4.0
        else:
            if vol_spike_soft:
                votes += 1.6
            if impulse_soft:
                votes += 1.2
            max_votes += 3.0

        if distance_ok:
            votes += 2.2
        max_votes += 2.2

        if trend_long or trend_short or trending:
            votes += 2.4
        max_votes += 2.4

        if ha_mom_5m_ok:
            votes += 2.2
        max_votes += 2.2

        if rsi_oversold or rsi_overbought:
            votes += 1.2
        max_votes += 1.2

        if stoch_long or stoch_short:
            votes += 1.2
        max_votes += 1.2

        if bullish_div or bearish_div or hidden_bullish or hidden_bearish:
            votes += 2.4
        max_votes += 2.4

        if pattern_long or pattern_short:
            votes += 1.8
        max_votes += 1.8

        if not choppy:
            votes += 1.0
        max_votes += 1.0

        if high_volume_session:
            votes += 0.8
        max_votes += 0.8

        if funding_bias:
            votes += 0.6
        max_votes += 0.6

        if TREND_CONFIRM:
            if trend_confirm_long or trend_confirm_short:
                votes += 1.4
            max_votes += 1.4

        confidence = votes / max_votes if max_votes > 0 else 0.0
        volume_multiplier = 1.0 + vol_ratio_clamped * VOL_RATIO_WEIGHT
        volume_multiplier = max(0.1, volume_multiplier)
        if long_signal or short_signal:
            confidence *= volume_multiplier
        conf_raw = float(confidence)
        if CONF_POWER and CONF_POWER > 0 and confidence > 0:
            confidence = confidence ** CONF_POWER
        conf_powered = float(confidence)
        if CONF_MAX > 0:
            confidence = max(CONF_MIN, min(CONF_MAX, confidence))
        conf_clamped = float(confidence)
        confidence = round(confidence, 2)

        if os.getenv("SCALPER_CONFIDENCE_DIAG", "").strip().lower() in ("1", "true", "yes", "y", "on"):
            _diag_throttled(
                f"{k}:confdiag",
                (
                    f"SIGNAL DIAG {k}: conf raw={conf_raw:.4f} "
                    f"pow({CONF_POWER:.2f})={conf_powered:.4f} "
                    f"clamp[{CONF_MIN:.2f},{CONF_MAX:.2f}]={conf_clamped:.4f}"
                ),
                every_sec=6.0,
            )

        if QUALITY_MODE and (long_signal or short_signal) and confidence < QUALITY_CONF_MIN:
            long_signal = False
            short_signal = False

        # Optional confidence gate for micro (disabled by default)
        if micro_profile and MICRO_CONF_TH > 0:
            if long_signal and confidence < MICRO_CONF_TH:
                long_signal = False
            if short_signal and confidence < MICRO_CONF_TH:
                short_signal = False

        # ---------------------------------------------------
        # FORCE ENTRY TEST (new, plumbing validation)
        # ---------------------------------------------------
        if FORCE_TEST:
            if (not IS_DRY) and (not ALLOW_FORCE_LIVE):
                _diag_throttled(f"{k}:forceblock", f"SIGNAL DIAG {k}: FORCE_ENTRY_TEST blocked (not dry-run).", every_sec=10.0)
            else:
                fired_count = sum(1 for x in _FORCE_FIRED if x.startswith(k + ":"))
                if fired_count < max(1, FORCE_MAX_PER_SYM):
                    base_ok = (not choppy) and ha_mom_5m_ok and (trend_long or trend_short or trending)

                    force_long = base_ok and long_mom and above_vwap
                    force_short = base_ok and short_mom and (not above_vwap)

                    if confidence >= FORCE_MIN_CONF and (force_long or force_short):
                        long_signal = bool(force_long)
                        short_signal = bool(force_short)
                        _FORCE_FIRED.add(f"{k}:{int(time.time())}")
                        log.critical(f"FORCE_ENTRY_TEST FIRED {k} dir={'LONG' if long_signal else 'SHORT'} conf={confidence:.2f} dry={IS_DRY}")

        # Trend confirmation gate (optional)
        if TREND_CONFIRM and TREND_CONFIRM_MODE == "gate" and (long_signal or short_signal):
            if long_signal and (not trend_confirm_long):
                _diag_throttled(
                    f"{k}:trendconfirm",
                    f"SIGNAL DIAG {k}: trend confirm gate blocked LONG tf={TREND_CONFIRM_TF}",
                    every_sec=8.0,
                )
                long_signal = False
            if short_signal and (not trend_confirm_short):
                _diag_throttled(
                    f"{k}:trendconfirm",
                    f"SIGNAL DIAG {k}: trend confirm gate blocked SHORT tf={TREND_CONFIRM_TF}",
                    every_sec=8.0,
                )
                short_signal = False

        # Near-miss explanation (throttled)
        if not (long_signal or short_signal) and confidence >= float(NEAR_MISS_CONF):
            blockers = []
            if not (long_mom or short_mom):
                blockers.append("mom")
            if not (vol_spike_hard or vol_spike_soft or impulse_soft):
                blockers.append(f"volZ<{VOL_Z_SOFT_TH:.2f}&atr<{ATR_PCT_SOFT_TH:.3%}")
            if choppy:
                blockers.append("chop")
            if not distance_ok:
                blockers.append(f"vwapD<{dist_th:.4f}")
            if not confirm_move:
                blockers.append("confirm_move")
            if not (trig_long or trig_short):
                trig_bits = []
                if pattern_long or pattern_short:
                    trig_bits.append("pattern")
                if bullish_div or bearish_div or hidden_bullish or hidden_bearish:
                    trig_bits.append("div")
                if (rsi_oversold and stoch_long) or (rsi_overbought and stoch_short):
                    trig_bits.append("rsi_stoch")
                blockers.append(f"trigger:{'|'.join(trig_bits) or 'none'}")
            if ENHANCED and (not DEBUG_LOOSE):
                if range_atr and range_atr < RANGE_ATR_MIN:
                    blockers.append(f"rangeATR<{RANGE_ATR_MIN:.2f}")
                if not vol_floor_ok:
                    blockers.append(f"vol<{VOL_MIN_MULT:.2f}x")
            if not quality_ok:
                blockers.append("quality")
            if long_mom and not above_vwap:
                blockers.append("below_vwap")
            if short_mom and above_vwap:
                blockers.append("above_vwap")
            if not ha_mom_5m_ok:
                blockers.append("5m_mom")
            if micro_profile:
                if long_mom and not base_long:
                    blockers.append("base_long")
                if short_mom and not base_short:
                    blockers.append("base_short")
            else:
                if long_mom and not base_long:
                    blockers.append("base_long")
                if short_mom and not base_short:
                    blockers.append("base_short")
            if not micro_profile:
                if long_mom and not persistence_long and not vol_spike_hard and not impulse_soft:
                    blockers.append("persist/impulse")
                if short_mom and not persistence_short and not vol_spike_hard and not impulse_soft:
                    blockers.append("persist/impulse")
            if TREND_CONFIRM and TREND_CONFIRM_MODE == "gate":
                if long_mom and (not trend_confirm_long):
                    blockers.append(f"trend_confirm_{TREND_CONFIRM_TF}")
                if short_mom and (not trend_confirm_short):
                    blockers.append(f"trend_confirm_{TREND_CONFIRM_TF}")
            if not trending and not micro_profile:
                blockers.append(f"adx<{ADX_TH:.1f}")

            _diag_throttled(
                f"{k}:nearmiss",
                (
                    f"NEAR MISS {k} | conf={confidence:.2f} | blockers={','.join(blockers) or 'unknown'} | "
                    f"mom={mom:+.2%} dyn={dynamic_mom:.3%} volZ={vol_z:.1f} adx={adx:.1f} "
                    f"vwapD={vwap_distance:.2%} bbW={bb_width_pct:.2%} vwapW={VWAP_WIN} divSep={DIV_MAX_SEP} "
                    f"micro={micro_profile}"
                ),
                every_sec=6.0,
            )
        else:
            blockers = []

        if long_signal or short_signal:
            direction = "LONG" if long_signal else "SHORT"
            div_flag = (bullish_div or bearish_div or hidden_bullish or hidden_bearish)
            log.critical(
                f"SIGNAL {direction} {k} | conf={confidence:.2f} | micro={micro_profile}\n"
                f"mom={mom:+.2%} dyn={dynamic_mom:.3%} volZ={vol_z:.1f} adx={adx:.1f} regime={vol_regime}\n"
                f"vwap={'ABOVE' if above_vwap else 'BELOW'} dist={vwap_distance:.2%} (th={dist_th:.2%}) "
                f"bbW={bb_width_pct:.2%} choppy={choppy} vwapW={VWAP_WIN}\n"
                f"rsi={rsi:.1f} stochK={stoch_k:.1f} div={div_flag} pattern={(pattern_long or pattern_short)} mom5m={mom_5m:+.2%} "
                f"move={'HARD' if vol_spike_hard else ('SOFT' if vol_spike_soft else ('ATR' if impulse_soft else 'NO'))}"
            )

        # Strategy audit (CSV)
        if _audit_on():
            audit_data = {
                "mom": round(float(mom), 6),
                "dyn_mom": round(float(dynamic_mom), 6),
                "vol_z": round(float(vol_z), 3),
                "vol_ratio": round(float(vol_ratio), 3),
                "adx": round(float(adx), 2),
                "vwap_dist": round(float(vwap_distance), 6),
                "atr_pct": round(float(atr_pct), 6),
                "bb_width_pct": round(float(bb_width_pct), 6),
                "above_vwap": int(bool(above_vwap)),
                "choppy": int(bool(choppy)),
                "long_mom": int(bool(long_mom)),
                "short_mom": int(bool(short_mom)),
                "distance_ok": int(bool(distance_ok)),
                "vol_hard": int(bool(vol_spike_hard)),
                "vol_soft": int(bool(vol_spike_soft)),
                "impulse": int(bool(impulse_soft)),
                "trend_long": int(bool(trend_long)),
                "trend_short": int(bool(trend_short)),
                "quality": int(bool(quality_ok)),
                "vol_conf_mult": round(float(volume_multiplier), 3),
            }
            if long_signal or short_signal:
                _audit_emit(k, "signal_long" if long_signal else "signal_short", confidence, audit_data, [])
            elif confidence >= float(NEAR_MISS_CONF):
                _audit_emit(k, "near_miss", confidence, audit_data, blockers)

        return bool(long_signal), bool(short_signal), float(confidence)

    except Exception as e:
        log.error(f"SIGNAL ERROR {k}: {e}")
        return False, False, 0.0
