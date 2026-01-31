# execution/data_loop.py — SCALPER ETERNAL — DATA LOOP — 2026 v1.5 (RAW-SYMBOL AUTO-FALLBACK)
# Patch vs v1.4:
# - ✅ FIX: Do NOT “lock in” a bad raw symbol. Try multiple ccxt symbol formats until one works.
# - ✅ FIX: Prefer what the exchange *actually* lists (ex.markets) — e.g. MATICUSDT on Binance futures.
# - ✅ Keeps: MiniDataCache, resample rules ('min'/'h'), speed knobs, guardian-safe never-raise behavior.
# - ✅ Result: MATIC-style symbols stop showing df=None; cache fills reliably.

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.logging import log_entry, log_core

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit_throttled, emit  # type: ignore
except Exception:
    emit_throttled = None
    emit = None


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        return getattr(getattr(bot, "cfg", None), name, default)
    except Exception:
        return default


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
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


def _pick_symbols(bot) -> list[str]:
    try:
        s = getattr(bot, "active_symbols", None)
        if isinstance(s, set) and s:
            return sorted(list(s))
        if isinstance(s, (list, tuple)) and s:
            return [str(x) for x in s if str(x).strip()]
    except Exception:
        pass

    try:
        s2 = getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(s2, (list, tuple)) and s2:
            return [str(x) for x in s2 if str(x).strip()]
    except Exception:
        pass

    return ["BTCUSDT"]


def _ensure_shutdown_event(bot) -> asyncio.Event:
    ev = getattr(bot, "_shutdown", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot._shutdown = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


def _ensure_data_ready_event(bot) -> asyncio.Event:
    ev = getattr(bot, "data_ready", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot.data_ready = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


# ----------------------------
# Minimal DataCache shim
# ----------------------------

@dataclass
class _MiniDataCache:
    """
    Minimal cache used when a "real" DataCache module isn't present.
    Provides get_df(sym, tf) for strategies.
    """
    price: Dict[str, float]
    ohlcv: Dict[str, List[list]]
    last_poll_ts: Dict[str, float]
    raw_symbol: Dict[str, str]          # canonical -> raw
    lock: asyncio.Lock
    base_timeframe: str = "1m"

    def get_price(self, k: str, in_position: bool = False) -> float:
        return _safe_float(self.price.get(k), 0.0)

    def get_cache_age(self, k: str, tf: str = "1m") -> float:
        key = f"{_symkey(k)}_{tf}" if tf else _symkey(k)
        ts = _safe_float(self.last_poll_ts.get(key), 0.0)
        if ts <= 0 and tf:
            # fallback to non-tf key if present
            ts = _safe_float(self.last_poll_ts.get(_symkey(k)), 0.0)
        if ts <= 0:
            return float("inf")
        return max(0.0, _now() - ts)

    def get_df(self, sym: str, tf: str = "1m") -> Optional[pd.DataFrame]:
        k = _symkey(sym)
        rows = self.ohlcv.get(k) or self.ohlcv.get(sym) or []
        if not rows or len(rows) < 10:
            return None

        try:
            df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
            for col in ("o", "h", "l", "c", "v"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["o", "h", "l", "c", "v"])
            if df.empty:
                return None

            df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("dt").sort_index()

            tf = str(tf or "1m").lower().strip()
            if tf in ("1m", "1min", "1"):
                return df[["o", "h", "l", "c", "v"]].copy()

            rule = None
            if tf in ("5m", "5min", "5"):
                rule = "5min"
            elif tf in ("15m", "15min", "15"):
                rule = "15min"
            elif tf in ("30m", "30min", "30"):
                rule = "30min"
            elif tf in ("1h", "60m", "60"):
                rule = "1h"

            if rule is None:
                return df[["o", "h", "l", "c", "v"]].copy()

            ohlc = df[["o", "h", "l", "c", "v"]].resample(rule).agg(
                {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
            ).dropna()

            return ohlc if not ohlc.empty else None
        except Exception:
            return None


def _ensure_data(bot) -> Any:
    d = getattr(bot, "data", None)
    if d is None:
        d = _MiniDataCache(price={}, ohlcv={}, last_poll_ts={}, raw_symbol={}, lock=asyncio.Lock())
        try:
            bot.data = d
        except Exception:
            pass

    # Ensure required fields exist (even for custom DataCache)
    try:
        if not hasattr(d, "price") or not isinstance(getattr(d, "price", None), dict):
            d.price = {}
    except Exception:
        pass
    try:
        if not hasattr(d, "ohlcv") or not isinstance(getattr(d, "ohlcv", None), dict):
            d.ohlcv = {}
    except Exception:
        pass
    try:
        if not hasattr(d, "last_poll_ts") or not isinstance(getattr(d, "last_poll_ts", None), dict):
            d.last_poll_ts = {}
    except Exception:
        pass
    try:
        if not hasattr(d, "raw_symbol") or not isinstance(getattr(d, "raw_symbol", None), dict):
            d.raw_symbol = {}
    except Exception:
        pass
    try:
        if not hasattr(d, "lock") or not isinstance(getattr(d, "lock", None), asyncio.Lock):
            d.lock = asyncio.Lock()
    except Exception:
        pass
    try:
        if not hasattr(d, "base_timeframe"):
            d.base_timeframe = "1m"
    except Exception:
        pass

    return d


# ----------------------------
# Raw symbol candidates (ccxt-friendly)
# ----------------------------

def _guess_raw_candidates(k: str) -> list[str]:
    """
    From canonical BTCUSDT -> common possibilities:
      - BTCUSDT               (Binance style, often works on futures too)
      - BTC/USDT:USDT         (ccxt perp style for many exchanges)
      - BTC/USDT              (spot style)
    NOTE: Do NOT assume which one exists; we will probe.
    """
    kk = _symkey(k)
    if not kk.endswith("USDT") or len(kk) <= 4:
        return [k]

    base = kk[:-4]
    return [
        kk,
        f"{base}/USDT:USDT",
        f"{base}/USDT",
    ]


def _candidate_raw_symbols(ex, d, k: str) -> list[str]:
    """
    Build an ordered, de-duplicated candidate list.
    Priority:
      1) previously-known mapping (d.raw_symbol[k])
      2) anything the exchange explicitly lists in ex.markets (best truth source)
      3) heuristic candidates
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(x: Optional[str]) -> None:
        if not x:
            return
        s = str(x).strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    # 1) cached mapping first
    try:
        raw_map = getattr(d, "raw_symbol", None)
        if isinstance(raw_map, dict):
            _add(raw_map.get(k))
    except Exception:
        pass

    # 2) exchange markets truth
    markets = None
    try:
        markets = getattr(ex, "markets", None)
    except Exception:
        markets = None

    # wrapper path: ex.exchange.markets
    if not isinstance(markets, dict):
        try:
            inner = getattr(ex, "exchange", None)
            markets = getattr(inner, "markets", None) if inner is not None else None
        except Exception:
            markets = None

    if isinstance(markets, dict) and markets:
        # Prefer exact hits among our likely candidates, but only if exchange lists them
        for cand in _guess_raw_candidates(k):
            if cand in markets:
                _add(cand)

        # Some exchanges use different keys; also try to find any market whose id matches k-ish
        # (safe, best-effort; never required)
        try:
            kk = _symkey(k)
            # Common: market id like "MATICUSDT" might be stored as markets["MATIC/USDT"] etc.
            for sym_key, m in list(markets.items())[:5000]:
                if not isinstance(m, dict):
                    continue
                mid = str(m.get("id", "") or "").upper().strip()
                msym = str(m.get("symbol", "") or "").strip()
                if mid == kk:
                    _add(msym or sym_key)
        except Exception:
            pass

    # 3) heuristic fallbacks
    for cand in _guess_raw_candidates(k):
        _add(cand)

    # ultimate fallback
    if not out:
        out = [k]

    return out


# ----------------------------
# CCXT fetch helpers (with fallback)
# ----------------------------

async def _fetch_ticker_price_one(ex, sym_raw: str) -> Optional[float]:
    fn = getattr(ex, "fetch_ticker", None)
    if not callable(fn):
        return None
    try:
        t = await fn(sym_raw)
        if isinstance(t, dict):
            for key in ("last", "close"):
                v = _safe_float(t.get(key), 0.0)
                if v > 0:
                    return v
            info = t.get("info") or {}
            if isinstance(info, dict):
                v = _safe_float(info.get("lastPrice"), 0.0)
                if v > 0:
                    return v
        return None
    except Exception:
        return None


async def _fetch_ticker_price(ex, candidates: list[str]) -> Tuple[Optional[float], Optional[str]]:
    for sym_raw in candidates:
        px = await _fetch_ticker_price_one(ex, sym_raw)
        if px and px > 0:
            return float(px), sym_raw
    return None, None


async def _fetch_ohlcv_one(ex, sym_raw: str, timeframe: str, limit: int) -> Optional[List[list]]:
    fn = getattr(ex, "fetch_ohlcv", None)
    if not callable(fn):
        return None
    try:
        rows = await fn(sym_raw, timeframe=timeframe, limit=int(limit))
        if isinstance(rows, list) and rows:
            out: List[list] = []
            for r in rows:
                if isinstance(r, (list, tuple)) and len(r) >= 6:
                    ts = _safe_float(r[0], 0.0)
                    if ts > 0:
                        out.append([int(ts), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
            return out if out else None
        return None
    except Exception:
        return None


async def _fetch_ohlcv(ex, candidates: list[str], timeframe: str, limit: int) -> Tuple[Optional[List[list]], Optional[str]]:
    for sym_raw in candidates:
        rows = await _fetch_ohlcv_one(ex, sym_raw, timeframe=timeframe, limit=limit)
        if rows:
            return rows, sym_raw
    return None, None


def _merge_ohlcv(existing: List[list], incoming: List[list], max_keep: int) -> List[list]:
    if not existing:
        return incoming[-max_keep:] if max_keep > 0 else incoming

    m: Dict[int, list] = {}
    for r in existing:
        try:
            m[int(r[0])] = r
        except Exception:
            continue
    for r in incoming:
        try:
            m[int(r[0])] = r
        except Exception:
            continue

    ts_sorted = sorted(m.keys())
    merged = [m[ts] for ts in ts_sorted]
    if max_keep > 0 and len(merged) > max_keep:
        merged = merged[-max_keep:]
    return merged


# ----------------------------
# Main loop
# ----------------------------

async def data_loop(bot) -> None:
    shutdown_ev = _ensure_shutdown_event(bot)
    data_ready_ev = _ensure_data_ready_event(bot)

    ex = getattr(bot, "ex", None)
    if ex is None:
        log_core.warning("DATA_LOOP: bot.ex missing; idling")
        while not shutdown_ev.is_set():
            await asyncio.sleep(1.0)
        return

    d = _ensure_data(bot)

    poll_sec = float(_cfg(bot, "DATA_POLL_SEC", 1.0) or 1.0)
    ticker_every = float(_cfg(bot, "DATA_TICKER_EVERY_SEC", 1.0) or 1.0)
    ohlcv_every = float(_cfg(bot, "DATA_OHLCV_EVERY_SEC", 5.0) or 5.0)

    timeframe = str(_cfg(bot, "DATA_TIMEFRAME", "1m") or "1m")
    ohlcv_limit = int(_cfg(bot, "DATA_OHLCV_LIMIT", 200) or 200)
    ohlcv_keep = int(_cfg(bot, "DATA_OHLCV_KEEP", max(ohlcv_limit, 200)) or max(ohlcv_limit, 200))

    try:
        setattr(d, "base_timeframe", timeframe)
    except Exception:
        pass

    per_symbol_gap = float(_cfg(bot, "DATA_PER_SYMBOL_GAP_SEC", 0.12) or 0.12)
    jitter = float(_cfg(bot, "DATA_JITTER_SEC", 0.05) or 0.05)

    last_ticker_ts: Dict[str, float] = {}
    last_ohlcv_ts: Dict[str, float] = {}

    log_core.info(f"DATA_LOOP ONLINE — pumping prices + OHLCV | bot.data={type(d).__name__} tf={timeframe}")

    # mark ready as soon as we get ANY real data
    data_ready_set = data_ready_ev.is_set()

    while not shutdown_ev.is_set():
        try:
            start = _now()
            syms = _pick_symbols(bot)
            if not syms:
                await asyncio.sleep(max(0.25, poll_sec))
                continue

            for sym in syms:
                if shutdown_ev.is_set():
                    break

                k = _symkey(sym)
                if not k:
                    continue

                cands = _candidate_raw_symbols(ex, d, k)

                # 1) Ticker price (try fallbacks)
                lt = _safe_float(last_ticker_ts.get(k), 0.0)
                if ticker_every <= 0 or (_now() - lt) >= ticker_every:
                    px, used = await _fetch_ticker_price(ex, cands)
                    if px and px > 0 and used:
                        try:
                            async with getattr(d, "lock", asyncio.Lock()):
                                d.price[k] = float(px)
                                d.price[used] = float(px)
                                d.last_poll_ts[k] = _now()
                                # only set raw_symbol if it WORKED
                                d.raw_symbol[k] = used
                        except Exception:
                            try:
                                d.price[k] = float(px)
                                d.last_poll_ts[k] = _now()
                                d.raw_symbol[k] = used
                            except Exception:
                                pass

                        last_ticker_ts[k] = _now()

                        if not data_ready_set:
                            data_ready_set = True
                            try:
                                data_ready_ev.set()
                            except Exception:
                                pass
                    else:
                        if callable(emit_throttled):
                            try:
                                await emit_throttled(
                                    bot,
                                    "data.ticker_missing",
                                    key=f"{k}",
                                    cooldown_sec=30,
                                    data={"symbol": k, "cand": cands[:3]},
                                    level="info",
                                    symbol=k,
                                )
                            except Exception:
                                pass

                # 2) OHLCV (try fallbacks)
                lo = _safe_float(last_ohlcv_ts.get(k), 0.0)
                if ohlcv_every <= 0 or (_now() - lo) >= ohlcv_every:
                    rows, used = await _fetch_ohlcv(ex, cands, timeframe=timeframe, limit=ohlcv_limit)
                    if rows and used:
                        try:
                            async with getattr(d, "lock", asyncio.Lock()):
                                prev = d.ohlcv.get(k) or []
                                merged = _merge_ohlcv(prev, rows, max_keep=ohlcv_keep)
                                d.ohlcv[k] = merged
                                d.ohlcv[used] = merged
                                d.last_poll_ts[f"{k}_{timeframe}"] = _now()
                                # only set raw_symbol if it WORKED
                                d.raw_symbol[k] = used
                        except Exception:
                            try:
                                prev = d.ohlcv.get(k) or []
                                d.ohlcv[k] = _merge_ohlcv(prev, rows, max_keep=ohlcv_keep)
                                d.last_poll_ts[f"{k}_{timeframe}"] = _now()
                                d.raw_symbol[k] = used
                            except Exception:
                                pass

                        last_ohlcv_ts[k] = _now()

                        if not data_ready_set:
                            data_ready_set = True
                            try:
                                data_ready_ev.set()
                            except Exception:
                                pass
                    else:
                        if callable(emit_throttled):
                            try:
                                await emit_throttled(
                                    bot,
                                    "data.ohlcv_missing",
                                    key=f"{k}:{timeframe}",
                                    cooldown_sec=30,
                                    data={"symbol": k, "tf": timeframe, "cand": cands[:3]},
                                    level="info",
                                    symbol=k,
                                )
                            except Exception:
                                pass

                await asyncio.sleep(max(0.0, per_symbol_gap + random.uniform(0.0, jitter)))

            # optional heartbeat
            try:
                setattr(ex, "last_health_check", _now())
            except Exception:
                pass

            elapsed = _now() - start
            if poll_sec > 0 and elapsed < poll_sec:
                await asyncio.sleep(max(0.05, poll_sec - elapsed))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"DATA_LOOP outer error: {e}")
            if callable(emit_throttled):
                try:
                    await emit_throttled(
                        bot,
                        "data.loop_error",
                        key="data_loop",
                        cooldown_sec=10,
                        data={"err": repr(e)[:300]},
                        level="critical",
                    )
                except Exception:
                    pass
            await asyncio.sleep(1.0)

    log_core.critical("DATA_LOOP OFFLINE — shutdown flag set")
