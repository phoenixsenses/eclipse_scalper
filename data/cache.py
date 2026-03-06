# data/cache.py — SCALPER ETERNAL — COSMIC DATA TRUTH ORACLE — 2026 v4.5 (MONOTONIC AGE + STALE REPORT + MARKETS BOOTSTRAP)
# Patch vs v4.4:
# - ✅ Monotonic-safe cache age (no wall-clock jump poison)
# - ✅ get_stale_report(): exact stale symbols/TFs + thresholds + last errors
# - ✅ bootstrap_markets(): populate raw_symbol aggressively from exchange markets (prefers futures form)
# - ✅ last_error + fail_streak telemetry (better debugging + adaptive backoff hooks)
# - ✅ Optional truth guard: get_df(..., require_fresh=True) to block stale indicator poison

import asyncio
import time
import os
import json
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from utils.logging import log_data

CACHE_PATH = os.path.expanduser("~/.blade_cosmic_cache.json")

CACHE_VERSION = "cosmic-truth-oracle-ascendant-v4.5-2026-jan06"
ACCEPTED_CACHE_VERSIONS = {
    "cosmic-truth-oracle-ascendant-v4.2-2026-jan05",
    "cosmic-truth-oracle-ascendant-v4.3-2026-jan06",
    "cosmic-truth-oracle-ascendant-v4.4-2026-jan06",
    CACHE_VERSION,
}

# Safety caps
MAX_CANDLES = 1200
MAX_FUNDING_HIST = 12

# Staleness thresholds (seconds) — conservative defaults
PRICE_STALE_SEC_IN_POS = 15.0
PRICE_STALE_SEC_IDLE = 60.0
OHLCV_STALE_SEC_1M = 120.0
OHLCV_STALE_SEC_5M = 600.0
OHLCV_STALE_SEC_15M = 1800.0


def _symkey(sym: str) -> str:
    """
    Canonical symbol key used throughout the bot: 'BTCUSDT'.
    Handles ccxt forms like 'BTC/USDT:USDT' and 'BTC/USDT'.
    Also collapses 'BTCUSDTUSDT' -> 'BTCUSDT' to match brain/state canon law.
    """
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


class GodEmperorDataOracle:
    """
    COSMIC DATA TRUTH ORACLE — v4.5
    Real-time, gap-aware, persistent market cache with symbol-key unification.
    Adds monotonic-safe age + stale reporting + exchange markets bootstrap.
    """

    def __init__(self):
        # Multi-timeframe caches keyed by canonical symkey, stored as rows: [ts_ms, o, h, l, c, v]
        self.ohlcv: Dict[str, List[List[Any]]] = {}
        self.ohlcv_5m: Dict[str, List[List[Any]]] = {}
        self.ohlcv_15m: Dict[str, List[List[Any]]] = {}

        # Ticker caches keyed by canonical symkey
        self.price: Dict[str, float] = {}
        self.bidask: Dict[str, Tuple[float, float]] = {}
        self.funding: Dict[str, float] = {}
        self.funding_history: Dict[str, List[float]] = {}

        # Health / telemetry (wall-clock timestamps, for humans + persistence)
        self.last_poll: Dict[str, float] = {}  # key: f"{k}_{tf}" or k for ticker
        self.gap_count: Dict[str, int] = {}
        self.success_ratio: Dict[str, float] = {}

        # Monotonic timestamps (for reliable age)
        self._last_poll_mono: Dict[str, float] = {}  # same keys as last_poll

        # Error telemetry
        self.last_error: Dict[str, str] = {}         # key -> last exception string
        self.fail_streak: Dict[str, int] = {}        # key -> consecutive failure count

        # Base intervals (seconds)
        self.base_intervals = {"1m": 11, "5m": 45, "15m": 120}

        # Task registry (optional)
        self.poll_tasks: Dict[str, asyncio.Task] = {}

        # Raw symbol mapping:
        # canonical key -> raw ccxt market symbol used to call exchange methods
        self.raw_symbol: Dict[str, str] = {}

        # throttle higher tf derivation per symbol
        self._last_derive_ts: Dict[str, float] = {}

    # ---------- time helpers ----------

    @staticmethod
    def _now_wall() -> float:
        return time.time()

    @staticmethod
    def _now_mono() -> float:
        return time.monotonic()

    def _mark_success(self, key: str):
        noww = self._now_wall()
        nowm = self._now_mono()
        self.last_poll[key] = noww
        self._last_poll_mono[key] = nowm
        self.last_error.pop(key, None)
        self.fail_streak[key] = 0

    def _mark_fail(self, key: str, err: Exception):
        self.last_error[key] = str(err)
        self.fail_streak[key] = int(self.fail_streak.get(key, 0) or 0) + 1

    # ---------- helpers ----------

    @staticmethod
    def _tf_expected_ms(tf: str) -> int:
        return {"1m": 60_000, "5m": 300_000, "15m": 900_000}.get(tf, 60_000)

    @staticmethod
    def _tf_stale_sec(tf: str) -> float:
        if tf == "1m":
            return OHLCV_STALE_SEC_1M
        if tf == "5m":
            return OHLCV_STALE_SEC_5M
        if tf == "15m":
            return OHLCV_STALE_SEC_15M
        return OHLCV_STALE_SEC_1M

    @staticmethod
    def _normalize_ohlcv_rows(raw_rows: List[List[Any]]) -> List[List[Any]]:
        """
        Ensure rows are [ts_ms(int), o, h, l, c, v] and sorted unique by ts.
        Fast enough for 1200 rows, robust against weird inputs.
        """
        if not raw_rows:
            return []

        try:
            df = pd.DataFrame(raw_rows, columns=["ts", "o", "h", "l", "c", "v"])
        except Exception:
            return []

        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"])
        df["ts"] = df["ts"].astype("int64")

        for col in ["o", "h", "l", "c", "v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["o", "h", "l", "c", "v"])

        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")

        return df[["ts", "o", "h", "l", "c", "v"]].values.tolist()

    def _storage_for_tf(self, tf: str) -> Dict[str, List[List[Any]]]:
        return {"1m": self.ohlcv, "5m": self.ohlcv_5m, "15m": self.ohlcv_15m}.get(tf, self.ohlcv)

    def _register_symbol(self, sym: str):
        """Remember canonical<->raw mapping for consistent polling."""
        k = _symkey(sym)
        raw = str(sym or "")

        if not k or not raw:
            return

        cur = self.raw_symbol.get(k) or ""
        if not cur:
            self.raw_symbol[k] = raw
            return

        # Prefer the most "futures ccxt form"
        if ("/USDT:USDT" in raw) and ("/USDT:USDT" not in cur):
            self.raw_symbol[k] = raw
            return
        if (":USDT" in raw) and (":USDT" not in cur):
            self.raw_symbol[k] = raw
            return

    def _resolve_raw(self, k: str, fallback: str) -> str:
        return str(self.raw_symbol.get(k) or fallback or k)

    # ---------- new: exchange markets bootstrap ----------

    async def bootstrap_markets(self, bot) -> bool:
        """
        Populate raw_symbol using exchange markets. Call once at startup after load_markets().
        This makes raw_symbol consistent even before poll loops warm up.
        """
        ex = getattr(bot, "ex", None)
        if ex is None:
            return False

        markets_obj = None
        try:
            if hasattr(ex, "load_markets"):
                markets_obj = await ex.load_markets()
        except Exception:
            markets_obj = None

        if not markets_obj:
            try:
                markets_obj = await ex.fetch_markets()
            except Exception:
                markets_obj = None

        if not markets_obj:
            return False

        count = 0

        def consider(sym_raw: str):
            nonlocal count
            if not sym_raw:
                return
            k = _symkey(sym_raw)
            if not k:
                return
            self._register_symbol(sym_raw)
            count += 1

        try:
            if isinstance(markets_obj, dict):
                for raw_sym, info in markets_obj.items():
                    if isinstance(info, dict):
                        s = str(info.get("symbol") or raw_sym or "")
                    else:
                        s = str(raw_sym or "")
                    consider(s)

            elif isinstance(markets_obj, list):
                for info in markets_obj:
                    if not isinstance(info, dict):
                        continue
                    s = str(info.get("symbol") or "")
                    consider(s)

            if count > 0:
                log_data.critical(f"COSMIC MARKETS BOOTSTRAP — raw_symbol mapped: {len(self.raw_symbol)}")
                return True
        except Exception:
            return False

        return False

    # ---------- public API ----------

    def get_df(self, sym: str, tf: str, *, require_fresh: bool = False) -> pd.DataFrame:
        """
        Accepts raw or canonical, returns dataframe with columns: ts,o,h,l,c,v
        ts is datetime64[ns, UTC]
        If require_fresh=True, returns empty df when tf cache is stale (prevents indicator poison).
        """
        k = _symkey(sym)
        if require_fresh:
            age = self.get_cache_age(k, tf)
            if age == float("inf") or age > float(self._tf_stale_sec(tf)):
                return pd.DataFrame()

        storage = self._storage_for_tf(tf)
        data = storage.get(k, [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["ts", "o", "h", "l", "c", "v"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        return df

    def get_cache_age(self, sym: str, tf: str = "1m") -> float:
        """
        Monotonic-safe cache age. Immune to system clock jumps.
        Uses _last_poll_mono if available; falls back to wall-clock if not.
        """
        k = _symkey(sym)
        key = f"{k}_{tf}" if tf else k

        last_m = self._last_poll_mono.get(key, 0.0) or 0.0
        if last_m > 0:
            return max(0.0, self._now_mono() - float(last_m))

        last_w = self.last_poll.get(key, 0.0) or 0.0
        if last_w > 0:
            # best effort fallback
            return max(0.0, self._now_wall() - float(last_w))

        return float("inf")

    def get_price(self, sym: str, *, in_position: bool = False) -> float:
        """
        Best-effort price getter with staleness protection.
        Returns 0.0 if stale/unknown.
        """
        k = _symkey(sym)
        age = self.get_cache_age(k, tf="")  # ticker key uses just k
        stale_limit = PRICE_STALE_SEC_IN_POS if in_position else PRICE_STALE_SEC_IDLE
        if age > stale_limit:
            return 0.0

        px = _safe_float(self.price.get(k, 0.0), 0.0)
        if px > 0:
            return px

        ba = self.bidask.get(k)
        if ba and ba[0] > 0 and ba[1] > 0:
            return (float(ba[0]) + float(ba[1])) / 2.0

        return 0.0

    def get_bidask(self, sym: str) -> Tuple[float, float]:
        k = _symkey(sym)
        ba = self.bidask.get(k)
        if not ba:
            px = _safe_float(self.price.get(k, 0.0), 0.0)
            return (px, px) if px > 0 else (0.0, 0.0)
        return (float(ba[0]), float(ba[1]))

    def get_funding(self, sym: str) -> float:
        k = _symkey(sym)
        return _safe_float(self.funding.get(k, 0.0), 0.0)

    def get_funding_trend(self, sym: str) -> str:
        k = _symkey(sym)
        history = self.funding_history.get(k, [])
        if len(history) < 3:
            return "unknown"
        trend = float(history[-1]) - float(history[-3])
        if trend > 0.0001:
            return "rising"
        elif trend < -0.0001:
            return "falling"
        return "stable"

    # ---------- new: stale reporting ----------

    def get_stale_report(self, *, active_symbols: Optional[set] = None, in_positions: Optional[set] = None) -> dict:
        """
        Returns a structured stale report useful for kill-switch / logs.
        active_symbols: set of canonical symbols to check (recommended).
        in_positions: set of canonical symbols currently in positions (optional; affects ticker threshold).

        Output:
          {
            "ts": <wall_time>,
            "stale": [
              {"k":"BTCUSDT","kind":"ohlcv","tf":"1m","age":123.4,"limit":120.0,"error":"...","fail_streak":2},
              {"k":"BTCUSDT","kind":"ticker","tf":"","age":70.1,"limit":60.0,"error":"...","fail_streak":1},
              ...
            ],
            "ok_count": N,
            "stale_count": M
          }
        """
        active = set(active_symbols or [])
        posset = set(in_positions or [])

        out = []
        ok = 0

        def push(k: str, kind: str, tf: str, age: float, limit: float):
            key = f"{k}_{tf}" if (kind == "ohlcv") else k
            if age <= limit:
                return False
            out.append({
                "k": k,
                "kind": kind,
                "tf": tf,
                "age": float(age),
                "limit": float(limit),
                "error": str(self.last_error.get(key) or ""),
                "fail_streak": int(self.fail_streak.get(key, 0) or 0),
            })
            return True

        for k in sorted(active):
            # OHLCV staleness for known TFs
            for tf in ("1m", "5m", "15m"):
                key = f"{k}_{tf}"
                age = self.get_cache_age(k, tf)
                lim = float(self._tf_stale_sec(tf))
                if not push(k, "ohlcv", tf, age, lim):
                    ok += 1

            # Ticker staleness (threshold depends on in_position)
            age_t = self.get_cache_age(k, tf="")
            lim_t = float(PRICE_STALE_SEC_IN_POS if (k in posset) else PRICE_STALE_SEC_IDLE)
            if not push(k, "ticker", "", age_t, lim_t):
                ok += 1

        return {
            "ts": float(self._now_wall()),
            "stale": out,
            "ok_count": int(ok),
            "stale_count": int(len(out)),
        }

    # ---------- internal ----------

    async def _heal_gaps(self, bot, *, k: str, raw_sym: str, tf: str, storage: Dict[str, List[List[Any]]]):
        df = self.get_df(k, tf)
        if df is None or df.empty or len(df) < 3:
            return

        expected_ms = self._tf_expected_ms(tf)
        diffs_ms = df["ts"].diff().dt.total_seconds() * 1000.0
        gaps = diffs_ms > (expected_ms * 1.5)
        if not gaps.any():
            return

        gap_indices = df[gaps].index.tolist()
        for idx in gap_indices:
            prev_ts = df["ts"].iloc[idx - 1]
            since = int(prev_ts.timestamp() * 1000) + int(expected_ms)
            try:
                raw = await bot.ex.fetch_ohlcv(raw_sym, tf, since=since, limit=300)
                rows = self._normalize_ohlcv_rows(raw or [])
                if rows:
                    base = storage.get(k, [])
                    merged = self._normalize_ohlcv_rows((base or []) + rows)
                    storage[k] = merged[-MAX_CANDLES:]
                    self.gap_count[f"{k}_{tf}"] = self.gap_count.get(f"{k}_{tf}", 0) + 1
                    log_data.critical(f"COSMIC GAP HEALED {k} {tf} — truth restored")
            except Exception as e:
                log_data.error(f"Gap healing failed {k} {tf}: {e}")

    async def _derive_higher_tf(self, k: str):
        if k not in self.ohlcv or len(self.ohlcv[k]) < 200:
            return

        now = time.time()
        last = self._last_derive_ts.get(k, 0.0)
        if now - last < 30.0:
            return
        self._last_derive_ts[k] = now

        df_1m = self.get_df(k, "1m")
        if df_1m is None or df_1m.empty:
            return

        df_1m = df_1m.set_index("ts")

        df_5m = (
            df_1m.resample("5min")
            .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
            .dropna()
        )

        try:
            if len(df_5m) > 2:
                last_ts = df_5m.index[-1]
                now_utc = pd.Timestamp.now(tz="UTC")
                if (now_utc - last_ts).total_seconds() < 60:
                    df_5m = df_5m.iloc[:-1]
        except Exception:
            pass

        df_5m = df_5m.reset_index()
        df_5m_store = df_5m.copy()
        df_5m_store["ts"] = (df_5m_store["ts"].astype("int64") // 10**6).astype("int64")
        self.ohlcv_5m[k] = df_5m_store[["ts", "o", "h", "l", "c", "v"]].tail(MAX_CANDLES).values.tolist()

        df_15m = (
            df_5m.set_index("ts")
            .resample("15min")
            .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
            .dropna()
        )

        try:
            if len(df_15m) > 2:
                last_ts = df_15m.index[-1]
                now_utc = pd.Timestamp.now(tz="UTC")
                if (now_utc - last_ts).total_seconds() < 120:
                    df_15m = df_15m.iloc[:-1]
        except Exception:
            pass

        df_15m = df_15m.reset_index()
        df_15m_store = df_15m.copy()
        df_15m_store["ts"] = (df_15m_store["ts"].astype("int64") // 10**6).astype("int64")
        self.ohlcv_15m[k] = df_15m_store[["ts", "o", "h", "l", "c", "v"]].tail(MAX_CANDLES).values.tolist()

    async def poll_ohlcv(self, bot, sym: str, tf: str, storage: Dict[str, List[List[Any]]], interval: int = 11):
        self._register_symbol(sym)
        k = _symkey(sym)
        raw_sym = self._resolve_raw(k, sym)

        key = f"{k}_{tf}"
        backoff = 0.0
        success_count = 0
        total_count = 0

        while (k in bot.active_symbols) and (not bot._shutdown.is_set()):
            base = float(self.base_intervals.get(tf, float(interval)))
            in_pos = (k in bot.state.positions)
            dyn = base * (0.7 if in_pos else 1.8)

            # If we are failing repeatedly, expand sleep more aggressively
            streak = int(self.fail_streak.get(key, 0) or 0)
            streak_penalty = min(6.0, 1.0 + 0.35 * streak)

            sleep_s = max(1.0, (dyn + backoff) * streak_penalty)
            await asyncio.sleep(sleep_s)

            total_count += 1

            try:
                raw = await bot.ex.fetch_ohlcv(raw_sym, tf, limit=MAX_CANDLES)
                rows = self._normalize_ohlcv_rows(raw or [])
                if rows:
                    storage[k] = rows[-MAX_CANDLES:]
                    self._mark_success(key)
                    success_count += 1
                    backoff = 0.0

                    await self._heal_gaps(bot, k=k, raw_sym=raw_sym, tf=tf, storage=storage)
                    if tf == "1m":
                        await self._derive_higher_tf(k)
                else:
                    log_data.debug(f"OHLCV empty {key}")

            except Exception as e:
                self._mark_fail(key, e)
                es = str(e)
                if "429" in es or "418" in es:
                    backoff = min(backoff + 60.0, 900.0)
                    log_data.critical(f"COSMIC RATE LIMIT {key} — sleeping +{backoff:.0f}s")
                else:
                    backoff = min(backoff + 10.0, 180.0)
                    log_data.error(f"OHLCV fail {key}: {e}")

            if total_count % 30 == 0:
                ratio = (success_count / total_count) if total_count else 0.0
                self.success_ratio[key] = ratio
                if ratio < 0.6:
                    backoff = min(backoff + 30.0, 300.0)

    def _parse_funding_from_ccxt(self, fr: Any) -> float:
        if fr is None:
            return 0.0
        if isinstance(fr, (int, float)):
            return _safe_float(fr, 0.0)
        if isinstance(fr, dict):
            if "fundingRate" in fr:
                return _safe_float(fr.get("fundingRate"), 0.0)
            if "rate" in fr:
                return _safe_float(fr.get("rate"), 0.0)
            info = fr.get("info") or {}
            if isinstance(info, dict):
                for kk in ("fundingRate", "lastFundingRate", "rate"):
                    if kk in info:
                        return _safe_float(info.get(kk), 0.0)
        return 0.0

    async def poll_ticker(self, bot, sym: str):
        self._register_symbol(sym)
        k = _symkey(sym)
        raw_sym = self._resolve_raw(k, sym)

        backoff = 0.0
        while (k in bot.active_symbols) and (not bot._shutdown.is_set()):
            in_pos = (k in bot.state.positions)
            base_sleep = 3.0 if in_pos else 10.0

            # failure streak penalty
            streak = int(self.fail_streak.get(k, 0) or 0)
            streak_penalty = min(6.0, 1.0 + 0.35 * streak)

            sleep_s = max(1.0, (base_sleep + backoff) * streak_penalty)
            await asyncio.sleep(sleep_s)

            try:
                t = await bot.ex.fetch_ticker(raw_sym)

                last = _safe_float((t or {}).get("last"), 0.0)
                if last <= 0:
                    last = _safe_float((t or {}).get("close"), 0.0)
                if last > 0:
                    self.price[k] = last
                    bid = _safe_float((t or {}).get("bid"), last)
                    ask = _safe_float((t or {}).get("ask"), last)
                    self.bidask[k] = (bid, ask)

                funding = 0.0
                try:
                    if hasattr(bot.ex, "fetch_funding_rate"):
                        fr = await bot.ex.fetch_funding_rate(raw_sym)
                        funding = self._parse_funding_from_ccxt(fr)
                    if funding == 0.0:
                        funding = _safe_float((t or {}).get("fundingRate"), 0.0)
                except Exception:
                    funding = _safe_float((t or {}).get("fundingRate"), 0.0)

                self.funding[k] = funding
                hist = self.funding_history.setdefault(k, [])
                hist.append(float(funding))
                self.funding_history[k] = hist[-MAX_FUNDING_HIST:]

                self._mark_success(k)
                backoff = 0.0

                if abs(funding) > 0.01:
                    log_data.warning(f"EXTREME FUNDING {k}: {funding:.4%}")

            except Exception as e:
                self._mark_fail(k, e)
                es = str(e)
                if "429" in es or "418" in es:
                    backoff = min(backoff + 120.0, 1200.0)
                    log_data.critical(f"TICKER COSMIC BAN {k} — backoff {backoff:.0f}s")
                else:
                    backoff = min(backoff + 10.0, 180.0)
                    log_data.error(f"Ticker fail {k}: {e}")

    # ---------- persistence ----------

    async def save_cache(self):
        try:
            cache_data = {
                "version": CACHE_VERSION,
                "timestamp": self._now_wall(),
                "ohlcv": {k: (v[-MAX_CANDLES:] if isinstance(v, list) else []) for k, v in (self.ohlcv or {}).items()},
                "ohlcv_5m": {k: (v[-MAX_CANDLES:] if isinstance(v, list) else []) for k, v in (self.ohlcv_5m or {}).items()},
                "ohlcv_15m": {k: (v[-MAX_CANDLES:] if isinstance(v, list) else []) for k, v in (self.ohlcv_15m or {}).items()},
                "funding_history": {
                    k: (list(map(float, v))[-MAX_FUNDING_HIST:] if isinstance(v, list) else [])
                    for k, v in (self.funding_history or {}).items()
                },
                "raw_symbol": {k: str(v) for k, v in (self.raw_symbol or {}).items()},
            }
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            log_data.critical("COSMIC CACHE PRESERVED — truth eternal")
        except Exception as e:
            log_data.error(f"Cache preservation failed: {e}")

    async def load_cache(self) -> bool:
        if not os.path.exists(CACHE_PATH):
            return False

        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)

            v = str(data.get("version") or "")
            if v not in ACCEPTED_CACHE_VERSIONS:
                return False

            self.ohlcv = {}
            for k, rows in (data.get("ohlcv", {}) or {}).items():
                ck = _symkey(k)
                if not ck:
                    continue
                self.ohlcv[ck] = self._normalize_ohlcv_rows(rows or [])[-MAX_CANDLES:]

            self.ohlcv_5m = {}
            for k, rows in (data.get("ohlcv_5m", {}) or {}).items():
                ck = _symkey(k)
                if not ck:
                    continue
                self.ohlcv_5m[ck] = self._normalize_ohlcv_rows(rows or [])[-MAX_CANDLES:]

            self.ohlcv_15m = {}
            for k, rows in (data.get("ohlcv_15m", {}) or {}).items():
                ck = _symkey(k)
                if not ck:
                    continue
                self.ohlcv_15m[ck] = self._normalize_ohlcv_rows(rows or [])[-MAX_CANDLES:]

            self.funding_history = {}
            fh = data.get("funding_history", {}) or {}
            for k, vlist in fh.items():
                ck = _symkey(k)
                if not ck:
                    continue
                if isinstance(vlist, list):
                    try:
                        self.funding_history[ck] = [float(x) for x in vlist][-MAX_FUNDING_HIST:]
                    except Exception:
                        self.funding_history[ck] = []
                else:
                    self.funding_history[ck] = []

            self.raw_symbol = {}
            rs = data.get("raw_symbol", {}) or {}
            for k, raw in rs.items():
                ck = _symkey(k)
                rv = str(raw or "")
                if not ck or not rv:
                    continue
                self.raw_symbol[ck] = rv

            log_data.critical("COSMIC CACHE RESURRECTED — truth reborn from eternity")
            return True

        except Exception as e:
            log_data.error(f"Cache resurrection failed: {e}")
            return False
