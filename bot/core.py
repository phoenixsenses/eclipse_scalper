#!/usr/bin/env python3
# bot/core.py — SCALPER ETERNAL — GOD-EMPEROR CORE — 2026 v4.8 (RUNNER-COHERENT + DATA_READY + ENV SYMBOL RESPECT)

from __future__ import annotations

import asyncio
import time
import os
import random
from datetime import datetime, timezone
from typing import Set, Dict, List, Optional, Any

from utils.logging import log, log_core, log_data, log_brain
from brain.state import PsycheState
from brain.persistence import save_brain, load_brain
from data.cache import GodEmperorDataOracle as DataCache
from notifications.telegram import Notifier
from exchanges import get_exchange

from execution.entry import try_enter
from execution.emergency import emergency_flat
from execution.exit import handle_exit
from strategies.risk import portfolio_heat

# Kill switch
from risk.kill_switch import evaluate as ks_evaluate, trade_allowed as ks_trade_allowed, is_halted as ks_is_halted


FALLBACK_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "SHIBUSDT", "WIFUSDT",
    "BNBUSDT", "TONUSDT", "TRXUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "NEARUSDT", "LTCUSDT", "APTUSDT",
    "SUIUSDT", "HBARUSDT", "UNIUSDT", "FILUSDT", "INJUSDT",
    "OPUSDT", "ARBUSDT", "RUNEUSDT", "ATOMUSDT", "SEIUSDT",
]

CORRELATION_GROUPS = {
    "MAJOR": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "LAYER1": ["SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "TONUSDT"],
    "DEFi": ["UNIUSDT", "LINKUSDT", "AAVEUSDT", "MKRUSDT", "CRVUSDT"],
    "MEME": ["DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "WIFUSDT"],
}

CORE_VERSION = "god-emperor-ascendant-absolute-v4.8-2026-jan"
STARTUP_TIMESTAMP = time.time()


def _is_micro_mode(cfg) -> bool:
    try:
        return str(getattr(cfg, "CONFIG_VERSION", "")).startswith("micro")
    except Exception:
        return False


def _parse_whitelist(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.split(",")]
    return [p for p in parts if p]


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _extract_usdt_equity(balance: Any) -> float:
    if not isinstance(balance, dict):
        return 0.0

    total = balance.get("total")
    if isinstance(total, dict):
        v = _safe_float(total.get("USDT"), 0.0)
        if v > 0:
            return v

    info = balance.get("info")
    if isinstance(info, dict):
        for k in ("totalWalletBalance", "totalMarginBalance", "walletBalance", "equity"):
            v = _safe_float(info.get(k), 0.0)
            if v > 0:
                return v

        a = info.get("assets") or info.get("balances")
        if isinstance(a, list):
            for row in a:
                if not isinstance(row, dict):
                    continue
                if str(row.get("asset") or row.get("currency") or "").upper() == "USDT":
                    for kk in ("walletBalance", "balance", "equity", "totalWalletBalance"):
                        v = _safe_float(row.get(kk), 0.0)
                        if v > 0:
                            return v

    usdt = balance.get("USDT")
    if isinstance(usdt, dict):
        v = _safe_float(usdt.get("total") or usdt.get("free") or usdt.get("used"), 0.0)
        if v > 0:
            return v

    return 0.0


class EclipseEternal:
    """
    Runner-compatible bot core.

    Key contract for bootstrap:
      - bot.ex / bot.exchange
      - bot.data (DataCache)
      - bot.state
      - bot.cfg
      - bot._shutdown asyncio.Event
      - bot.data_ready asyncio.Event  (NEW: bootstrap gating)
      - bot.active_symbols set[str] canonical
    """

    def __init__(self):
        self.ex = get_exchange()
        self.exchange = self.ex  # alias for runner compatibility

        self.state = PsycheState()
        self.data = DataCache()

        self.notify = (
            Notifier(token=os.getenv("TELEGRAM_TOKEN"), chat_id=os.getenv("TELEGRAM_CHAT_ID"))
            if os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")
            else None
        )

        self.active_symbols: Set[str] = set()     # canonical keys only
        self.min_amounts: Dict[str, float] = {}   # canonical keys only

        self.session_peak_equity: float = 0.0

        self._shutdown = asyncio.Event()
        self._shutdown_once = asyncio.Event()

        # ✅ bootstrap gating expects this
        self.data_ready = asyncio.Event()

        self.cfg = None

        self._tasks: Dict[str, asyncio.Task] = {}
        self._entry_semaphore = asyncio.Semaphore(10)
        self._last_entry_attempt: Dict[str, float] = {}

        # only set once
        self._data_ready_set_once = False

    async def speak(self, text: str, priority: str = "critical"):
        if self.notify:
            try:
                await self.notify.speak(text, priority)
            except Exception:
                pass

    # ---- Kill-switch wrappers (sync/async safe) ----
    async def _halted(self) -> bool:
        try:
            v = ks_is_halted(self)
            if asyncio.iscoroutine(v):
                return bool(await v)
            return bool(v)
        except Exception:
            return False

    async def _trade_allowed(self) -> bool:
        try:
            v = ks_trade_allowed(self)
            if asyncio.iscoroutine(v):
                return bool(await v)
            return bool(v)
        except Exception:
            return False  # fail-closed for entry safety

    # ---- Task helpers ----
    def _track_task(self, name: str, coro) -> None:
        old = self._tasks.get(name)
        if old and not old.done():
            old.cancel()
        self._tasks[name] = asyncio.create_task(coro, name=name)

    async def _cancel_all_tasks(self):
        for t in list(self._tasks.values()):
            if t and not t.done():
                t.cancel()
        try:
            await asyncio.wait(list(self._tasks.values()), timeout=2.5)
        except Exception:
            pass
        self._tasks.clear()

    async def _safe_loop(self, name: str, coro):
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_core.error(f"Task {name} crashed: {e}")

    def _resolve_raw_symbol(self, k: str, fallback: str) -> str:
        try:
            raw_map = getattr(self.data, "raw_symbol", {}) or {}
            if isinstance(raw_map, dict) and raw_map.get(k):
                return str(raw_map[k])
        except Exception:
            pass
        return fallback

    def _maybe_set_data_ready(self) -> None:
        """
        Best-effort: set once when we have at least *some* usable market data.
        bootstrap waits on this to start entry loop.
        """
        if self._data_ready_set_once:
            return
        try:
            # price map exists?
            price_map = getattr(self.data, "price", None)
            if isinstance(price_map, dict) and any(_safe_float(v, 0.0) > 0 for v in price_map.values()):
                self._data_ready_set_once = True
                self.data_ready.set()
                return
        except Exception:
            pass

        # fallback: any df not-empty
        try:
            for k in list(self.active_symbols)[:4]:
                df = None
                try:
                    df = self.data.get_df(k, "1m")
                except Exception:
                    df = None
                if df is not None and getattr(df, "empty", True) is False and len(df) >= 10:
                    self._data_ready_set_once = True
                    self.data_ready.set()
                    return
        except Exception:
            pass

    # ---------------------------
    # Symbol discovery
    # ---------------------------
    async def _load_dynamic_symbols(self):
        """
        Only used if runner/bootstrap didn't already set active_symbols.
        """
        try:
            markets_obj = None
            try:
                if hasattr(self.ex, "load_markets"):
                    markets_obj = await self.ex.load_markets()
            except Exception:
                markets_obj = None

            if not markets_obj:
                try:
                    markets_obj = await self.ex.fetch_markets()
                except Exception:
                    markets_obj = None

            whitelist = _parse_whitelist(os.getenv("MICRO_SYMBOL_WHITELIST"))
            use_whitelist = _is_micro_mode(self.cfg) and len(whitelist) > 0

            norm_to_info: Dict[str, dict] = {}

            if isinstance(markets_obj, dict):
                for raw_sym, info in markets_obj.items():
                    if not isinstance(info, dict):
                        continue
                    sym = str(info.get("symbol") or raw_sym or "")
                    if not sym or info.get("active") is False:
                        continue
                    mtype = (info.get("type") or "").lower()
                    is_swap = info.get("swap") is True or mtype == "swap"
                    if not is_swap:
                        continue
                    quote = str(info.get("quote") or info.get("settle") or "").upper()
                    if quote != "USDT":
                        continue
                    norm_to_info[_symkey(sym)] = info

            elif isinstance(markets_obj, list):
                for info in markets_obj:
                    if not isinstance(info, dict):
                        continue
                    sym = str(info.get("symbol") or "")
                    if not sym or info.get("active") is False:
                        continue
                    mtype = (info.get("type") or "").lower()
                    is_swap = info.get("swap") is True or mtype == "swap"
                    if not is_swap:
                        continue
                    quote = str(info.get("quote") or info.get("settle") or "").upper()
                    if quote != "USDT":
                        continue
                    norm_to_info[_symkey(sym)] = info

            if not norm_to_info:
                raise RuntimeError("No usable USDT swap markets discovered")

            chosen: List[str] = []
            if use_whitelist:
                for want in whitelist:
                    if want in norm_to_info:
                        chosen.append(want)
                if not chosen:
                    log_core.critical("MICRO_SYMBOL_WHITELIST set but none matched — falling back")

            if not chosen:
                all_norm = list(norm_to_info.keys())
                limit = int(os.getenv("MICRO_SYMBOL_LIMIT", "12")) if _is_micro_mode(self.cfg) else int(os.getenv("SYMBOL_LIMIT", "60"))
                fallback_set = set(FALLBACK_SYMBOLS)

                def _rank_key(k: str):
                    info = norm_to_info.get(k, {}) or {}
                    vol = _safe_float(info.get("quoteVolume") or info.get("volume") or 0.0, 0.0)
                    boost = 1_000_000_000.0 if k in fallback_set else 0.0
                    return boost + vol

                ranked = sorted(all_norm, key=_rank_key, reverse=True)
                chosen = ranked[:max(1, int(limit))]

                log_core.critical(f"DISCOVERY — {len(chosen)} perpetuals (limit={limit}, micro={_is_micro_mode(self.cfg)})")

            self.active_symbols = set(chosen)
            for k in self.active_symbols:
                info = norm_to_info.get(k, {}) or {}
                m = 0.0
                try:
                    lim = (info.get("limits") or {}) if isinstance(info.get("limits"), dict) else {}
                    amt = (lim.get("amount") or {}) if isinstance(lim.get("amount"), dict) else {}
                    m = _safe_float(amt.get("min"), 0.0)
                except Exception:
                    m = 0.0
                self.min_amounts[k] = m if m > 0 else 0.0

        except Exception as e:
            log_core.critical(f"Dynamic discovery failed: {e} — fallback invoked")
            self.active_symbols = set(FALLBACK_SYMBOLS)
            for sym in self.active_symbols:
                self.min_amounts[sym] = 0.0

    # ---------------------------
    # Core lifecycle
    # ---------------------------
    async def start(self):
        """
        Runner/bootstrap may call this OR may just use loops directly.
        This method is safe either way.
        """
        try:
            self.state.version = CORE_VERSION
        except Exception:
            pass

        log_core.critical(f"GOD-EMPEROR CORE AWAKENED — VERSION {CORE_VERSION}")

        # default: do NOT spawn internal watchers (bootstrap/guardian owns supervision)
        spawn_internal = bool(getattr(self.cfg, "CORE_SPAWN_INTERNAL_WATCHERS", False) or False)

        max_pos = int(getattr(self.cfg, "MAX_CONCURRENT_POSITIONS", 5) or 5)
        self._entry_semaphore = asyncio.Semaphore(max(2, max_pos * 2))

        # Exchange init
        log_core.info("Ascending to Binance Futures...")
        try:
            if hasattr(self.ex, "load_markets"):
                await self.ex.load_markets()
            else:
                await self.ex.fetch_markets()

            # Hedge mode if supported
            try:
                if hasattr(self.ex, "set_position_mode"):
                    await self.ex.set_position_mode(True)
            except Exception:
                pass

        except Exception as e:
            log_core.critical(f"Ascension failed: {e}")
            await self.speak("ASCENSION FAILED — CONNECTION/INIT ERROR", "critical")
            return

        # Load cache + brain
        try:
            await self.data.load_cache()
        except Exception:
            pass

        # Bootstrap raw_symbol map from exchange markets (best effort)
        try:
            bm = getattr(self.data, "bootstrap_markets", None)
            if callable(bm):
                await bm(self)
        except Exception:
            pass

        restored = await load_brain(self.state)

        try:
            self.state.validate()
        except Exception:
            pass
        try:
            self.state.recompute_derived()
        except Exception:
            pass

        try:
            self.state.version = CORE_VERSION
        except Exception:
            pass

        if restored:
            log_brain.critical(f"BRAIN RESURRECTED — {len(self.state.positions)} positions reborn")
        else:
            log_brain.info("Fresh consciousness")

        # ✅ Respect symbols pre-set by bootstrap/env
        if not self.active_symbols:
            await self._load_dynamic_symbols()
        else:
            # ensure canonical set + min map exists
            self.active_symbols = set(_symkey(s) for s in self.active_symbols if s)
            for k in self.active_symbols:
                self.min_amounts.setdefault(k, 0.0)

        log_core.critical(f"{len(self.active_symbols)} symbols active")

        # PRIME equity once
        equity = 0.0
        try:
            bal = await self.ex.fetch_balance()
            equity = _extract_usdt_equity(bal)
        except Exception:
            equity = 0.0

        if equity <= 0 and self.state.current_equity > 0:
            equity = float(self.state.current_equity)

        if equity > 0:
            self.state.current_equity = equity
            self.state.peak_equity = max(float(self.state.peak_equity or 0.0), equity)
            self.state.start_of_day_equity = equity
            self.session_peak_equity = max(float(self.session_peak_equity or 0.0), equity)
        else:
            self.state.current_equity = 0.0

        self.state.current_day = datetime.now(timezone.utc).date()

        # ✅ internal loops only if explicitly enabled
        if spawn_internal:
            self._track_task("watcher", self._safe_loop("watcher", self._watcher_loop()))
            self._track_task("cache_saver", self._safe_loop("cache_saver", self._cache_saver_loop()))
            log_core.critical("CORE ONLINE — internal watcher loops spawned")
        else:
            log_core.critical("CORE ONLINE — runner-supervised mode (no internal watcher loops)")

        # keep alive until shutdown (useful if start() is used as entrypoint)
        while not self._shutdown.is_set():
            await asyncio.sleep(5)

    # ---------------------------
    # Runner-facing loops
    # ---------------------------
    async def data_loop(self):
        """
        Runner will spawn this. It starts per-symbol polling tasks.
        """
        if not self.active_symbols:
            # if runner starts data_loop without calling start(), we still need symbols
            await self._load_dynamic_symbols()

        # Start polling tasks using CANONICAL symbols
        for k in sorted(self.active_symbols):
            log_data.info(f"Polling → {k}")
            self._track_task(
                f"ohlcv:{k}",
                self._safe_loop(
                    f"ohlcv:{k}",
                    self.data.poll_ohlcv(bot=self, sym=k, tf="1m", storage=self.data.ohlcv, interval=11),
                ),
            )
            self._track_task(
                f"ticker:{k}",
                self._safe_loop(f"ticker:{k}", self.data.poll_ticker(bot=self, sym=k)),
            )

        # Keep alive until shutdown
        while not self._shutdown.is_set():
            self._maybe_set_data_ready()
            await asyncio.sleep(1.0)

    async def signal_loop(self):
        """
        Runner will spawn this. Responsible for scheduling entries at cadence.
        """
        cycle_count = 0
        while not self._shutdown.is_set():
            await asyncio.sleep(47)
            cycle_count += 1

            hour = datetime.now(timezone.utc).hour
            heat = portfolio_heat(self.state.positions, self.state.current_equity) if self.state.current_equity > 0 else 0.0

            log_core.info(
                f"CYCLE {cycle_count:06d} | UTC {hour:02d}h | "
                f"Pos:{len(self.state.positions):2d} Heat:{heat:6.1%} | "
                f"Equity:${self.state.current_equity:11,.0f} | "
                f"Daily:${self.state.daily_pnl:+10,.0f} | "
                f"Active:{len(self.active_symbols)}"
            )

            if await self._halted():
                continue

            if not await self._correlation_heat_check():
                continue

            await self._schedule_entries()

    async def _correlation_heat_check(self) -> bool:
        if not self.state.current_equity or self.state.current_equity <= 0:
            return False

        eq = float(self.state.current_equity)
        lev = float(getattr(self.cfg, "LEVERAGE", 1) or 1)

        expo: Dict[str, float] = {}
        for sym, pos in (self.state.positions or {}).items():
            try:
                k = _symkey(sym)
                expo[k] = expo.get(k, 0.0) + abs(float(pos.size) * float(pos.entry_price)) / max(1.0, lev)
            except Exception:
                continue

        cap = float(getattr(self.cfg, "CORRELATION_HEAT_CAP", 0.35))
        for group, symbols in CORRELATION_GROUPS.items():
            group_exposure = sum(expo.get(s, 0.0) for s in symbols)
            heat = group_exposure / eq if eq > 0 else 0.0
            if heat > cap:
                log_core.warning(f"Correlation heat {group}: {heat:.1%} — blocked")
                return False

        return True

    async def _schedule_entries(self):
        if not self.state.current_equity or self.state.current_equity <= 0:
            return
        if not await self._trade_allowed():
            return

        now = time.time()
        debounce_s = float(getattr(self.cfg, "ENTRY_DEBOUNCE_SEC", 6.0))
        spawn_both = bool(getattr(self.cfg, "SPAWN_BOTH_SIDES", False))

        for k in list(self.active_symbols):
            if self._shutdown.is_set():
                return
            if k in self.state.positions:
                continue
            if self.state.blacklist.get(k, 0) > now:
                continue
            last_exit = self.state.last_exit_time.get(k, 0.0)
            if now - last_exit < float(getattr(self.cfg, "SYMBOL_COOLDOWN_MINUTES", 0.0)) * 60.0:
                continue

            last = self._last_entry_attempt.get(k, 0.0)
            if now - last < debounce_s:
                continue
            self._last_entry_attempt[k] = now

            age_1m = self.data.get_cache_age(k, "1m")
            if age_1m > 120.0:
                continue

            df = self.data.get_df(k, "1m")
            if df is None or df.empty or len(df) < 120:
                continue

            if spawn_both:
                asyncio.create_task(self._guarded_entry(k, "long"))
                asyncio.create_task(self._guarded_entry(k, "short"))
            else:
                side = "long" if (hash(k) ^ int(now // max(1.0, debounce_s))) % 2 == 0 else "short"
                asyncio.create_task(self._guarded_entry(k, side))

    async def _guarded_entry(self, sym: str, side: str):
        try:
            async with self._entry_semaphore:
                await try_enter(self, sym, side)
        except Exception as e:
            log_core.error(f"Entry task failed {sym} {side}: {e}")

    # ---------------------------
    # Optional internal watcher loops (OFF by default)
    # ---------------------------
    async def _cache_saver_loop(self):
        interval = float(os.getenv("CACHE_SAVE_SEC", "180"))
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(interval)
                await self.data.save_cache()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def _watcher_loop(self):
        log_core.info("Watcher active (internal)")

        orders_poll_sec = float(getattr(self.cfg, "CLOSED_ORDERS_POLL_SEC", 25.0))
        last_orders_poll = 0.0

        while not self._shutdown.is_set():
            try:
                # Equity truth
                try:
                    bal = await self.ex.fetch_balance()
                    total = _extract_usdt_equity(bal)
                except Exception:
                    total = float(self.state.current_equity or 0.0)

                if total > 0:
                    self.state.current_equity = total

                if total > float(self.state.peak_equity or 0.0):
                    self.state.peak_equity = total
                    log_core.critical(f"NEW PEAK: ${total:,.0f}")
                    await self.speak(f"NEW PEAK — ${total:,.0f}", "critical")

                today = datetime.now(timezone.utc).date()
                if today != self.state.current_day:
                    await self.speak(
                        f"DAILY REPORT\nEquity: ${total:,.0f}\nDaily PnL: ${self.state.daily_pnl:+,.0f}\nTrades: {self.state.total_trades}",
                        "critical",
                    )
                    self.state.start_of_day_equity = total
                    self.state.current_day = today
                    self.session_peak_equity = total
                    self.state.daily_pnl = 0.0
                else:
                    if self.state.start_of_day_equity > 0:
                        self.state.daily_pnl = total - self.state.start_of_day_equity

                # Central kill-switch evaluation (log reason)
                try:
                    v = ks_evaluate(self)
                    ok_reason = await v if asyncio.iscoroutine(v) else v
                    if isinstance(ok_reason, tuple):
                        ok, reason = ok_reason
                        if not ok and reason:
                            log_core.critical(f"KILL_SWITCH: {reason}")
                except Exception:
                    pass

                now = time.time()
                if now - last_orders_poll >= orders_poll_sec:
                    last_orders_poll = now
                    await self._poll_closed_orders_for_exits()

                try:
                    self.state.recompute_derived()
                except Exception:
                    pass

            except Exception as e:
                log_core.error(f"Watcher error: {e}")

            await asyncio.sleep(23)

    async def _poll_closed_orders_for_exits(self):
        """
        Internal-only exit polling.
        If you're using guardian Exit Watcher, leave CORE_SPAWN_INTERNAL_WATCHERS=False and this won't run.
        """
        symbols = set(_symkey(s) for s in (self.state.positions or {}).keys())
        extra = int(os.getenv("CLOSED_ORDERS_EXTRA_SYMBOLS", "8"))
        if extra > 0 and self.active_symbols:
            sample = random.sample(list(self.active_symbols), k=min(extra, len(self.active_symbols)))
            for k in sample:
                symbols.add(k)

        orders: List[dict] = []
        for k in list(symbols):
            try:
                sym_raw = self._resolve_raw_symbol(k, k)
                recent = await self.ex.fetch_closed_orders(sym_raw, limit=20)
                if isinstance(recent, list):
                    orders.extend(recent)
            except Exception:
                pass

        for order in orders:
            try:
                await handle_exit(self, order)
            except Exception:
                pass

    async def shutdown(self):
        if self._shutdown_once.is_set():
            return
        self._shutdown_once.set()

        log_core.critical("CORE SHUTDOWN")
        self._shutdown.set()
        try:
            self.data_ready.set()
        except Exception:
            pass

        try:
            await self._cancel_all_tasks()
        except Exception:
            pass

        try:
            await emergency_flat(self)
        except Exception:
            pass

        try:
            await save_brain(self.state)
        except Exception:
            pass

        try:
            await self.ex.close()
        except Exception:
            pass
