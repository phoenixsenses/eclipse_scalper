# execution/bootstrap.py — SCALPER ETERNAL — ONE TRUE ENTRYPOINT — 2026 v1.8 (OPTIONAL EXIT/POSITION MANAGER TASKS)
# Patch vs v1.7:
# - ✅ NEW: Starts optional exit/position manager loops as tasks (single bot/ex/state — no double-brain)
# - ✅ NEW: Starts protection loops BEFORE entry (so stops/management are online first)
# - ✅ NEW: Clear warnings when optional loops are missing (no silent “why no exits?”)
# - ✅ KEEP: .env load, UTF-8 hardening, ACTIVE_SYMBOLS bridge, data_ready gating, robust cache support
# - ✅ Guardian-safe: never raises from shutdown paths

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any, Optional, Callable, Awaitable, List

# ----------------------------
# UTF-8 hardening (Windows / piping)
# ----------------------------

def _force_utf8_io_best_effort() -> None:
    """
    Prevent UnicodeEncodeError on Windows (cp1254/cp1252 consoles), especially when piping:
      python -m execution.bootstrap | Select-String ...
    """
    try:
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:
        pass

    # Python 3.7+ stream reconfigure
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


_force_utf8_io_best_effort()

# ----------------------------
# Load .env (critical for your BINANCE_API_* keys)
# ----------------------------

def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


_load_dotenv_best_effort()

# ----------------------------
# Logging helpers (best effort)
# ----------------------------

def _get_logger():
    try:
        from utils.logging import log_core  # type: ignore
        return log_core
    except Exception:
        return SimpleNamespace(
            info=print,
            warning=print,
            error=print,
            critical=print,
        )


log_core = _get_logger()


def _opt_import(mod: str) -> Optional[Any]:
    try:
        return importlib.import_module(mod)
    except Exception as e:
        # keep these as warnings; they are optional by design in your framework
        log_core.warning(f"[bootstrap] optional import missing: {mod} ({e})")
        return None


def _callable(obj: Any, name: str) -> Optional[Callable]:
    fn = getattr(obj, name, None)
    return fn if callable(fn) else None


# ----------------------------
# ENV helpers
# ----------------------------

def _parse_env_symbols(raw: str) -> list[str]:
    """
    Accepts:
      "BTCUSDT,ETHUSDT"
      "BTCUSDT;ETHUSDT"
      " BTCUSDT , ETHUSDT "
    Returns uppercase unique list preserving order.
    """
    try:
        s = (raw or "").strip()
        if not s:
            return []
        parts = [p.strip().upper() for p in s.replace(";", ",").split(",")]
        out: list[str] = []
        seen: set[str] = set()
        for p in parts:
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out
    except Exception:
        return []


def _apply_env_symbols_to_cfg(cfg: Any) -> list[str]:
    """
    Bridge for PowerShell:
      $env:ACTIVE_SYMBOLS="BTCUSDT,ETHUSDT,..."
    -> cfg.ACTIVE_SYMBOLS = [...]
    """
    raw = os.getenv("ACTIVE_SYMBOLS", "").strip()
    syms = _parse_env_symbols(raw)
    if not syms:
        return []
    try:
        setattr(cfg, "ACTIVE_SYMBOLS", syms)
    except Exception:
        pass
    return syms


def _apply_env_overrides_to_cfg(cfg: Any) -> None:
    def _env_float(name: str) -> Optional[float]:
        try:
            v = os.getenv(name, "")
            if v is not None and str(v).strip() != "":
                return float(v)
        except Exception:
            return None
        return None

    for key in ("MIN_CONFIDENCE", "ENTRY_MIN_CONFIDENCE"):
        val = _env_float(key)
        if val is not None:
            try:
                setattr(cfg, key, val)
            except Exception:
                pass


def _apply_env_symbols_to_bot(bot: Any) -> list[str]:
    """
    Also expose as bot.active_symbols (entry_loop/data_loop prefer this if present).
    """
    raw = os.getenv("ACTIVE_SYMBOLS", "").strip()
    syms = _parse_env_symbols(raw)
    if not syms:
        return []
    try:
        bot.active_symbols = set(syms)
    except Exception:
        pass
    return syms


def _env_any(keys: list[str], default: str = "") -> str:
    """
    Return first non-empty environment variable among keys.
    """
    for k in keys:
        v = os.getenv(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


# ----------------------------
# Config load (defensive)
# ----------------------------

def _load_cfg() -> Any:
    for modname in ("config.settings", "config", "settings"):
        m = _opt_import(modname)
        if not m:
            continue
        if hasattr(m, "CFG"):
            return getattr(m, "CFG")
        if hasattr(m, "cfg"):
            return getattr(m, "cfg")
        # If module exposes Config class, instantiate it
        try:
            cfg_cls = getattr(m, "Config", None)
            if callable(cfg_cls):
                return cfg_cls()
        except Exception:
            pass
        # If module doesn't provide config, try next
        continue
    return SimpleNamespace()


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    try:
        return getattr(cfg, name, default)
    except Exception:
        return default


# ----------------------------
# Exchange init (ccxt async)
# ----------------------------

async def _init_exchange(cfg: Any) -> Any:
    try:
        import ccxt.async_support as ccxt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ccxt.async_support missing: {e}")

    ex_name = str(_cfg(cfg, "EXCHANGE", os.getenv("EXCHANGE", "binance"))).strip()
    ex_cls = getattr(ccxt, ex_name, None)
    if ex_cls is None:
        raise RuntimeError(f"Unknown ccxt exchange: {ex_name}")

    # ✅ IMPORTANT: Support BOTH your framework keys and the common .env keys you tested
    api_key = (
        str(_cfg(cfg, "API_KEY", "")).strip()
        or _env_any(["BINANCE_API_KEY", "API_KEY"])
    )
    api_secret = (
        str(_cfg(cfg, "API_SECRET", "")).strip()
        or _env_any(["BINANCE_API_SECRET", "API_SECRET"])
    )
    api_password = (
        str(_cfg(cfg, "API_PASSWORD", "")).strip()
        or _env_any(["BINANCE_API_PASSWORD", "API_PASSWORD"])
    )

    params: dict[str, Any] = {
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_password,
        "enableRateLimit": True,
        "options": {},
    }

    default_type = str(_cfg(cfg, "DEFAULT_TYPE", os.getenv("DEFAULT_TYPE", "future"))).strip()
    if default_type:
        params["options"]["defaultType"] = default_type  # "spot" or "future"

    headers = _cfg(cfg, "EXCHANGE_HEADERS", None)
    if isinstance(headers, dict) and headers:
        params["headers"] = headers

    exchange = ex_cls(params)

    # quick visibility
    try:
        log_core.info(
            f"[bootstrap] exchange={ex_name} defaultType={params['options'].get('defaultType')} "
            f"apiKey={'YES' if bool(api_key) else 'NO'} secret={'YES' if bool(api_secret) else 'NO'}"
        )
    except Exception:
        pass

    try:
        await exchange.load_markets()
    except Exception as e:
        log_core.warning(f"[bootstrap] load_markets failed (continuing): {e}")

    return exchange


# ----------------------------
# State/Brain init (best effort)
# ----------------------------

def _ensure_state_shape(st: Any) -> Any:
    if asyncio.iscoroutine(st) or inspect.isawaitable(st):
        return st

    if st is None:
        st = SimpleNamespace()

    try:
        _ = st.__dict__  # type: ignore[attr-defined]
    except Exception:
        st = SimpleNamespace(_wrapped=repr(st)[:500])

    try:
        if not hasattr(st, "positions") or not isinstance(getattr(st, "positions", None), dict):
            st.positions = {}
    except Exception:
        pass

    for name, default in (
        ("blacklist", {}),
        ("blacklist_reason", {}),
        ("last_exit_time", {}),
        ("consecutive_losses", {}),
        ("known_exit_order_ids", set()),
        ("current_equity", 0.0),
        ("start_of_day_equity", 0.0),
        ("daily_pnl", 0.0),
        ("win_streak", 0),
        ("total_trades", 0),
        ("total_wins", 0),
        ("session_peak_equity", 0.0),
        ("SYMBOL_COOLDOWN_MINUTES", 12),
    ):
        try:
            if not hasattr(st, name):
                setattr(st, name, default)
            else:
                cur = getattr(st, name)
                if isinstance(default, dict) and not isinstance(cur, dict):
                    setattr(st, name, {})
                if isinstance(default, set) and not isinstance(cur, set):
                    setattr(st, name, set())
        except Exception:
            pass

    try:
        if not hasattr(st, "symbol_performance") or not isinstance(getattr(st, "symbol_performance", None), dict):
            st.symbol_performance = {}
    except Exception:
        pass

    try:
        if not hasattr(st, "run_context") or not isinstance(getattr(st, "run_context", None), dict):
            st.run_context = {}
    except Exception:
        pass

    try:
        if not hasattr(st, "halt"):
            st.halt = False
    except Exception:
        pass

    for k, v in (
        ("shutdown_reason", ""),
        ("shutdown_source", ""),
        ("shutdown_ts", 0.0),
    ):
        try:
            if not hasattr(st, k):
                setattr(st, k, v)
        except Exception:
            pass

    return st


async def _maybe_call_persistence(fn: Callable, st_seed: Any) -> Any:
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    out = None

    if sig is not None:
        params = list(sig.parameters.values())
        required = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        ]
        if len(required) >= 1:
            out = fn(st_seed)
        else:
            out = fn()
    else:
        try:
            out = fn()
        except TypeError:
            out = fn(st_seed)

    if asyncio.iscoroutine(out) or inspect.isawaitable(out):
        out = await out

    return out


async def _load_state_best_effort() -> Any:
    st_seed = _ensure_state_shape(SimpleNamespace())

    p = _opt_import("brain.persistence")
    if not p:
        return _ensure_state_shape(st_seed)

    for fn_name in ("load_brain", "load_state"):
        fn = _callable(p, fn_name)
        if not fn:
            continue
        try:
            loaded = await _maybe_call_persistence(fn, st_seed)
            if loaded is not None:
                return _ensure_state_shape(loaded)
        except Exception as e:
            log_core.warning(f"[bootstrap] {fn_name} failed: {e}")

    return _ensure_state_shape(st_seed)


async def _save_state_best_effort(state: Any) -> None:
    p = _opt_import("brain.persistence")
    if not p:
        return

    for fn_name in ("save_brain", "save_state"):
        fn = _callable(p, fn_name)
        if not fn:
            continue
        try:
            out = None
            try:
                out = fn(state)
            except TypeError:
                out = fn()
            if asyncio.iscoroutine(out) or inspect.isawaitable(out):
                await out
            return
        except Exception as e:
            log_core.warning(f"[bootstrap] {fn_name} failed: {e}")


# ----------------------------
# Diagnostics banner (best effort)
# ----------------------------

def _run_diagnostics_best_effort(bot: Any) -> None:
    d = _opt_import("execution.diagnostics")
    if not d:
        return

    for fn_name in ("diagnostics_banner", "print_banner"):
        fn = _callable(d, fn_name)
        if fn:
            try:
                fn()
                return
            except Exception as e:
                log_core.warning(f"[bootstrap] {fn_name} failed: {e}")

    fn = _callable(d, "diagnostics_check")
    if fn:
        try:
            fn(bot)
        except Exception as e:
            log_core.warning(f"[bootstrap] diagnostics_check failed: {e}")


# ----------------------------
# Optional component builders
# ----------------------------

def _build_data_best_effort(bot: Any) -> None:
    for modname, clsname in (
        ("data.cache", "DataCache"),
        ("data.datacache", "DataCache"),
        ("data.data_cache", "DataCache"),
        ("execution.data_cache", "DataCache"),
        ("execution.datacache", "DataCache"),
    ):
        m = _opt_import(modname)
        if not m:
            continue
        C = getattr(m, clsname, None)
        if C:
            try:
                bot.data = C(bot)
                return
            except Exception as e:
                log_core.warning(f"[bootstrap] {modname}.{clsname} init failed: {e}")
                return


def _build_notify_best_effort(bot: Any) -> None:
    for modname, clsname in (
        ("utils.notify", "Notify"),
        ("utils.notifications", "Notify"),
        ("execution.notify", "Notify"),
    ):
        m = _opt_import(modname)
        if not m:
            continue
        C = getattr(m, clsname, None)
        if C:
            try:
                bot.notify = C(bot)
                return
            except Exception as e:
                log_core.warning(f"[bootstrap] {modname}.{clsname} init failed: {e}")
                return


# ----------------------------
# Data validation (robust)
# ----------------------------

def _data_is_valid(obj: Any) -> bool:
    """
    Accept multiple DataCache “shapes”, because different versions exist.

    Valid if ANY of these:
      - get_df(sym, tf) callable
      - get_price(sym, ...) callable
      - has dict-like .ohlcv and/or .price
    """
    if obj is None:
        return False

    try:
        if callable(getattr(obj, "get_df", None)):
            return True
    except Exception:
        pass

    try:
        if callable(getattr(obj, "get_price", None)):
            return True
    except Exception:
        pass

    try:
        ohlcv = getattr(obj, "ohlcv", None)
        price = getattr(obj, "price", None)
        if isinstance(ohlcv, dict) or isinstance(price, dict):
            return True
    except Exception:
        pass

    return False


def _data_has_any_market_data(obj: Any) -> bool:
    """
    Stronger "ready" condition: data structure exists AND has at least one price or ohlcv row.
    """
    if obj is None:
        return False
    try:
        price = getattr(obj, "price", None)
        if isinstance(price, dict) and any((float(v) > 0) for v in price.values() if isinstance(v, (int, float))):
            return True
    except Exception:
        pass
    try:
        ohlcv = getattr(obj, "ohlcv", None)
        if isinstance(ohlcv, dict):
            for v in ohlcv.values():
                if isinstance(v, list) and len(v) >= 5:
                    return True
    except Exception:
        pass
    return False


def _validate_data_or_null(bot: Any) -> None:
    d = getattr(bot, "data", None)
    if d is None:
        return
    if _data_is_valid(d):
        return
    log_core.warning(f"[bootstrap] bot.data invalid (type={type(d).__name__}) -> forcing None")
    try:
        bot.data = None
    except Exception:
        pass


async def _wait_for_data_ready(bot: Any, timeout_sec: float = 8.0, poll: float = 0.10) -> bool:
    """
    Gate entry loop until data exists & has at least one market datapoint.
    Prefers bot.data_ready Event if present.
    """
    t0 = time.time()

    ev = getattr(bot, "data_ready", None)
    if isinstance(ev, asyncio.Event):
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout_sec)
            return True
        except Exception:
            pass  # fall back to polling below

    while (time.time() - t0) < timeout_sec:
        _validate_data_or_null(bot)
        d = getattr(bot, "data", None)
        if _data_is_valid(d) and _data_has_any_market_data(d):
            return True
        await asyncio.sleep(poll)

    d = getattr(bot, "data", None)
    return _data_is_valid(d) and _data_has_any_market_data(d)


# ----------------------------
# Task runner helpers
# ----------------------------

async def _safe_task(name: str, coro: Awaitable[None]) -> None:
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log_core.error(f"[bootstrap] task crashed: {name} ({e})")


def _maybe_start_task(tasks: List[asyncio.Task], name: str, fn: Optional[Callable], *args) -> None:
    if not callable(fn):
        return
    try:
        t = asyncio.create_task(_safe_task(name, fn(*args)))
        tasks.append(t)
        log_core.info(f"[bootstrap] started: {name}")
    except Exception as e:
        log_core.warning(f"[bootstrap] failed to start {name}: {e}")


# ----------------------------
# Entry loop gating wrapper
# ----------------------------

async def _gated_entry_loop(bot: Any, entry_fn: Callable[..., Awaitable[None]]) -> None:
    timeout = float(
        _cfg(
            getattr(bot, "cfg", None),
            "BOOT_DATA_READY_TIMEOUT_SEC",
            float(os.getenv("BOOT_DATA_READY_TIMEOUT_SEC", "8")),
        )
    )
    ok = await _wait_for_data_ready(bot, timeout_sec=timeout)
    log_core.info(
        f"[bootstrap] data_ready={ok} bot.data={type(getattr(bot,'data',None)).__name__} "
        f"has_market_data={_data_has_any_market_data(getattr(bot,'data',None))}"
    )
    await entry_fn(bot)


async def _run_maintenance_oneshot(bot: Any) -> None:
    """Run a single reconcile maintenance tick and return."""
    reconcile_mod = _opt_import("execution.reconcile")
    reconcile_tick = _callable(reconcile_mod, "reconcile_tick") if reconcile_mod else None
    if not callable(reconcile_tick):
        log_core.warning("[bootstrap] maintenance oneshot: reconcile_tick unavailable")
        return
    try:
        await reconcile_tick(bot)
        log_core.info("[bootstrap] maintenance oneshot: reconcile_tick complete")
    except Exception as e:
        log_core.warning(f"[bootstrap] maintenance oneshot reconcile failed: {e}")


# ----------------------------
# Signal handlers (installed inside running loop)
# ----------------------------

def _install_signal_handlers_running_loop(loop: asyncio.AbstractEventLoop) -> None:
    def _stop():
        try:
            for task in asyncio.all_tasks(loop):
                task.cancel()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except Exception:
            # Windows / some environments won’t support this.
            pass


# ----------------------------
# Main
# ----------------------------

async def main() -> None:
    # install handlers inside the live loop (best effort)
    try:
        _install_signal_handlers_running_loop(asyncio.get_running_loop())
    except Exception:
        pass

    cfg = _load_cfg()
    _apply_env_overrides_to_cfg(cfg)

    # ENV -> cfg bridge BEFORE bot is built
    env_syms_cfg = _apply_env_symbols_to_cfg(cfg)
    if env_syms_cfg:
        log_core.info(f"[bootstrap] ENV ACTIVE_SYMBOLS -> cfg.ACTIVE_SYMBOLS ({len(env_syms_cfg)}): {env_syms_cfg[:12]}")

    bot = SimpleNamespace()
    bot.cfg = cfg
    bot.state = await _load_state_best_effort()

    # common coordination events
    bot._shutdown = asyncio.Event()
    bot.data_ready = asyncio.Event()

    bot.ex = await _init_exchange(cfg)
    bot.exchange = bot.ex

    bot.data = getattr(bot, "data", None)
    bot.notify = getattr(bot, "notify", None)

    factory_mod = _opt_import("execution.bot_factory") or _opt_import("bot_factory")
    if factory_mod:
        fn = _callable(factory_mod, "build_bot")
        if fn:
            try:
                maybe_bot = fn(bot)
                if asyncio.iscoroutine(maybe_bot) or inspect.isawaitable(maybe_bot):
                    maybe_bot = await maybe_bot
                if maybe_bot is not None:
                    bot = maybe_bot
            except Exception as e:
                log_core.warning(f"[bootstrap] bot_factory.build_bot failed: {e}")

    # re-assert critical fields (factory might replace bot)
    bot.cfg = getattr(bot, "cfg", cfg)
    bot.ex = getattr(bot, "ex", bot.ex)
    bot.exchange = getattr(bot, "exchange", bot.ex)
    bot.state = _ensure_state_shape(getattr(bot, "state", bot.state))

    # ensure events still exist even if factory swapped bot
    if not isinstance(getattr(bot, "_shutdown", None), asyncio.Event):
        bot._shutdown = asyncio.Event()
    if not isinstance(getattr(bot, "data_ready", None), asyncio.Event):
        bot.data_ready = asyncio.Event()

    # ENV -> bot bridge AFTER factory (so it can't wipe it)
    env_syms_bot = _apply_env_symbols_to_bot(bot)
    if env_syms_bot:
        log_core.info(f"[bootstrap] ENV ACTIVE_SYMBOLS -> bot.active_symbols ({len(env_syms_bot)}): {env_syms_bot[:12]}")

    # validate + build data/notify
    _validate_data_or_null(bot)
    if getattr(bot, "data", None) is None:
        _build_data_best_effort(bot)
    _validate_data_or_null(bot)

    if getattr(bot, "notify", None) is None:
        _build_notify_best_effort(bot)

    # Optional: bootstrap raw_symbol map from exchange markets
    try:
        bm = getattr(getattr(bot, "data", None), "bootstrap_markets", None)
        if callable(bm):
            await bm(bot)
    except Exception as e:
        log_core.warning(f"[bootstrap] data.bootstrap_markets failed: {e}")

    # Boot diagnostics
    log_core.info(
        f"[bootstrap] bot.data={type(getattr(bot,'data',None)).__name__} data_valid={_data_is_valid(getattr(bot,'data',None))}"
    )

    # Optional startup rebuild: reconstruct local execution state from exchange state.
    try:
        do_rebuild = os.getenv("BOOT_REBUILD_ON_START", "1").strip().lower() not in ("0", "false", "no", "off")
        if do_rebuild:
            rebuild_mod = _opt_import("execution.rebuild")
            rebuild_fn = _callable(rebuild_mod, "rebuild_local_state") if rebuild_mod else None
            if callable(rebuild_fn):
                freeze_on_orphans = os.getenv("BOOT_REBUILD_FREEZE_ON_ORPHANS", "0").strip().lower() in ("1", "true", "yes", "on")
                fill_window_sec = float(os.getenv("BOOT_REBUILD_FILL_WINDOW_SEC", "3600") or 3600.0)
                summary = await rebuild_fn(
                    bot,
                    fill_window_sec=fill_window_sec,
                    adopt_orphans=True,
                    freeze_on_orphans=freeze_on_orphans,
                )
                if isinstance(summary, dict):
                    log_core.info(
                        "[bootstrap] rebuild positions=%s prev=%s orphans=%s halted=%s"
                        % (
                            int(summary.get("positions_rebuilt", 0) or 0),
                            int(summary.get("positions_prev", 0) or 0),
                            int(summary.get("orphans", 0) or 0),
                            bool(summary.get("halted", False)),
                        )
                    )
    except Exception as e:
        log_core.warning(f"[bootstrap] rebuild failed: {e}")

    try:
        syms = getattr(bot, "active_symbols", None) or getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(syms, (list, tuple, set)):
            syms_list = list(syms)
            log_core.info(f"[bootstrap] symbols={syms_list[:12]}{'...' if len(syms_list) > 12 else ''}")
    except Exception:
        pass

    _run_diagnostics_best_effort(bot)

    maintenance_oneshot = os.getenv("BOOT_MAINTENANCE_ONESHOT", "0").strip().lower() in ("1", "true", "yes", "on")
    if maintenance_oneshot:
        log_core.warning("[bootstrap] BOOT_MAINTENANCE_ONESHOT=1 -> rebuild + single reconcile tick, then exit")
        try:
            await _run_maintenance_oneshot(bot)
        finally:
            try:
                await _save_state_best_effort(bot.state)
            except Exception:
                pass
            try:
                ex = getattr(bot, "ex", None)
                if ex is not None and hasattr(ex, "close"):
                    await ex.close()
            except Exception:
                pass
        log_core.critical("[bootstrap] OFFLINE (maintenance oneshot)")
        return

    # ------------------------------------------------------------
    # Required loops
    # ------------------------------------------------------------
    guardian_mod = _opt_import("execution.guardian")
    guardian_loop = _callable(guardian_mod, "guardian_loop") if guardian_mod else None
    if not callable(guardian_loop):
        raise RuntimeError("execution.guardian.guardian_loop is required but missing")

    # Entry loop selection (authoritative):
    # execution.entry_loop.entry_loop is the only supported runtime orchestrator.
    entry_mode = os.getenv("ENTRY_LOOP_MODE", "").strip().lower()
    if entry_mode in ("full", "risk", "advanced", "basic", "simple", "lite"):
        log_core.warning(
            f"[bootstrap] ENTRY_LOOP_MODE={entry_mode} is deprecated; forcing execution.entry_loop.entry_loop"
        )
    # Guard legacy direct try_enter paths at runtime unless explicitly re-enabled.
    if os.getenv("ENTRY_ENABLE_LEGACY_TRY_ENTER", "").strip() == "":
        os.environ["ENTRY_ENABLE_LEGACY_TRY_ENTER"] = "0"
    entry_loop_mod = _opt_import("execution.entry_loop")
    entry_loop = _callable(entry_loop_mod, "entry_loop") if entry_loop_mod else None
    if entry_loop:
        log_core.info("[bootstrap] entry loop: AUTHORITATIVE (execution.entry_loop.entry_loop)")
    else:
        log_core.warning("[bootstrap] entry loop: NONE (execution.entry_loop.entry_loop missing)")

    data_loop_mod = _opt_import("execution.data_loop")
    data_loop = _callable(data_loop_mod, "data_loop") if data_loop_mod else None

    # ------------------------------------------------------------
    # Optional loops (best effort)
    # ------------------------------------------------------------
    pm_mod = _opt_import("execution.position_manager")
    position_manager_loop = None
    if pm_mod:
        position_manager_loop = (
            _callable(pm_mod, "position_manager_loop")
            or _callable(pm_mod, "position_manager")
            or _callable(pm_mod, "run")
        )

    exit_mod = _opt_import("execution.exit")
    exit_loop = None
    if exit_mod:
        exit_loop = (
            _callable(exit_mod, "exit_loop")
            or _callable(exit_mod, "exit")
            or _callable(exit_mod, "run")
        )

    _opt_import("execution.telemetry")

    tasks: List[asyncio.Task] = []

    # Start guardian + data first
    _maybe_start_task(tasks, "guardian.guardian_loop", guardian_loop, bot)
    _maybe_start_task(tasks, "data_loop.data_loop", data_loop, bot)

    # Start protection loops BEFORE entry (so management is online first)
    if callable(position_manager_loop):
        _maybe_start_task(tasks, "position_manager.loop", position_manager_loop, bot)
    else:
        log_core.warning("[bootstrap] optional loop missing: execution.position_manager.(position_manager_loop/run)")
        if os.getenv("BOOT_REQUIRE_POSMGR", "1").strip().lower() not in ("0", "false", "no", "off"):
            raise RuntimeError("bootstrap requires execution.position_manager (set BOOT_REQUIRE_POSMGR=0 to bypass)")

    if callable(exit_loop):
        _maybe_start_task(tasks, "exit.loop", exit_loop, bot)
    else:
        log_core.warning("[bootstrap] optional loop missing: execution.exit.(exit_loop/run)")
        if os.getenv("BOOT_REQUIRE_EXIT", "1").strip().lower() not in ("0", "false", "no", "off"):
            raise RuntimeError("bootstrap requires execution.exit (set BOOT_REQUIRE_EXIT=0 to bypass)")

    # Start entry with gating wrapper
    if callable(entry_loop):
        try:
            t = asyncio.create_task(_safe_task("entry_loop.entry_loop", _gated_entry_loop(bot, entry_loop)))
            tasks.append(t)
            log_core.info("[bootstrap] started: entry_loop.entry_loop (gated)")
        except Exception as e:
            log_core.warning(f"[bootstrap] failed to start entry_loop.entry_loop (gated): {e}")

    log_core.critical(f"[bootstrap] ONLINE | tasks={len(tasks)} | exchange={type(bot.ex).__name__}")

    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc:
                raise exc
    finally:
        # cooperative shutdown flag
        try:
            ev = getattr(bot, "_shutdown", None)
            if isinstance(ev, asyncio.Event):
                ev.set()
        except Exception:
            pass

        # cancel + drain tasks
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # persist best effort
        try:
            await _save_state_best_effort(bot.state)
        except Exception:
            pass

        # close exchange best effort
        try:
            ex = getattr(bot, "ex", None)
            if ex is not None and hasattr(ex, "close"):
                await ex.close()
        except Exception:
            pass

        log_core.critical("[bootstrap] OFFLINE")


if __name__ == "__main__":
    # Windows sometimes needs Selector policy for subprocess/signal weirdness
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except Exception:
            pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ctrl+C on Windows often lands here; keep it clean.
        pass
