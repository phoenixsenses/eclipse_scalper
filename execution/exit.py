# execution/exit.py — SCALPER ETERNAL — COSMIC REAPER ASCENDANT — 2026 v5.0d
# (FULL FIX: _get_min_amount defined + hedge_side_hint for ALL exit orders)
#
# Patch vs v5.0c:
# - ✅ FIX: Defines _get_min_amount() (NameError resolved).
# - ✅ HARDEN: All exit-side create_order() calls include hedge_side_hint=pos_side when hedge mode is ON
#             (router v2.3 requires hedge_side_hint for exits in hedge mode).
# - ✅ KEEP: env fallback + bool parse fix + bootstrap loop adapters.

import time
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


from utils.logging import log_entry, log
from brain.persistence import save_brain

from execution.order_router import create_order, cancel_order  # ✅ ROUTER
from execution.position_lock import atomic_position_remove  # ✅ ATOMIC LOCK
from ta.volatility import AverageTrueRange


try:
    from execution.telemetry import emit_throttled, emit  # type: ignore
except Exception:
    emit_throttled = None
    emit = None

try:
    from execution.error_codes import map_reason, ERR_UNKNOWN, EXIT_MOM, EXIT_VWAP, EXIT_TIME, EXIT_STAGNATION  # type: ignore
except Exception:
    map_reason = None
    ERR_UNKNOWN = 'ERR_UNKNOWN'
    EXIT_MOM = 'EXIT_MOM'
    EXIT_VWAP = 'EXIT_VWAP'
    EXIT_TIME = 'EXIT_TIME'
    EXIT_STAGNATION = 'EXIT_STAGNATION'

# Enhanced exit strategy (optional)
try:
    from strategies.exit_strategy import (
        ExitStrategy,
        ExitPlan,
        ExitSignal,
        ExitType,
        TrailMode,
        get_exit_strategy,
    )
    EXIT_STRATEGY_AVAILABLE = True
except Exception:
    EXIT_STRATEGY_AVAILABLE = False
    get_exit_strategy = None

# Enhanced exit plan cache (per position)
_EXIT_PLANS: dict[str, ExitPlan] = {}

_EXIT_LOCKS: dict[str, asyncio.Lock] = {}


_TELEMETRY_ACTIONS_PATH = Path(os.getenv("TELEMETRY_ANOMALY_ACTIONS", "logs/telemetry_anomaly_actions.json"))
_TELEMETRY_ACTIONS_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}}
_TELEMETRY_CACHE_TTL = 5.0
_GUARD_HISTORY_ACTIONS_PATH = Path(
    os.getenv("TELEMETRY_GUARD_HISTORY_ACTIONS", "logs/telemetry_guard_history_actions.json")
)
_GUARD_HISTORY_ACTIONS_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}}
_GUARD_HISTORY_CACHE_TTL = 5.0
_SIGNAL_FEEDBACK_PATH = Path(os.getenv("EXIT_SIGNAL_FEEDBACK_PATH", "logs/signal_exit_feedback.json"))
_SIGNAL_FEEDBACK_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}}
_SIGNAL_FEEDBACK_CACHE_TTL = 5.0


def _get_exit_lock(k: str) -> asyncio.Lock:
    lock = _EXIT_LOCKS.get(k)
    if lock is None:
        lock = asyncio.Lock()
        _EXIT_LOCKS[k] = lock
    return lock


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _load_telemetry_actions() -> dict[str, Any]:
    now = time.time()
    cache = _TELEMETRY_ACTIONS_CACHE
    if now - float(cache.get("ts", 0.0) or 0.0) < _TELEMETRY_CACHE_TTL:
        return cache.get("data") or {}
    data: dict[str, Any] = {}
    try:
        text = _TELEMETRY_ACTIONS_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception:
        data = {}
    cache["ts"] = now
    cache["data"] = data
    return data


def _load_guard_history_actions() -> dict[str, Any]:
    now = time.time()
    cache = _GUARD_HISTORY_ACTIONS_CACHE
    if now - float(cache.get("ts", 0.0) or 0.0) < _GUARD_HISTORY_CACHE_TTL:
        return cache.get("data") or {}
    data: dict[str, Any] = {}
    try:
        text = _GUARD_HISTORY_ACTIONS_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception:
        data = {}
    cache["ts"] = now
    cache["data"] = data
    return data


def _load_signal_feedback() -> dict[str, Any]:
    now = time.time()
    cache = _SIGNAL_FEEDBACK_CACHE
    if now - float(cache.get("ts", 0.0) or 0.0) < _SIGNAL_FEEDBACK_CACHE_TTL:
        return cache.get("data") or {}
    data: dict[str, Any] = {}
    try:
        text = _SIGNAL_FEEDBACK_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception:
        data = {}
    cache["ts"] = now
    cache["data"] = data
    return data


def _get_entry_signal_meta(bot, k: str) -> dict[str, Any] | None:
    rc = getattr(bot.state, "run_context", None)
    if not isinstance(rc, dict):
        return None
    signals = rc.get("last_entry_signal")
    if not isinstance(signals, dict):
        return None
    meta = signals.get(k)
    if not isinstance(meta, dict):
        return None
    return meta


def _exit_signal_data(bot, k: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    meta = _get_entry_signal_meta(bot, k)
    if meta:
        confidence = meta.get("confidence")
        ts = float(meta.get("ts") or 0.0)
        if confidence is not None:
            try:
                data["entry_confidence"] = float(confidence)
            except Exception:
                pass
        if ts > 0:
            data["entry_signal_age_sec"] = round(max(0.0, time.time() - ts), 2)
        side = str(meta.get("side") or "").lower().strip()
        if side:
            data["entry_signal_side"] = side
        for key, value in meta.items():
            if not isinstance(key, str):
                continue
            if not key.startswith("entry_"):
                continue
            if key in data:
                continue
            data[key] = value
    if extra:
        data.update(extra)
    return data


def _schedule_exit_event(
    bot,
    event: str,
    *,
    symbol: Optional[str] = None,
    reason: str,
    data: Optional[Dict[str, Any]] = None,
    code: Optional[str] = None,
    level: str = "info",
    throttle_sec: float = 0.0,
    key: Optional[str] = None,
) -> None:
    if not callable(emit) and not callable(emit_throttled):
        return
    payload: Dict[str, Any] = dict(data or {})
    if symbol:
        payload.setdefault("symbol", symbol)
    if reason:
        payload.setdefault("reason", reason)
    if "code" not in payload:
        try:
            payload["code"] = code or (map_reason(reason) if callable(map_reason) else ERR_UNKNOWN)
        except Exception:
            payload["code"] = ERR_UNKNOWN

    try:
        if throttle_sec > 0 and callable(emit_throttled):
            asyncio.create_task(
                emit_throttled(
                    bot,
                    event,
                    key=key or f"{symbol or 'exit'}:{reason}",
                    cooldown_sec=throttle_sec,
                    data=payload,
                    symbol=symbol,
                    level=level,
                )
            )
        elif callable(emit):
            asyncio.create_task(
                emit(bot, event, data=payload if payload else {}, symbol=symbol, level=level)
            )
    except Exception:
        pass


def _emit_position_closed(bot, symbol: str, payload: dict[str, Any]) -> None:
    if not callable(emit):
        return
    try:
        asyncio.create_task(emit(bot, "position.closed", data=payload, symbol=symbol, level="info"))
    except Exception:
        pass


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


async def _safe_speak(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _truthy(x) -> bool:
    # string "0" must be False
    if x is True:
        return True
    if x is False or x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "t", "on")
    return False


def _cfg_env(bot, name: str, default):
    """
    Prefer bot.cfg.NAME, fallback to env var NAME, else default.
    """
    try:
        cfg = getattr(bot, "cfg", None)
        if cfg is not None and hasattr(cfg, name):
            v = getattr(cfg, name)
            if v is not None:
                return v
    except Exception:
        pass

    try:
        ev = os.getenv(name, None)
        if ev is not None and str(ev).strip() != "":
            return ev
    except Exception:
        pass

    return default


def _env_float_sym(name: str, default: float, symkey: str) -> float:
    try:
        sk = _symkey(symkey)
        base = sk[:-4] if sk.endswith("USDT") and len(sk) > 4 else sk
        for suffix in (base, sk):
            ev = os.getenv(f"{name}_{suffix}", None)
            if ev is None or str(ev).strip() == "":
                continue
            return float(ev)
    except Exception:
        pass
    return float(default)


def _is_reduce_only(order: dict) -> bool:
    """
    React only to reduce-only or closePosition intent.
    """
    try:
        if _truthy(order.get("reduceOnly")):
            return True

        info = order.get("info") or {}
        if _truthy(info.get("reduceOnly")):
            return True
        if _truthy(info.get("closePosition")):
            return True

        params = order.get("params") or {}
        if _truthy(params.get("reduceOnly")):
            return True
        if _truthy(params.get("closePosition")):
            return True
    except Exception:
        pass
    return False


def _get_realized_pnl(order: dict) -> float:
    info = order.get("info") or {}
    for k in ("realizedPnl", "realizedProfit", "rp", "pnl"):
        if k in info:
            return _safe_float(info.get(k), 0.0)
    if "pnl" in order:
        return _safe_float(order.get("pnl"), 0.0)
    return 0.0


def _get_filled(order: dict) -> float:
    filled = _safe_float(order.get("filled"), 0.0)
    if filled > 0:
        return filled
    info = order.get("info") or {}
    return _safe_float(info.get("executedQty"), 0.0)


def _get_exit_price(order: dict, fallback: float = 0.0) -> float:
    try:
        avg = _safe_float(order.get("average"), 0.0)
        if avg > 0:
            return avg
        px = _safe_float(order.get("price"), 0.0)
        if px > 0:
            return px
        info = order.get("info") or {}
        avg2 = _safe_float(info.get("avgPrice"), 0.0)
        if avg2 > 0:
            return avg2
        fill_px = _safe_float(info.get("fillPrice"), 0.0)
        if fill_px > 0:
            return fill_px
    except Exception:
        return float(fallback or 0.0)
    return float(fallback or 0.0)


def _ensure_sym_perf(state, k: str) -> dict:
    sp = getattr(state, "symbol_performance", None)
    if not isinstance(sp, dict):
        try:
            state.symbol_performance = {}
        except Exception:
            pass
        sp = getattr(state, "symbol_performance", None)

    if not isinstance(sp, dict):
        return {}

    perf = sp.get(k)
    if not isinstance(perf, dict):
        perf = {"pnl": 0.0, "wins": 0, "losses": 0, "last_win": 0.0}
        sp[k] = perf

    perf.setdefault("pos_realized_pnl", 0.0)
    perf.setdefault("entry_size_abs", 0.0)
    perf.setdefault("mfe_pct", 0.0)
    perf.setdefault("trailing_order_ids", [])
    perf.setdefault("last_trail_ts", 0.0)  # debounce
    return perf


def _ensure_known_exit_ids(state):
    try:
        s = getattr(state, "known_exit_order_ids", None)
        if not isinstance(s, set):
            state.known_exit_order_ids = set()
    except Exception:
        try:
            state.known_exit_order_ids = set()
        except Exception:
            pass


async def _cancel_order_safe(bot, order_id: Optional[str], symbol_any: str):
    if not order_id:
        return
    try:
        await cancel_order(bot, order_id, symbol_any)
    except Exception:
        _schedule_exit_event(
            bot,
            "exit.blocked",
            symbol=_symkey(symbol_any),
            reason="cancel_failed",
            level="warning",
            throttle_sec=15.0,
            key=f"{_symkey(symbol_any)}:cancel",
        )


def _get_current_price(bot, k: str, sym_any: str) -> float:
    """
    sym_any may be canonical or raw; we just use it to check price caches.
    """
    try:
        data = getattr(bot, "data", None)
        if data is None:
            return 0.0

        gp = getattr(data, "get_price", None)
        if callable(gp):
            px = _safe_float(gp(k, in_position=True), 0.0)
            if px > 0:
                return px

        price_map = getattr(data, "price", {}) or {}
        if not isinstance(price_map, dict):
            return 0.0

        for key in (k, sym_any, _symkey(sym_any)):
            if key in price_map:
                px = _safe_float(price_map.get(key), 0.0)
                if px > 0:
                    return px

        target = k
        for kk, vv in price_map.items():
            if _symkey(kk) == target:
                px = _safe_float(vv, 0.0)
                if px > 0:
                    return px
    except Exception:
        pass
    return 0.0


def _resolve_raw_symbol(bot, k: str, fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return fallback


def _get_df_any(bot, k: str, sym_any: str, tf: str):
    try:
        data = getattr(bot, "data", None)
        if data is None:
            return None
        fn = getattr(data, "get_df", None)
        if not callable(fn):
            return None
        df = fn(k, tf)
        if df is not None:
            return df
        df = fn(sym_any, tf)
        if df is not None:
            return df
        raw_map = getattr(data, "raw_symbol", {}) or {}
        if isinstance(raw_map, dict):
            raw = raw_map.get(k) or raw_map.get(_symkey(sym_any))
            if raw:
                return fn(raw, tf)
    except Exception:
        return None
    return None


def _ha_momentum(df) -> float:
    try:
        if df is None or len(df) < 3:
            return 0.0
        ha_c = (df["o"] + df["h"] + df["l"] + df["c"]) / 4.0
        mom = ha_c.pct_change(2).iloc[-1]
        return _safe_float(mom, 0.0)
    except Exception:
        return 0.0


def _get_or_create_exit_plan(
    k: str,
    entry_price: float,
    direction: str,
    stop_loss: float,
    atr: float,
) -> Optional["ExitPlan"]:
    """Get existing exit plan or create a new one."""
    if not EXIT_STRATEGY_AVAILABLE or not callable(get_exit_strategy):
        return None

    try:
        # Check for existing plan
        if k in _EXIT_PLANS:
            plan = _EXIT_PLANS[k]
            # Validate plan is still for same position
            if abs(plan.entry_price - entry_price) < entry_price * 0.001:
                return plan
            # Different position, create new plan
            del _EXIT_PLANS[k]

        # Create new plan
        exit_mgr = get_exit_strategy()
        plan = exit_mgr.create_plan(
            symbol=k,
            entry_price=entry_price,
            direction=direction.upper(),
            stop_loss=stop_loss,
            atr=atr,
        )
        _EXIT_PLANS[k] = plan
        return plan

    except Exception as e:
        log.warning(f"[exit] Failed to create exit plan for {k}: {e}")
        return None


def _check_enhanced_exit(
    bot,
    k: str,
    sym_any: str,
    current_price: float,
    plan: "ExitPlan",
) -> Optional["ExitSignal"]:
    """Check for enhanced exit signals."""
    if not EXIT_STRATEGY_AVAILABLE or not callable(get_exit_strategy):
        return None

    try:
        exit_mgr = get_exit_strategy()

        # Get 1m DataFrame for momentum analysis
        df = _get_df_any(bot, k, sym_any, "1m")

        signal = exit_mgr.check_exit(
            plan=plan,
            current_price=current_price,
            df=df,
        )

        return signal

    except Exception as e:
        log.warning(f"[exit] Enhanced exit check failed for {k}: {e}")
        return None


def _clear_exit_plan(k: str) -> None:
    """Clear exit plan when position is closed."""
    if k in _EXIT_PLANS:
        del _EXIT_PLANS[k]


def _momentum_exit_signal(
    bot,
    k: str,
    sym_any: str,
    side: str,
    *,
    min_mom: float,
    require_both: bool,
    tf_fast: str = "5m",
    tf_slow: str = "15m",
):
    df_fast = _get_df_any(bot, k, sym_any, tf_fast)
    df_slow = _get_df_any(bot, k, sym_any, tf_slow)
    if df_fast is None or df_slow is None:
        return False, 0.0, 0.0
    mom_fast = _ha_momentum(df_fast)
    mom_slow = _ha_momentum(df_slow)
    if str(side).lower().strip() == "long":
        hit_fast = mom_fast <= -abs(min_mom)
        hit_slow = mom_slow <= -abs(min_mom)
    else:
        hit_fast = mom_fast >= abs(min_mom)
        hit_slow = mom_slow >= abs(min_mom)
    if require_both:
        return (hit_fast and hit_slow), mom_fast, mom_slow
    return (hit_fast or hit_slow), mom_fast, mom_slow


def _vwap_cross_exit_signal(
    bot,
    k: str,
    sym_any: str,
    side: str,
    *,
    tf: str,
    window: int,
    require_cross: bool,
):
    df = _get_df_any(bot, k, sym_any, tf)
    if df is None or len(df) < max(10, int(window)):
        return False, 0.0, 0.0
    try:
        tp = (df["h"] + df["l"] + df["c"]) / 3.0
        v = df["v"]
        vwap_s = (tp * v).rolling(window).sum() / v.rolling(window).sum()
        vwap = _safe_float(vwap_s.iloc[-1], 0.0)
        vwap_prev = _safe_float(vwap_s.iloc[-2], 0.0)
        c = _safe_float(df["c"].iloc[-1], 0.0)
        c_prev = _safe_float(df["c"].iloc[-2], 0.0)
        if vwap <= 0 or vwap_prev <= 0:
            return False, vwap, c
        if str(side).lower().strip() == "long":
            if require_cross:
                hit = (c_prev >= vwap_prev) and (c < vwap)
            else:
                hit = c < vwap
        else:
            if require_cross:
                hit = (c_prev <= vwap_prev) and (c > vwap)
            else:
                hit = c > vwap
        return bool(hit), vwap, c
    except Exception:
        return False, 0.0, 0.0


# ----------------------------
# Hedge helpers (best-effort)
# ----------------------------

def _hedge_enabled(bot) -> bool:
    ex = getattr(bot, "ex", None)
    if ex is None:
        return True
    for attr in ("hedge_mode", "position_mode", "hedgeMode"):
        try:
            v = getattr(ex, attr, None)
            if isinstance(v, bool):
                return v
        except Exception:
            pass
    return True


def _position_side_for_trade(side: str) -> str:
    s = str(side).lower().strip()
    if s == "long":
        return "LONG"
    if s == "short":
        return "SHORT"
    return ""


# ----------------------------
# Markets / min amount  ✅ (_get_min_amount FIX)
# ----------------------------

async def _ensure_markets_loaded(bot) -> None:
    ex = getattr(bot, "ex", None)
    if ex is None:
        return

    # wrapper path
    try:
        inner = getattr(ex, "exchange", None)
        mk = getattr(inner, "markets", None) if inner is not None else None
        if isinstance(mk, dict) and len(mk) > 0:
            return
    except Exception:
        pass

    # plain path
    try:
        mk = getattr(ex, "markets", None)
        if isinstance(mk, dict) and len(mk) > 0:
            return
    except Exception:
        pass

    try:
        fm = getattr(ex, "fetch_markets", None)
        if callable(fm):
            await fm()
            return
    except Exception:
        pass

    try:
        lm = getattr(ex, "load_markets", None)
        if callable(lm):
            await lm()
            return
    except Exception:
        pass


async def _get_min_amount(bot, sym_any: str, k: str) -> float:
    """
    Returns min order amount for symbol (best-effort).
    - tries exchange markets limits first (wrapper or plain)
    - falls back to bot.min_amounts[k]
    """
    # 1) exchange market limits
    try:
        await _ensure_markets_loaded(bot)
        ex = getattr(bot, "ex", None)
        if ex is None:
            raise RuntimeError("no exchange")

        markets = None
        inner = getattr(ex, "exchange", None)
        if inner is not None:
            markets = getattr(inner, "markets", None)
        if not isinstance(markets, dict):
            markets = getattr(ex, "markets", None)

        if isinstance(markets, dict) and markets:
            keys_to_try = [sym_any]
            # if wrapper has resolver, try resolved raw too
            try:
                resolver = getattr(ex, "_resolve_symbol", None)
                if callable(resolver):
                    rr = resolver(sym_any)
                    if rr and rr not in keys_to_try:
                        keys_to_try.append(rr)
            except Exception:
                pass

            for raw_key in keys_to_try:
                if raw_key in markets:
                    lim = (markets[raw_key] or {}).get("limits") or {}
                    amt = (lim.get("amount") or {}).get("min")
                    m = _safe_float(amt, 0.0)
                    if m > 0:
                        return m
    except Exception:
        pass

    # 2) bot min map (canonical)
    try:
        m2 = _safe_float(getattr(bot, "min_amounts", {}).get(k, 0.0), 0.0)
        if m2 > 0:
            return m2
    except Exception:
        pass

    return 0.0


# ----------------------------
# Stop / trailing ladders (router-safe)
# ----------------------------

async def _place_stop_ladder(
    bot,
    *,
    sym_any: str,
    side: str,
    qty: float,
    stop_price: float,
    k: str,
    pos_side: Optional[str] = None,
):
    stop_side = "sell" if side == "long" else "buy"

    base_params = {}
    if pos_side:
        base_params["positionSide"] = pos_side

    # A) amount + reduceOnly (normal best path)
    try:
        return await create_order(
            bot,
            symbol=sym_any,
            type="STOP_MARKET",
            side=stop_side,
            amount=float(qty),
            price=None,
            params=dict(base_params),
            intent_reduce_only=True,
            intent_close_position=False,
            stop_price=float(stop_price),
            hedge_side_hint=pos_side,  # ✅ router hedge exit requirement
            retries=6,
        )
    except Exception as e1:
        log_entry.warning(f"BE stop A failed {k}: {e1}")

    # B) closePosition=True — router expects amount=None and NO reduceOnly
    try:
        p = dict(base_params)
        p["closePosition"] = True
        return await create_order(
            bot,
            symbol=sym_any,
            type="STOP_MARKET",
            side=stop_side,
            amount=None,                 # ✅ omit amount for closePosition path
            price=None,
            params=p,
            intent_reduce_only=False,     # ✅ do NOT request reduceOnly (prevents -1106)
            intent_close_position=True,
            stop_price=float(stop_price),
            hedge_side_hint=pos_side,     # ✅ router hedge exit requirement
            retries=6,
        )
    except Exception as e2:
        log_entry.error(f"BE stop B failed {k}: {e2}")

    return None


async def _place_trailing_ladder(
    bot,
    *,
    sym_any: str,
    side: str,
    qty: float,
    activation_price: float,
    callback_rate: float,
    k: str,
    pos_side: Optional[str] = None,
):
    close_side = "sell" if side == "long" else "buy"
    cb = _clamp(float(callback_rate), 0.1, 5.0)

    base_params = {}
    if pos_side:
        base_params["positionSide"] = pos_side

    # A) activation + callback + reduceOnly
    try:
        return await create_order(
            bot,
            symbol=sym_any,
            type="TRAILING_STOP_MARKET",
            side=close_side,
            amount=float(qty),
            price=None,
            params=dict(base_params),
            intent_reduce_only=True,
            activation_price=float(activation_price),
            callback_rate=float(cb),
            hedge_side_hint=pos_side,  # ✅
            retries=6,
        )
    except Exception as e1:
        log_entry.warning(f"Trailing A failed {k}: {e1}")

    # B) drop callbackRate
    try:
        return await create_order(
            bot,
            symbol=sym_any,
            type="TRAILING_STOP_MARKET",
            side=close_side,
            amount=float(qty),
            price=None,
            params=dict(base_params),
            intent_reduce_only=True,
            activation_price=float(activation_price),
            hedge_side_hint=pos_side,  # ✅
            retries=6,
        )
    except Exception as e2:
        log_entry.error(f"Trailing B failed {k}: {e2}")

    return None


# ----------------------------
# Exit handler (called by your fill/updates pipeline)
# ----------------------------

async def handle_exit(bot, order: dict):
    if not isinstance(order, dict):
        return

    oid = order.get("id")
    if not oid:
        return

    sym_any_in = order.get("symbol")
    if not sym_any_in:
        return

    # k must be canonical because state is canonical
    k = _symkey(sym_any_in)
    sym_any = sym_any_in  # router can handle canonical/raw

    lock = _get_exit_lock(k)
    async with lock:
        _ensure_known_exit_ids(bot.state)
        if oid in bot.state.known_exit_order_ids:
            return
        bot.state.known_exit_order_ids.add(oid)

        try:
            pos = (bot.state.positions or {}).get(k)
        except Exception:
            pos = None

        # Phantom exit
        if not pos:
            reduced = _get_filled(order)
            if reduced <= 0:
                return
            pnl = _get_realized_pnl(order)
            if abs(pnl) >= 1.0:
                await _safe_speak(bot, f"PHANTOM EXIT {k} | PnL ${pnl:+,.0f}", "critical")
            return

        # Allow opposite-side exits when reduceOnly flags are missing
        if not _is_reduce_only(order):
            try:
                pos_side = str(getattr(pos, "side", "") or "").lower().strip()
                order_side = str(order.get("side") or (order.get("info") or {}).get("side") or "").lower().strip()
                if not (
                    (pos_side == "long" and order_side == "sell")
                    or (pos_side == "short" and order_side == "buy")
                ):
                    return
            except Exception:
                return

        reduced = _get_filled(order)
        if reduced <= 0:
            return

        pnl = _get_realized_pnl(order)

        cfg = bot.cfg
        perf = _ensure_sym_perf(bot.state, k)

        # Hedge side hint (ONLY if hedge mode)
        pos_side = _position_side_for_trade(getattr(pos, "side", "")) if _hedge_enabled(bot) else ""
        if not pos_side:
            pos_side = None

        # Track entry size once (absolute)
        if _safe_float(perf.get("entry_size_abs", 0.0), 0.0) <= 0.0:
            perf["entry_size_abs"] = abs(_safe_float(getattr(pos, "size", 0.0), 0.0))

        perf["pos_realized_pnl"] = float(perf.get("pos_realized_pnl", 0.0)) + float(pnl)

        # size treated as absolute remaining
        remaining_before = abs(_safe_float(getattr(pos, "size", 0.0), 0.0))
        remaining_after = max(0.0, remaining_before - float(reduced))
        pos.size = remaining_after

        min_amount = await _get_min_amount(bot, sym_any, k)
        is_full_close = (remaining_after <= 0.0) or (min_amount > 0.0 and remaining_after < min_amount)
        if is_full_close:
            pos.size = 0.0

        duration_seconds = time.time() - float(getattr(pos, "entry_ts", time.time()))
        current_price = _get_current_price(bot, k, sym_any)
        entry_px = _safe_float(getattr(pos, "entry_price", 0.0), 0.0)

        # MFE (best effort)
        if current_price and entry_px > 0:
            if pos.side == "long":
                favorable = (current_price - entry_px) / entry_px
            else:
                favorable = (entry_px - current_price) / entry_px
            perf["mfe_pct"] = max(float(perf.get("mfe_pct", 0.0)), favorable * 100.0)

        # Partial notify
        if pnl and not is_full_close:
            log_entry.info(
                f"PARTIAL EXIT {pos.side.upper()} {k} | PnL ${pnl:+,.0f} | Remaining {remaining_after:.6f}"
            )
            await _safe_speak(
                bot,
                f"PARTIAL EXIT {pos.side.upper()} {k} | PnL ${pnl:+,.0f} | Remaining {remaining_after:.6f}",
                "info",
            )

        # Breakeven move after <= 50% remaining
        if not is_full_close:
            entry_abs = float(perf.get("entry_size_abs", 0.0) or 0.0)
            if entry_abs > 0:
                remaining_pct = remaining_after / entry_abs
                if remaining_pct <= 0.5 and not getattr(pos, "breakeven_moved", False):
                    try:
                        atr = float(getattr(pos, "atr", 0.0) or 0.0)
                        buffer = atr * float(cfg.BREAKEVEN_BUFFER_ATR_MULT)
                        be_price = entry_px + buffer if pos.side == "long" else entry_px - buffer
                        if be_price <= 0:
                            raise RuntimeError("invalid BE price")

                        if getattr(pos, "hard_stop_order_id", None):
                            await _cancel_order_safe(bot, pos.hard_stop_order_id, sym_any)

                        new_stop = await _place_stop_ladder(
                            bot,
                            sym_any=sym_any,
                            side=pos.side,
                            qty=remaining_after,
                            stop_price=float(be_price),
                            k=k,
                            pos_side=pos_side,
                        )
                        if new_stop and isinstance(new_stop, dict) and new_stop.get("id"):
                            pos.hard_stop_order_id = new_stop.get("id")
                            pos.breakeven_moved = True
                            await _safe_speak(bot, f"BREAKEVEN ASCENDED {k}", "critical")
                            _schedule_exit_event(
                                bot,
                                "exit.success",
                                symbol=k,
                                reason="breakeven",
                                level="info",
                                throttle_sec=10.0,
                                key=f"{k}:breakeven",
                            )
                        else:
                            log_entry.error(f"Breakeven stop placement failed (all fallbacks) {k}")
                    except Exception as e:
                        log_entry.error(f"Breakeven move failed {k}: {e}")
                        _schedule_exit_event(
                            bot,
                            "exit.blocked",
                            symbol=k,
                            reason="breakeven_fail",
                            data={"err": repr(e)[:200]},
                            level="warning",
                            throttle_sec=30.0,
                            key=f"{k}:breakeven_fail",
                        )

        # Velocity exit (full flatten if rapid drawdown early)
        if not is_full_close and current_price and entry_px > 0:
            if pos.side == "long":
                unrealized_pct = (current_price - entry_px) / entry_px
            else:
                unrealized_pct = (entry_px - current_price) / entry_px

            if (
                unrealized_pct < -float(cfg.VELOCITY_DRAWDOWN_PCT)
                and duration_seconds < float(cfg.VELOCITY_MINUTES) * 60.0
            ):
                try:
                    params = {}
                    if pos_side:
                        params["positionSide"] = pos_side

                    await create_order(
                        bot,
                        symbol=sym_any,
                        type="MARKET",
                        side="sell" if pos.side == "long" else "buy",
                        amount=float(remaining_after),
                        price=None,
                        params=params,
                        intent_reduce_only=True,
                        hedge_side_hint=pos_side,  # ✅
                        retries=6,
                    )
                    log_entry.critical(f"VELOCITY EXIT {k} — rapid drawdown detected")
                    await _safe_speak(bot, f"VELOCITY FORCED EXIT {k}", "critical")
                    _schedule_exit_event(
                        bot,
                        "exit.success",
                        symbol=k,
                        reason="velocity",
                        level="info",
                        throttle_sec=10.0,
                        key=f"{k}:velocity",
                    )
                except Exception as e:
                    log_entry.error(f"Velocity exit failed {k}: {e}")
                    _schedule_exit_event(
                        bot,
                        "exit.blocked",
                        symbol=k,
                        reason="velocity_exit_failed",
                        data={"err": repr(e)[:200]},
                        level="warning",
                        throttle_sec=30.0,
                        key=f"{k}:velocity_fail",
                    )

        # Full close
        if is_full_close:
            full_pnl = float(perf.get("pos_realized_pnl", 0.0))
            exit_ts = time.time()
            entry_qty = float(perf.get("entry_size_abs", 0.0) or 0.0)
            exit_qty = float(remaining_before or 0.0)
            exit_price = _get_exit_price(order, current_price)
            entry_notional = float(entry_px) * float(entry_qty) if entry_px > 0 and entry_qty > 0 else 0.0
            exit_notional = float(exit_price) * float(exit_qty) if exit_price > 0 and exit_qty > 0 else 0.0
            pnl_pct = (full_pnl / entry_notional) if entry_notional > 0 else 0.0
            exit_reason = str(order.get("reason") or (order.get("info") or {}).get("reason") or "order_fill")
            exit_type = str(order.get("type") or (order.get("info") or {}).get("type") or "")
            exit_side = str(order.get("side") or (order.get("info") or {}).get("side") or "")

            if getattr(pos, "hard_stop_order_id", None):
                await _cancel_order_safe(bot, pos.hard_stop_order_id, sym_any)

            trailing_ids = list(perf.get("trailing_order_ids", []) or [])
            for tid in trailing_ids:
                await _cancel_order_safe(bot, tid, sym_any)
            perf["trailing_order_ids"] = []

            try:
                if full_pnl > 0:
                    bot.state.total_wins += 1
                    bot.state.win_streak += 1
                    bot.state.consecutive_losses[k] = 0

                    perf["pnl"] = float(perf.get("pnl", 0.0)) + full_pnl
                    perf["wins"] = int(perf.get("wins", 0)) + 1
                    perf["last_win"] = time.time()

                    if cfg.BLACKLIST_AUTO_RESET_ON_PROFIT and k in bot.state.blacklist:
                        try:
                            del bot.state.blacklist[k]
                            await _safe_speak(bot, f"{k} BLACKLIST LIFTED — profit redemption", "critical")
                        except Exception:
                            pass
                else:
                    bot.state.win_streak = 0
                    bot.state.consecutive_losses[k] = bot.state.consecutive_losses.get(k, 0) + 1
                    loss_streak = bot.state.consecutive_losses[k]

                    perf["pnl"] = float(perf.get("pnl", 0.0)) + full_pnl
                    perf["losses"] = int(perf.get("losses", 0)) + 1

                    if loss_streak >= cfg.CONSECUTIVE_LOSS_BLACKLIST_COUNT:
                        blacklist_until = time.time() + float(cfg.SYMBOL_BLACKLIST_DURATION_HOURS) * 3600.0
                        bot.state.blacklist[k] = blacklist_until
                        await _safe_speak(
                            bot,
                            f"{k} BLACKLISTED — {cfg.SYMBOL_BLACKLIST_DURATION_HOURS}h exile",
                            "critical",
                        )

                bot.state.last_exit_time[k] = time.time()
                bot.state.win_rate = (
                    bot.state.total_wins / bot.state.total_trades
                    if bot.state.total_trades > 0
                    else 0.0
                )
            except Exception:
                pass

            mfe_pct = float(perf.get("mfe_pct", 0.0) or 0.0)
            _emit_position_closed(
                bot,
                k,
                _exit_signal_data(
                    bot,
                    k,
                    extra={
                        "entry_ts": float(getattr(pos, "entry_ts", 0.0) or 0.0),
                        "exit_ts": exit_ts,
                        "duration_sec": float(duration_seconds),
                        "entry_price": float(entry_px),
                        "exit_price": float(exit_price),
                        "entry_qty": float(entry_qty),
                        "exit_qty": float(exit_qty),
                        "entry_notional": float(entry_notional),
                        "exit_notional": float(exit_notional),
                        "pnl_usdt": float(full_pnl),
                        "pnl_pct": float(pnl_pct),
                        "mfe_pct": float(mfe_pct),
                        "exit_reason": exit_reason,
                        "exit_order_id": str(oid),
                        "exit_side": exit_side,
                        "exit_type": exit_type,
                    },
                ),
            )
            log_entry.info(
                f"FINAL EXIT {pos.side.upper()} {k} | Total PnL ${full_pnl:+,.0f} | "
                f"Duration {time.strftime('%Hh%Mm%Ss', time.gmtime(duration_seconds))} | "
                f"MFE {mfe_pct:+.1f}%"
            )
            await _safe_speak(
                bot,
                f"FINAL EXIT {pos.side.upper()} {k} | Total PnL ${full_pnl:+,.0f}\n"
                f"Duration: {time.strftime('%Hh%Mm%Ss', time.gmtime(duration_seconds))}\n"
                f"MFE: {mfe_pct:+.1f}% | Win Rate: {getattr(bot.state, 'win_rate', 0.0):.1%}\n"
                f"Equity: ${getattr(bot.state, 'current_equity', 0):,.0f}\n"
                f"ASCENDANT REAPER COMPLETE",
                "critical",
            )

            # Atomic position remove (prevents race with reconcile/position_manager)
            await atomic_position_remove(bot, k)

            perf["pos_realized_pnl"] = 0.0
            perf["entry_size_abs"] = 0.0
            perf["mfe_pct"] = 0.0
            perf["trailing_order_ids"] = []
            perf["last_trail_ts"] = 0.0

            try:
                await save_brain(bot.state)
            except Exception as pe:
                log_entry.warning(f"Brain save failed: {pe}")

            return

        # Trailing management while still open (debounced)
        if float(getattr(cfg, "TRAILING_ACTIVATION_RR", 0.0) or 0.0) <= 0:
            return

        now_ts = time.time()
        if now_ts - float(perf.get("last_trail_ts", 0.0) or 0.0) < float(
            getattr(cfg, "TRAILING_REBUILD_DEBOUNCE_SEC", 15.0) or 15.0
        ):
            return
        perf["last_trail_ts"] = now_ts

        remaining = remaining_after
        min_amount = await _get_min_amount(bot, sym_any, k)
        if remaining <= 0.0 or (min_amount > 0.0 and remaining < min_amount):
            return

        trailing_ids = list(perf.get("trailing_order_ids", []) or [])
        for tid in trailing_ids:
            await _cancel_order_safe(bot, tid, sym_any)
        perf["trailing_order_ids"] = []

        try:
            atr = float(getattr(pos, "atr", 0.0) or 0.0)
            stop_pct = (atr * float(cfg.STOP_ATR_MULT)) / entry_px if entry_px > 0 else 0.0
        except Exception:
            stop_pct = 0.0

        activation_pct = stop_pct * float(cfg.TRAILING_ACTIVATION_RR)
        if activation_pct <= 0 or entry_px <= 0:
            return

        activation_price = (
            entry_px * (1.0 + activation_pct)
            if pos.side == "long"
            else entry_px * (1.0 - activation_pct)
        )

        cb_main = _clamp(float(cfg.TRAILING_CALLBACK_RATE), 0.1, 5.0)
        cb_tight = _clamp(float(cfg.TRAILING_TIGHT_PCT), 0.1, 5.0)
        cb_loose = _clamp(float(cfg.TRAILING_LOOSE_PCT), 0.1, 5.0)

        try:
            if cfg.DUAL_TRAILING:
                tight_amount = remaining * 0.5
                loose_amount = remaining - tight_amount

                if tight_amount > 0 and (min_amount <= 0 or tight_amount >= min_amount):
                    tight_order = await _place_trailing_ladder(
                        bot,
                        sym_any=sym_any,
                        side=pos.side,
                        qty=float(tight_amount),
                        activation_price=float(activation_price),
                        callback_rate=float(cb_tight),
                        k=k,
                        pos_side=pos_side,
                    )
                    if tight_order and isinstance(tight_order, dict) and tight_order.get("id"):
                        perf["trailing_order_ids"].append(tight_order["id"])

                if loose_amount > 0 and (min_amount <= 0 or loose_amount >= min_amount):
                    loose_order = await _place_trailing_ladder(
                        bot,
                        sym_any=sym_any,
                        side=pos.side,
                        qty=float(loose_amount),
                        activation_price=float(activation_price),
                        callback_rate=float(cb_loose),
                        k=k,
                        pos_side=pos_side,
                    )
                    if loose_order and isinstance(loose_order, dict) and loose_order.get("id"):
                        perf["trailing_order_ids"].append(loose_order["id"])
            else:
                trail_order = await _place_trailing_ladder(
                    bot,
                    sym_any=sym_any,
                    side=pos.side,
                    qty=float(remaining),
                    activation_price=float(activation_price),
                    callback_rate=float(cb_main),
                    k=k,
                    pos_side=pos_side,
                )
                if trail_order and isinstance(trail_order, dict) and trail_order.get("id"):
                    perf["trailing_order_ids"].append(trail_order["id"])

        except Exception as e:
                        log_entry.error(f"Trailing placement failed {k}: {e}")
                        _schedule_exit_event(
                            bot,
                            "exit.blocked",
                            symbol=k,
                            reason="trailing_failed",
                            data={"err": repr(e)[:200]},
                            level="warning",
                            throttle_sec=30.0,
                            key=f"{k}:trailing_fail",
                        )


# ----------------------------
# BOOTSTRAP LOOP ADAPTERS
# ----------------------------

async def exit_loop(bot) -> None:
    """
    Bootstrap entrypoint. Lightweight maintenance loop.
    handle_exit() remains event-driven; this loop is for housekeeping.
    """
    if not _truthy(_cfg_env(bot, "EXIT_ENABLED", True)):
        log.info("EXIT LOOP disabled (EXIT_ENABLED=0)")
        return

    tick_sec = _safe_float(_cfg_env(bot, "EXIT_TICK_SEC", 2.0), 2.0)
    tick_sec = max(0.5, float(tick_sec))
    max_hold_sec = _safe_float(_cfg_env(bot, "EXIT_MAX_HOLD_SEC", 0.0), 0.0)
    exit_cooldown = _safe_float(_cfg_env(bot, "EXIT_TIME_COOLDOWN_SEC", 10.0), 10.0)
    stagnation_sec = _safe_float(_cfg_env(bot, "EXIT_STAGNATION_SEC", 0.0), 0.0)
    stagnation_atr = _safe_float(_cfg_env(bot, "EXIT_STAGNATION_ATR", 0.15), 0.15)
    exit_mom_enabled = _truthy(_cfg_env(bot, "EXIT_MOM_ENABLED", False))
    exit_mom_min = _safe_float(_cfg_env(bot, "EXIT_MOM_MIN", 0.0015), 0.0015)
    exit_mom_require_both = _truthy(_cfg_env(bot, "EXIT_MOM_REQUIRE_BOTH", True))
    exit_vwap_enabled = _truthy(_cfg_env(bot, "EXIT_VWAP_ENABLED", False))
    exit_vwap_tf = str(_cfg_env(bot, "EXIT_VWAP_TF", "5m") or "5m").strip().lower()
    exit_vwap_window = int(_safe_float(_cfg_env(bot, "EXIT_VWAP_WINDOW", 240), 240))
    exit_vwap_require_cross = _truthy(_cfg_env(bot, "EXIT_VWAP_REQUIRE_CROSS", True))
    exit_atr_scale_enabled = _truthy(_cfg_env(bot, "EXIT_ATR_SCALE_ENABLED", False))
    exit_atr_scale_ref = _safe_float(_cfg_env(bot, "EXIT_ATR_SCALE_REF_PCT", 0.003), 0.003)
    exit_atr_scale_min = _safe_float(_cfg_env(bot, "EXIT_ATR_SCALE_MIN", 0.6), 0.6)
    exit_atr_scale_max = _safe_float(_cfg_env(bot, "EXIT_ATR_SCALE_MAX", 1.6), 1.6)
    exit_telemetry_high_exposure = _safe_float(_cfg_env(bot, "EXIT_TELEMETRY_HIGH_EXPOSURE_USDT", 0.0), 0.0)
    exit_telemetry_force_hold = _safe_float(_cfg_env(bot, "EXIT_TELEMETRY_FORCE_HOLD_SEC", 0.0), 0.0)
    exit_telemetry_cooldown_mult = _safe_float(_cfg_env(bot, "EXIT_TELEMETRY_COOLDOWN_MULT", 0.5), 0.5)
    exit_telemetry_alert_interval = _safe_float(
        _cfg_env(bot, "EXIT_TELEMETRY_ALERT_INTERVAL_SEC", 300), 300
    )
    exit_guard_history_hold_scale = _safe_float(_cfg_env(bot, "EXIT_GUARD_HISTORY_HOLD_SCALE", 0.7), 0.7)
    exit_guard_history_stagnation_scale = _safe_float(_cfg_env(bot, "EXIT_GUARD_HISTORY_STAGNATION_SCALE", 0.7), 0.7)
    exit_signal_feedback_ratio = _safe_float(_cfg_env(bot, "EXIT_SIGNAL_FEEDBACK_MIN_RATIO", 0.25), 0.25)
    exit_signal_feedback_count = int(_safe_float(_cfg_env(bot, "EXIT_SIGNAL_FEEDBACK_MIN_COUNT", 3), 3))
    exit_signal_feedback_hold_scale = _safe_float(_cfg_env(bot, "EXIT_SIGNAL_FEEDBACK_HOLD_SCALE", 0.7), 0.7)
    exit_signal_feedback_stagnation_scale = _safe_float(_cfg_env(bot, "EXIT_SIGNAL_FEEDBACK_STAGNATION_SCALE", 0.7), 0.7)

    log.info(f"EXIT LOOP ONLINE — tick_sec={tick_sec}")

    while True:
        try:
            # Intentionally lightweight; handle_exit() is the real worker.
            st = getattr(bot, "state", None)
            pos_map = getattr(st, "positions", None) if st is not None else None
            if isinstance(pos_map, dict) and pos_map and max_hold_sec > 0:
                now = time.time()
                rc = getattr(st, "run_context", None)
                if not isinstance(rc, dict):
                    st.run_context = {}
                    rc = st.run_context
                last_map = rc.get("exit_time_last")
                if not isinstance(last_map, dict):
                    last_map = {}
                    rc["exit_time_last"] = last_map

                last_px_ts = rc.get("exit_stagnation_last_ts")
                if not isinstance(last_px_ts, dict):
                    last_px_ts = {}
                    rc["exit_stagnation_last_ts"] = last_px_ts

                last_mom_ts = rc.get("exit_momentum_last_ts")
                if not isinstance(last_mom_ts, dict):
                    last_mom_ts = {}
                    rc["exit_momentum_last_ts"] = last_mom_ts

                actions = _load_telemetry_actions()
                telemetry_exposures = float(actions.get("exposures") or 0.0)
                telemetry_guard_active = (
                    exit_telemetry_high_exposure > 0 and telemetry_exposures >= exit_telemetry_high_exposure
                )
                cooldown_current = exit_cooldown
                max_hold_current = max_hold_sec
                stagnation_current = stagnation_sec

                guard_actions = _load_guard_history_actions()
                guard_ts = float(guard_actions.get("ts") or 0.0)
                guard_window = max(
                    0.0,
                    float(guard_actions.get("confidence_duration") or 0.0),
                    float(guard_actions.get("leverage_duration") or 0.0),
                    float(guard_actions.get("notional_duration") or 0.0),
                )
                guard_active = guard_ts > 0 and guard_window > 0 and (now - guard_ts) <= guard_window
                if guard_active:
                    max_hold_current = max(0.0, max_hold_current * exit_guard_history_hold_scale)
                    stagnation_current = max(0.0, stagnation_current * exit_guard_history_stagnation_scale)
                    if callable(emit_throttled):
                        try:
                            asyncio.create_task(
                                emit_throttled(
                                    bot,
                                    "exit.guard_history_scaled",
                                    key="exit.guard_history_scaled",
                                    cooldown_sec=120.0,
                                    data={
                                        "hold_scale": exit_guard_history_hold_scale,
                                        "stagnation_scale": exit_guard_history_stagnation_scale,
                                        "window_sec": guard_window,
                                    },
                                    level="warning",
                                )
                            )
                        except Exception:
                            pass
                feedback = _load_signal_feedback()
                feedback_ratio = _safe_float(feedback.get("low_confidence_ratio"), 0.0)
                feedback_count = int(_safe_float(feedback.get("low_confidence_exits"), 0))
                feedback_triggered = (
                    feedback_ratio >= exit_signal_feedback_ratio and feedback_count >= exit_signal_feedback_count
                )
                if feedback_triggered:
                    max_hold_current = max(0.0, max_hold_current * exit_signal_feedback_hold_scale)
                    stagnation_current = max(0.0, stagnation_current * exit_signal_feedback_stagnation_scale)
                    if callable(emit_throttled):
                        try:
                            asyncio.create_task(
                                emit_throttled(
                                    bot,
                                    "exit.signal_feedback_scaled",
                                    key="exit.signal_feedback_scaled",
                                    cooldown_sec=120.0,
                                    data={
                                        "hold_scale": exit_signal_feedback_hold_scale,
                                        "stagnation_scale": exit_signal_feedback_stagnation_scale,
                                        "ratio": feedback_ratio,
                                        "count": feedback_count,
                                    },
                                    level="warning",
                                )
                            )
                        except Exception:
                            pass
                if telemetry_guard_active:
                    cooldown_current = max(0.5, exit_cooldown * exit_telemetry_cooldown_mult)
                    last_alert = float(rc.get("exit_telemetry_last_alert", 0.0) or 0.0)
                    if (now - last_alert) >= exit_telemetry_alert_interval:
                        rc["exit_telemetry_last_alert"] = now
                        _schedule_exit_event(
                            bot,
                            "exit.telemetry_guard",
                            reason="high_exposure",
                            data={"exposures": telemetry_exposures},
                            level="warning",
                            throttle_sec=exit_telemetry_alert_interval,
                            key="telemetry_guard",
                        )

                base_max_hold_current = max_hold_current
                base_stagnation_current = stagnation_current
                for k, pos in list(pos_map.items()):
                    try:
                        entry_ts = float(getattr(pos, "entry_ts", 0.0) or 0.0)
                        if entry_ts <= 0:
                            continue
                        age = now - entry_ts
                        max_hold_current = base_max_hold_current
                        stagnation_current = base_stagnation_current
                        max_hold_current = _env_float_sym("EXIT_MAX_HOLD_SEC", max_hold_current, k)
                        stagnation_current = _env_float_sym("EXIT_STAGNATION_SEC", stagnation_current, k)
                        stagnation_atr_sym = _env_float_sym("EXIT_STAGNATION_ATR", stagnation_atr, k)
                        sym_any = _resolve_raw_symbol(bot, k, k)
                        pos_side = _position_side_for_trade(getattr(pos, "side", "")) if _hedge_enabled(bot) else None
                        close_side = "sell" if str(getattr(pos, "side", "")).lower().strip() == "long" else "buy"
                        qty = abs(_safe_float(getattr(pos, "size", 0.0), 0.0))
                        if qty <= 0:
                            continue
                        exit_sent = False

                        if exit_atr_scale_enabled and (max_hold_current > 0 or stagnation_current > 0):
                            entry_px = _safe_float(getattr(pos, "entry_price", 0.0), 0.0)
                            entry_atr = _safe_float(getattr(pos, "atr", 0.0), 0.0)
                            if entry_px > 0 and entry_atr > 0 and exit_atr_scale_ref > 0:
                                atr_ref = _env_float_sym("EXIT_ATR_SCALE_REF_PCT", exit_atr_scale_ref, k)
                                atr_min = _env_float_sym("EXIT_ATR_SCALE_MIN", exit_atr_scale_min, k)
                                atr_max = _env_float_sym("EXIT_ATR_SCALE_MAX", exit_atr_scale_max, k)
                                atr_pct = entry_atr / entry_px
                                scale = atr_pct / max(1e-9, atr_ref)
                                scale = _clamp(scale, float(atr_min), float(atr_max))
                                if scale != 1.0:
                                    max_hold_current = max(0.0, max_hold_current * scale)
                                    stagnation_current = max(0.0, stagnation_current * scale)
                                    if callable(emit_throttled):
                                        try:
                                            asyncio.create_task(
                                                emit_throttled(
                                                    bot,
                                                    "exit.atr_scaled",
                                                    key=f"{k}:exit_atr_scaled",
                                                    cooldown_sec=120.0,
                                                    data={
                                                        "atr_pct": atr_pct,
                                                        "ref_pct": atr_ref,
                                                        "scale": scale,
                                                        "hold_sec": max_hold_current,
                                                        "stagnation_sec": stagnation_current,
                                                    },
                                                    symbol=k,
                                                    level="info",
                                                )
                                            )
                                        except Exception:
                                            pass

                        # Momentum fade exit (5m + 15m)
                        if exit_mom_enabled:
                            last_ts_mom = float(last_mom_ts.get(k, 0.0) or 0.0)
                            if (now - last_ts_mom) >= cooldown_current:
                                hit, mom5, mom15 = _momentum_exit_signal(
                                    bot,
                                    k,
                                    sym_any,
                                    getattr(pos, "side", ""),
                                    min_mom=exit_mom_min,
                                    require_both=exit_mom_require_both,
                                    tf_fast="5m",
                                    tf_slow="15m",
                                )
                                if hit:
                                    last_mom_ts[k] = now
                                    await create_order(
                                        bot,
                                        symbol=sym_any,
                                        type="MARKET",
                                        side=close_side,
                                        amount=float(qty),
                                        price=None,
                                        params=({"positionSide": pos_side} if pos_side else {}),
                                        intent_reduce_only=True,
                                        hedge_side_hint=pos_side,
                                        retries=4,
                                    )
                                    log_entry.info(
                                        f"MOMENTUM EXIT ??? {k} mom5m={mom5:+.4f} mom15m={mom15:+.4f} qty={qty:.6f}"
                                    )
                                    _schedule_exit_event(
                                        bot,
                                        "exit.success",
                                        symbol=k,
                                        reason="momentum",
                                        level="info",
                                        code=EXIT_MOM,
                                        throttle_sec=5.0,
                                        key=f"{k}:exit_mom",
                                        data=_exit_signal_data(bot, k, extra={"mom5": mom5, "mom15": mom15}),
                                    )
                                    exit_sent = True

                        # VWAP cross exit
                        if exit_vwap_enabled and not exit_sent:
                            last_ts_vwap = float(last_map.get(f"{k}:vwap", 0.0) or 0.0)
                            if (now - last_ts_vwap) >= cooldown_current:
                                hit, vwap, px = _vwap_cross_exit_signal(
                                    bot,
                                    k,
                                    sym_any,
                                    getattr(pos, "side", ""),
                                    tf=exit_vwap_tf,
                                    window=max(10, exit_vwap_window),
                                    require_cross=exit_vwap_require_cross,
                                )
                                if hit:
                                    last_map[f"{k}:vwap"] = now
                                    await create_order(
                                        bot,
                                        symbol=sym_any,
                                        type="MARKET",
                                        side=close_side,
                                        amount=float(qty),
                                        price=None,
                                        params=({"positionSide": pos_side} if pos_side else {}),
                                        intent_reduce_only=True,
                                        hedge_side_hint=pos_side,
                                        retries=4,
                                    )
                                    log_entry.info(
                                        f"VWAP EXIT ??? {k} px={px:.6f} vwap={vwap:.6f} tf={exit_vwap_tf} qty={qty:.6f}"
                                    )
                                    _schedule_exit_event(
                                        bot,
                                        "exit.success",
                                        symbol=k,
                                        reason="vwap_cross",
                                        level="info",
                                        code=EXIT_VWAP,
                                        throttle_sec=5.0,
                                        key=f"{k}:exit_vwap",
                                        data=_exit_signal_data(bot, k, extra={"vwap": vwap, "px": px}),
                                    )
                                    exit_sent = True

                        # Time exit
                        if (not exit_sent) and max_hold_current > 0 and age >= max_hold_current:
                            last_ts = float(last_map.get(k, 0.0) or 0.0)
                            if (now - last_ts) >= cooldown_current:
                                last_map[k] = now
                                if qty > 0:
                                    await create_order(
                                        bot,
                                        symbol=sym_any,
                                        type="MARKET",
                                        side=close_side,
                                        amount=float(qty),
                                        price=None,
                                        params=({"positionSide": pos_side} if pos_side else {}),
                                        intent_reduce_only=True,
                                        hedge_side_hint=pos_side,
                                        retries=4,
                                    )
                                    log_entry.info(f"TIME EXIT ??? {k} age={age:.0f}s qty={qty:.6f}")
                                    _schedule_exit_event(
                                        bot,
                                        "exit.success",
                                        symbol=k,
                                        reason="time",
                                        level="info",
                                        code=EXIT_TIME,
                                        throttle_sec=5.0,
                                        key=f"{k}:exit_time",
                                        data=_exit_signal_data(bot, k),
                                    )

                        # Stagnation exit (no progress vs entry)
                        if (not exit_sent) and stagnation_current > 0 and age >= stagnation_current:
                            last_ts = float(last_px_ts.get(k, 0.0) or 0.0)
                            if (now - last_ts) < max(1.0, stagnation_sec / 2.0):
                                continue
                            last_px_ts[k] = now

                            px = 0.0
                            try:
                                px = _safe_float(getattr(getattr(bot, "data", None), "price", {}).get(k), 0.0)
                            except Exception:
                                px = 0.0
                            if px <= 0:
                                continue

                            entry_px = _safe_float(getattr(pos, "entry_price", 0.0), 0.0)
                            if entry_px <= 0:
                                continue

                            atr = 0.0
                            try:
                                df = getattr(getattr(bot, "data", None), "get_df", None)
                                if callable(df):
                                    dfx = df(k, "1m")
                                    if dfx is not None and len(dfx) >= 20:
                                        atr = _safe_float(
                                            AverageTrueRange(dfx["h"], dfx["l"], dfx["c"], window=14).average_true_range().iloc[-1],
                                            0.0,
                                        )
                            except Exception:
                                atr = 0.0

                            if atr <= 0:
                                continue

                            move = abs(px - entry_px)
                            if move <= (atr * stagnation_atr_sym):
                                last_ts2 = float(last_map.get(k, 0.0) or 0.0)
                                if (now - last_ts2) < cooldown_current:
                                    continue
                                last_map[k] = now

                                if qty <= 0:
                                    continue

                                await create_order(
                                    bot,
                                    symbol=sym_any,
                                    type="MARKET",
                                    side=close_side,
                                    amount=float(qty),
                                    price=None,
                                    params=({"positionSide": pos_side} if pos_side else {}),
                                    intent_reduce_only=True,
                                    hedge_side_hint=pos_side,
                                    retries=4,
                                )
                                log_entry.info(
                                    f"STAGNATION EXIT ??? {k} age={age:.0f}s move={move:.6f} atr={atr:.6f} qty={qty:.6f}"
                                )
                                _schedule_exit_event(
                                    bot,
                                    "exit.success",
                                    symbol=k,
                                    reason="stagnation",
                                    level="info",
                                    code=EXIT_STAGNATION,
                                    throttle_sec=5.0,
                                    key=f"{k}:exit_stagnation",
                                    data=_exit_signal_data(bot, k, extra={"move": move, "atr": atr}),
                                )

                        # Telemetry guard forced exit
                        if (
                            not exit_sent
                            and telemetry_guard_active
                            and exit_telemetry_force_hold > 0
                            and age >= exit_telemetry_force_hold
                            and qty > 0
                        ):
                            try:
                                await create_order(
                                    bot,
                                    symbol=sym_any,
                                    type="MARKET",
                                    side=close_side,
                                    amount=float(qty),
                                    price=None,
                                    params=({"positionSide": pos_side} if pos_side else {}),
                                    intent_reduce_only=True,
                                    hedge_side_hint=pos_side,
                                    retries=4,
                                )
                                _schedule_exit_event(
                                    bot,
                                    "exit.success",
                                    symbol=k,
                                    reason="telemetry_force_close",
                                    level="critical",
                                    throttle_sec=5.0,
                                    key=f"{k}:exit_telemetry_force",
                                    data=_exit_signal_data(
                                        bot,
                                        k,
                                        extra={"guard": "telemetry", "exposures": telemetry_exposures},
                                    ),
                                )
                                exit_sent = True
                            except Exception as exc:
                                _schedule_exit_event(
                                    bot,
                                    "exit.blocked",
                                    symbol=k,
                                    reason="telemetry_force_close_failed",
                                    level="warning",
                                    throttle_sec=30.0,
                                    key=f"{k}:exit_telemetry_force_fail",
                                    data={"err": repr(exc)[:200]},
                                )
                    except Exception as e:
                        log_entry.error(f"TIME EXIT failed {k}: {e}")
                        _schedule_exit_event(
                            bot,
                            "exit.blocked",
                            symbol=k,
                            reason="exit_loop_symbol_error",
                            data={"err": repr(e)[:200]},
                            level="warning",
                            throttle_sec=30.0,
                            key=f"{k}:exit_loop",
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"exit_loop error: {e}")
            _schedule_exit_event(
                bot,
                "exit.blocked",
                reason="exit_loop_error",
                data={"err": repr(e)[:200]},
                level="warning",
                throttle_sec=30.0,
                key="exit_loop:error",
            )

        await asyncio.sleep(tick_sec)


async def run(bot) -> None:
    return await exit_loop(bot)
