# execution/entry_loop.py — SCALPER ETERNAL — ENTRY LOOP — 2026 v1.6 (PENDING-LOCK + COOLDOWN + OPEN-ORDERS ADOPT)
# Patch vs v1.5:
# - ✅ FIX: Per-symbol "pending entry" lock so you cannot machine-gun entries (even if reconcile is lagging)
# - ✅ FIX: Cooldown after ANY submitted entry attempt (success OR fail) to avoid rapid re-fire loops
# - ✅ HARDEN: Optional open-orders / open-position probe (best-effort) to detect real exposure even if brain-state is stale
# - ✅ SAFETY: backoff on margin-insufficient (-2019) retained
# - ✅ Keeps: ENV-first overrides, sizing resolver, tuple adapter, throttled logs, guardian-safe (never raises)

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional, Callable, Tuple

from utils.logging import log_entry, log_core
from execution.order_router import create_order

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit  # type: ignore
except Exception:
    emit = None

# Optional entry_watch (never fatal)
try:
    from execution.entry_watch import register_entry_watch  # type: ignore
except Exception:
    register_entry_watch = None

# Optional kill-switch gate (never fatal)
try:
    from risk.kill_switch import trade_allowed  # type: ignore
except Exception:
    trade_allowed = None


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


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "on"):
        return True
    return False


def _env_get(name: str) -> str:
    try:
        return str(os.getenv(name, "")).strip()
    except Exception:
        return ""


def _cfg_env_float(bot, name: str, default: float) -> float:
    """
    ENV wins, then cfg, then default.
    """
    v = _env_get(name)
    if v != "":
        try:
            return float(v)
        except Exception:
            pass
    try:
        return float(_cfg(bot, name, default) or default)
    except Exception:
        return float(default)


def _cfg_env_bool(bot, name: str, default: Any = False) -> bool:
    """
    ENV wins, then cfg.
    Accepts strings like "1/true/on".
    """
    v = _env_get(name)
    if v != "":
        return _truthy(v)
    return _truthy(_cfg(bot, name, default))


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


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


async def _safe_speak(bot, text: str, priority: str = "info") -> None:
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _pick_symbols(bot) -> list[str]:
    """
    Best-effort symbol universe:
    1) bot.active_symbols (set/list)
    2) cfg.ACTIVE_SYMBOLS (list)
    3) fallback ["BTCUSDT"]
    """
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


def _in_position_brain(bot, k: str) -> bool:
    """
    Cheap check: state.positions contains k with nonzero size.
    Note: if reconcile never adopts exchange positions, this may remain false.
    """
    try:
        st = getattr(bot, "state", None)
        posd = getattr(st, "positions", None)
        if not isinstance(posd, dict):
            return False
        p = posd.get(k)
        if p is None:
            return False
        sz = getattr(p, "size", 0.0)
        return abs(float(sz)) > 0.0
    except Exception:
        return False


def _resolve_raw_symbol(bot, k: str, fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return fallback


# ----------------------------
# ENV + CFG sizing resolver
# ----------------------------

def _env_float(*names: str, default: float = 0.0) -> float:
    for n in names:
        try:
            v = os.getenv(n, "")
            s = str(v).strip()
            if s != "":
                return float(s)
        except Exception:
            pass
    return float(default)


def _cfg_float(cfg_obj, *names: str, default: float = 0.0) -> float:
    for n in names:
        try:
            if cfg_obj is None:
                continue
            v = getattr(cfg_obj, n, None)
            if v is None:
                continue
            s = float(v)
            if s != 0.0:
                return s
        except Exception:
            pass
    return float(default)


def _resolve_sizing(bot) -> tuple[float, float]:
    """
    Returns (fixed_qty, fixed_notional_usdt)
    Priority:
      1) ENV (supports aliases)
      2) cfg (supports aliases)
    """
    cfg_obj = getattr(bot, "cfg", None)

    fixed_qty = _env_float("FIXED_QTY", "ORDER_QTY", "QTY", default=0.0)
    fixed_notional = _env_float(
        "FIXED_NOTIONAL_USDT",
        "ORDER_NOTIONAL_USDT",
        "FIXED_NOTIONAL",
        "NOTIONAL_USDT",
        "FIXED_USDT",
        "BASE_NOTIONAL_USDT",
        default=0.0,
    )

    if fixed_qty <= 0:
        fixed_qty = _cfg_float(cfg_obj, "FIXED_QTY", "ORDER_QTY", "QTY", default=0.0)

    if fixed_notional <= 0:
        fixed_notional = _cfg_float(
            cfg_obj,
            "FIXED_NOTIONAL_USDT",
            "ORDER_NOTIONAL_USDT",
            "FIXED_NOTIONAL",
            "NOTIONAL_USDT",
            "FIXED_USDT",
            "BASE_NOTIONAL_USDT",
            default=0.0,
        )

    return float(fixed_qty), float(fixed_notional)


# ----------------------------
# Throttled logging (no-trade spam killer)
# ----------------------------

_LAST_LOG_TS: Dict[str, float] = {}


def _throttled_log(key: str, every_sec: float, fn: Callable[[str], None], msg: str) -> None:
    """
    Log msg at most once per `every_sec` per key.
    `fn` is e.g. log_entry.info / log_entry.warning.
    """
    try:
        every_sec = float(every_sec)
        if every_sec <= 0:
            fn(msg)
            return
        now = _now()
        last = float(_LAST_LOG_TS.get(key, 0.0) or 0.0)
        if (now - last) >= every_sec:
            _LAST_LOG_TS[key] = now
            fn(msg)
    except Exception:
        pass


# ----------------------------
# Strategy signal adapter
# ----------------------------

def _load_signal_fn() -> Optional[Callable]:
    """
    Locate a signal function without hard dependency.
    Preferred: strategies.eclipse_scalper.scalper_signal
    """
    try:
        from strategies.eclipse_scalper import scalper_signal as fn  # type: ignore
        if callable(fn):
            return fn
    except Exception:
        pass
    return None


def _tuple_to_sig_dict(
    sym: str,
    out: Any,
    diag: bool = False,
    no_trade_log_every: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """
    Accepts tuple/list like (long_bool, short_bool, confidence)
    Returns a dict compatible with the rest of this entry loop.
    """
    try:
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            return None

        long_sig = bool(out[0])
        short_sig = bool(out[1])
        conf = float(out[2]) if out[2] is not None else 0.0

        if long_sig and not short_sig:
            return {"action": "buy", "confidence": conf, "type": "market", "symbol": sym}
        if short_sig and not long_sig:
            return {"action": "sell", "confidence": conf, "type": "market", "symbol": sym}

        if diag:
            _throttled_log(
                key=f"tuple_no_trade:{sym}",
                every_sec=no_trade_log_every,
                fn=log_entry.info,
                msg=f"ENTRY_LOOP tuple no-trade {sym}: long={long_sig} short={short_sig} conf={conf}",
            )
        return None
    except Exception:
        return None


async def _maybe_call_signal(fn: Callable, bot, symbol: str, diag: bool = False) -> Optional[Dict[str, Any]]:
    """
    Supports:
    - dict-returning strategies
    - tuple-returning scalper_signal() -> (long, short, confidence)
    Tries multiple signatures, including canonical: fn(sym, data=bot.data, cfg=bot.cfg)
    """
    no_trade_log_every = float(_cfg_env_float(bot, "ENTRY_NO_TRADE_LOG_EVERY_SEC", 5.0) or 5.0)

    # 1) Preferred: fn(symbol, data=bot.data, cfg=bot.cfg)
    try:
        out = fn(symbol, data=getattr(bot, "data", None), cfg=getattr(bot, "cfg", None))
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, dict):
            return out
        t = _tuple_to_sig_dict(symbol, out, diag=diag, no_trade_log_every=no_trade_log_every)
        if t:
            return t
    except TypeError:
        pass
    except Exception as e:
        log_entry.warning(f"ENTRY_LOOP signal failed {symbol}: {e}")
        return None

    # 2) fn(bot, symbol)
    try:
        out = fn(bot, symbol)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, dict):
            return out
        t = _tuple_to_sig_dict(symbol, out, diag=diag, no_trade_log_every=no_trade_log_every)
        if t:
            return t
    except TypeError:
        pass
    except Exception as e:
        log_entry.warning(f"ENTRY_LOOP signal failed {symbol}: {e}")
        return None

    # 3) fn(symbol, bot=bot)
    try:
        out = fn(symbol, bot=bot)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, dict):
            return out
        t = _tuple_to_sig_dict(symbol, out, diag=diag, no_trade_log_every=no_trade_log_every)
        if t:
            return t
    except Exception as e:
        log_entry.warning(f"ENTRY_LOOP signal failed {symbol}: {e}")
        return None

    return None


def _parse_action(sig: Dict[str, Any]) -> Optional[str]:
    a = str(sig.get("action") or sig.get("side") or "").strip().lower()
    if a in ("long", "buy"):
        return "buy"
    if a in ("short", "sell"):
        return "sell"
    return None


def _parse_order_type(sig: Dict[str, Any]) -> str:
    t = str(sig.get("type") or sig.get("order_type") or "market").strip().lower()
    return "limit" if t in ("limit",) else "market"


def _parse_amount(sig: Dict[str, Any]) -> Optional[float]:
    for key in ("amount", "qty", "size"):
        if key in sig:
            try:
                v = float(sig.get(key))
                return v if v > 0 else None
            except Exception:
                return None
    return None


def _parse_price(sig: Dict[str, Any]) -> Optional[float]:
    for key in ("price", "limit_price"):
        if key in sig:
            try:
                v = float(sig.get(key))
                return v if v > 0 else None
            except Exception:
                return None
    return None


def _get_price(bot, symbol: str) -> float:
    """
    Best-effort last price getter for qty-from-notional conversion.
    """
    k = _symkey(symbol)
    px = 0.0
    try:
        data = getattr(bot, "data", None)
        gp = getattr(data, "get_price", None) if data is not None else None
        if callable(gp):
            try:
                px = float(gp(k, in_position=False) or 0.0)
            except TypeError:
                px = float(gp(k) or 0.0)
    except Exception:
        px = 0.0

    if px <= 0:
        try:
            price_map = getattr(getattr(bot, "data", None), "price", {}) or {}
            if isinstance(price_map, dict):
                px = float(price_map.get(k, 0.0) or 0.0)
        except Exception:
            px = 0.0

    return float(px) if px > 0 else 0.0


def _sizing_fallback_amount(bot, symbol: str) -> Optional[float]:
    """
    Minimal fallback sizing:
    - If (ENV/CFG) FIXED_QTY is set => use that
    - Else if (ENV/CFG) FIXED_NOTIONAL_USDT and we can get price => qty = notional/price
    - Else None (skip)
    """
    fixed_qty, notional = _resolve_sizing(bot)

    try:
        if fixed_qty > 0:
            return float(fixed_qty)
    except Exception:
        pass

    if notional <= 0:
        return None

    px = _get_price(bot, symbol)
    if px <= 0:
        return None

    qty = float(notional) / float(px)

    try:
        min_qty = float(_cfg(bot, "MIN_ENTRY_QTY", 0.0) or 0.0)
        if min_qty > 0 and qty < min_qty:
            qty = min_qty
    except Exception:
        pass

    return qty if qty > 0 else None


# ----------------------------
# Anti-spam: pending entry tracking (locks + cooldown)
# ----------------------------

_ENTRY_LOCKS: Dict[str, asyncio.Lock] = {}
_PENDING_UNTIL: Dict[str, float] = {}          # k -> ts until which entries are blocked
_PENDING_ORDER_ID: Dict[str, str] = {}         # k -> last submitted order id (best-effort)


def _get_entry_lock(k: str) -> asyncio.Lock:
    lk = _ENTRY_LOCKS.get(k)
    if isinstance(lk, asyncio.Lock):
        return lk
    lk = asyncio.Lock()
    _ENTRY_LOCKS[k] = lk
    return lk


def _set_pending(k: str, *, sec: float, order_id: Optional[str] = None) -> None:
    try:
        _PENDING_UNTIL[k] = _now() + max(0.0, float(sec))
        if order_id:
            _PENDING_ORDER_ID[k] = str(order_id)
    except Exception:
        pass


def _pending_active(k: str) -> bool:
    try:
        until = float(_PENDING_UNTIL.get(k, 0.0) or 0.0)
        return _now() < until
    except Exception:
        return False


async def _has_open_entry_order(bot, k: str, sym_raw: str) -> bool:
    """
    Best-effort: detect open orders from exchange to avoid stacking while reconcile lags.
    This is intentionally conservative and never fatal.
    """
    try:
        enabled = _cfg_env_bool(bot, "ENTRY_PROBE_OPEN_ORDERS", True)
        if not enabled:
            return False

        ex = getattr(bot, "ex", None)
        if ex is None:
            return False

        fn = getattr(ex, "fetch_open_orders", None)
        if not callable(fn):
            return False

        # Some exchanges want raw symbol; we try sym_raw first, then k.
        try:
            orders = await fn(sym_raw)
        except Exception:
            try:
                orders = await fn(k)
            except Exception:
                return False

        if not isinstance(orders, (list, tuple)) or not orders:
            return False

        # If we can, ignore reduceOnly/closePosition orders (exits); entry loop should only care about entry-ish orders.
        for o in orders:
            if not isinstance(o, dict):
                continue
            info = o.get("info") or {}
            params = o.get("params") or {}

            ro = info.get("reduceOnly", params.get("reduceOnly"))
            cp = info.get("closePosition", params.get("closePosition"))
            if _truthy(ro) or _truthy(cp):
                continue

            status = str(o.get("status") or "").lower()
            if status in ("open", "new", "partially_filled", ""):
                return True

        return False
    except Exception:
        return False


async def _has_open_position_exchange(bot, k: str, sym_raw: str) -> bool:
    """
    Best-effort: detect if exchange reports an open position.
    This helps when brain-state isn't adopted yet.
    """
    try:
        enabled = _cfg_env_bool(bot, "ENTRY_PROBE_EXCHANGE_POSITIONS", False)
        if not enabled:
            return False

        ex = getattr(bot, "ex", None)
        if ex is None:
            return False

        fn = getattr(ex, "fetch_positions", None)
        if not callable(fn):
            return False

        try:
            poss = await fn([sym_raw])
        except Exception:
            try:
                poss = await fn([k])
            except Exception:
                return False

        if not isinstance(poss, (list, tuple)) or not poss:
            return False

        for p in poss:
            if not isinstance(p, dict):
                continue
            # ccxt position formats vary; try common fields:
            sz = p.get("contracts")
            if sz is None:
                sz = p.get("contractSize")
            if sz is None:
                sz = p.get("size")
            if sz is None:
                sz = p.get("positionAmt")
            try:
                if sz is not None and abs(float(sz)) > 0.0:
                    return True
            except Exception:
                continue

        return False
    except Exception:
        return False


# ----------------------------
# Entry loop
# ----------------------------

async def entry_loop(bot) -> None:
    """
    Main entry loop.
    Places NEW entries only. Exits/stop management handled elsewhere (exit/posmgr).
    """
    shutdown_ev = _ensure_shutdown_event(bot)

    # ENV overrides for safety knobs
    poll_sec = _cfg_env_float(bot, "ENTRY_POLL_SEC", 1.0)
    per_symbol_gap_sec = _cfg_env_float(bot, "ENTRY_PER_SYMBOL_GAP_SEC", 2.5)
    local_cooldown_sec = _cfg_env_float(bot, "ENTRY_LOCAL_COOLDOWN_SEC", 8.0)
    min_conf = _cfg_env_float(bot, "ENTRY_MIN_CONFIDENCE", 0.0)

    # NEW: pending-block window after submit to stop stacking while reconcile adopts
    pending_block_sec = _cfg_env_float(bot, "ENTRY_PENDING_BLOCK_SEC", 30.0)

    respect_kill = bool(_cfg_env_bool(bot, "ENTRY_RESPECT_KILL_SWITCH", True))

    # Hedge hint mode can be set via ENV too
    hedge_hint_mode = bool(
        _cfg_env_bool(bot, "HEDGE_MODE", False)
        or _cfg_env_bool(bot, "HEDGE_SAFE", False)
        or _truthy(_cfg(bot, "HEDGE_MODE", False) or _cfg(bot, "HEDGE_SAFE", False))
    )

    # diag can be controlled via ENV, else cfg
    diag = _cfg_env_bool(bot, "SCALPER_SIGNAL_DIAG", _cfg(bot, "SCALPER_SIGNAL_DIAG", "0"))

    sizing_warn_every = _cfg_env_float(bot, "ENTRY_SIZING_WARN_EVERY_SEC", 30.0)
    last_sizing_warn_ts = 0.0

    # Backoff on margin insufficient to prevent spam
    margin_backoff_sec = _cfg_env_float(bot, "ENTRY_MARGIN_INSUFFICIENT_BACKOFF_SEC", 900.0)  # 15m default
    backoff_until_by_sym: Dict[str, float] = {}

    # Local cooldown memory
    last_attempt_by_sym: Dict[str, float] = {}
    last_symbol_tick = 0.0

    sig_fn = _load_signal_fn()
    if not callable(sig_fn):
        log_core.warning("ENTRY_LOOP: strategy signal missing (strategies.eclipse_scalper.scalper_signal). Loop will idle.")

    log_core.info("ENTRY_LOOP ONLINE — scanning for new entries")

    while not shutdown_ev.is_set():
        try:
            now = _now()

            # optional kill-switch gate
            if respect_kill and callable(trade_allowed):
                try:
                    ok = await trade_allowed(bot)
                    if not ok:
                        await asyncio.sleep(max(0.25, poll_sec))
                        continue
                except Exception:
                    await asyncio.sleep(max(0.25, poll_sec))
                    continue

            # avoid super tight spin if poll_sec tiny
            if poll_sec > 0 and (now - last_symbol_tick) < poll_sec:
                await asyncio.sleep(max(0.05, poll_sec - (now - last_symbol_tick)))
            last_symbol_tick = _now()

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

                # margin-insufficient backoff gate
                bo = float(backoff_until_by_sym.get(k, 0.0) or 0.0)
                if bo > 0 and _now() < bo:
                    continue

                # hard pending gate (prevents stacking during adopt/reconcile lag)
                if _pending_active(k):
                    continue

                # skip if already in position (brain-state)
                if _in_position_brain(bot, k):
                    continue

                # local cooldown
                la = float(last_attempt_by_sym.get(k, 0.0) or 0.0)
                if local_cooldown_sec > 0 and (_now() - la) < local_cooldown_sec:
                    continue

                # must have signal function to do anything
                if not callable(sig_fn):
                    continue

                # resolve raw symbol once (needed for exchange probes)
                sym_raw = _resolve_raw_symbol(bot, k, k)

                # best-effort exchange probes (optional)
                try:
                    if await _has_open_entry_order(bot, k, sym_raw):
                        _set_pending(k, sec=max(5.0, pending_block_sec * 0.5))
                        continue
                except Exception:
                    pass

                try:
                    if await _has_open_position_exchange(bot, k, sym_raw):
                        _set_pending(k, sec=max(5.0, pending_block_sec * 0.5))
                        continue
                except Exception:
                    pass

                # Acquire per-symbol entry lock to prevent concurrent submit storms
                lk = _get_entry_lock(k)
                if lk.locked():
                    continue

                async with lk:
                    # re-check gates after lock (race-safe)
                    if shutdown_ev.is_set():
                        break
                    if _pending_active(k):
                        continue
                    if _in_position_brain(bot, k):
                        continue
                    bo = float(backoff_until_by_sym.get(k, 0.0) or 0.0)
                    if bo > 0 and _now() < bo:
                        continue

                    # mark attempt NOW to enforce cooldown even if we error later
                    last_attempt_by_sym[k] = _now()

                    sig = await _maybe_call_signal(sig_fn, bot, k, diag=diag)
                    if not isinstance(sig, dict) or not sig:
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    action = _parse_action(sig)
                    if action not in ("buy", "sell"):
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    # confidence gate
                    try:
                        conf = float(sig.get("confidence", sig.get("conf", 0.0)) or 0.0)
                    except Exception:
                        conf = 0.0
                    if min_conf > 0 and conf < min_conf:
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    otype = _parse_order_type(sig)
                    price = _parse_price(sig) if otype == "limit" else None
                    amt = _parse_amount(sig)
                    if amt is None:
                        amt = _sizing_fallback_amount(bot, k)

                    if amt is None or amt <= 0:
                        if sizing_warn_every > 0 and (_now() - last_sizing_warn_ts) >= sizing_warn_every:
                            last_sizing_warn_ts = _now()
                            fixed_qty, fixed_notional = _resolve_sizing(bot)
                            log_entry.warning(
                                "ENTRY_LOOP: sizing missing; set FIXED_QTY or FIXED_NOTIONAL_USDT. "
                                f"(FIXED_QTY={fixed_qty}, FIXED_NOTIONAL_USDT={fixed_notional})"
                            )
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    # Hedge hint: router accepts "long"/"short" for entries
                    hedge_side_hint = None
                    if hedge_hint_mode:
                        hedge_side_hint = "long" if action == "buy" else "short"

                    # Submit order
                    try:
                        if otype == "limit":
                            if price is None or price <= 0:
                                log_entry.warning(f"ENTRY_LOOP: limit signal missing price for {k}")
                                await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                continue

                            res = await create_order(
                                bot,
                                symbol=sym_raw,
                                type="LIMIT",
                                side=action,
                                amount=float(amt),
                                price=float(price),
                                params={},
                                intent_reduce_only=False,
                                intent_close_position=False,
                                hedge_side_hint=hedge_side_hint,
                                retries=int(_cfg_env_float(bot, "ENTRY_ROUTER_RETRIES", 6) or 6),
                            )
                        else:
                            res = await create_order(
                                bot,
                                symbol=sym_raw,
                                type="MARKET",
                                side=action,
                                amount=float(amt),
                                price=None,
                                params={},
                                intent_reduce_only=False,
                                intent_close_position=False,
                                hedge_side_hint=hedge_side_hint,
                                retries=int(_cfg_env_float(bot, "ENTRY_ROUTER_RETRIES", 4) or 4),
                            )

                        oid = None
                        if isinstance(res, dict):
                            oid = res.get("id") or (res.get("info") or {}).get("orderId")

                        if res is None:
                            log_entry.warning(f"ENTRY_LOOP: create_order returned None for {k}")
                            if callable(emit):
                                try:
                                    await emit(
                                        bot,
                                        "entry.order_failed",
                                        data={"symbol": k, "action": action, "type": otype},
                                        symbol=k,
                                        level="critical",
                                    )
                                except Exception:
                                    pass
                            # even if failed, block briefly to prevent rapid spam while exchange is angry
                            _set_pending(k, sec=max(3.0, pending_block_sec * 0.25))
                            await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                            continue

                        # ✅ key anti-stack: once we submitted ANY entry, block more entries for a while
                        _set_pending(k, sec=max(5.0, pending_block_sec), order_id=str(oid) if oid else None)

                        log_core.critical(f"ENTRY_LOOP: ORDER SUBMITTED {k} {action.upper()} type={otype} amt={amt} id={oid}")

                        # Register watch for limit/pending orders (optional)
                        if otype == "limit" and callable(register_entry_watch) and oid:
                            try:
                                await register_entry_watch(
                                    bot,
                                    symbol=sym_raw,
                                    order_id=str(oid),
                                    side=("long" if action == "buy" else "short"),
                                    amount=float(amt),
                                    price=float(price or 0.0),
                                    meta={"confidence": conf, "signal": {kk: vv for kk, vv in sig.items() if kk not in ("raw",)}},
                                )
                            except Exception:
                                pass

                        # Telemetry (optional)
                        if callable(emit):
                            try:
                                await emit(
                                    bot,
                                    "entry.submitted",
                                    data={
                                        "symbol": k,
                                        "action": action,
                                        "type": otype,
                                        "amount": float(amt),
                                        "price": float(price) if price is not None else None,
                                        "confidence": conf,
                                        "order_id": oid,
                                        "pending_block_sec": float(pending_block_sec),
                                    },
                                    symbol=k,
                                    level="info",
                                )
                            except Exception:
                                pass

                        if _truthy(_cfg(bot, "ENTRY_NOTIFY", False)):
                            await _safe_speak(bot, f"ENTRY {k} {action.upper()} {otype} amt={amt}", "info")

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # try to detect margin insufficient and backoff
                        msg = str(e)
                        if "Margin is insufficient" in msg or '"code":-2019' in msg or "code': -2019" in msg:
                            until = _now() + max(60.0, float(margin_backoff_sec))
                            backoff_until_by_sym[k] = until
                            log_entry.critical(f"ENTRY_LOOP: margin insufficient → backing off {k} for {int(margin_backoff_sec)}s")
                            # also block short-term so we don't keep hammering before backoff map is checked again
                            _set_pending(k, sec=max(10.0, pending_block_sec * 0.5))
                        else:
                            log_entry.error(f"ENTRY_LOOP: order submit failed {k}: {e}")
                            _set_pending(k, sec=max(3.0, pending_block_sec * 0.25))

                        if callable(emit):
                            try:
                                await emit(
                                    bot,
                                    "entry.exception",
                                    data={"symbol": k, "err": repr(e)[:300]},
                                    symbol=k,
                                    level="critical",
                                )
                            except Exception:
                                pass

                await asyncio.sleep(max(0.01, per_symbol_gap_sec))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"ENTRY_LOOP outer error: {e}")
            await asyncio.sleep(1.0)

    log_core.critical("ENTRY_LOOP OFFLINE — shutdown flag set")
