# execution/entry_loop.py — SCALPER ETERNAL — ENTRY LOOP — 2026 v1.4 (ENV+CFG SIZING RESOLVER)
# Patch vs v1.3:
# - ✅ FIX: Entry sizing reads ENV + cfg (supports FIXED_NOTIONAL_USDT + common aliases)
# - ✅ FIX: Sizing warning prints resolved values (not cfg-only zeros)
# - ✅ Keeps: tuple no-trade throttle, diag throttle, sizing warn throttle
# - ✅ Guardian-safe: never raises

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional, Callable

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


def _in_position(bot, k: str) -> bool:
    """
    Cheap check: state.positions contains k with nonzero size.
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
        # never fatal
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

        # both false OR both true => treat as no actionable signal
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
    # how often to print "no-trade" tuple diag per symbol
    no_trade_log_every = float(_cfg(bot, "ENTRY_NO_TRADE_LOG_EVERY_SEC", 5.0) or 5.0)

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

    # 2) fn(bot, symbol) (legacy pattern)
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

    # 3) fn(symbol, bot=bot) (older adapter path)
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
            # get_price signature varies; keep it super safe
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
# Entry loop
# ----------------------------

async def entry_loop(bot) -> None:
    """
    Main entry loop.
    Places NEW entries only. Exits/stop management handled elsewhere (exit/posmgr).
    """
    shutdown_ev = _ensure_shutdown_event(bot)

    poll_sec = float(_cfg(bot, "ENTRY_POLL_SEC", 1.0) or 1.0)
    per_symbol_gap_sec = float(_cfg(bot, "ENTRY_PER_SYMBOL_GAP_SEC", 2.5) or 2.5)

    local_cooldown_sec = float(_cfg(bot, "ENTRY_LOCAL_COOLDOWN_SEC", 8.0) or 8.0)
    min_conf = float(_cfg(bot, "ENTRY_MIN_CONFIDENCE", 0.0) or 0.0)

    respect_kill = bool(_truthy(_cfg(bot, "ENTRY_RESPECT_KILL_SWITCH", True)))
    hedge_hint_mode = bool(_truthy(_cfg(bot, "HEDGE_MODE", False) or _cfg(bot, "HEDGE_SAFE", False)))

    diag = _truthy(os.getenv("SCALPER_SIGNAL_DIAG", _cfg(bot, "SCALPER_SIGNAL_DIAG", "0")))

    # Throttle: sizing warning every N seconds
    sizing_warn_every = float(_cfg(bot, "ENTRY_SIZING_WARN_EVERY_SEC", 30.0) or 30.0)
    last_sizing_warn_ts = 0.0

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

                # skip if already in position
                if _in_position(bot, k):
                    continue

                # local per-symbol cooldown
                la = float(last_attempt_by_sym.get(k, 0.0) or 0.0)
                if local_cooldown_sec > 0 and (_now() - la) < local_cooldown_sec:
                    continue

                # must have a signal function to do anything
                if not callable(sig_fn):
                    continue

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
                    # Throttled warning: you have signals but no sizing configured
                    if sizing_warn_every > 0 and (_now() - last_sizing_warn_ts) >= sizing_warn_every:
                        last_sizing_warn_ts = _now()
                        fixed_qty, fixed_notional = _resolve_sizing(bot)
                        log_entry.warning(
                            "ENTRY_LOOP: sizing missing; set FIXED_QTY or FIXED_NOTIONAL_USDT. "
                            f"(FIXED_QTY={fixed_qty}, FIXED_NOTIONAL_USDT={fixed_notional})"
                        )
                    last_attempt_by_sym[k] = _now()
                    await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                    continue

                sym_raw = _resolve_raw_symbol(bot, k, k)

                # Hedge hint: positionSide wants LONG/SHORT; router accepts "long"/"short"
                hedge_side_hint = None
                if hedge_hint_mode:
                    hedge_side_hint = "long" if action == "buy" else "short"

                last_attempt_by_sym[k] = _now()

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
                            retries=int(_cfg(bot, "ENTRY_ROUTER_RETRIES", 6) or 6),
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
                            retries=int(_cfg(bot, "ENTRY_ROUTER_RETRIES", 4) or 4),
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
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

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
                                },
                                symbol=k,
                                level="info",
                            )
                        except Exception:
                            pass

                    # optional notify
                    if _truthy(_cfg(bot, "ENTRY_NOTIFY", False)):
                        await _safe_speak(bot, f"ENTRY {k} {action.upper()} {otype} amt={amt}", "info")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    log_entry.error(f"ENTRY_LOOP: order submit failed {k}: {e}")
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
