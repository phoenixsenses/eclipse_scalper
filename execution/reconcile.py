# execution/reconcile.py — SCALPER ETERNAL — REALITY RECONCILER — 2026 v1.6 (DIAGNOSTIC-AWARE + NO SILENT OPTIONALS)
# Patch vs v1.5:
# - ✅ Wires diagnostics helper into optional imports (explicit "OPTIONAL MISSING" logs)
# - ✅ Makes kill-switch optional import loud (no silent None)
# - ✅ Optional one-shot module banner (keeps logs readable)
# - ✅ No logic change to reconcile itself

import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List

from utils.logging import log_entry, log_core
from execution.order_router import create_order, cancel_order  # ✅ ROUTER


# ─────────────────────────────────────────────────────────────────────
# Diagnostics wiring (never fatal)
# ─────────────────────────────────────────────────────────────────────

_RECONCILE_ONCE = False


def _banner_once() -> None:
    global _RECONCILE_ONCE
    if _RECONCILE_ONCE:
        return
    _RECONCILE_ONCE = True
    log_core.info("RECONCILE ONLINE — reality reconciling is armed")


def _optional_import(module: str, attr: Optional[str] = None):
    """
    Best-effort optional import that logs explicitly what's missing.
    Returns module or attribute; returns None if unavailable.
    """
    try:
        mod = __import__(module, fromlist=[attr] if attr else [])
        if attr is None:
            return mod
        return getattr(mod, attr)
    except Exception:
        what = f"{module}.{attr}" if attr else module
        log_core.warning(f"OPTIONAL MISSING — {what}")
        return None


# Optional diagnostics dump (if present)
_print_diagnostics = _optional_import("execution.diagnostics", "print_diagnostics")


def _diag_dump(bot, note: str) -> None:
    try:
        if callable(_print_diagnostics):
            log_core.warning(f"DIAG DUMP — {note}")
            _print_diagnostics(bot)
    except Exception:
        pass


# Optional kill-switch gate (used only to skip creating new protective stops)
is_halted = _optional_import("risk.kill_switch", "is_halted")


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "t", "on"):
        return True
    return False


def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
    except Exception:
        return default


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


def _ensure_run_context(bot) -> dict:
    rc = getattr(getattr(bot, "state", None), "run_context", None)
    if not isinstance(rc, dict):
        try:
            bot.state.run_context = {}
        except Exception:
            pass
        rc = getattr(getattr(bot, "state", None), "run_context", None)
    return rc if isinstance(rc, dict) else {}


def _resolve_raw_symbol(bot, k: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return k  # fallback canonical


def _extract_pos_size_side(pos: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    side = None

    contracts = _safe_float(pos.get("contracts"), 0.0)
    if contracts:
        side = pos.get("side")
        if isinstance(side, str):
            side = side.lower().strip()
        return abs(contracts), side if side in ("long", "short") else None

    info = pos.get("info") or {}
    amt = info.get("positionAmt")
    if amt is not None:
        signed = _safe_float(amt, 0.0)
        if signed != 0:
            side = "long" if signed > 0 else "short"
            return abs(signed), side

    amt2 = _safe_float(pos.get("amount"), 0.0)
    if amt2 != 0:
        return abs(amt2), None

    return 0.0, None


def _extract_entry_price(pos: Dict[str, Any]) -> float:
    for kk in ("entryPrice", "entry_price", "average"):
        v = _safe_float(pos.get(kk), 0.0)
        if v > 0:
            return v
    info = pos.get("info") or {}
    for kk in ("entryPrice", "avgPrice"):
        v = _safe_float(info.get(kk), 0.0)
        if v > 0:
            return v
    return 0.0


def _is_reduce_only(order: Dict[str, Any]) -> bool:
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


def _is_stop_like(order: Dict[str, Any]) -> bool:
    try:
        t = str(order.get("type") or "").upper()
        if "STOP" in t:
            return True
        info = order.get("info") or {}
        it = str(info.get("type") or info.get("orderType") or "").upper()
        if "STOP" in it:
            return True
        if (order.get("stopPrice") is not None) or (info.get("stopPrice") is not None):
            return True
    except Exception:
        pass
    return False


def _is_tp_like(order: Dict[str, Any]) -> bool:
    try:
        t = str(order.get("type") or "").upper()
        if "TAKE_PROFIT" in t or "TP" in t:
            return True
        info = order.get("info") or {}
        it = str(info.get("type") or info.get("orderType") or "").upper()
        if "TAKE_PROFIT" in it:
            return True
    except Exception:
        pass
    return False


async def _safe_speak(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _alert_ok(bot, key: str, msg: str, cooldown_sec: float) -> bool:
    """
    Dedup + cooldown by 'key'. Stores last ts and last msg.
    """
    rc = _ensure_run_context(bot)
    store = rc.get("reconcile_alerts")
    if not isinstance(store, dict):
        store = {}
        rc["reconcile_alerts"] = store

    now = _now()
    prev = store.get(key)
    if not isinstance(prev, dict):
        prev = {"ts": 0.0, "msg": ""}
        store[key] = prev

    last_ts = _safe_float(prev.get("ts"), 0.0)
    last_msg = str(prev.get("msg") or "")

    if msg == last_msg and (now - last_ts) < cooldown_sec:
        return False

    if (now - last_ts) < cooldown_sec and msg != last_msg:
        return False

    prev["ts"] = now
    prev["msg"] = msg
    return True


# ----------------------------
# Exchange fetch helpers
# ----------------------------

async def _fetch_open_orders_best_effort(bot, sym_raw: str) -> List[dict]:
    ex = getattr(bot, "ex", None)
    if ex is None:
        return []
    fn = getattr(ex, "fetch_open_orders", None)
    if not callable(fn):
        return []
    try:
        res = await fn(sym_raw)
        return res if isinstance(res, list) else []
    except Exception:
        return []


async def _fetch_positions_best_effort(bot, symbols: Optional[List[str]] = None) -> List[dict]:
    """
    Some ccxt wrappers accept fetch_positions([symbols]); some don't.
    If symbols is None -> fetch all positions (for orphan scan).
    """
    ex = getattr(bot, "ex", None)
    if ex is None:
        return []
    fp = getattr(ex, "fetch_positions", None)
    if not callable(fp):
        return []

    if symbols is None:
        try:
            res = await ex.fetch_positions()
            return res if isinstance(res, list) else []
        except Exception:
            return []

    try:
        res = await ex.fetch_positions(symbols)
        if isinstance(res, list):
            return res
    except Exception:
        pass

    try:
        res = await ex.fetch_positions()
        if not isinstance(res, list):
            return []
        want = set(_symkey(s) for s in symbols)
        out = []
        for p in res:
            if isinstance(p, dict) and _symkey(p.get("symbol") or "") in want:
                out.append(p)
        return out
    except Exception:
        return []


async def _cancel_reduce_only_open_orders(bot, sym_raw: str, *, cancel_stops: bool = True, cancel_tps: bool = True):
    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        for o in oo:
            if not isinstance(o, dict):
                continue
            if not _is_reduce_only(o):
                continue

            if (not cancel_stops) and _is_stop_like(o):
                continue
            if (not cancel_tps) and _is_tp_like(o):
                continue

            oid = o.get("id")
            if oid:
                try:
                    await cancel_order(bot, str(oid), sym_raw)
                except Exception:
                    pass
    except Exception:
        pass


# ----------------------------
# Protective stop (ladder)
# ----------------------------

async def _place_stop_ladder_router(
    bot,
    *,
    sym_raw: str,
    side: str,             # "long"/"short"
    qty: float,
    stop_price: float,
    hedge_side_hint: Optional[str],
    k: str,
) -> Optional[str]:
    """
    Ladder stop restore like entry.py:
    A) amount + reduceOnly
    B) closePosition fallbacks (amount variations)
    """
    stop_side = "sell" if side == "long" else "buy"

    try:
        o = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=float(qty),
            price=None,
            params={},
            intent_reduce_only=True,
            intent_close_position=False,
            stop_price=float(stop_price),
            hedge_side_hint=hedge_side_hint,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e1:
        log_entry.warning(f"{k} reconcile stop A failed: {e1}")

    last = None
    for amt_try in (float(qty), 0.0, 0, "0", "0.0"):
        try:
            o = await create_order(
                bot,
                symbol=sym_raw,
                type="STOP_MARKET",
                side=stop_side,
                amount=amt_try,
                price=None,
                params={"closePosition": True},
                intent_reduce_only=True,
                intent_close_position=True,
                stop_price=float(stop_price),
                hedge_side_hint=hedge_side_hint,
                retries=6,
            )
            if isinstance(o, dict) and o.get("id"):
                return str(o.get("id"))
        except Exception as e2:
            last = e2
            continue

    log_entry.critical(f"{k} reconcile stop B failed (all fallbacks): {last}")
    return None


async def _ensure_protective_stop(bot, k: str, pos_obj, ex_side_hint: Optional[str], ex_size: float):
    if not _cfg(bot, "GUARDIAN_ENSURE_STOP", True):
        return

    if _truthy(_cfg(bot, "GUARDIAN_RESPECT_KILL_SWITCH", True)) and callable(is_halted):
        try:
            if is_halted(bot):
                return
        except Exception:
            pass

    sym_raw = _resolve_raw_symbol(bot, k)

    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        for o in oo:
            if isinstance(o, dict) and _is_reduce_only(o) and _is_stop_like(o):
                return
    except Exception:
        pass

    side = None
    try:
        side = str(getattr(pos_obj, "side", "") or "").lower().strip()
    except Exception:
        side = None
    if side not in ("long", "short"):
        side = ex_side_hint if ex_side_hint in ("long", "short") else None
    if side not in ("long", "short"):
        msg = f"RECONCILE: {k} no side detected — stop not placed"
        if _alert_ok(bot, f"{k}:no_side", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
        return

    hedge_side_hint = side  # router maps to positionSide LONG/SHORT if hedge enabled

    entry_px = _safe_float(getattr(pos_obj, "entry_price", 0.0), 0.0)
    atr = _safe_float(getattr(pos_obj, "atr", 0.0), 0.0)
    if entry_px <= 0:
        msg = f"RECONCILE: {k} missing entry_price — stop not placed"
        if _alert_ok(bot, f"{k}:no_entry", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
        return

    stop_atr_mult = float(_cfg(bot, "STOP_ATR_MULT", 2.0))
    buffer_atr_mult = float(_cfg(bot, "GUARDIAN_STOP_BUFFER_ATR_MULT", 0.0))

    if atr > 0:
        dist = atr * (stop_atr_mult + buffer_atr_mult)
    else:
        dist = entry_px * float(_cfg(bot, "GUARDIAN_STOP_FALLBACK_PCT", 0.0035))

    stop_price = entry_px - dist if side == "long" else entry_px + dist
    if stop_price <= 0:
        return

    oid = await _place_stop_ladder_router(
        bot,
        sym_raw=sym_raw,
        side=side,
        qty=float(ex_size),
        stop_price=float(stop_price),
        hedge_side_hint=hedge_side_hint,
        k=k,
    )

    if oid:
        msg = f"RECONCILE: PROTECTIVE STOP RESTORED {k}"
        if _alert_ok(bot, f"{k}:stop_restored", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "critical")
    else:
        msg = f"RECONCILE: STOP RESTORE FAILED {k}"
        if _alert_ok(bot, f"{k}:stop_failed", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "critical")
        # Optional: uncomment when debugging stop restore failures
        # _diag_dump(bot, f"{k} stop restore failed")


# ----------------------------
# Orphan reduceOnly order cleanup (optional)
# ----------------------------

async def _cancel_orphan_reduce_only_orders_if_no_position(bot, k: str):
    if not _truthy(_cfg(bot, "RECONCILE_CANCEL_ORPHAN_REDUCE_ONLY_ORDERS", True)):
        return

    sym_raw = _resolve_raw_symbol(bot, k)
    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        if not oo:
            return
        any_ro = any(isinstance(o, dict) and _is_reduce_only(o) for o in oo)
        if not any_ro:
            return

        await _cancel_reduce_only_open_orders(bot, sym_raw, cancel_stops=True, cancel_tps=True)

        msg = f"RECONCILE: ORPHAN reduceOnly orders canceled {k}"
        if _alert_ok(bot, f"{k}:orphan_ro_canceled", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
    except Exception:
        pass


# ----------------------------
# Main reconcile tick
# ----------------------------

async def reconcile_tick(bot):
    """
    Single pass: reconcile bot.state.positions vs exchange positions.
    Called by execution/guardian.guardian_loop().
    """
    _banner_once()

    drift_abs = float(_cfg(bot, "GUARDIAN_SIZE_DRIFT_ABS", 0.0))
    drift_pct = float(_cfg(bot, "GUARDIAN_SIZE_DRIFT_PCT", 0.05))
    auto_flat_orphans = bool(_cfg(bot, "GUARDIAN_AUTO_FLAT_ORPHANS", False))
    full_scan = _truthy(_cfg(bot, "RECONCILE_FULL_SCAN_ORPHANS", True))

    _ensure_shutdown_event(bot)

    try:
        if not hasattr(bot.state, "positions") or not isinstance(getattr(bot.state, "positions", None), dict):
            bot.state.positions = {}
    except Exception:
        pass

    state_positions: Dict[str, Any] = getattr(bot.state, "positions", {}) or {}
    active = getattr(bot, "active_symbols", set()) or set()

    tracked_syms = set(_symkey(s) for s in state_positions.keys())
    tracked_syms |= set(_symkey(s) for s in active if s)

    if full_scan:
        ex_positions = await _fetch_positions_best_effort(bot, None)
    else:
        if not tracked_syms:
            return
        ex_positions = await _fetch_positions_best_effort(bot, list(tracked_syms))

    ex_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(ex_positions, list):
        for p in ex_positions:
            if not isinstance(p, dict):
                continue
            sym = p.get("symbol")
            if not sym:
                continue
            kk = _symkey(sym)
            size, _ = _extract_pos_size_side(p)
            if size > 0:
                ex_map[kk] = p

    # 1) ORPHAN EXCHANGE POSITIONS
    for k, p in ex_map.items():
        if k not in state_positions:
            size, side = _extract_pos_size_side(p)
            entry_px = _extract_entry_price(p)

            msg = f"RECONCILE: ORPHAN EXCHANGE POSITION {k} | side={side} | size={size:.6f} | entry={entry_px:.6f}"
            log_core.critical(msg)
            if _alert_ok(bot, f"{k}:orphan_pos", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                await _safe_speak(bot, msg, "critical")

            if auto_flat_orphans and side in ("long", "short") and size > 0:
                try:
                    sym_raw = _resolve_raw_symbol(bot, k)
                    close_side = "sell" if side == "long" else "buy"
                    await create_order(
                        bot,
                        symbol=sym_raw,
                        type="MARKET",
                        side=close_side,
                        amount=float(size),
                        params={},
                        intent_reduce_only=True,
                        hedge_side_hint=side,
                        retries=6,
                    )
                    msg2 = f"RECONCILE: ORPHAN FLATTENED {k}"
                    if _alert_ok(bot, f"{k}:orphan_flat", msg2, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                        await _safe_speak(bot, msg2, "critical")
                except Exception as e:
                    log_entry.error(f"Reconcile orphan flatten failed {k}: {e}")
                    msg3 = f"RECONCILE: ORPHAN FLATTEN FAILED {k} — {e}"
                    if _alert_ok(bot, f"{k}:orphan_flat_fail", msg3, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                        await _safe_speak(bot, msg3, "critical")

    # 2) PHANTOM STATE POSITIONS
    for k in list(state_positions.keys()):
        if k not in ex_map:
            sym_raw = _resolve_raw_symbol(bot, k)
            log_core.warning(f"RECONCILE: PHANTOM STATE POSITION {k} — clearing + cancel reduceOnly orders")
            msg = f"RECONCILE: PHANTOM STATE POSITION {k} — cleared"
            if _alert_ok(bot, f"{k}:phantom_cleared", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                await _safe_speak(bot, msg, "info")
            await _cancel_reduce_only_open_orders(bot, sym_raw)
            try:
                state_positions.pop(k, None)
            except Exception:
                pass

    # 2b) ORPHAN reduceOnly orders (no position exists)
    for k in list(tracked_syms):
        if k and k not in ex_map and k not in state_positions:
            await _cancel_orphan_reduce_only_orders_if_no_position(bot, k)

    # 3) DRIFT reconcile + entry_price patch + protective stop
    for k, pos_obj in list(state_positions.items()):
        ex_p = ex_map.get(k)
        if not isinstance(ex_p, dict):
            continue

        ex_size, ex_side = _extract_pos_size_side(ex_p)
        if ex_size <= 0:
            continue

        st_size = abs(_safe_float(getattr(pos_obj, "size", 0.0), 0.0))
        thresh = max(drift_abs, st_size * drift_pct) if st_size > 0 else drift_abs

        if st_size <= 0 or (thresh > 0 and abs(st_size - ex_size) > thresh):
            try:
                setattr(pos_obj, "size", float(ex_size))
            except Exception:
                pass
            msg = f"RECONCILE: SIZE SYNC {k} state={st_size:.6f} -> ex={ex_size:.6f}"
            if _alert_ok(bot, f"{k}:size_sync", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                await _safe_speak(bot, msg, "info")

        if _safe_float(getattr(pos_obj, "entry_price", 0.0), 0.0) <= 0:
            ep = _extract_entry_price(ex_p)
            if ep > 0:
                try:
                    setattr(pos_obj, "entry_price", float(ep))
                except Exception:
                    pass

        await _ensure_protective_stop(bot, k, pos_obj, ex_side, ex_size)


# Backward compatibility for older code that still imports guardian_loop from reconcile.py
async def guardian_loop(bot):
    poll_sec = float(_cfg(bot, "GUARDIAN_POLL_SEC", 15.0))
    shutdown_ev = _ensure_shutdown_event(bot)

    log_core.critical("RECONCILE LEGACY LOOP ONLINE — (guardian.py should own the loop)")

    while not shutdown_ev.is_set():
        try:
            await reconcile_tick(bot)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"Reconcile legacy loop error: {e}")
            # Optional: uncomment if you want auto-diag on legacy crashes
            # _diag_dump(bot, f"legacy loop exception: {e}")
        await asyncio.sleep(poll_sec)
