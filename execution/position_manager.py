# execution/position_manager.py — SCALPER ETERNAL — POSITION MANAGER — 2026 v1.5
# Patch vs v1.4:
# - ✅ FIX: STOP restore uses position-stable clientOrderId base (prevents duplicate spam)
# - ✅ FIX: If Binance returns -4116 (duplicate clientOrderId), ADOPT existing open order id
# - ✅ FIX: closePosition stop now uses intent_reduce_only=False (router + Binance consistent)
# - ✅ FIX: hedge_side_hint normalized to LONG/SHORT (router expects this for hedge exits)
# - ✅ KEEP: v1.4 logic (tick loop, env fallback, throttles, BE adoption, trailing rebuild)

import asyncio
import time
import os
import hashlib
from typing import Optional, List, Tuple, Dict, Any

from utils.logging import log_entry, log_core
from execution.order_router import create_order, cancel_order
from execution.position_lock import PositionLockManager  # ✅ ATOMIC POSITION OPS


_POSMGR_LOCKS: dict[str, asyncio.Lock] = {}

# Per-symbol open-orders throttle cache
_OPEN_ORDERS_CACHE: dict[str, Dict[str, Any]] = {}


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


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


def _truthy(x) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "t", "on")
    return False


def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
    except Exception:
        return default


def _cfg_env(bot, name: str, default):
    """
    Prefer bot.cfg.NAME, fallback to env var NAME, else default.
    This makes your PowerShell $env:POSMGR_* always apply even if cfg wiring is imperfect.
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


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _get_lock(k: str) -> asyncio.Lock:
    lk = _POSMGR_LOCKS.get(k)
    if lk is None:
        lk = asyncio.Lock()
        _POSMGR_LOCKS[k] = lk
    return lk


def _resolve_raw_symbol(bot, k: str, fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return fallback


def _pos_side_hint(side: str) -> str:
    s = str(side or "").lower().strip()
    if s == "long":
        return "LONG"
    if s == "short":
        return "SHORT"
    return ""


def _stable_stop_client_id_base(
    *,
    k: str,
    hedge_hint: str,
    entry_ts: float,
    entry_px: float,
    size: float,
) -> str:
    """
    Build a stable stop clientOrderId base across restarts.
    If entry_ts is missing, derive a deterministic suffix from entry price + size.
    """
    if entry_ts > 0:
        suffix = str(int(entry_ts * 1000))
    else:
        blob = f"{k}|{hedge_hint}|{entry_px:.8f}|{size:.8f}"
        suffix = f"X{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:10]}"
    return f"SE_STOP_{k}_{hedge_hint}_{suffix}"


def _get_current_price(bot, k: str, sym_any: str) -> float:
    """
    Prefer bot.data.get_price(k), fallback to bot.data.price maps.
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


async def _safe_speak(bot, text: str, priority: str = "info"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


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

    perf.setdefault("entry_size_abs", 0.0)
    perf.setdefault("mfe_pct", 0.0)

    # trailing state
    perf.setdefault("trailing_order_ids", [])
    perf.setdefault("last_trail_ts", 0.0)

    # stop checks
    perf.setdefault("posmgr_last_stop_check_ts", 0.0)
    perf.setdefault("posmgr_last_stop_place_ts", 0.0)
    perf.setdefault("posmgr_last_stop_id", None)
    perf.setdefault("posmgr_last_stop_id_ts", 0.0)

    return perf


def _is_reduce_only(order: dict) -> bool:
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


def _is_stop_like(order: dict) -> bool:
    try:
        t = str(order.get("type") or "").upper()
        if "STOP" in t:
            return True
        info = order.get("info") or {}
        it = str(info.get("type") or info.get("orderType") or "").upper()
        if "STOP" in it:
            return True
        if (order.get("stopPrice") is not None) or ((order.get("info") or {}).get("stopPrice") is not None):
            return True
    except Exception:
        pass
    return False


def _extract_order_id(o: dict) -> Optional[str]:
    try:
        oid = o.get("id")
        if oid:
            return str(oid)
        info = o.get("info") or {}
        oid2 = info.get("orderId") or info.get("id")
        if oid2:
            return str(oid2)
    except Exception:
        pass
    return None


def _extract_stop_price(o: dict) -> float:
    try:
        sp = o.get("stopPrice")
        v = _safe_float(sp, 0.0)
        if v > 0:
            return v
        info = o.get("info") or {}
        v2 = _safe_float(info.get("stopPrice"), 0.0)
        if v2 > 0:
            return v2
    except Exception:
        pass
    return 0.0


def _find_existing_stop(open_orders: List[dict], pos_side: str) -> Tuple[Optional[str], float]:
    """
    Returns (order_id, stop_price).
    Accepts reduceOnly/closePosition stops, or stop-like orders opposite to position side.
    """
    want_side = "sell" if str(pos_side).lower().strip() == "long" else "buy"
    for o in open_orders:
        if not isinstance(o, dict):
            continue
        if _is_stop_like(o) and _is_reduce_only(o):
            return _extract_order_id(o), _extract_stop_price(o)
        if _is_stop_like(o):
            try:
                side = str(o.get("side") or (o.get("info") or {}).get("side") or "").lower().strip()
                if side and side == want_side:
                    return _extract_order_id(o), _extract_stop_price(o)
            except Exception:
                pass
    return None, 0.0


def _looks_like_dup_client_order_id(err: Exception) -> bool:
    s = repr(err)
    return ("-4116" in s) or ("ClientOrderId is duplicated" in s) or ("clientorderid is duplicated" in s.lower())


def _order_client_id_any(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    info = order.get("info") or {}
    for kk in ("clientOrderId", "clientOrderID", "origClientOrderId", "origClientOrderID"):
        v = order.get(kk) or info.get(kk)
        if v:
            return str(v)
    return ""


async def _cancel_best_effort(bot, order_id: str, sym_raw: str, *, why: str = "") -> bool:
    if not order_id or not sym_raw:
        return False
    try:
        ok = await cancel_order(bot, str(order_id), str(sym_raw))
        if not ok and why:
            log_entry.warning(f"POSMGR cancel failed ({why}) id={order_id} sym={sym_raw}")
        return bool(ok)
    except Exception as e:
        if why:
            log_entry.warning(f"POSMGR cancel exception ({why}) id={order_id} sym={sym_raw}: {e}")
        return False


async def _fetch_open_orders_best_effort(
    bot,
    sym_raw: str,
    *,
    min_interval_sec: float = 8.0,
    cache_key: Optional[str] = None,
) -> List[dict]:
    """
    Throttled per-symbol fetch to avoid hammering exchange.
    """
    if not sym_raw:
        return []

    k = str(cache_key or sym_raw)
    now = _now()
    ent = _OPEN_ORDERS_CACHE.get(k) or {}
    last_ts = _safe_float(ent.get("ts", 0.0), 0.0)
    cached = ent.get("orders")
    last_sym = ent.get("sym_raw")

    if (
        cached is not None
        and (now - last_ts) < max(0.0, float(min_interval_sec))
        and last_sym == sym_raw
    ):
        return cached if isinstance(cached, list) else []

    ex = getattr(bot, "ex", None)
    if ex is None:
        return []

    fn = getattr(ex, "fetch_open_orders", None)
    if not callable(fn):
        return []

    try:
        res = await fn(sym_raw)
        arr = res if isinstance(res, list) else []
        _OPEN_ORDERS_CACHE[k] = {"ts": now, "orders": arr, "sym_raw": sym_raw}
        return arr
    except Exception:
        if isinstance(cached, list) and last_sym == sym_raw:
            return cached
        return []


async def _adopt_open_order_id_by_client_id(
    bot,
    sym_raw: str,
    client_id: str,
    *,
    min_interval_sec: float = 2.0,
    cache_key: Optional[str] = None,
) -> Optional[str]:
    if not client_id or not sym_raw:
        return None
    oo = await _fetch_open_orders_best_effort(bot, sym_raw, min_interval_sec=min_interval_sec, cache_key=cache_key)
    for o in oo:
        try:
            if _order_client_id_any(o) == str(client_id):
                oid = _extract_order_id(o)
                if oid:
                    return str(oid)
        except Exception:
            continue
    return None


async def _place_stop_ladder_router(
    bot,
    *,
    sym_raw: str,
    side: str,          # "long"/"short"
    qty: float,
    stop_price: float,
    hedge_side_hint: str,   # "LONG"/"SHORT"
    k: str,
    stop_client_id_base: str,
) -> Optional[str]:
    stop_side = "sell" if side == "long" else "buy"

    cid_a = f"{stop_client_id_base}_A"
    cid_b = f"{stop_client_id_base}_B"

    # A) amount + reduceOnly
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
            client_order_id=cid_a,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e1:
        if _looks_like_dup_client_order_id(e1):
            adopted = await _adopt_open_order_id_by_client_id(bot, sym_raw, cid_a, cache_key=k)
            if adopted:
                log_entry.warning(f"{k} posmgr stop A duplicate → ADOPTED id={adopted}")
                return adopted
        log_entry.warning(f"{k} posmgr stop A failed: {e1}")

    # B) closePosition ladder (IMPORTANT: intent_reduce_only=False)
    try:
        o = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=None,
            price=None,
            params={"closePosition": True},
            intent_reduce_only=False,       # ✅ FIX
            intent_close_position=True,
            stop_price=float(stop_price),
            hedge_side_hint=hedge_side_hint,
            client_order_id=cid_b,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e2:
        if _looks_like_dup_client_order_id(e2):
            adopted = await _adopt_open_order_id_by_client_id(bot, sym_raw, cid_b, cache_key=k)
            if adopted:
                log_entry.warning(f"{k} posmgr stop B duplicate → ADOPTED id={adopted}")
                return adopted
        log_entry.warning(f"{k} posmgr stop B failed: {e2}")

    log_entry.critical(f"{k} posmgr stop failed (all fallbacks exhausted)")
    return None


async def _place_trailing_router(
    bot,
    *,
    sym_raw: str,
    side: str,
    qty: float,
    activation_price: float,
    callback_rate: float,
    hedge_side_hint: str,   # "LONG"/"SHORT"
    k: str,
) -> Optional[str]:
    close_side = "sell" if side == "long" else "buy"
    cb = _clamp(float(callback_rate), 0.1, 5.0)
    try:
        o = await create_order(
            bot,
            symbol=sym_raw,
            type="TRAILING_STOP_MARKET",
            side=close_side,
            amount=float(qty),
            price=None,
            params={},
            intent_reduce_only=True,
            activation_price=float(activation_price),
            callback_rate=float(cb),
            hedge_side_hint=hedge_side_hint,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e:
        log_entry.warning(f"{k} posmgr trailing failed: {e}")
    return None


# ----------------------------
# Optional halt hook (guardian calls it)
# ----------------------------

async def on_halt(bot) -> None:
    return


# ----------------------------
# Main tick
# ----------------------------

async def position_manager_tick(bot) -> None:
    """
    Tick-based position management.
    Never raises (guardian-safe).
    """
    try:
        if not _truthy(_cfg_env(bot, "POSMGR_ENABLED", True)):
            return

        st = getattr(bot, "state", None)
        if st is None:
            return

        positions = getattr(st, "positions", None)
        if not isinstance(positions, dict) or not positions:
            return

        # ✅ ATOMIC: Get position lock manager for atomic operations
        _pos_lock_mgr = PositionLockManager.get(bot) if PositionLockManager.is_enabled(bot) else None

        rr_be = float(_cfg_env(bot, "BREAKEVEN_RR_TRIGGER", 1.0))
        rr_trail = float(_cfg_env(bot, "TRAILING_ACTIVATION_RR", 0.0))
        trail_debounce = float(_cfg_env(bot, "TRAILING_REBUILD_DEBOUNCE_SEC", 15.0))

        stop_check_sec = float(_cfg_env(bot, "POSMGR_STOP_CHECK_SEC", 30.0))
        ensure_stop = _truthy(_cfg_env(bot, "POSMGR_ENSURE_STOP", True))

        stop_restore_cooldown = float(_cfg_env(bot, "POSMGR_STOP_RESTORE_COOLDOWN_SEC", 90.0))
        be_stop_tolerance_pct = float(_cfg_env(bot, "POSMGR_BE_STOP_TOLERANCE_PCT", 0.0005))

        open_orders_min_interval = float(_cfg_env(bot, "POSMGR_OPEN_ORDERS_MIN_INTERVAL_SEC", 8.0))

        stop_atr_mult = float(_cfg_env(bot, "STOP_ATR_MULT", 2.0))
        be_buffer_atr_mult = float(_cfg_env(bot, "BREAKEVEN_BUFFER_ATR_MULT", 0.0))

        cb_main = float(_cfg_env(bot, "TRAILING_CALLBACK_RATE", 1.0))
        dual_trailing = _truthy(_cfg_env(bot, "DUAL_TRAILING", False))
        cb_tight = float(_cfg_env(bot, "TRAILING_TIGHT_PCT", 0.8))
        cb_loose = float(_cfg_env(bot, "TRAILING_LOOSE_PCT", 1.5))
        min_trail_qty = float(_cfg_env(bot, "POSMGR_MIN_TRAIL_QTY", 0.0))

        # ✅ ATOMIC: Get position items snapshot atomically if lock manager enabled
        if _pos_lock_mgr:
            items = await _pos_lock_mgr.atomic_items_snapshot(st, timeout=2.0)
        else:
            items = list(positions.items())

        for k, pos in items:
            try:
                if not k or pos is None:
                    continue

                k2 = _symkey(str(k))
                if k2 != k:
                    k = k2

                lk = _get_lock(k)
                if lk.locked():
                    continue

                async with lk:
                    pos2 = (getattr(st, "positions", {}) or {}).get(k)
                    if pos2 is None:
                        continue
                    pos = pos2

                    side = str(getattr(pos, "side", "") or "").lower().strip()
                    if side not in ("long", "short"):
                        continue

                    hedge_hint = _pos_side_hint(side)
                    if not hedge_hint:
                        continue

                    size = abs(_safe_float(getattr(pos, "size", 0.0), 0.0))
                    if size <= 0:
                        continue

                    entry_px = _safe_float(getattr(pos, "entry_price", 0.0), 0.0)
                    atr = _safe_float(getattr(pos, "atr", 0.0), 0.0)
                    if entry_px <= 0:
                        continue

                    sym_raw = _resolve_raw_symbol(bot, k, k)

                    px = _get_current_price(bot, k, sym_raw)
                    if px <= 0:
                        continue

                    r_dist = atr * stop_atr_mult if atr > 0 else entry_px * float(_cfg_env(bot, "GUARDIAN_STOP_FALLBACK_PCT", 0.0035))
                    if r_dist <= 0:
                        continue

                    rr = (px - entry_px) / r_dist if side == "long" else (entry_px - px) / r_dist
                    perf = _ensure_sym_perf(st, k)

                    # A stable per-position client id base
                    try:
                        ets = float(getattr(pos, "entry_ts", 0.0) or 0.0)
                    except Exception:
                        ets = 0.0
                    stop_client_id_base = _stable_stop_client_id_base(
                        k=k,
                        hedge_hint=hedge_hint,
                        entry_ts=ets,
                        entry_px=entry_px,
                        size=size,
                    )

                    # ------------------------
                    # 1) Ensure protective STOP exists
                    # ------------------------
                    if ensure_stop and isinstance(perf, dict):
                        last_chk = _safe_float(perf.get("posmgr_last_stop_check_ts", 0.0), 0.0)
                        if (_now() - last_chk) >= max(5.0, stop_check_sec):
                            perf["posmgr_last_stop_check_ts"] = _now()

                            last_place = _safe_float(perf.get("posmgr_last_stop_place_ts", 0.0), 0.0)
                            oo = await _fetch_open_orders_best_effort(
                                bot,
                                sym_raw,
                                min_interval_sec=open_orders_min_interval,
                                cache_key=k,
                            )

                            stop_id, _stop_sp = _find_existing_stop(oo, side)
                            if stop_id:
                                try:
                                    if not getattr(pos, "hard_stop_order_id", None):
                                        pos.hard_stop_order_id = str(stop_id)
                                    pos.hard_stop_order_ts = _now()
                                except Exception:
                                    pass
                            else:
                                if (_now() - last_place) < max(10.0, stop_restore_cooldown):
                                    continue
                                try:
                                    last_id = getattr(pos, "hard_stop_order_id", None)
                                    last_ts = _safe_float(getattr(pos, "hard_stop_order_ts", 0.0), 0.0)
                                    if last_id and ( _now() - last_ts ) < max(10.0, stop_restore_cooldown):
                                        continue
                                except Exception:
                                    pass

                                buffer = atr * be_buffer_atr_mult if atr > 0 else 0.0
                                stop_px = (entry_px - (r_dist - buffer)) if side == "long" else (entry_px + (r_dist - buffer))
                                if stop_px > 0:
                                    oid = await _place_stop_ladder_router(
                                        bot,
                                        sym_raw=sym_raw,
                                        side=side,
                                        qty=float(size),
                                        stop_price=float(stop_px),
                                        hedge_side_hint=hedge_hint,
                                        k=k,
                                        stop_client_id_base=stop_client_id_base,
                                    )
                                    if oid:
                                        perf["posmgr_last_stop_place_ts"] = _now()
                                        perf["posmgr_last_stop_id"] = str(oid)
                                        perf["posmgr_last_stop_id_ts"] = _now()
                                        try:
                                            pos.hard_stop_order_id = str(oid)
                                            pos.hard_stop_order_ts = _now()
                                        except Exception:
                                            pass
                                        log_core.critical(f"POSMGR: STOP RESTORED {k} ({side})")
                                        await _safe_speak(bot, f"POSMGR: PROTECTIVE STOP RESTORED {k}", "critical")

                    # ------------------------
                    # 2) Breakeven move (adopt if already BE-ish)
                    # ------------------------
                    if rr_be > 0 and rr >= rr_be and not _truthy(getattr(pos, "breakeven_moved", False)):
                        buffer = atr * be_buffer_atr_mult if atr > 0 else 0.0
                        be_px = (entry_px + buffer) if side == "long" else (entry_px - buffer)
                        if be_px <= 0:
                            continue

                        oo = await _fetch_open_orders_best_effort(
                            bot,
                            sym_raw,
                            min_interval_sec=open_orders_min_interval,
                            cache_key=k,
                        )
                        stop_id, stop_sp = _find_existing_stop(oo, side)
                        if stop_id and stop_sp > 0:
                            tol = max(1e-9, entry_px * be_stop_tolerance_pct)
                            be_ok = (stop_sp >= (be_px - tol)) if side == "long" else (stop_sp <= (be_px + tol))
                            if be_ok:
                                try:
                                    pos.hard_stop_order_id = str(stop_id)
                                    pos.breakeven_moved = True
                                except Exception:
                                    pass
                                log_core.critical(f"POSMGR: BREAKEVEN ADOPTED {k} rr={rr:.2f}")
                                continue

                        old = getattr(pos, "hard_stop_order_id", None)
                        if old:
                            ok = await _cancel_best_effort(bot, str(old), sym_raw, why="be_replace_old_stop")
                            if not ok:
                                oo2 = await _fetch_open_orders_best_effort(
                                    bot,
                                    sym_raw,
                                    min_interval_sec=open_orders_min_interval,
                                    cache_key=k,
                                )
                                sid2, _ = _find_existing_stop(oo2, side)
                                if sid2:
                                    try:
                                        pos.hard_stop_order_id = str(sid2)
                                        pos.hard_stop_order_ts = _now()
                                    except Exception:
                                        pass
                                continue

                        new_id = await _place_stop_ladder_router(
                            bot,
                            sym_raw=sym_raw,
                            side=side,
                            qty=float(size),
                            stop_price=float(be_px),
                            hedge_side_hint=hedge_hint,
                            k=k,
                            stop_client_id_base=stop_client_id_base,
                        )
                        if new_id:
                            perf2 = _ensure_sym_perf(st, k)
                            if isinstance(perf2, dict):
                                perf2["posmgr_last_stop_place_ts"] = _now()
                                perf2["posmgr_last_stop_id"] = str(new_id)
                                perf2["posmgr_last_stop_id_ts"] = _now()
                            try:
                                pos.hard_stop_order_id = str(new_id)
                                pos.hard_stop_order_ts = _now()
                                pos.breakeven_moved = True
                            except Exception:
                                pass
                            log_core.critical(f"POSMGR: BREAKEVEN ASCENDED {k} rr={rr:.2f}")
                            await _safe_speak(bot, f"BREAKEVEN ASCENDED {k} | rr={rr:.2f}", "critical")

                    # ------------------------
                    # 3) Trailing rebuild (optional)
                    # ------------------------
                    if rr_trail and rr_trail > 0 and rr >= rr_trail and isinstance(perf, dict):
                        last_tr = _safe_float(perf.get("last_trail_ts", 0.0), 0.0)
                        if (_now() - last_tr) >= max(5.0, trail_debounce):
                            perf["last_trail_ts"] = _now()

                            try:
                                tids = [str(x) for x in (perf.get("trailing_order_ids", []) or []) if x]
                                for tid in tids:
                                    await _cancel_best_effort(bot, tid, sym_raw, why="trail_rebuild_cancel")
                                perf["trailing_order_ids"] = []
                            except Exception:
                                perf["trailing_order_ids"] = []

                            activation_price = (
                                entry_px + (r_dist * rr_trail)
                                if side == "long"
                                else entry_px - (r_dist * rr_trail)
                            )
                            if activation_price <= 0:
                                continue

                            def _qty_ok(q: float) -> bool:
                                return q > 0 and (min_trail_qty <= 0 or q >= min_trail_qty)

                            try:
                                if dual_trailing:
                                    tight_amt = float(size) * 0.5
                                    loose_amt = float(size) - tight_amt

                                    if _qty_ok(tight_amt):
                                        t1 = await _place_trailing_router(
                                            bot,
                                            sym_raw=sym_raw,
                                            side=side,
                                            qty=float(tight_amt),
                                            activation_price=float(activation_price),
                                            callback_rate=float(cb_tight),
                                            hedge_side_hint=hedge_hint,
                                            k=k,
                                        )
                                        if t1:
                                            perf["trailing_order_ids"].append(str(t1))

                                    if _qty_ok(loose_amt):
                                        t2 = await _place_trailing_router(
                                            bot,
                                            sym_raw=sym_raw,
                                            side=side,
                                            qty=float(loose_amt),
                                            activation_price=float(activation_price),
                                            callback_rate=float(cb_loose),
                                            hedge_side_hint=hedge_hint,
                                            k=k,
                                        )
                                        if t2:
                                            perf["trailing_order_ids"].append(str(t2))
                                else:
                                    if _qty_ok(float(size)):
                                        t = await _place_trailing_router(
                                            bot,
                                            sym_raw=sym_raw,
                                            side=side,
                                            qty=float(size),
                                            activation_price=float(activation_price),
                                            callback_rate=float(cb_main),
                                            hedge_side_hint=hedge_hint,
                                            k=k,
                                        )
                                        if t:
                                            perf["trailing_order_ids"].append(str(t))

                                if perf.get("trailing_order_ids"):
                                    log_entry.info(f"POSMGR: TRAILING REBUILT {k} rr={rr:.2f} ids={len(perf['trailing_order_ids'])}")
                                else:
                                    log_entry.warning(f"POSMGR: trailing rebuild placed 0 orders {k} rr={rr:.2f}")
                            except Exception as e:
                                log_entry.warning(f"POSMGR trailing rebuild failed {k}: {e}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log_entry.error(f"POSMGR symbol tick failed {k}: {e}")
                continue

    except asyncio.CancelledError:
        raise
    except Exception as e:
        log_entry.error(f"POSMGR tick failed: {e}")
        return


# ----------------------------
# BOOTSTRAP LOOP ADAPTERS
# ----------------------------

async def position_manager_loop(bot) -> None:
    """
    Bootstrap entrypoint.
    Runs forever, calling position_manager_tick().
    """
    tick_sec = _safe_float(_cfg_env(bot, "POSMGR_TICK_SEC", 2.0), 2.0)
    tick_sec = max(0.5, float(tick_sec))

    log_core.info(f"POSITION_MANAGER ONLINE — tick_sec={tick_sec}")

    while True:
        try:
            await position_manager_tick(bot)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"POSMGR loop error: {e}")
        await asyncio.sleep(tick_sec)


async def run(bot) -> None:
    """
    Some bootstraps look for run().
    """
    return await position_manager_loop(bot)
