# execution/position_manager.py — SCALPER ETERNAL — POSITION MANAGER — 2026 v1.1 (ROUTER-SIGNATURE SAFE + STOP ADOPTION + HALT HOOK)
# PURPOSE:
# - Tick-based position management (runs even when no fills happen)
# - Moves stop to breakeven when RR threshold reached
# - Rebuilds trailing stops on debounce when activation RR reached
# - Ensures a protective reduceOnly STOP exists (best effort, throttled)
#
# Notes:
# - Uses execution/order_router.py for ALL create/cancel
# - Uses DataCache.raw_symbol when available
# - Stores debounces + last-checks in bot.state.symbol_performance + bot.state.run_context
# - Never raises; guardian-safe

import asyncio
import time
from typing import Optional, List

from utils.logging import log_entry, log_core
from execution.order_router import create_order, cancel_order


_POSMGR_LOCKS: dict[str, asyncio.Lock] = {}


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
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
    perf.setdefault("trailing_order_ids", [])
    perf.setdefault("last_trail_ts", 0.0)
    perf.setdefault("posmgr_last_stop_check_ts", 0.0)
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


async def _cancel_best_effort(bot, order_id: str, sym_raw: str, *, why: str = "") -> bool:
    """
    Router signature (your v1.6):
      cancel_order(bot, order_id, symbol) -> bool
    """
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


async def _place_stop_ladder_router(
    bot,
    *,
    sym_raw: str,
    side: str,          # "long"/"short"
    qty: float,
    stop_price: float,
    hedge_side_hint: Optional[str],
    k: str,
) -> Optional[str]:
    stop_side = "sell" if side == "long" else "buy"

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
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e1:
        log_entry.warning(f"{k} posmgr stop A failed: {e1}")

    # B) closePosition ladder
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

    log_entry.critical(f"{k} posmgr stop B failed (all fallbacks): {last}")
    return None


async def _place_trailing_router(
    bot,
    *,
    sym_raw: str,
    side: str,
    qty: float,
    activation_price: float,
    callback_rate: float,
    hedge_side_hint: Optional[str],
    k: str,
) -> Optional[str]:
    close_side = "sell" if side == "long" else "buy"

    # A) activation + callback
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
            callback_rate=float(callback_rate),
            hedge_side_hint=hedge_side_hint,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e1:
        log_entry.warning(f"{k} posmgr trailing A failed: {e1}")

    # B) drop callbackRate
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
            hedge_side_hint=hedge_side_hint,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e2:
        log_entry.warning(f"{k} posmgr trailing B failed: {e2}")

    return None


# ----------------------------
# Optional halt hook (guardian calls it)
# ----------------------------

async def on_halt(bot) -> None:
    """
    Optional hook invoked by guardian when kill-switch is active.
    Default policy: do nothing (posmgr continues to ensure protection).
    You can extend this later (e.g., cancel trailing, freeze upgrades, etc.)
    """
    return


# ----------------------------
# Main tick
# ----------------------------

async def position_manager_tick(bot) -> None:
    """
    Tick-based position management.
    Safe to call every guardian cycle.
    """
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return

        positions = getattr(st, "positions", None)
        if not isinstance(positions, dict) or not positions:
            return

        cfg = getattr(bot, "cfg", None)
        if cfg is None:
            return

        # Tunables (safe defaults)
        rr_be = float(_cfg(bot, "BREAKEVEN_RR_TRIGGER", 1.0))
        rr_trail = float(_cfg(bot, "TRAILING_ACTIVATION_RR", 0.0))
        trail_debounce = float(_cfg(bot, "TRAILING_REBUILD_DEBOUNCE_SEC", 15.0))
        stop_check_sec = float(_cfg(bot, "POSMGR_STOP_CHECK_SEC", 30.0))
        ensure_stop = bool(_cfg(bot, "POSMGR_ENSURE_STOP", True))

        stop_atr_mult = float(_cfg(bot, "STOP_ATR_MULT", 2.0))
        be_buffer_atr_mult = float(_cfg(bot, "BREAKEVEN_BUFFER_ATR_MULT", 0.0))
        cb_main = float(_cfg(bot, "TRAILING_CALLBACK_RATE", 1.0))
        dual_trailing = bool(_cfg(bot, "DUAL_TRAILING", False))
        cb_tight = float(_cfg(bot, "TRAILING_TIGHT_PCT", 0.8))
        cb_loose = float(_cfg(bot, "TRAILING_LOOSE_PCT", 1.5))

        # Iterate snapshot (avoid runtime mutation issues)
        items = list(positions.items())

        for k, pos in items:
            try:
                if not k or pos is None:
                    continue
                k2 = _symkey(str(k))
                if k2 != k:
                    k = k2  # don't rename state; just operate on canonical key

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

                    # Risk distance proxy (R): ATR * STOP_ATR_MULT (fallback: entry * 0.35%)
                    r_dist = atr * stop_atr_mult if atr > 0 else entry_px * float(_cfg(bot, "GUARDIAN_STOP_FALLBACK_PCT", 0.0035))
                    if r_dist <= 0:
                        continue

                    # Current RR
                    if side == "long":
                        rr = (px - entry_px) / r_dist
                    else:
                        rr = (entry_px - px) / r_dist

                    # Track MFE
                    perf = _ensure_sym_perf(st, k)
                    if isinstance(perf, dict):
                        mfe_pct = float(perf.get("mfe_pct", 0.0) or 0.0)
                        favorable = (px - entry_px) / entry_px if side == "long" else (entry_px - px) / entry_px
                        perf["mfe_pct"] = max(mfe_pct, favorable * 100.0)

                        if _safe_float(perf.get("entry_size_abs", 0.0), 0.0) <= 0:
                            perf["entry_size_abs"] = float(size)

                    # --------
                    # 1) Ensure protective STOP exists (throttled)
                    # --------
                    if ensure_stop and isinstance(perf, dict):
                        last_chk = _safe_float(perf.get("posmgr_last_stop_check_ts", 0.0), 0.0)
                        if (_now() - last_chk) >= max(5.0, stop_check_sec):
                            perf["posmgr_last_stop_check_ts"] = _now()

                            try:
                                oo = await _fetch_open_orders_best_effort(bot, sym_raw)

                                have_stop = False
                                found_stop_id: Optional[str] = None

                                for o in oo:
                                    if isinstance(o, dict) and _is_reduce_only(o) and _is_stop_like(o):
                                        have_stop = True
                                        found_stop_id = _extract_order_id(o)
                                        break

                                # Adopt existing stop id so later BE replace can cancel it
                                if have_stop and found_stop_id:
                                    try:
                                        if not getattr(pos, "hard_stop_order_id", None):
                                            pos.hard_stop_order_id = str(found_stop_id)
                                    except Exception:
                                        pass

                                if not have_stop:
                                    # place a conservative stop based on current entry + R distance
                                    be_buffer = atr * be_buffer_atr_mult if atr > 0 else 0.0
                                    stop_px = (entry_px - (r_dist - be_buffer)) if side == "long" else (entry_px + (r_dist - be_buffer))
                                    if stop_px > 0:
                                        oid = await _place_stop_ladder_router(
                                            bot,
                                            sym_raw=sym_raw,
                                            side=side,
                                            qty=float(size),
                                            stop_price=float(stop_px),
                                            hedge_side_hint=side,
                                            k=k,
                                        )
                                        if oid:
                                            try:
                                                pos.hard_stop_order_id = oid
                                            except Exception:
                                                pass
                                            log_core.critical(f"POSMGR: STOP RESTORED {k} ({side})")
                                            await _safe_speak(bot, f"POSMGR: PROTECTIVE STOP RESTORED {k}", "critical")
                            except Exception as e:
                                log_entry.warning(f"POSMGR stop check failed {k}: {e}")

                    # --------
                    # 2) Breakeven move (without requiring a partial fill)
                    # --------
                    if rr_be > 0 and rr >= rr_be and not _truthy(getattr(pos, "breakeven_moved", False)):
                        try:
                            buffer = atr * be_buffer_atr_mult if atr > 0 else 0.0
                            be_px = (entry_px + buffer) if side == "long" else (entry_px - buffer)
                            if be_px > 0:
                                old = getattr(pos, "hard_stop_order_id", None)
                                if old:
                                    await _cancel_best_effort(bot, str(old), sym_raw, why="be_replace_old_stop")

                                new_id = await _place_stop_ladder_router(
                                    bot,
                                    sym_raw=sym_raw,
                                    side=side,
                                    qty=float(size),
                                    stop_price=float(be_px),
                                    hedge_side_hint=side,
                                    k=k,
                                )
                                if new_id:
                                    try:
                                        pos.hard_stop_order_id = new_id
                                        pos.breakeven_moved = True
                                    except Exception:
                                        pass
                                    log_core.critical(f"POSMGR: BREAKEVEN ASCENDED {k} rr={rr:.2f}")
                                    await _safe_speak(bot, f"BREAKEVEN ASCENDED {k} | rr={rr:.2f}", "critical")
                        except Exception as e:
                            log_entry.warning(f"POSMGR breakeven failed {k}: {e}")

                    # --------
                    # 3) Trailing rebuild (without requiring a partial fill)
                    # --------
                    if rr_trail and rr_trail > 0 and rr >= rr_trail and isinstance(perf, dict):
                        last_tr = _safe_float(perf.get("last_trail_ts", 0.0), 0.0)
                        if (_now() - last_tr) >= max(5.0, trail_debounce):
                            perf["last_trail_ts"] = _now()

                            try:
                                # cancel prior trailing ids (best effort)
                                tids = [str(x) for x in (perf.get("trailing_order_ids", []) or []) if x]
                                for tid in tids:
                                    await _cancel_best_effort(bot, tid, sym_raw, why="trail_rebuild_cancel")
                                perf["trailing_order_ids"] = []

                                # activation price based on RR trigger distance
                                activation_price = (
                                    entry_px + (r_dist * rr_trail)
                                    if side == "long"
                                    else entry_px - (r_dist * rr_trail)
                                )
                                if activation_price <= 0:
                                    continue

                                # Guard: don't place trailing for dust
                                min_qty = float(_cfg(bot, "POSMGR_MIN_TRAIL_QTY", 0.0))
                                def _qty_ok(q: float) -> bool:
                                    return q > 0 and (min_qty <= 0 or q >= min_qty)

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
                                            hedge_side_hint=side,
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
                                            hedge_side_hint=side,
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
                                            hedge_side_hint=side,
                                            k=k,
                                        )
                                        if t:
                                            perf["trailing_order_ids"].append(str(t))

                                if perf.get("trailing_order_ids"):
                                    log_entry.info(
                                        f"POSMGR: TRAILING REBUILT {k} rr={rr:.2f} ids={len(perf['trailing_order_ids'])}"
                                    )
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
