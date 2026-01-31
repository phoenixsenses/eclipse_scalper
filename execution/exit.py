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
import os
from typing import Optional

from utils.logging import log_entry, log
from brain.persistence import save_brain

from execution.order_router import create_order, cancel_order  # ✅ ROUTER


_EXIT_LOCKS: dict[str, asyncio.Lock] = {}


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
        pass


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
                        else:
                            log_entry.error(f"Breakeven stop placement failed (all fallbacks) {k}")
                    except Exception as e:
                        log_entry.error(f"Breakeven move failed {k}: {e}")

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
                except Exception as e:
                    log_entry.error(f"Velocity exit failed {k}: {e}")

        # Full close
        if is_full_close:
            full_pnl = float(perf.get("pos_realized_pnl", 0.0))

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

            try:
                bot.state.positions.pop(k, None)
            except Exception:
                pass

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

    log.info(f"EXIT LOOP ONLINE — tick_sec={tick_sec}")

    while True:
        try:
            # Intentionally lightweight; handle_exit() is the real worker.
            st = getattr(bot, "state", None)
            pos_map = getattr(st, "positions", None) if st is not None else None
            if isinstance(pos_map, dict) and pos_map:
                pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"exit_loop error: {e}")

        await asyncio.sleep(tick_sec)


async def run(bot) -> None:
    return await exit_loop(bot)
