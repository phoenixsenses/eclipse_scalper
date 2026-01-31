# execution/entry.py — SCALPER ETERNAL — THE BLADE ASCENDANT — 2026 v4.8c (STOP DUPLICATE FIX + ADOPT EXISTING)
# Patch vs v4.8b:
# - ✅ FIX: STOP orders use position-stable clientOrderId (per-position, not per-call)
# - ✅ FIX: If Binance returns -4116 (duplicate clientOrderId), treat as "already placed" and ADOPT existing open order id
# - ✅ HARDEN: stop A/B use different clientOrderId suffixes (prevents internal collisions)
# - ✅ Keeps: TP hedge-safe hints, router v2.2 closePosition behavior, safety flatten, skip throttling

import asyncio
import time
import numpy as np
from typing import Optional

from utils.logging import log_entry, log, log_risk
from strategies.eclipse_scalper import scalper_signal as generate_signal
from strategies.risk import portfolio_heat
from brain.state import Position
from brain.persistence import save_brain
from ta.volatility import AverageTrueRange

from execution.order_router import create_order, cancel_order
from risk.kill_switch import trade_allowed

try:
    from execution.entry_watch import register_entry_watch  # type: ignore
except Exception:
    register_entry_watch = None


_SYMBOL_LOCKS: dict[str, asyncio.Lock] = {}

# Skip logging throttle (prevents spam)
_SKIP_LAST: dict[str, float] = {}


def _symkey(sym: str) -> str:
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _get_symbol_lock(k: str) -> asyncio.Lock:
    lock = _SYMBOL_LOCKS.get(k)
    if lock is None:
        lock = asyncio.Lock()
        _SYMBOL_LOCKS[k] = lock
    return lock


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


async def _safe_speak(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _resolve_raw_symbol(bot, k: str, sym_fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict):
            v = raw_map.get(k)
            if v:
                return str(v)
    except Exception:
        pass
    return sym_fallback


def _hedge_enabled(bot) -> bool:
    """
    Best-effort hedge mode detection:
    - If exchange wrapper exposes hedge_mode or position_mode
    - Else assume hedge is ON because core tries to enable it.
    """
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
    return "LONG" if str(side).lower().strip() == "long" else "SHORT"


async def _ensure_markets_loaded(bot) -> None:
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return

        mk = getattr(ex, "markets", None)
        if isinstance(mk, dict) and len(mk) > 0:
            return

        fm = getattr(ex, "fetch_markets", None)
        if callable(fm):
            await fm()
            return

        lm = getattr(ex, "load_markets", None)
        if callable(lm):
            await lm()
    except Exception:
        pass


async def _get_min_amount(bot, sym_raw: str, k: str) -> float:
    try:
        await _ensure_markets_loaded(bot)
        ex = getattr(bot, "ex", None)
        markets = getattr(ex, "markets", None)
        if isinstance(markets, dict) and sym_raw in markets:
            lim = (markets[sym_raw] or {}).get("limits") or {}
            amt = (lim.get("amount") or {}).get("min")
            m = _safe_float(amt, 0.0)
            if m > 0:
                return m
    except Exception:
        pass

    try:
        m2 = _safe_float(getattr(bot, "min_amounts", {}).get(k, 0.0), 0.0)
        if m2 > 0:
            return m2
    except Exception:
        pass

    return 0.0


def _order_filled(order: dict) -> float:
    if not isinstance(order, dict):
        return 0.0

    filled = _safe_float(order.get("filled", None), default=0.0)
    if filled > 0:
        return filled

    info = order.get("info") or {}
    filled2 = _safe_float(info.get("executedQty", None), default=0.0)
    if filled2 > 0:
        return filled2

    return 0.0


def _order_avg_price(order: dict, fallback: float) -> float:
    if not isinstance(order, dict):
        return fallback

    avg = _safe_float(order.get("average", None), default=0.0)
    if avg > 0:
        return avg

    price = _safe_float(order.get("price", None), default=0.0)
    if price > 0:
        return price

    info = order.get("info") or {}
    avg2 = _safe_float(info.get("avgPrice", None), default=0.0)
    if avg2 > 0:
        return avg2

    quote = _safe_float(info.get("cummulativeQuoteQty", None), default=0.0)
    qty = _safe_float(info.get("executedQty", None), default=0.0)
    if quote > 0 and qty > 0:
        return quote / qty

    return fallback


def _skip(bot, k: str, side: str, reason: str):
    try:
        throttle = float(getattr(getattr(bot, "cfg", None), "SKIP_LOG_THROTTLE_SEC", 12.0) or 12.0)
        if throttle < 0:
            throttle = 0.0
    except Exception:
        throttle = 12.0

    now = time.time()
    key = f"{k}|{side}|{reason}"
    last = float(_SKIP_LAST.get(key, 0.0) or 0.0)

    if throttle == 0.0 or (now - last) >= throttle:
        _SKIP_LAST[key] = now
        log_entry.info(f"SKIP {k} {side.upper()} — {reason}")


def _parse_funding_rate_any(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return _safe_float(x, 0.0)
    if isinstance(x, dict):
        if "fundingRate" in x:
            return _safe_float(x.get("fundingRate"), 0.0)
        if "rate" in x:
            return _safe_float(x.get("rate"), 0.0)
        info = x.get("info") or {}
        if isinstance(info, dict):
            for kk in ("fundingRate", "lastFundingRate", "rate"):
                if kk in info:
                    return _safe_float(info.get(kk), 0.0)
    return 0.0


def _record_last_entry_attempt(bot, k: str) -> None:
    try:
        rc = getattr(bot.state, "run_context", None)
        if not isinstance(rc, dict):
            bot.state.run_context = {}
            rc = bot.state.run_context
        m = rc.get("last_entry_attempt")
        if not isinstance(m, dict):
            m = {}
            rc["last_entry_attempt"] = m
        m[k] = float(time.time())
    except Exception:
        pass


def _record_pending_limit_entry(bot, k: str, payload: dict) -> None:
    try:
        rc = getattr(bot.state, "run_context", None)
        if not isinstance(rc, dict):
            bot.state.run_context = {}
            rc = bot.state.run_context
        pe = rc.get("pending_limit_entries")
        if not isinstance(pe, dict):
            pe = {}
            rc["pending_limit_entries"] = pe
        pe[k] = dict(payload)
    except Exception:
        pass


async def _get_fresh_last(bot, sym_raw: str, fallback: float) -> float:
    try:
        if hasattr(bot.ex, "fetch_ticker"):
            t = await bot.ex.fetch_ticker(sym_raw)
            last = _safe_float((t or {}).get("last", None), default=0.0)
            if last > 0:
                return last
    except Exception:
        pass
    return fallback


async def _refresh_equity_if_needed(bot) -> float:
    """
    Best-effort equity refresh from exchange if state is empty.
    Returns current_equity (may be 0.0 if unavailable).
    """
    try:
        cur = float(getattr(bot.state, "current_equity", 0.0) or 0.0)
        sod = float(getattr(bot.state, "start_of_day_equity", 0.0) or 0.0)
        if cur > 0 or sod > 0:
            return cur or sod
    except Exception:
        pass

    ex = getattr(bot, "ex", None)
    if ex is None:
        return 0.0
    try:
        fn = getattr(ex, "fetch_balance", None)
        if not callable(fn):
            return 0.0
        bal = await fn()
        total = (bal or {}).get("total") or {}
        for key in ("USDT", "USD"):
            v = total.get(key)
            eq = _safe_float(v, 0.0)
            if eq > 0:
                bot.state.current_equity = eq
                if _safe_float(getattr(bot.state, "start_of_day_equity", 0.0), 0.0) <= 0:
                    bot.state.start_of_day_equity = eq
                return eq
        # fallback: try info->totalWalletBalance
        info = (bal or {}).get("info") or {}
        eq = _safe_float(info.get("totalWalletBalance") or info.get("totalWalletBalanceUSD"), 0.0)
        if eq > 0:
            bot.state.current_equity = eq
            if _safe_float(getattr(bot.state, "start_of_day_equity", 0.0), 0.0) <= 0:
                bot.state.start_of_day_equity = eq
            return eq
    except Exception:
        pass
    return 0.0


async def _get_funding_rate_safely(bot, k: str, sym_raw: str) -> float:
    cached = 0.0
    try:
        cached = float(getattr(bot.data, "funding", {}).get(k, 0.0) or 0.0)
    except Exception:
        cached = 0.0

    max_stale = float(getattr(bot.cfg, "FUNDING_MAX_STALE_SEC", 120.0))
    age_ok = True
    try:
        last_ts = float(getattr(bot.data, "last_poll", {}).get(k, 0.0) or 0.0)
        age_ok = (time.time() - last_ts) <= max_stale
    except Exception:
        age_ok = True

    if cached != 0.0 and age_ok:
        return cached

    try:
        if hasattr(bot.ex, "fetch_funding_rate"):
            live = await bot.ex.fetch_funding_rate(sym_raw)
            return _parse_funding_rate_any(live)
    except Exception:
        pass

    return cached


def _looks_like_dup_client_order_id(err: Exception) -> bool:
    s = repr(err)
    # Binance futures duplicate clientOrderId => code -4116
    return ("-4116" in s) or ("ClientOrderId is duplicated" in s) or ("clientorderid is duplicated" in s.lower())


async def _fetch_open_orders_safe(bot, sym_raw: str) -> list:
    try:
        fn = getattr(bot.ex, "fetch_open_orders", None)
        if callable(fn):
            res = await fn(sym_raw)
            return res if isinstance(res, list) else []
    except Exception:
        pass
    return []


def _order_client_id_any(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    info = order.get("info") or {}
    for k in ("clientOrderId", "clientOrderID", "origClientOrderId", "origClientOrderID"):
        v = order.get(k) or info.get(k)
        if v:
            return str(v)
    return ""


async def _adopt_open_order_id_by_client_id(bot, sym_raw: str, client_id: str) -> Optional[str]:
    if not client_id:
        return None
    oo = await _fetch_open_orders_safe(bot, sym_raw)
    for o in oo:
        try:
            if _order_client_id_any(o) == str(client_id):
                oid = (o or {}).get("id") or ((o or {}).get("info") or {}).get("orderId")
                return str(oid) if oid else None
        except Exception:
            continue
    return None


async def _emergency_flatten(bot, sym_raw: str, side: str, qty: float, why: str, k: str, pos_side: Optional[str] = None):
    for attempt in range(2):
        try:
            await create_order(
                bot,
                symbol=sym_raw,
                type="MARKET",
                side="sell" if side == "long" else "buy",
                amount=float(qty),
                price=None,
                params=({"positionSide": pos_side} if pos_side else {}),
                intent_reduce_only=True,
                hedge_side_hint=pos_side,
                retries=6,
            )
            return True
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(0.15)
                continue
            log_entry.critical(f"EMERGENCY FLATTEN FAILED {k}: {why} | {e}")

    try:
        ttl = float(getattr(bot.cfg, "BLACKLIST_ON_EMERGENCY_SEC", 600))
        bot.state.blacklist[k] = time.time() + ttl
        bot.state.blacklist_reason[k] = f"emergency_flat_fail: {why}"
    except Exception:
        pass

    await _safe_speak(bot, f"EMERGENCY FLATTEN FAILED {k}\n{why}", "critical")
    return False


async def _place_stop_ladder(
    bot,
    *,
    sym_raw: str,
    side: str,
    qty: float,
    stop_price: float,
    k: str,
    pos_side: Optional[str],
    stop_client_id_base: str,
):
    stop_side = "sell" if side == "long" else "buy"

    base_params = {}
    if pos_side:
        base_params["positionSide"] = pos_side

    # Make client IDs stable per position, different for A/B
    cid_a = f"{stop_client_id_base}_A"
    cid_b = f"{stop_client_id_base}_B"

    # A) amount + reduceOnly (standard protective stop)
    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=float(qty),
            price=None,
            params=dict(base_params),
            intent_reduce_only=True,
            intent_close_position=False,
            stop_price=float(stop_price),
            hedge_side_hint=pos_side,
            client_order_id=cid_a,  # ✅ stable id
            retries=6,
        )
        if isinstance(order, dict) and order.get("id"):
            return order.get("id")
    except Exception as e1:
        # ✅ If duplicate clientOrderId, adopt existing open order id
        if _looks_like_dup_client_order_id(e1):
            adopted = await _adopt_open_order_id_by_client_id(bot, sym_raw, cid_a)
            if adopted:
                log_entry.warning(f"{k} stop A duplicate → ADOPTED open order id={adopted}")
                return adopted
        log_entry.warning(f"{k} stop A failed: {e1}")

    # B) closePosition stop (router will force amount=0.0 anyway)
    try:
        p = dict(base_params)
        p["closePosition"] = True

        order = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=None,
            price=None,
            params=p,
            intent_reduce_only=False,
            intent_close_position=True,
            stop_price=float(stop_price),
            hedge_side_hint=pos_side,
            client_order_id=cid_b,  # ✅ stable id, different from A
            retries=6,
        )
        if isinstance(order, dict) and order.get("id"):
            return order.get("id")
    except Exception as e2:
        if _looks_like_dup_client_order_id(e2):
            adopted = await _adopt_open_order_id_by_client_id(bot, sym_raw, cid_b)
            if adopted:
                log_entry.warning(f"{k} stop B duplicate → ADOPTED open order id={adopted}")
                return adopted
        log_entry.warning(f"{k} stop B failed: {e2}")

    log_entry.critical(f"{k} stop failed (all fallbacks exhausted)")
    return None


async def _place_trailing_ladder(
    bot,
    *,
    sym_raw: str,
    side: str,
    qty: float,
    activation_price: float,
    callback_rate: float,
    k: str,
    pos_side: Optional[str],
):
    tside = "sell" if side == "long" else "buy"
    cb = _clamp(float(callback_rate), 0.1, 5.0)

    base_params = {}
    if pos_side:
        base_params["positionSide"] = pos_side

    # A) activation + callback + reduceOnly
    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="TRAILING_STOP_MARKET",
            side=tside,
            amount=float(qty),
            price=None,
            params=dict(base_params),
            intent_reduce_only=True,
            activation_price=float(activation_price),
            callback_rate=float(cb),
            hedge_side_hint=pos_side,
            retries=6,
        )
        return bool(order)
    except Exception as e1:
        log_entry.warning(f"{k} trailing A failed: {e1}")

    # B) drop callbackRate
    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="TRAILING_STOP_MARKET",
            side=tside,
            amount=float(qty),
            price=None,
            params=dict(base_params),
            intent_reduce_only=True,
            activation_price=float(activation_price),
            hedge_side_hint=pos_side,
            retries=6,
        )
        return bool(order)
    except Exception as e2:
        log_entry.warning(f"{k} trailing B failed: {e2}")

    return False


async def try_enter(bot, sym: str, side: str):
    cfg = bot.cfg

    k = _symkey(sym)
    sym_raw = _resolve_raw_symbol(bot, k, sym)

    if k not in getattr(bot, "active_symbols", set()):
        return

    lock = _get_symbol_lock(k)
    if lock.locked():
        _skip(bot, k, side, "symbol lock busy")
        return

    async with lock:
        if k not in getattr(bot, "active_symbols", set()):
            return

        try:
            if not await trade_allowed(bot):
                _skip(bot, k, side, "kill_switch halted")
                return
        except Exception as e:
            log_entry.warning(f"{k} trade_allowed() error — allowing by exception: {e}")

        _record_last_entry_attempt(bot, k)

        if bot.state.blacklist.get(k, 0) > time.time():
            _skip(bot, k, side, "blacklisted")
            return

        log_entry.info(f"SCALPER SCAN → {k} {side.upper()}")

        if k in bot.state.positions:
            _skip(bot, k, side, "already in position")
            return

        last_exit = bot.state.last_exit_time.get(k, 0)
        if time.time() - last_exit < cfg.SYMBOL_COOLDOWN_MINUTES * 60:
            _skip(bot, k, side, "cooldown")
            return

        equity = float(bot.state.current_equity or bot.state.start_of_day_equity or 0.0)
        if equity <= 0:
            equity = float(await _refresh_equity_if_needed(bot) or 0.0)
        if equity <= 0:
            log_entry.warning(f"{k} equity unknown/zero — trading blocked")
            return

        current_heat = portfolio_heat(bot.state.positions, equity)

        if len(bot.state.positions) >= cfg.MAX_CONCURRENT_POSITIONS:
            _skip(bot, k, side, f"max positions {cfg.MAX_CONCURRENT_POSITIONS} reached")
            return

        if current_heat > cfg.MAX_PORTFOLIO_HEAT:
            _skip(bot, k, side, f"portfolio heat {current_heat:.1%} > {cfg.MAX_PORTFOLIO_HEAT:.1%}")
            return

        session_peak = float(getattr(bot, "session_peak_equity", 0.0) or 0.0)
        session_dd = (session_peak - equity) / session_peak if session_peak > 0 else 0.0
        if session_dd > cfg.SESSION_EQUITY_PEAK_PROTECTION_PCT:
            _skip(bot, k, side, f"session drawdown {session_dd:.1%} > {cfg.SESSION_EQUITY_PEAK_PROTECTION_PCT:.1%}")
            return

        sod = float(bot.state.start_of_day_equity or equity)
        if bot.state.daily_pnl < -cfg.MAX_DAILY_LOSS_PCT * sod:
            _skip(bot, k, side, "daily loss limit exceeded")
            return

        long_sig, short_sig, confidence = generate_signal(sym_raw, data=bot.data, cfg=cfg)
        confidence = float(confidence or 0.0)

        if confidence < cfg.MIN_CONFIDENCE:
            _skip(bot, k, side, f"confidence {confidence:.2f} < {cfg.MIN_CONFIDENCE:.2f}")
            return

        if (side == "long" and not long_sig) or (side == "short" and not short_sig):
            _skip(bot, k, side, "signal not present")
            return

        tf = getattr(cfg, "TIMEFRAME", "1m")
        df = bot.data.get_df(k, tf)
        if df is None or getattr(df, "empty", True):
            _skip(bot, k, side, "no dataframe")
            return
        if len(df) < 200:
            _skip(bot, k, side, f"insufficient bars {len(df)} < 200")
            return

        try:
            close_series = df["c"]
            high = df["h"]
            low = df["l"]
        except Exception:
            _skip(bot, k, side, "missing OHLC columns")
            return

        close_df = _safe_float(close_series.iloc[-1], default=0.0)
        if close_df <= 0:
            _skip(bot, k, side, "close <= 0")
            return

        atr = AverageTrueRange(high, low, close_series, window=14).average_true_range().iloc[-1]
        atr = _safe_float(atr, default=0.0)
        if atr <= 0:
            _skip(bot, k, side, "atr <= 0")
            return

        atr_pct = atr / close_df
        if atr_pct < cfg.MIN_ATR_PCT_FOR_ENTRY:
            _skip(bot, k, side, f"atr_pct {atr_pct:.3%} < {cfg.MIN_ATR_PCT_FOR_ENTRY:.3%}")
            return

        funding_rate = float(await _get_funding_rate_safely(bot, k, sym_raw) or 0.0)
        if (side == "long" and funding_rate > cfg.MAX_FUNDING_LONG) or (
            side == "short" and funding_rate < cfg.MIN_FUNDING_SHORT
        ):
            _skip(bot, k, side, f"funding filter ({funding_rate:+.5f})")
            return

        use_fresh = bool(getattr(cfg, "SLIPPAGE_USE_FRESH_TICKER", True))
        ref_px = await _get_fresh_last(bot, sym_raw, close_df) if use_fresh else close_df
        if ref_px <= 0:
            _skip(bot, k, side, "ref_px <= 0")
            return

        max_stop_pct = float(cfg.MAX_STOP_PCT)
        if confidence > 0.9:
            max_stop_pct *= 1.2

        base_risk = float(cfg.MAX_RISK_PER_TRADE)

        if cfg.ADAPTIVE_RISK_SCALING:
            streak_factor = 1.0 + (bot.state.win_streak * 0.05) if bot.state.win_streak > 0 else 1.0
            dd_factor = 1.0 - (session_dd * 2.0) if session_dd > 0 else 1.0
            scaled_risk = base_risk * streak_factor * dd_factor
        else:
            scaled_risk = base_risk

        if cfg.CONFIDENCE_SCALING:
            confidence_multiplier = 0.8 + float(confidence) * 0.4
            scaled_risk *= confidence_multiplier

        risk_amount = equity * scaled_risk
        available_risk = equity * max(0.0, (cfg.MAX_PORTFOLIO_HEAT - current_heat))
        risk_amount = min(risk_amount, available_risk)

        if risk_amount < cfg.MIN_RISK_DOLLARS:
            _skip(bot, k, side, f"risk_amount ${risk_amount:.2f} < MIN_RISK_DOLLARS ${cfg.MIN_RISK_DOLLARS:.2f}")
            return

        stop_distance = atr * float(cfg.STOP_ATR_MULT)
        stop_pct = stop_distance / close_df

        if stop_pct <= 0 or np.isnan(stop_pct) or stop_pct > 0.25:
            _skip(bot, k, side, f"stop_pct invalid ({stop_pct})")
            return
        if stop_pct > max_stop_pct:
            _skip(bot, k, side, f"stop_pct {stop_pct:.2%} > max_stop_pct {max_stop_pct:.2%}")
            return

        notional = risk_amount / stop_pct
        amount = notional / ref_px

        min_notional = float(getattr(cfg, "MIN_NOTIONAL_USDT", 0.0) or 0.0)
        if min_notional > 0 and notional < min_notional:
            notional = min_notional
            amount = notional / ref_px

        min_margin = float(getattr(cfg, "MIN_MARGIN_USDT", 0.0) or 0.0)
        if min_margin > 0:
            est_margin = notional / float(cfg.LEVERAGE)
            if est_margin < min_margin:
                notional = min_margin * float(cfg.LEVERAGE)
                amount = notional / ref_px

        min_amount_val = float(await _get_min_amount(bot, sym_raw, k) or 0.0)
        original_amount = float(amount)
        amount = max(float(amount), float(min_amount_val or 0.0))

        effective_notional = float(amount) * ref_px
        effective_risk_amount = effective_notional * stop_pct
        effective_heat_add = effective_risk_amount / equity

        if amount > original_amount:
            log_entry.info(f"{k} min qty enforced — amount {original_amount:.6f} → {amount:.6f}")

        if effective_risk_amount > risk_amount * 1.8:
            _skip(bot, k, side, f"min bump inflates risk ${effective_risk_amount:.2f} vs planned ${risk_amount:.2f}")
            return

        if current_heat + effective_heat_add > cfg.MAX_PORTFOLIO_HEAT:
            _skip(bot, k, side, f"would exceed heat {(current_heat + effective_heat_add):.1%}")
            return

        risk_amount = effective_risk_amount
        notional = effective_notional
        scaled_risk = risk_amount / equity

        log_risk.critical(
            f"SCALPER {k} | Risk {scaled_risk:.2%} (${risk_amount:.2f}) | "
            f"Notional ${notional:.2f} | Amount {amount:.6f} | Conf {confidence:.2f} | Funding {funding_rate:+.5f}"
        )

        pos_side = _position_side_for_trade(side) if _hedge_enabled(bot) else None

        try:
            log_entry.critical(f"THE BLADE STRIKES → {side.upper()} {k}")

            entry_type = str(getattr(cfg, "ENTRY_ORDER_TYPE", "MARKET") or "MARKET").upper()
            entry_side = "buy" if side == "long" else "sell"

            if entry_type == "LIMIT":
                off = float(getattr(cfg, "ENTRY_LIMIT_OFFSET_PCT", 0.0) or 0.0)
                limit_px = ref_px * (1.0 - off) if side == "long" else ref_px * (1.0 + off)

                p = {"timeInForce": "GTC"}
                if pos_side:
                    p["positionSide"] = pos_side

                order = await create_order(
                    bot,
                    symbol=sym_raw,
                    type="LIMIT",
                    side=entry_side,
                    amount=float(amount),
                    price=float(limit_px),
                    params=p,
                    intent_reduce_only=False,
                    retries=6,
                )

                oid = (order or {}).get("id") if isinstance(order, dict) else None

                _record_pending_limit_entry(
                    bot,
                    k,
                    {
                        "order_id": str(oid) if oid else None,
                        "sym_raw": sym_raw,
                        "side": side,
                        "requested_qty": float(amount),
                        "limit_px": float(limit_px),
                        "created_ts": float(time.time()),
                        "positionSide": pos_side,
                    },
                )

                if oid and callable(register_entry_watch):
                    try:
                        await register_entry_watch(
                            bot,
                            symbol=sym_raw,
                            order_id=str(oid),
                            side=side,
                            requested_qty=float(amount),
                            created_ts=time.time(),
                            replace_price=float(limit_px),
                        )
                        log_entry.info(f"ENTRY WATCH ARMED → {k} {side.upper()} oid={oid}")
                    except Exception:
                        pass

                log_entry.info(f"LIMIT ENTRY PLACED → {k} {side.upper()} qty={amount:.6f} px={limit_px:.6f}")
                return

            p_entry = {}
            if pos_side:
                p_entry["positionSide"] = pos_side

            order = await create_order(
                bot,
                symbol=sym_raw,
                type="MARKET",
                side=entry_side,
                amount=float(amount),
                price=None,
                params=p_entry,
                intent_reduce_only=False,
                retries=6,
            )

            filled = _order_filled(order)
            if filled <= 0:
                _skip(bot, k, side, "order not filled")
                return

            req_amt = max(float(amount), 1e-12)
            if (filled / req_amt) < cfg.MIN_FILL_RATIO:
                _skip(bot, k, side, f"partial fill {filled/req_amt:.1%} < {cfg.MIN_FILL_RATIO:.1%} — flatten")
                ok = await _emergency_flatten(
                    bot, sym_raw, side, float(filled), "Partial fill below MIN_FILL_RATIO", k, pos_side=pos_side
                )
                if not ok:
                    _skip(bot, k, side, "partial flatten failed (blacklisted)")
                return

            entry_price = _order_avg_price(order, ref_px)
            log_entry.info(f"FILLED: {filled:.6f} @ {entry_price:.6f}")

            slippage_pct = abs(entry_price - ref_px) / ref_px if ref_px > 0 else 0.0
            if slippage_pct > cfg.SLIPPAGE_MAX_PCT:
                _skip(bot, k, side, f"slippage {slippage_pct:.2%} > {cfg.SLIPPAGE_MAX_PCT:.2%} — flatten")
                await _emergency_flatten(bot, sym_raw, side, float(filled), "Slippage exceeded", k, pos_side=pos_side)
                return

            pos = Position(
                side=side,
                size=abs(float(filled)),
                entry_price=float(entry_price),
                atr=float(atr),
                leverage=int(cfg.LEVERAGE),
                entry_ts=time.time(),
                confidence=float(confidence),
            )
            bot.state.positions[k] = pos

            stop_price = entry_price * (1.0 - stop_pct) if side == "long" else entry_price * (1.0 + stop_pct)

            # ✅ position-stable STOP client id (same across retries for this position)
            entry_ts_ms = int(pos.entry_ts * 1000)
            stop_client_id_base = f"SE_STOP_{k}_{pos_side or 'ONEWAY'}_{entry_ts_ms}"

            stop_id = await _place_stop_ladder(
                bot,
                sym_raw=sym_raw,
                side=side,
                qty=float(filled),
                stop_price=float(stop_price),
                k=k,
                pos_side=pos_side,
                stop_client_id_base=stop_client_id_base,
            )
            pos.hard_stop_order_id = stop_id

            if stop_id is None:
                bot.state.positions.pop(k, None)
                await _emergency_flatten(bot, sym_raw, side, float(filled), "STOP placement failed", k, pos_side=pos_side)
                return

            tp1_pct = stop_pct * float(cfg.TP1_RR_MULT)
            tp2_pct = stop_pct * float(cfg.TP2_RR_MULT)

            tp1_price = entry_price * (1.0 + tp1_pct) if side == "long" else entry_price * (1.0 - tp1_pct)
            tp2_price = entry_price * (1.0 + tp2_pct) if side == "long" else entry_price * (1.0 - tp2_pct)

            min_amt = float(min_amount_val or 0.0)
            tp1_amount = float(filled) * 0.5
            tp2_amount = float(filled) * 0.3

            close_side = "sell" if side == "long" else "buy"

            tp_params = {"timeInForce": "GTC"}
            if pos_side:
                tp_params["positionSide"] = pos_side

            if tp1_amount > 0 and (min_amt <= 0 or tp1_amount >= min_amt):
                await create_order(
                    bot,
                    symbol=sym_raw,
                    type="LIMIT",
                    side=close_side,
                    amount=float(tp1_amount),
                    price=float(tp1_price),
                    params=dict(tp_params),
                    intent_reduce_only=True,
                    hedge_side_hint=pos_side,
                    retries=6,
                )
            else:
                log_entry.warning(f"{k} TP1 skipped (amount {tp1_amount:.6f} < min {min_amt:.6f})")

            if tp2_amount > 0 and (min_amt <= 0 or tp2_amount >= min_amt):
                await create_order(
                    bot,
                    symbol=sym_raw,
                    type="LIMIT",
                    side=close_side,
                    amount=float(tp2_amount),
                    price=float(tp2_price),
                    params=dict(tp_params),
                    intent_reduce_only=True,
                    hedge_side_hint=pos_side,
                    retries=6,
                )
            else:
                log_entry.warning(f"{k} TP2 skipped (amount {tp2_amount:.6f} < min {min_amt:.6f})")

            remaining = float(filled) * 0.2
            if remaining > 0 and (cfg.DYNAMIC_TRAILING_FULL or cfg.DUAL_TRAILING):
                if min_amt <= 0 or remaining >= min_amt:
                    trail_activation_pct = stop_pct * float(cfg.TRAILING_ACTIVATION_RR)
                    activation_price = (
                        entry_price * (1.0 + trail_activation_pct)
                        if side == "long"
                        else entry_price * (1.0 - trail_activation_pct)
                    )

                    cb = _safe_float(getattr(cfg, "TRAILING_CALLBACK_RATE", 1), default=1.0)
                    ok = await _place_trailing_ladder(
                        bot,
                        sym_raw=sym_raw,
                        side=side,
                        qty=float(remaining),
                        activation_price=float(activation_price),
                        callback_rate=float(cb),
                        k=k,
                        pos_side=pos_side,
                    )
                    if not ok:
                        log_entry.warning(f"{k} trailing skipped (fallbacks exhausted) — hard stop + TPs still active")
                else:
                    log_entry.warning(f"{k} trailing skipped (amount {remaining:.6f} < min {min_amt:.6f})")

            bot.state.total_trades += 1

            await _safe_speak(
                bot,
                f"SCALPER {side.upper()} {k}\n"
                f"Risk ${risk_amount:.2f} | Conf {confidence:.2f} | {cfg.LEVERAGE}x\n"
                f"Notional ${notional:.2f} | Qty {filled:.6f}\n"
                f"STOP OK | THE BLADE STRIKES",
                "critical",
            )

            try:
                await save_brain(bot.state)
            except Exception as pe:
                log_entry.warning(f"Brain save failed: {pe}")

            log_entry.critical(f"SCALPER {side.upper()} {k} ASCENDED — MANAGEMENT ACTIVE")

        except Exception as e:
            log_entry.critical(f"SCALPER FAILED {k} {side.upper()}: {e}")
            log.error(f"Full error: {repr(e)}")
            try:
                bot.state.positions.pop(k, None)
            except Exception:
                pass
