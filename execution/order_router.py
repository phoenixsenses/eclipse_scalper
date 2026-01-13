# execution/order_router.py — SCALPER ETERNAL — ORDER ROUTER — 2026 v1.8 (LIVE-SAFE SETTINGS + EXCHANGE FILTER VALIDATION)
# Patch vs v1.7b:
# - ✅ Adds explicit leverage + margin mode enforcement (best-effort, cached, never fatal)
# - ✅ Adds exchange filter validation (minQty / step / minNotional) + precision normalization before sending
# - ✅ Adds "FIRST_LIVE_SAFE" guardrails (caps leverage, forces isolated, caps per-order notional, optional symbol allowlist)
# - ✅ Blocks invalid orders early with clear CRITICAL logs + telemetry (canonical k)

import asyncio
import random
import time
import hashlib
import os
from typing import Any, Dict, Optional, Tuple

from utils.logging import log_entry

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit_order_create, emit_order_cancel, emit  # type: ignore
except Exception:
    emit_order_create = None
    emit_order_cancel = None
    emit = None


# ----------------------------
# Helpers
# ----------------------------

def _cfg(bot, name: str, default: Any) -> Any:
    try:
        return getattr(getattr(bot, "cfg", None), name, default)
    except Exception:
        return default


def _cfg_env(bot, name: str, default: Any) -> Any:
    """
    Prefer bot.cfg.NAME, fallback to env var NAME, else default.
    """
    try:
        v = _cfg(bot, name, None)
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_ex(bot):
    return getattr(bot, "ex", None)


def _resolve_raw_symbol(bot, k: str, fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return fallback


def _to_float_if_possible(x: Any) -> Any:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return x
        return float(s)
    except Exception:
        return x


def _price_to_precision_safe(ex, sym_raw: str, price: float) -> float:
    p = float(price)
    try:
        fn = getattr(ex, "price_to_precision", None)
        if callable(fn):
            out = fn(sym_raw, p)
            out2 = _to_float_if_possible(out)
            return float(out2) if isinstance(out2, (int, float)) else p
    except Exception:
        pass
    try:
        inner = getattr(ex, "exchange", None)
        if inner is not None:
            out = inner.price_to_precision(sym_raw, p)
            out2 = _to_float_if_possible(out)
            return float(out2) if isinstance(out2, (int, float)) else p
    except Exception:
        pass
    return p


def _amount_to_precision_safe(ex, sym_raw: str, amount: float) -> float:
    a = float(amount)
    try:
        fn = getattr(ex, "amount_to_precision", None)
        if callable(fn):
            out = fn(sym_raw, a)
            out2 = _to_float_if_possible(out)
            return float(out2) if isinstance(out2, (int, float)) else a
    except Exception:
        pass
    try:
        inner = getattr(ex, "exchange", None)
        if inner is not None:
            out = inner.amount_to_precision(sym_raw, a)
            out2 = _to_float_if_possible(out)
            return float(out2) if isinstance(out2, (int, float)) else a
    except Exception:
        pass
    return a


def _merge_params(base: Optional[dict], extra: Optional[dict]) -> dict:
    p: dict = {}
    if isinstance(base, dict):
        p.update(base)
    if isinstance(extra, dict):
        p.update(extra)
    return p


def _normalize_type_for_ccxt(type_u: str) -> str:
    tu = (type_u or "").upper().strip()

    if tu == "MARKET":
        return "market"
    if tu == "LIMIT":
        return "limit"
    if tu in ("STOP_MARKET", "STOP", "STOPMARKET"):
        return "stop_market"
    if tu in ("TAKE_PROFIT_MARKET", "TP_MARKET", "TAKEPROFITMARKET"):
        return "take_profit_market"
    if tu in ("TRAILING_STOP_MARKET", "TRAILING", "TRAILINGSTOPMARKET"):
        return "trailing_stop_market"

    return tu.lower() if tu.isupper() else tu


def _is_number_like(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _strip_none_params(p: dict) -> dict:
    out = {}
    for k, v in (p or {}).items():
        if v is None:
            continue
        out[k] = v
    return out


def _normalize_bool_params(p: dict, keys: Tuple[str, ...]) -> None:
    for k in keys:
        if k in p:
            p[k] = bool(_truthy(p.get(k)))


def _infer_position_side(side_hint: Optional[str]) -> Optional[str]:
    if not side_hint:
        return None
    s = str(side_hint).strip()
    if not s:
        return None

    u = s.upper()
    if u in ("LONG", "SHORT"):
        return u

    l = s.lower()
    if l == "long":
        return "LONG"
    if l == "short":
        return "SHORT"
    if l == "buy":
        return "LONG"
    if l == "sell":
        return "SHORT"
    return None


def _make_client_order_id(
    *,
    prefix: str,
    sym_raw: str,
    type_norm: str,
    side_l: str,
    amount: Any,
    price: Any,
    stop_price: Any,
) -> str:
    blob = f"{sym_raw}|{type_norm}|{side_l}|{amount}|{price}|{stop_price}"
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{h}"


def _telemetry_task(coro) -> None:
    try:
        asyncio.create_task(coro)
    except Exception:
        pass


# ----------------------------
# Live trading safety: leverage/margin + filters
# ----------------------------

_MARKETS_LOAD_LOCK = asyncio.Lock()
_MARKETS_LOADED_AT: float = 0.0

# cache to avoid hammering leverage/margin endpoints
_SYMBOL_SETTINGS_DONE: dict[str, float] = {}  # raw_symbol -> ts


async def _ensure_markets_loaded(ex) -> None:
    """
    Best-effort: load markets once (cached). Never raises.
    """
    global _MARKETS_LOADED_AT
    try:
        # If already loaded and not ancient, skip.
        mk = getattr(ex, "markets", None)
        if isinstance(mk, dict) and len(mk) > 0 and (time.time() - _MARKETS_LOADED_AT) < 60 * 30:
            return
        async with _MARKETS_LOAD_LOCK:
            mk2 = getattr(ex, "markets", None)
            if isinstance(mk2, dict) and len(mk2) > 0 and (time.time() - _MARKETS_LOADED_AT) < 60 * 30:
                return
            fn = getattr(ex, "load_markets", None)
            if callable(fn):
                await fn(True)
                _MARKETS_LOADED_AT = time.time()
    except Exception:
        # never fatal
        return


def _binance_filters_from_market(market: dict) -> dict:
    """
    Parse Binance filters from ccxt market['info']['filters'].
    """
    out: dict[str, Any] = {}
    try:
        info = market.get("info") or {}
        filters = info.get("filters") or []
        for f in filters:
            ftype = f.get("filterType")
            if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                out["minQty"] = f.get("minQty")
                out["stepSize"] = f.get("stepSize")
            elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                out["minNotional"] = f.get("minNotional") or f.get("notional")
    except Exception:
        pass
    return out


def _market_limits(market: dict) -> Tuple[float, float]:
    """
    returns (min_amount, min_cost) from ccxt market limits (best effort)
    """
    try:
        lim = market.get("limits") or {}
        min_amt = _safe_float(((lim.get("amount") or {}).get("min")), 0.0)
        min_cost = _safe_float(((lim.get("cost") or {}).get("min")), 0.0)
        return min_amt, min_cost
    except Exception:
        return 0.0, 0.0


def _market_lookup(ex, sym_raw: str) -> Optional[dict]:
    """
    Best-effort get market from loaded markets.
    """
    try:
        mk = getattr(ex, "markets", None)
        if isinstance(mk, dict) and mk.get(sym_raw):
            return mk.get(sym_raw)
    except Exception:
        pass
    try:
        fn = getattr(ex, "market", None)
        if callable(fn):
            return fn(sym_raw)
    except Exception:
        pass
    return None


async def _fetch_last_price(ex, sym_raw: str) -> Optional[float]:
    try:
        fn = getattr(ex, "fetch_ticker", None)
        if callable(fn):
            t = await fn(sym_raw)
            last = _safe_float(t.get("last") or t.get("close"), 0.0)
            return last if last > 0 else None
    except Exception:
        return None
    return None


async def _validate_and_normalize_order(
    ex,
    *,
    sym_raw: str,
    amount: Any,
    price: Any,
    log,
) -> Tuple[bool, Any, Any, str]:
    """
    Precision normalize + exchange filters:
    - amount_to_precision
    - price_to_precision (if price provided)
    - minQty / minNotional
    Never raises. Returns (ok, amount_norm, price_norm, reason)
    """
    try:
        await _ensure_markets_loaded(ex)
        market = _market_lookup(ex, sym_raw)
        if not market:
            # If we can't validate, don't hard-block; but still normalize precision.
            amt_norm = _amount_to_precision_safe(ex, sym_raw, _safe_float(amount, 0.0))
            px_norm = None
            if price is not None and _is_number_like(price):
                px_norm = _price_to_precision_safe(ex, sym_raw, float(price))
            else:
                px_norm = price
            return True, amt_norm, px_norm, "ok_no_market"

        # normalize precision first
        amt_f = _safe_float(amount, 0.0)
        amt_norm: Any = _amount_to_precision_safe(ex, sym_raw, amt_f)

        px_norm: Any = None
        if price is not None:
            if _is_number_like(price):
                px_norm = _price_to_precision_safe(ex, sym_raw, float(price))
            else:
                px_norm = price

        # gather limits/filters
        min_amt_ccxt, min_cost_ccxt = _market_limits(market)
        bf = _binance_filters_from_market(market)
        min_qty = max(min_amt_ccxt, _safe_float(bf.get("minQty"), 0.0))
        min_notional = max(min_cost_ccxt, _safe_float(bf.get("minNotional"), 0.0))

        amt_norm_f = _safe_float(amt_norm, 0.0)
        if amt_norm_f <= 0:
            return False, amt_norm, px_norm, "amount<=0"

        if min_qty and amt_norm_f < min_qty:
            return False, amt_norm, px_norm, f"amount<{min_qty}"

        # notional needs price; for market orders, fetch last
        px_for_notional: Optional[float] = None
        if px_norm is not None and _is_number_like(px_norm):
            px_for_notional = _safe_float(px_norm, 0.0) or None
        if px_for_notional is None:
            px_for_notional = await _fetch_last_price(ex, sym_raw)

        if min_notional and px_for_notional and px_for_notional > 0:
            notion = amt_norm_f * float(px_for_notional)
            if notion < min_notional:
                return False, amt_norm, px_norm, f"notional<{min_notional} (got {notion:.4f})"

        return True, amt_norm, px_norm, "ok"
    except Exception as e:
        # if validation explodes, fail-open but keep original values
        log(f"[router] WARN validation crashed: {e}")
        return True, amount, price, "ok_validation_error_failopen"


async def _ensure_futures_settings(
    ex,
    *,
    sym_raw: str,
    leverage: int,
    margin_mode: str,
    log,
) -> bool:
    """
    Best-effort: enforce margin mode + leverage (ccxt methods if present).
    Cached to avoid rate limits. Never raises.
    """
    try:
        now = time.time()
        last = _SYMBOL_SETTINGS_DONE.get(sym_raw, 0.0)
        if now - last < 60 * 30:  # 30 min
            return True

        # margin mode
        if margin_mode:
            try:
                fn = getattr(ex, "set_margin_mode", None)
                if callable(fn):
                    await fn(margin_mode, sym_raw)
                    log(f"[router] margin_mode set: {sym_raw} -> {margin_mode}")
            except Exception as e:
                log(f"[router] WARN margin_mode failed for {sym_raw}: {e}")

        # leverage
        if leverage and leverage > 0:
            try:
                fn = getattr(ex, "set_leverage", None)
                if callable(fn):
                    await fn(int(leverage), sym_raw)
                    log(f"[router] leverage set: {sym_raw} -> {leverage}x")
            except Exception as e:
                log(f"[router] WARN leverage failed for {sym_raw}: {e}")

        _SYMBOL_SETTINGS_DONE[sym_raw] = now
        return True
    except Exception as e:
        log(f"[router] WARN _ensure_futures_settings crashed: {e}")
        return False


def _first_live_safe_enabled(bot) -> bool:
    return bool(_truthy(_cfg_env(bot, "FIRST_LIVE_SAFE", False)))


def _allowed_symbols_set(bot) -> Optional[set[str]]:
    """
    Optional allowlist when FIRST_LIVE_SAFE is enabled.
    Accepts env/bot cfg FIRST_LIVE_SYMBOLS="BTCUSDT,ETHUSDT"
    """
    try:
        s = _cfg_env(bot, "FIRST_LIVE_SYMBOLS", "")
        if not s:
            return None
        parts = [p.strip().upper() for p in str(s).replace(";", ",").split(",") if p.strip()]
        return set(parts) if parts else None
    except Exception:
        return None


# ----------------------------
# Cancel order (router)
# ----------------------------

async def _cancel_order_raw(ex, order_id: str, sym_raw: str) -> bool:
    if not order_id:
        return False
    try:
        fn = getattr(ex, "cancel_order", None)
        if callable(fn):
            await fn(order_id, sym_raw)
            return True
    except Exception:
        return False
    return False


async def cancel_order(bot, order_id: str, symbol: str) -> bool:
    ex = _get_ex(bot)
    if ex is None or not order_id:
        return False

    k = _symkey(symbol)
    sym_raw = _resolve_raw_symbol(bot, k, symbol)

    ok = await _cancel_order_raw(ex, order_id, sym_raw)
    if not ok and sym_raw != symbol:
        ok = await _cancel_order_raw(ex, order_id, symbol)
    if not ok and k not in (symbol, sym_raw):
        ok = await _cancel_order_raw(ex, order_id, k)

    # Telemetry (best effort, non-blocking) — use canonical symbol
    if callable(emit_order_cancel):
        _telemetry_task(emit_order_cancel(bot, k, order_id, ok, why="router"))

    return ok


# ----------------------------
# Create order (router)
# ----------------------------

def _default_retry_policy() -> Tuple[int, float, float]:
    return 6, 0.25, 0.20


def _is_dry_run(bot) -> bool:
    ex = _get_ex(bot)
    try:
        fn = getattr(ex, "_is_dry_run", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    try:
        v = os.getenv("SCALPER_DRY_RUN", "")
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _dry_run_order_stub(symbol: str, type_: str, side: str, amount: Any, price: Any, params: dict) -> dict:
    amt = _safe_float(amount, 0.0)
    return {
        "id": None,  # IMPORTANT: never persist fake IDs
        "symbol": symbol,
        "type": type_,
        "side": side,
        "amount": amt,
        "filled": 0.0,
        "status": "canceled",
        "average": None,
        "reduceOnly": bool((params or {}).get("reduceOnly", False)),
        "info": {"dry_run": True, "router": True, "params": dict(params or {})},
    }


async def create_order(
    bot,
    *,
    symbol: str,
    type: str,
    side: str,
    amount: Any,
    price: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
    intent_reduce_only: bool = False,
    intent_close_position: bool = False,
    stop_price: Optional[Any] = None,
    trigger_price: Optional[Any] = None,
    activation_price: Optional[Any] = None,
    callback_rate: Optional[float] = None,
    client_order_id: Optional[str] = None,
    client_order_id_prefix: str = "SE_",
    auto_client_order_id: bool = False,
    hedge_side_hint: Optional[str] = None,
    hedge_mode: Optional[bool] = None,
    respect_kill_switch: bool = False,
    retries: Optional[int] = None,
) -> Optional[dict]:
    ex = _get_ex(bot)
    if ex is None:
        return None

    if respect_kill_switch:
        try:
            from risk.kill_switch import is_halted  # type: ignore
            if callable(is_halted) and is_halted(bot):
                log_entry.critical("ORDER ROUTER BLOCKED BY KILL_SWITCH (respect_kill_switch=True)")
                return None
        except Exception:
            pass

    type_u = str(type or "").upper().strip()
    type_norm = _normalize_type_for_ccxt(type_u)
    side_l = str(side or "").lower().strip()

    k = _symkey(symbol)
    sym_raw = _resolve_raw_symbol(bot, k, symbol)

    # FIRST LIVE SAFE allowlist (optional)
    if _first_live_safe_enabled(bot):
        allow = _allowed_symbols_set(bot)
        if allow is not None and k not in allow:
            log_entry.critical(f"FIRST_LIVE_SAFE BLOCKED → symbol not allowlisted: k={k} allow={sorted(list(allow))}")
            if callable(emit):
                _telemetry_task(
                    emit(bot, "order.blocked", data={"k": k, "why": "first_live_symbol_not_allowed"}, symbol=k, level="critical")
                )
            return None

    p = _merge_params(params, {})
    if intent_reduce_only:
        p.setdefault("reduceOnly", True)
    if intent_close_position:
        p.setdefault("closePosition", True)

    if hedge_mode is None:
        hedge_mode = (
            bool(_truthy(_cfg(bot, "HEDGE_MODE", False)))
            or bool(_truthy(_cfg(bot, "HEDGE_SAFE", False)))
            or bool(_truthy(_cfg(bot, "HEDGE_ENABLED", False)))
        )

    if hedge_mode:
        ps = _infer_position_side(hedge_side_hint)
        if ps:
            p.setdefault("positionSide", ps)

    if client_order_id:
        p["clientOrderId"] = str(client_order_id)
    elif auto_client_order_id and "clientOrderId" not in p:
        p["clientOrderId"] = _make_client_order_id(
            prefix=client_order_id_prefix,
            sym_raw=sym_raw,
            type_norm=type_norm,
            side_l=side_l,
            amount=amount,
            price=price,
            stop_price=(stop_price if stop_price is not None else trigger_price),
        )

    sp_any = stop_price if stop_price is not None else trigger_price
    if sp_any is not None:
        if _is_number_like(sp_any):
            sp_f = _price_to_precision_safe(ex, sym_raw, float(sp_any))
            p["stopPrice"] = float(sp_f)
        else:
            p["stopPrice"] = sp_any
        p.pop("triggerPrice", None)

    if activation_price is not None:
        if _is_number_like(activation_price):
            ap_f = _price_to_precision_safe(ex, sym_raw, float(activation_price))
            p["activationPrice"] = float(ap_f)
        else:
            p["activationPrice"] = activation_price

    if callback_rate is not None:
        p["callbackRate"] = _clamp(float(callback_rate), 0.1, 5.0)

    _normalize_bool_params(p, ("reduceOnly", "closePosition"))
    p = _strip_none_params(p)

    if _is_dry_run(bot):
        log_entry.critical(
            f"DRY_RUN ROUTER BLOCKED → k={k} raw={sym_raw} {type_norm} {side_l} amount={amount} price={price} params={p}"
        )
        return _dry_run_order_stub(sym_raw, type_norm, side_l, amount, price, p)

    # ----------------------------
    # LIVE SAFETY: enforce margin/leverage (best effort)
    # ----------------------------
    # base config
    margin_mode = str(_cfg_env(bot, "MARGIN_MODE", _cfg(bot, "MARGIN_MODE", "isolated"))).strip().lower()
    leverage = int(_safe_float(_cfg_env(bot, "LEVERAGE", _cfg(bot, "LEVERAGE", 1)), 1))

    if _first_live_safe_enabled(bot):
        # hard clamp in first-live mode
        margin_mode = "isolated"
        leverage = 1

    # Ensure settings (never fatal)
    try:
        await _ensure_futures_settings(ex, sym_raw=sym_raw, leverage=leverage, margin_mode=margin_mode, log=lambda s: log_entry.info(s))
    except Exception:
        pass

    # ----------------------------
    # Precision + exchange filter validation (minQty/minNotional)
    # ----------------------------
    ok, amt_norm, px_norm, why = await _validate_and_normalize_order(
        ex,
        sym_raw=sym_raw,
        amount=amount,
        price=price,
        log=lambda s: log_entry.info(s),
    )

    if not ok:
        log_entry.critical(
            f"ROUTER BLOCKED BY EXCHANGE FILTERS → k={k} raw={sym_raw} type={type_norm} side={side_l} amount={amount} price={price} why={why}"
        )
        if callable(emit):
            _telemetry_task(
                emit(
                    bot,
                    "order.blocked",
                    data={
                        "k": k,
                        "raw": sym_raw,
                        "type": type_u,
                        "side": side_l,
                        "amount": amount,
                        "price": price,
                        "why": why,
                    },
                    symbol=k,
                    level="critical",
                )
            )
        return None

    amt_prec: Any = amt_norm
    px_prec: Any = px_norm

    # FIRST LIVE SAFE: cap per-order notional (best-effort)
    if _first_live_safe_enabled(bot):
        cap = _safe_float(_cfg_env(bot, "FIRST_LIVE_MAX_NOTIONAL_USDT", 3.0), 3.0)
        try:
            # estimate notional using price if given, else last
            px_for_cap: Optional[float] = None
            if px_prec is not None and _is_number_like(px_prec):
                px_for_cap = _safe_float(px_prec, 0.0) or None
            if px_for_cap is None:
                px_for_cap = await _fetch_last_price(ex, sym_raw)
            if px_for_cap and px_for_cap > 0:
                notion = _safe_float(amt_prec, 0.0) * float(px_for_cap)
                if notion > cap:
                    log_entry.critical(
                        f"FIRST_LIVE_SAFE BLOCKED → notional cap exceeded: k={k} raw={sym_raw} notional={notion:.4f} cap={cap}"
                    )
                    if callable(emit):
                        _telemetry_task(
                            emit(bot, "order.blocked", data={"k": k, "why": "first_live_notional_cap", "notional": notion, "cap": cap}, symbol=k, level="critical")
                        )
                    return None
        except Exception:
            # fail-open on cap calc errors (don't brick routing)
            pass

    max_attempts, base_delay, jitter = _default_retry_policy()
    if retries is not None:
        max_attempts = max(1, int(retries))

    last_err: Optional[Exception] = None

    async def _attempt(raw_symbol: str, amt_try: Any, px_try: Any, p_try: dict) -> dict:
        fn = getattr(ex, "create_order", None)
        if not callable(fn):
            raise RuntimeError("exchange has no create_order()")

        if type_norm == "market":
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, params=p_try)

        if type_norm == "limit":
            if px_try is None:
                raise RuntimeError("limit order missing price")
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, price=px_try, params=p_try)

        if px_try is None:
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, params=p_try)

        return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, price=px_try, params=p_try)

    variants: list[tuple[str, Any, Any, dict]] = [(sym_raw, amt_prec, px_prec, dict(p))]

    if type_u in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "TP_MARKET"):
        if intent_close_position or _truthy(p.get("closePosition")):
            p2 = dict(p)
            p2.setdefault("reduceOnly", True)
            p2["closePosition"] = True
            variants.append((sym_raw, 0, px_prec, p2))

    if type_u == "TRAILING_STOP_MARKET" and "callbackRate" in p:
        p2 = dict(p)
        p2.pop("callbackRate", None)
        variants.append((sym_raw, amt_prec, px_prec, p2))

    if sym_raw != symbol:
        variants.append((symbol, amt_prec, px_prec, dict(p)))

    tries = 0

    for attempt in range(max_attempts):
        for (raw_sym, amt_try, px_try, p_try) in variants:
            tries += 1
            try:
                res = await _attempt(raw_sym, amt_try, px_try, p_try)

                # Telemetry (best effort, non-blocking) — canonical symbol
                if callable(emit_order_create):
                    _telemetry_task(emit_order_create(bot, k, res, intent=f"{type_u}:{side_l}"))

                return res
            except Exception as e:
                last_err = e

        delay = base_delay * (2 ** attempt) + random.uniform(0.0, jitter)
        await asyncio.sleep(delay)

    if callable(emit):
        _telemetry_task(
            emit(
                bot,
                "order.create_failed",
                data={
                    "k": k,
                    "raw": sym_raw,
                    "type": type_u,
                    "side": side_l,
                    "amount": amount,
                    "price": price,
                    "tries": tries,
                    "variants": len(variants),
                    "params": p,  # already sanitized/no Nones
                    "err": (repr(last_err)[:300] if last_err else "unknown"),
                },
                symbol=k,
                level="critical",
            )
        )

    log_entry.error(
        f"ORDER ROUTER FAILED → k={k} raw={sym_raw} {type_norm} {side_l} amount={amount} price={price} err={last_err}"
    )
    return None


# ----------------------------
# Convenience wrappers
# ----------------------------

async def create_market(
    bot,
    *,
    symbol: str,
    side: str,
    amount: float,
    reduce_only: bool = False,
    hedge_side_hint: Optional[str] = None,
) -> Optional[dict]:
    return await create_order(
        bot,
        symbol=symbol,
        type="MARKET",
        side=side,
        amount=amount,
        price=None,
        params={},
        intent_reduce_only=reduce_only,
        hedge_side_hint=hedge_side_hint,
        retries=4,
    )


async def create_stop_market(
    bot,
    *,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    reduce_only: bool = True,
    close_position: bool = False,
    hedge_side_hint: Optional[str] = None,
) -> Optional[dict]:
    return await create_order(
        bot,
        symbol=symbol,
        type="STOP_MARKET",
        side=side,
        amount=amount,
        price=None,
        params={},
        intent_reduce_only=reduce_only,
        intent_close_position=close_position,
        stop_price=stop_price,
        hedge_side_hint=hedge_side_hint,
        retries=6,
    )


async def create_trailing_stop_market(
    bot,
    *,
    symbol: str,
    side: str,
    amount: float,
    activation_price: float,
    callback_rate: float,
    reduce_only: bool = True,
    hedge_side_hint: Optional[str] = None,
) -> Optional[dict]:
    return await create_order(
        bot,
        symbol=symbol,
        type="TRAILING_STOP_MARKET",
        side=side,
        amount=amount,
        price=None,
        params={},
        intent_reduce_only=reduce_only,
        activation_price=activation_price,
        callback_rate=callback_rate,
        hedge_side_hint=hedge_side_hint,
        retries=6,
    )
