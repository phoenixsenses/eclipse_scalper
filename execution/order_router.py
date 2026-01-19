# execution/order_router.py — SCALPER ETERNAL — ORDER ROUTER — 2026 v2.1 (BINANCE CLOSEPOSITION + REDUCEONLY FIX)
# Patch vs v2.0:
# - ✅ FIX: Binance futures STOP/TP with closePosition=True MUST still pass amount to ccxt (amount is required arg) → we send amount=0.0
# - ✅ FIX: Validation no longer blocks amount<=0 when closePosition=True (router used to block it)
# - ✅ FIX: If closePosition=True, router strips reduceOnly ALWAYS (Binance -1106 “reduceonly not required”)
# - ✅ HARDEN: Adds retry variant that auto-removes reduceOnly when exchange complains (even if caller set it)
# - ✅ Keeps: FIRST_LIVE_SAFE entry-only caps, hedge-mode positionSide injection, filters, retries, telemetry

import asyncio
import random
import time
import hashlib
import os
from typing import Any, Dict, Optional, Tuple, List

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


def _is_futures_symbol(sym_raw: str) -> bool:
    # Binance futures linear symbols look like "DOGE/USDT:USDT"
    return (":USDT" in (sym_raw or "")) or (":USD" in (sym_raw or ""))


def _is_exit_intent(
    *,
    type_u: str,
    params: dict,
    intent_reduce_only: bool,
    intent_close_position: bool,
) -> bool:
    """
    Conservative: treat these as "exits" (must never be blocked by FIRST_LIVE_SAFE):
    - reduceOnly / closePosition flags
    - intent flags
    - common protective order types (STOP/TP/TRAILING)
    """
    if bool(_truthy(params.get("reduceOnly"))) or bool(_truthy(params.get("closePosition"))):
        return True
    if intent_reduce_only or intent_close_position:
        return True
    if type_u in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "TP_MARKET", "TRAILING_STOP_MARKET"):
        return True
    return False


def _looks_like_binance_reduceonly_not_required(err: Exception) -> bool:
    s = repr(err).lower()
    # Your exact message: Parameter 'reduceonly' sent when not required.
    return ("reduceonly" in s) and ("not required" in s or "sent when not required" in s or "parameter 'reduceonly'" in s)


# ----------------------------
# Binance Futures account-mode detection (cached)
# ----------------------------

_MODE_LOCK = asyncio.Lock()
_MODE_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "dualSidePosition": None,      # True/False
    "multiAssetsMargin": None,     # True/False
}


async def _detect_binance_futures_modes(ex, *, force: bool = False) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Best-effort: detect:
      - dualSidePosition (hedge mode)
      - multiAssetsMargin (multi-assets mode)
    Never fatal. Cached.
    """
    try:
        now = time.time()
        if (not force) and (_MODE_CACHE["ts"] and (now - float(_MODE_CACHE["ts"])) < 60 * 10):
            return _MODE_CACHE.get("dualSidePosition"), _MODE_CACHE.get("multiAssetsMargin")

        async with _MODE_LOCK:
            now = time.time()
            if (not force) and (_MODE_CACHE["ts"] and (now - float(_MODE_CACHE["ts"])) < 60 * 10):
                return _MODE_CACHE.get("dualSidePosition"), _MODE_CACHE.get("multiAssetsMargin")

            dual = None
            multi = None

            try:
                fn = getattr(ex, "fapiPrivateGetPositionSideDual", None)
                if callable(fn):
                    r = await fn()
                    if isinstance(r, dict) and "dualSidePosition" in r:
                        dual = bool(_truthy(r.get("dualSidePosition")))
            except Exception:
                pass

            try:
                fn = getattr(ex, "fapiPrivateGetMultiAssetsMargin", None)
                if callable(fn):
                    r = await fn()
                    if isinstance(r, dict) and "multiAssetsMargin" in r:
                        multi = bool(_truthy(r.get("multiAssetsMargin")))
            except Exception:
                pass

            _MODE_CACHE["ts"] = time.time()
            _MODE_CACHE["dualSidePosition"] = dual
            _MODE_CACHE["multiAssetsMargin"] = multi
            return dual, multi
    except Exception:
        return None, None


# ----------------------------
# Live trading safety: leverage/margin + filters
# ----------------------------

_MARKETS_LOAD_LOCK = asyncio.Lock()
_MARKETS_LOADED_AT: float = 0.0
_SYMBOL_SETTINGS_DONE: dict[str, float] = {}  # raw_symbol -> ts


async def _ensure_markets_loaded(ex) -> None:
    global _MARKETS_LOADED_AT
    try:
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
        return


def _binance_filters_from_market(market: dict) -> dict:
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
    try:
        lim = market.get("limits") or {}
        min_amt = _safe_float(((lim.get("amount") or {}).get("min")), 0.0)
        min_cost = _safe_float(((lim.get("cost") or {}).get("min")), 0.0)
        return min_amt, min_cost
    except Exception:
        return 0.0, 0.0


def _market_lookup(ex, sym_raw: str) -> Optional[dict]:
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
    params: Optional[dict],
    log,
) -> Tuple[bool, Any, Any, str]:
    """
    Returns: (ok, amount_norm, price_norm, why)

    IMPORTANT (Binance futures / ccxt):
    - If closePosition=True, ccxt still requires an amount argument in create_order signature.
      So we accept amount<=0 and allow it through validation.
    """
    try:
        await _ensure_markets_loaded(ex)

        p = params or {}
        is_close_pos = bool(_truthy(p.get("closePosition")))

        market = _market_lookup(ex, sym_raw)

        # normalize price
        px_norm: Any = None
        if price is not None:
            if _is_number_like(price):
                px_norm = _price_to_precision_safe(ex, sym_raw, float(price))
            else:
                px_norm = price

        # normalize amount
        amt_f = _safe_float(amount, 0.0)
        amt_norm: Any = _amount_to_precision_safe(ex, sym_raw, amt_f)
        amt_norm_f = _safe_float(amt_norm, 0.0)

        # closePosition: allow amount<=0 (router will send 0.0) and skip minQty/minNotional blocks
        if is_close_pos:
            return True, float(amt_norm_f), px_norm, "ok_closePosition_amount_can_be_zero"

        if not market:
            # no market info: fail-open
            return True, amt_norm, px_norm, "ok_no_market"

        min_amt_ccxt, min_cost_ccxt = _market_limits(market)
        bf = _binance_filters_from_market(market)
        min_qty = max(min_amt_ccxt, _safe_float(bf.get("minQty"), 0.0))
        min_notional = max(min_cost_ccxt, _safe_float(bf.get("minNotional"), 0.0))

        if amt_norm_f <= 0:
            return False, amt_norm, px_norm, "amount<=0"

        if min_qty and amt_norm_f < min_qty:
            return False, amt_norm, px_norm, f"amount<{min_qty}"

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
    try:
        now = time.time()
        last = _SYMBOL_SETTINGS_DONE.get(sym_raw, 0.0)
        if now - last < 60 * 30:
            return True

        if margin_mode:
            try:
                fn = getattr(ex, "set_margin_mode", None)
                if callable(fn):
                    await fn(margin_mode, sym_raw)
                    log(f"[router] margin_mode set: {sym_raw} -> {margin_mode}")
            except Exception as e:
                log(f"[router] WARN margin_mode failed for {sym_raw}: {e}")

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
    amt = _safe_float(amount, 0.0) if amount is not None else 0.0
    return {
        "id": None,
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

    p = _merge_params(params, {})

    # intents -> params
    if intent_close_position:
        p.setdefault("closePosition", True)

    # IMPORTANT:
    # - If closePosition=True, NEVER send reduceOnly (Binance -1106).
    # - If caller asked intent_reduce_only but also closePosition, closePosition wins.
    if bool(_truthy(p.get("closePosition"))):
        p.pop("reduceOnly", None)
    else:
        if intent_reduce_only:
            p.setdefault("reduceOnly", True)

    _normalize_bool_params(p, ("reduceOnly", "closePosition"))
    p = _strip_none_params(p)

    # Decide exit-vs-entry early (drives FIRST_LIVE_SAFE behavior)
    is_exit = _is_exit_intent(
        type_u=type_u,
        params=p,
        intent_reduce_only=intent_reduce_only,
        intent_close_position=intent_close_position,
    )

    # FIRST LIVE SAFE allowlist applies to ENTRIES only (never block exits)
    if _first_live_safe_enabled(bot) and (not is_exit):
        allow = _allowed_symbols_set(bot)
        if allow is not None and k not in allow:
            log_entry.critical(f"FIRST_LIVE_SAFE BLOCKED → symbol not allowlisted: k={k} allow={sorted(list(allow))}")
            if callable(emit):
                _telemetry_task(
                    emit(bot, "order.blocked", data={"k": k, "why": "first_live_symbol_not_allowed"}, symbol=k, level="critical")
                )
            return None

    # ----------------------------
    # Detect Binance hedge/multi-assets modes (best-effort)
    # ----------------------------
    is_fut = _is_futures_symbol(sym_raw)
    dual_side, multi_assets = (None, None)
    if is_fut:
        dual_side, multi_assets = await _detect_binance_futures_modes(ex)

    hedge_mode_effective = bool(dual_side) if dual_side is not None else bool(_truthy(hedge_mode))

    # ----------------------------
    # HEDGE MODE PARAMS
    # ----------------------------
    if is_fut and hedge_mode_effective:
        inferred = "LONG" if side_l == "buy" else "SHORT"
        is_exit2 = is_exit or bool(_truthy(p.get("reduceOnly"))) or bool(_truthy(p.get("closePosition")))

        if is_exit2:
            ps = _infer_position_side(hedge_side_hint)
            if not ps:
                log_entry.critical(
                    f"ROUTER BLOCKED → hedge exit requires hedge_side_hint LONG/SHORT | k={k} raw={sym_raw} type={type_norm} side={side_l} reduceOnly={p.get('reduceOnly')} closePosition={p.get('closePosition')}"
                )
                if callable(emit):
                    _telemetry_task(
                        emit(bot, "order.blocked", data={"k": k, "why": "missing_hedge_side_hint_for_exit"}, symbol=k, level="critical")
                    )
                return None

            p.setdefault("positionSide", ps)

            # Only force reduceOnly if NOT closePosition
            if not bool(_truthy(p.get("closePosition"))):
                p.setdefault("reduceOnly", True)
            else:
                p.pop("reduceOnly", None)
        else:
            p.setdefault("positionSide", inferred)

    # Client order id
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

    # stop / trigger normalization
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
    # If closePosition, strip reduceOnly again (defense in depth)
    if bool(_truthy(p.get("closePosition"))):
        p.pop("reduceOnly", None)
    p = _strip_none_params(p)

    # Recompute is_exit after full param assembly
    is_exit = _is_exit_intent(
        type_u=type_u,
        params=p,
        intent_reduce_only=intent_reduce_only,
        intent_close_position=intent_close_position,
    )

    # SHOW WHAT WILL BE SENT
    try:
        log_entry.info(
            f"[router] SEND k={k} raw={sym_raw} type={type_norm} side={side_l} amt={amount} px={price} "
            f"is_exit={is_exit} reduceOnly={p.get('reduceOnly')} closePosition={p.get('closePosition')} positionSide={p.get('positionSide')} params_keys={sorted(list(p.keys()))}"
        )
    except Exception:
        pass

    if _is_dry_run(bot):
        log_entry.critical(
            f"DRY_RUN ROUTER BLOCKED → k={k} raw={sym_raw} {type_norm} {side_l} amount={amount} price={price} params={p}"
        )
        return _dry_run_order_stub(sym_raw, type_norm, side_l, amount, price, p)

    # ----------------------------
    # LIVE SAFETY: enforce margin/leverage (best effort)
    # NOTE: We still do this even for exits; it's non-fatal best-effort.
    # ----------------------------
    margin_mode = str(_cfg_env(bot, "MARGIN_MODE", _cfg(bot, "MARGIN_MODE", "cross"))).strip().lower()
    leverage = int(_safe_float(_cfg_env(bot, "LEVERAGE", _cfg(bot, "LEVERAGE", 1)), 1))

    if _first_live_safe_enabled(bot):
        leverage = 1
        if bool(multi_assets):
            margin_mode = ""
        else:
            margin_mode = "isolated"

    try:
        await _ensure_futures_settings(
            ex,
            sym_raw=sym_raw,
            leverage=leverage,
            margin_mode=margin_mode,
            log=lambda s: log_entry.info(s),
        )
    except Exception:
        pass

    # ----------------------------
    # Precision + exchange filter validation
    # ----------------------------
    # BINANCE CLOSEPOSITION RULE:
    # ccxt requires amount arg; for closePosition exits we will send amount=0.0.
    amount_for_validation = amount
    if bool(_truthy(p.get("closePosition"))):
        # even if caller gave None, we use 0.0
        amount_for_validation = _safe_float(amount, 0.0)

    ok, amt_norm, px_norm, why = await _validate_and_normalize_order(
        ex,
        sym_raw=sym_raw,
        amount=amount_for_validation,
        price=price,
        params=p,
        log=lambda s: log_entry.info(s),
    )

    if not ok:
        log_entry.critical(
            f"ROUTER BLOCKED BY EXCHANGE FILTERS → k={k} raw={sym_raw} type={type_norm} side={side_l} amount={amount_for_validation} price={price} why={why}"
        )
        if callable(emit):
            _telemetry_task(
                emit(
                    bot,
                    "order.blocked",
                    data={"k": k, "raw": sym_raw, "type": type_u, "side": side_l, "amount": amount_for_validation, "price": price, "why": why},
                    symbol=k,
                    level="critical",
                )
            )
        return None

    amt_prec: Any = amt_norm
    px_prec: Any = px_norm

    # For closePosition, force amount to 0.0 at send time (ccxt signature requirement)
    if bool(_truthy(p.get("closePosition"))):
        amt_prec = 0.0

    # FIRST LIVE SAFE: cap per-order notional (ENTRIES ONLY, never exits)
    if _first_live_safe_enabled(bot) and (not is_exit):
        cap = _safe_float(_cfg_env(bot, "FIRST_LIVE_MAX_NOTIONAL_USDT", 5.0), 5.0)
        try:
            px_for_cap: Optional[float] = None
            if px_prec is not None and _is_number_like(px_prec):
                px_for_cap = _safe_float(px_prec, 0.0) or None
            if px_for_cap is None:
                px_for_cap = await _fetch_last_price(ex, sym_raw)

            if amt_prec is not None and px_for_cap and px_for_cap > 0:
                notion = _safe_float(amt_prec, 0.0) * float(px_for_cap)
                if notion > cap:
                    log_entry.critical(
                        f"FIRST_LIVE_SAFE BLOCKED → notional cap exceeded: k={k} raw={sym_raw} notional={notion:.4f} cap={cap}"
                    )
                    if callable(emit):
                        _telemetry_task(
                            emit(
                                bot,
                                "order.blocked",
                                data={"k": k, "why": "first_live_notional_cap", "notional": notion, "cap": cap},
                                symbol=k,
                                level="critical",
                            )
                        )
                    return None
        except Exception:
            pass

    max_attempts, base_delay, jitter = 6, 0.25, 0.20
    if retries is not None:
        max_attempts = max(1, int(retries))

    last_err: Optional[Exception] = None

    async def _attempt(raw_symbol: str, amt_try: Any, px_try: Any, p_try: dict) -> dict:
        fn = getattr(ex, "create_order", None)
        if not callable(fn):
            raise RuntimeError("exchange has no create_order()")

        # ccxt python requires amount parameter in signature for ALL order types, including stop/closePosition on binance
        if amt_try is None:
            amt_try = 0.0

        if type_norm == "market":
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, params=p_try)

        if type_norm == "limit":
            if px_try is None:
                raise RuntimeError("limit order missing price")
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, price=px_try, params=p_try)

        if px_try is None:
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, params=p_try)

        return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, price=px_try, params=p_try)

    # Variants: try normal first, then (if relevant) stripped reduceOnly, then symbol fallbacks
    variants: List[tuple[str, Any, Any, dict]] = [(sym_raw, amt_prec, px_prec, dict(p))]

    # If exchange rejects reduceOnly (common on binance for some orders), we try again with reduceOnly removed
    if "reduceOnly" in p:
        p_ro = dict(p)
        p_ro.pop("reduceOnly", None)
        variants.append((sym_raw, amt_prec, px_prec, p_ro))

    # STOP/TP closePosition: always ensure reduceOnly removed and amount present (=0.0)
    if type_u in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "TP_MARKET"):
        if bool(_truthy(p.get("closePosition"))) or intent_close_position:
            p2 = dict(p)
            p2["closePosition"] = True
            p2.pop("reduceOnly", None)
            variants.append((sym_raw, 0.0, px_prec, p2))

    # TRAILING: drop callbackRate variant (some venues)
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
                if callable(emit_order_create):
                    _telemetry_task(emit_order_create(bot, k, res, intent=f"{type_u}:{side_l}"))
                return res
            except Exception as e:
                last_err = e

                # If we see the binance -1106 reduceOnly not required, add a stripped variant dynamically
                if _looks_like_binance_reduceonly_not_required(e):
                    try:
                        if "reduceOnly" in p_try:
                            p3 = dict(p_try)
                            p3.pop("reduceOnly", None)
                            variants.append((raw_sym, amt_try, px_try, p3))
                    except Exception:
                        pass

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
                    "params": p,
                    "err": (repr(last_err)[:300] if last_err else "unknown"),
                },
                symbol=k,
                level="critical",
            )
        )

    log_entry.error(f"ORDER ROUTER FAILED → k={k} raw={sym_raw} {type_norm} {side_l} amount={amount} price={price} err={last_err}")
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
    amount: Optional[float],
    stop_price: float,
    reduce_only: bool = True,
    close_position: bool = False,
    hedge_side_hint: Optional[str] = None,
) -> Optional[dict]:
    # IMPORTANT (Binance futures + ccxt): if close_position=True, we still send amount=0.0 (ccxt signature requires it)
    amt = 0.0 if close_position else amount
    return await create_order(
        bot,
        symbol=symbol,
        type="STOP_MARKET",
        side=side,
        amount=amt,
        price=None,
        params={"closePosition": True} if close_position else {},
        intent_reduce_only=(reduce_only and (not close_position)),
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
