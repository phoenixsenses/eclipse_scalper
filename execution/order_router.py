# execution/order_router.py — SCALPER ETERNAL — ORDER ROUTER — 2026 v2.4 (BINANCE COID<36 + IDEMPOTENT CANCEL + BOUNDED RETRIES)
# Patch vs v2.3:
# - ✅ FIX: Binance -4015 "Client order id length should be less than 36 chars" → ALWAYS sanitize/trim/hash clientOrderId (<36)
# - ✅ HARDEN: Sanitize clientOrderId from *any* source (explicit arg, params, auto-id, duplicate retry suffix)
# - ✅ HARDEN: Duplicate (-4116) retry is bounded + deduped (variants won't explode)
# - ✅ FIX: cancel_order() idempotent: "unknown/already gone" (-2011 etc.) treated as success → POSMGR spam dies
# - ✅ Keeps: stopPrice required, closePosition amount=0.0, reduceOnly stripping for closePosition,
#           FIRST_LIVE_SAFE entry-only caps, hedge-mode positionSide injection, filters, retries, telemetry.

from __future__ import annotations

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

# Binance: clientOrderId MUST be < 36 chars (so max 35)
_BINANCE_CLIENT_ID_MAX = 35

# Prevent variant explosion
_MAX_VARIANTS = 12


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
    if x is False or x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "t", "on")
    return False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_callback_rate(val: Any) -> float:
    """
    Binance expects callbackRate in percent units within [0.1, 5.0].
    If user passes 45 (likely "45%"), normalize to 0.45 before clamping.
    """
    v = _safe_float(val, 0.0)
    if v > 5.0:
        v = v / 100.0
    return _clamp(v, 0.1, 5.0)


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
    s = sym_raw or ""
    return (":USDT" in s) or (":USD" in s) or ("PERP" in s.upper())


def _is_futures_symbol_ex(ex, sym_raw: str) -> bool:
    """
    Prefer market metadata when available; fallback to string heuristics.
    """
    try:
        market = _market_lookup(ex, sym_raw)
        if isinstance(market, dict):
            if bool(market.get("contract")) or bool(market.get("swap")) or bool(market.get("future")):
                return True
    except Exception:
        pass
    return _is_futures_symbol(sym_raw)


def _is_exit_intent(
    *,
    type_u: str,
    params: dict,
    intent_reduce_only: bool,
    intent_close_position: bool,
) -> bool:
    if bool(_truthy(params.get("reduceOnly"))) or bool(_truthy(params.get("closePosition"))):
        return True
    if intent_reduce_only or intent_close_position:
        return True
    # Do not assume STOP/TP/TRAILING are exits unless reduceOnly/closePosition is set.
    # This avoids misclassifying stop-entry orders as exits (safety + hedge-side rules).
    return False


def _looks_like_binance_reduceonly_not_required(err: Exception) -> bool:
    s = repr(err).lower()
    return ("reduceonly" in s) and ("not required" in s or "sent when not required" in s or "parameter 'reduceonly'" in s)


def _looks_like_binance_client_id_duplicate(err: Exception) -> bool:
    s = repr(err).lower()
    return ("-4116" in s) or ("clientorderid is duplicated" in s) or ("client order id is duplicated" in s)


def _looks_like_binance_client_id_too_long(err: Exception) -> bool:
    s = repr(err).lower()
    return ("-4015" in s) or ("client order id length" in s) or ("less than 36" in s)


def _looks_like_unknown_order(err: Exception) -> bool:
    """
    Binance/CCXT "already canceled / unknown order / order not found" patterns.
    Treat as idempotent success for cancel.
    """
    s = repr(err).lower()
    return (
        ("-2011" in s)  # Binance unknown order
        or ("unknown order" in s)
        or ("order does not exist" in s)
        or ("order not found" in s)
        or ("order_not_found" in s)
        or ("invalid order" in s and "id" in s)
        or ("cancel" in s and "already" in s and "order" in s)
    )


def _router_auto_client_id_enabled(bot) -> bool:
    v = _cfg_env(bot, "ROUTER_AUTO_CLIENT_ID", "1")
    return bool(_truthy(v))


def _sanitize_client_order_id(coid: Any, *, max_len: int = _BINANCE_CLIENT_ID_MAX) -> Optional[str]:
    """
    Binance requires clientOrderId length < 36 (use max 35).
    Keep only [A-Za-z0-9_-]. If too long, shorten deterministically via hash.
    """
    if coid is None:
        return None
    s = str(coid).strip()
    if not s:
        return None

    safe_chars = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            safe_chars.append(ch)
        # else: drop it completely (no "_" spam)

    s2 = "".join(safe_chars) or "SE"

    if len(s2) <= max_len:
        return s2

    # Deterministic shorten:
    # keep a bit of prefix for human readability + hash tail for uniqueness
    h = hashlib.sha1(s2.encode("utf-8")).hexdigest()  # deterministic
    # reserve 1 + 10 for "_" + 10 hash chars
    keep = max(1, max_len - (1 + 10))
    prefix = s2[:keep]
    compact = f"{prefix}_{h[:10]}"
    return compact[:max_len]


def _sanitize_client_id_fields(p: dict) -> dict:
    """
    Sanitize client id fields even if upstream passes them inside params.
    Handles common Binance keys: clientOrderId, newClientOrderId
    """
    if not isinstance(p, dict):
        return p
    for key in ("clientOrderId", "newClientOrderId"):
        if key in p:
            s = _sanitize_client_order_id(p.get(key))
            if s is None:
                p.pop(key, None)
            else:
                p[key] = s
    return p


def _freshen_client_order_id(existing: Any) -> str:
    """
    Keep <36 chars; append _XXXXXX. Always returns sanitized string.
    """
    base = _sanitize_client_order_id(existing) or "SE"
    # leave room for "_" + 6
    room = 1 + 6
    base = base[: max(1, _BINANCE_CLIENT_ID_MAX - room)]
    suffix = hashlib.sha1(f"{time.time()}|{random.random()}".encode("utf-8")).hexdigest()[:6]
    return _sanitize_client_order_id(f"{base}_{suffix}") or f"{base}_{suffix}"


def _variant_key(sym: str, amt: Any, px: Any, p: dict) -> str:
    """
    Stable-ish dedupe key. We stringify values to avoid unhashable cases.
    """
    try:
        items = sorted((str(k), str(v)) for k, v in (p or {}).items())
        return f"{sym}|{str(amt)}|{str(px)}|{items}"
    except Exception:
        return f"{sym}|{str(amt)}|{str(px)}|{repr(p)}"


def _push_variant(
    variants: List[tuple[str, Any, Any, dict]],
    seen: set[str],
    sym: str,
    amt: Any,
    px: Any,
    p: dict,
) -> None:
    if len(variants) >= _MAX_VARIANTS:
        return
    key = _variant_key(sym, amt, px, p)
    if key in seen:
        return
    seen.add(key)
    variants.append((sym, amt, px, p))


# ----------------------------
# Binance Futures account-mode detection (cached)
# ----------------------------

_MODE_LOCK = asyncio.Lock()
_MODE_CACHE: dict[str, Any] = {
    "ts": 0.0,
    "dualSidePosition": None,
    "multiAssetsMargin": None,
}


async def _detect_binance_futures_modes(ex, *, force: bool = False) -> Tuple[Optional[bool], Optional[bool]]:
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
_SYMBOL_SETTINGS_CACHE: dict[str, Tuple[float, int, str]] = {}  # raw_symbol -> (ts, leverage, margin_mode)


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
        inner = getattr(ex, "exchange", None)
        mk2 = getattr(inner, "markets", None) if inner is not None else None
        if isinstance(mk2, dict) and mk2.get(sym_raw):
            return mk2.get(sym_raw)
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
    try:
        await _ensure_markets_loaded(ex)

        p = params or {}
        is_close_pos = bool(_truthy(p.get("closePosition")))

        market = _market_lookup(ex, sym_raw)

        px_norm: Any = None
        if price is not None:
            if _is_number_like(price):
                px_norm = _price_to_precision_safe(ex, sym_raw, float(price))
            else:
                px_norm = price

        amt_f = _safe_float(amount, 0.0)
        amt_norm: Any = _amount_to_precision_safe(ex, sym_raw, amt_f)
        amt_norm_f = _safe_float(amt_norm, 0.0)

        if is_close_pos:
            return True, float(amt_norm_f), px_norm, "ok_closePosition_amount_can_be_zero"

        if not market:
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
        cached = _SYMBOL_SETTINGS_CACHE.get(sym_raw)
        if cached is not None:
            ts, lev_cached, mm_cached = cached
            if (now - ts) < 60 * 30 and lev_cached == leverage and mm_cached == margin_mode:
                return True
        elif now - last < 60 * 30:
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
        _SYMBOL_SETTINGS_CACHE[sym_raw] = (now, int(leverage), str(margin_mode))
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
        parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
        canon = [_symkey(p) for p in parts if _symkey(p)]
        return set(canon) if canon else None
    except Exception:
        return None


# ----------------------------
# Cancel order (router) — IDEMPOTENT
# ----------------------------

async def _cancel_order_raw(ex, order_id: str, sym_raw: str) -> Tuple[bool, Optional[Exception]]:
    if not order_id:
        return False, None
    try:
        fn = getattr(ex, "cancel_order", None)
        if callable(fn):
            await fn(order_id, sym_raw)
            return True, None
    except Exception as e:
        return False, e
    return False, None


async def cancel_order(bot, order_id: str, symbol: str) -> bool:
    """
    Idempotent cancel:
    - If order already gone/unknown -> return True (goal achieved: it's not live anymore)
    """
    ex = _get_ex(bot)
    if ex is None or not order_id:
        return False

    k = _symkey(symbol)
    sym_raw = _resolve_raw_symbol(bot, k, symbol)

    # try a few symbol spellings, but treat unknown-order as success
    candidates = []
    candidates.append(sym_raw)
    if symbol and symbol != sym_raw:
        candidates.append(symbol)
    if k and k not in (symbol, sym_raw):
        candidates.append(k)

    last_err: Optional[Exception] = None
    saw_unknown = False

    for sym_try in candidates:
        ok, err = await _cancel_order_raw(ex, order_id, sym_try)
        if ok:
            if callable(emit_order_cancel):
                _telemetry_task(emit_order_cancel(bot, k, order_id, True, why="router"))
            return True
        if err is not None:
            if _looks_like_unknown_order(err):
                saw_unknown = True
            else:
                last_err = err

    if saw_unknown and last_err is None:
        # ✅ idempotent success after exhausting symbol candidates
        log_entry.info(f"[router] cancel idempotent success (already gone) | k={k} id={order_id}")
        if callable(emit_order_cancel):
            _telemetry_task(emit_order_cancel(bot, k, order_id, True, why="already_gone"))
        return True

    if callable(emit_order_cancel):
        _telemetry_task(emit_order_cancel(bot, k, order_id, False, why=(repr(last_err)[:120] if last_err else "unknown")))

    if last_err is not None:
        log_entry.warning(f"[router] cancel failed | k={k} id={order_id} err={last_err}")
    return False


# ----------------------------
# Create order (router)
# ----------------------------

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

    if side_l not in ("buy", "sell"):
        log_entry.critical(f"ROUTER BLOCKED → invalid side '{side}' | k={_symkey(symbol)} raw={symbol}")
        return None
    if not type_u:
        log_entry.critical(f"ROUTER BLOCKED → missing order type | k={_symkey(symbol)} raw={symbol}")
        return None

    k = _symkey(symbol)
    sym_raw = _resolve_raw_symbol(bot, k, symbol)

    p = _merge_params(params, {})
    p = _strip_none_params(p)
    p = _sanitize_client_id_fields(p)

    # intents -> params
    if intent_close_position:
        p.setdefault("closePosition", True)

    # closePosition wins over reduceOnly
    if bool(_truthy(p.get("closePosition"))):
        p.pop("reduceOnly", None)
    else:
        if intent_reduce_only:
            p.setdefault("reduceOnly", True)

    _normalize_bool_params(p, ("reduceOnly", "closePosition"))
    p = _strip_none_params(p)
    p = _sanitize_client_id_fields(p)

    # exit-vs-entry early
    is_exit = _is_exit_intent(
        type_u=type_u,
        params=p,
        intent_reduce_only=intent_reduce_only,
        intent_close_position=intent_close_position,
    )

    # STOP/TP/TRAILING must have stopPrice
    if type_u in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "TP_MARKET", "TRAILING_STOP_MARKET"):
        # Trailing stops use activationPrice + callbackRate; stopPrice is NOT required.
        if type_u != "TRAILING_STOP_MARKET":
            if (stop_price is None) and (trigger_price is None) and ("stopPrice" not in p):
                log_entry.critical(f"ROUTER BLOCKED → {type_u} missing stopPrice | k={k} raw={sym_raw}")
                if callable(emit):
                    _telemetry_task(
                        emit(
                            bot,
                            "order.blocked",
                            data={"k": k, "why": "missing_stopPrice", "type": type_u},
                            symbol=k,
                            level="critical",
                        )
                    )
                return None
        if type_u == "TRAILING_STOP_MARKET":
            if (activation_price is None) and ("activationPrice" not in p):
                log_entry.critical(f"ROUTER BLOCKED → {type_u} missing activationPrice | k={k} raw={sym_raw}")
                if callable(emit):
                    _telemetry_task(
                        emit(
                            bot,
                            "order.blocked",
                            data={"k": k, "why": "missing_activationPrice", "type": type_u},
                            symbol=k,
                            level="critical",
                        )
                    )
                return None
            if (callback_rate is None) and ("callbackRate" not in p):
                log_entry.critical(f"ROUTER BLOCKED → {type_u} missing callbackRate | k={k} raw={sym_raw}")
                if callable(emit):
                    _telemetry_task(
                        emit(
                            bot,
                            "order.blocked",
                            data={"k": k, "why": "missing_callbackRate", "type": type_u},
                            symbol=k,
                            level="critical",
                        )
                    )
                return None

    # FIRST LIVE SAFE allowlist applies to ENTRIES only
    if _first_live_safe_enabled(bot) and (not is_exit):
        allow = _allowed_symbols_set(bot)
        if allow is not None and k not in allow:
            log_entry.critical(f"FIRST_LIVE_SAFE BLOCKED → symbol not allowlisted: k={k} allow={sorted(list(allow))}")
            if callable(emit):
                _telemetry_task(
                    emit(bot, "order.blocked", data={"k": k, "why": "first_live_symbol_not_allowed"}, symbol=k, level="critical")
                )
            return None

    # detect binance modes
    is_fut = _is_futures_symbol_ex(ex, sym_raw)
    dual_side, multi_assets = (None, None)
    if is_fut:
        dual_side, multi_assets = await _detect_binance_futures_modes(ex)

    hedge_mode_effective = bool(dual_side) if dual_side is not None else bool(_truthy(hedge_mode))

    # hedge mode positionSide rules
    if is_fut and hedge_mode_effective:
        inferred = "LONG" if side_l == "buy" else "SHORT"
        is_exit2 = is_exit or bool(_truthy(p.get("reduceOnly"))) or bool(_truthy(p.get("closePosition")))

        if is_exit2:
            ps = _infer_position_side(p.get("positionSide")) or _infer_position_side(hedge_side_hint)
            if not ps:
                log_entry.critical(
                    f"ROUTER BLOCKED → hedge exit requires hedge_side_hint LONG/SHORT | "
                    f"k={k} raw={sym_raw} type={type_norm} side={side_l} reduceOnly={p.get('reduceOnly')} closePosition={p.get('closePosition')}"
                )
                if callable(emit):
                    _telemetry_task(
                        emit(bot, "order.blocked", data={"k": k, "why": "missing_hedge_side_hint_for_exit"}, symbol=k, level="critical")
                    )
                return None

            p.setdefault("positionSide", ps)
            if not bool(_truthy(p.get("closePosition"))):
                p.setdefault("reduceOnly", True)
            else:
                p.pop("reduceOnly", None)
        else:
            p.setdefault("positionSide", inferred)

    # stop/trigger -> p['stopPrice']
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
        p["callbackRate"] = _normalize_callback_rate(callback_rate)
    elif "callbackRate" in p:
        p["callbackRate"] = _normalize_callback_rate(p.get("callbackRate"))

    _normalize_bool_params(p, ("reduceOnly", "closePosition"))

    # defense: closePosition strips reduceOnly
    if bool(_truthy(p.get("closePosition"))):
        p.pop("reduceOnly", None)

    p = _strip_none_params(p)
    p = _sanitize_client_id_fields(p)

    # recompute is_exit after param assembly
    is_exit = _is_exit_intent(
        type_u=type_u,
        params=p,
        intent_reduce_only=intent_reduce_only,
        intent_close_position=intent_close_position,
    )

    # ----------------------------
    # clientOrderId logic (ALWAYS sanitize, ALWAYS <36)
    # ----------------------------
    if client_order_id:
        s = _sanitize_client_order_id(client_order_id)
        if s:
            p["clientOrderId"] = s
        else:
            p.pop("clientOrderId", None)
    else:
        want_auto = bool(auto_client_order_id)
        if (not want_auto) and is_exit and _router_auto_client_id_enabled(bot) and ("clientOrderId" not in p):
            want_auto = True

        if want_auto and ("clientOrderId" not in p):
            stop_for_id = p.get("stopPrice")
            s = _make_client_order_id(
                prefix=client_order_id_prefix,
                sym_raw=sym_raw,
                type_norm=type_norm,
                side_l=side_l,
                amount=amount,
                price=price,
                stop_price=stop_for_id,
            )
            p["clientOrderId"] = _sanitize_client_order_id(s) or "SE"

    # final hard sanitize (fail-safe)
    p = _sanitize_client_id_fields(p)

    # log
    try:
        log_entry.info(
            f"[router] SEND k={k} raw={sym_raw} type={type_norm} side={side_l} amt={amount} px={price} "
            f"is_exit={is_exit} reduceOnly={p.get('reduceOnly')} closePosition={p.get('closePosition')} "
            f"positionSide={p.get('positionSide')} clientOrderId={p.get('clientOrderId')} params_keys={sorted(list(p.keys()))}"
        )
    except Exception:
        pass

    if _is_dry_run(bot):
        log_entry.critical(
            f"DRY_RUN ROUTER BLOCKED → k={k} raw={sym_raw} {type_norm} {side_l} amount={amount} price={price} params={p}"
        )
        return _dry_run_order_stub(sym_raw, type_norm, side_l, amount, price, p)

    # futures settings (best effort)
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

    # validation
    amount_for_validation = amount
    if bool(_truthy(p.get("closePosition"))):
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
            f"ROUTER BLOCKED BY EXCHANGE FILTERS → k={k} raw={sym_raw} type={type_norm} side={side_l} "
            f"amount={amount_for_validation} price={price} why={why}"
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

    # closePosition forces amount to 0.0 at send time (ccxt signature requirement)
    if bool(_truthy(p.get("closePosition"))):
        amt_prec = 0.0
        # and *absolute* strip reduceOnly
        p.pop("reduceOnly", None)

    # FIRST LIVE SAFE cap (entries only)
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

        if amt_try is None:
            amt_try = 0.0

        # FINAL safety: sanitize client ids right before send + closePosition strips reduceOnly
        p_try = _sanitize_client_id_fields(dict(p_try))
        if bool(_truthy(p_try.get("closePosition"))):
            p_try.pop("reduceOnly", None)

        if type_norm == "market":
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, params=p_try)

        if type_norm == "limit":
            if px_try is None:
                raise RuntimeError("limit order missing price")
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, price=px_try, params=p_try)

        if px_try is None:
            return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, params=p_try)

        return await fn(symbol=raw_symbol, type=type_norm, side=side_l, amount=amt_try, price=px_try, params=p_try)

    # ----------------------------
    # Variants (BOUNDED + DEDUPED)
    # ----------------------------
    variants: List[tuple[str, Any, Any, dict]] = []
    seen: set[str] = set()

    _push_variant(variants, seen, sym_raw, amt_prec, px_prec, dict(p))

    # reduceOnly stripped variant
    if "reduceOnly" in p:
        p_ro = dict(p)
        p_ro.pop("reduceOnly", None)
        _push_variant(variants, seen, sym_raw, amt_prec, px_prec, p_ro)

    # closePosition variant for stop/tp: enforce reduceOnly removed
    if type_u in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "TP_MARKET"):
        if bool(_truthy(p.get("closePosition"))) or intent_close_position:
            p2 = dict(p)
            p2["closePosition"] = True
            p2.pop("reduceOnly", None)
            _push_variant(variants, seen, sym_raw, 0.0, px_prec, p2)

    # trailing drop callback variant
    if type_u == "TRAILING_STOP_MARKET" and "callbackRate" in p:
        p2 = dict(p)
        p2.pop("callbackRate", None)
        _push_variant(variants, seen, sym_raw, amt_prec, px_prec, p2)

    # symbol fallback
    if sym_raw != symbol:
        _push_variant(variants, seen, symbol, amt_prec, px_prec, dict(p))

    tries = 0
    for attempt in range(max_attempts):
        for (raw_sym, amt_try, px_try, p_try) in list(variants):
            tries += 1
            try:
                res = await _attempt(raw_sym, amt_try, px_try, p_try)
                if callable(emit_order_create):
                    _telemetry_task(emit_order_create(bot, k, res, intent=f"{type_u}:{side_l}"))
                return res
            except Exception as e:
                last_err = e

                # -1106 reduceOnly not required → add stripped variant
                if _looks_like_binance_reduceonly_not_required(e):
                    if "reduceOnly" in p_try:
                        p3 = dict(p_try)
                        p3.pop("reduceOnly", None)
                        _push_variant(variants, seen, raw_sym, amt_try, px_try, p3)

                # -4015 id too long → sanitize again (hash/compact) and retry
                if _looks_like_binance_client_id_too_long(e):
                    coid = (p_try or {}).get("clientOrderId")
                    if coid:
                        p5 = dict(p_try)
                        p5["clientOrderId"] = _sanitize_client_order_id(coid)
                        _push_variant(variants, seen, raw_sym, amt_try, px_try, p5)
                        log_entry.warning(
                            f"[router] BINANCE CLIENT_ID TOO LONG → sanitized clientOrderId {coid} -> {p5.get('clientOrderId')} | k={k}"
                        )

                # ✅ -4116 clientOrderId duplicated → add fresh clientOrderId variant (ALWAYS short)
                if _looks_like_binance_client_id_duplicate(e):
                    coid = (p_try or {}).get("clientOrderId")
                    if coid:
                        p4 = dict(p_try)
                        p4["clientOrderId"] = _freshen_client_order_id(coid)
                        _push_variant(variants, seen, raw_sym, amt_try, px_try, p4)
                        log_entry.warning(
                            f"[router] BINANCE DUP CLIENT_ID → freshened clientOrderId {coid} -> {p4.get('clientOrderId')} | k={k}"
                        )

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
