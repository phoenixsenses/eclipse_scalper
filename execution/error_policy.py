from __future__ import annotations

import re
from typing import Any, Callable, Optional

ERROR_CLASS_RETRYABLE = "retryable"
ERROR_CLASS_RETRYABLE_MOD = "retryable_with_modification"
ERROR_CLASS_IDEMPOTENT = "idempotent_safe"
ERROR_CLASS_FATAL = "fatal"


def _error_text(err: Exception) -> str:
    try:
        return f"{repr(err)} {str(err)}".lower()
    except Exception:
        return str(err or "").lower()


def _map_code(reason: str, map_reason: Optional[Callable[[str], str]], default: str) -> str:
    if callable(map_reason):
        try:
            return str(map_reason(reason))
        except Exception:
            pass
    return str(default)


def _extract_binance_code(msg: str) -> Optional[int]:
    try:
        hits = re.findall(r"-\d{4,5}", msg)
        if not hits:
            return None
        return int(hits[0])
    except Exception:
        return None


def _binance_filter_reason(msg: str) -> Optional[str]:
    if "filter failure" not in msg:
        return None
    if "price_filter" in msg:
        return "price_filter"
    if "min_notional" in msg or "notional" in msg:
        return "min_notional"
    if "lot_size" in msg:
        return "lot_size"
    if "market_lot_size" in msg:
        return "market_lot_size"
    return "filter_failure"


def _is_futures_symbol(sym_raw: str) -> bool:
    s = str(sym_raw or "")
    return (":USDT" in s) or (":USD" in s) or ("PERP" in s.upper())


def _is_futures_symbol_ex(ex: Any, sym_raw: Optional[str]) -> bool:
    if not sym_raw:
        return False
    try:
        market = None
        fn = getattr(ex, "market", None)
        if callable(fn):
            market = fn(sym_raw)
        if isinstance(market, dict):
            if bool(market.get("contract")) or bool(market.get("swap")) or bool(market.get("future")):
                return True
    except Exception:
        pass
    return _is_futures_symbol(str(sym_raw))


def looks_like_binance_reduceonly_not_required(err: Exception) -> bool:
    s = repr(err).lower()
    return ("reduceonly" in s) and ("not required" in s or "sent when not required" in s or "parameter 'reduceonly'" in s)


def looks_like_binance_client_id_duplicate(err: Exception) -> bool:
    s = repr(err).lower()
    return ("-4116" in s) or ("clientorderid is duplicated" in s) or ("client order id is duplicated" in s)


def looks_like_binance_client_id_too_long(err: Exception) -> bool:
    s = repr(err).lower()
    return ("-4015" in s) or ("client order id length" in s) or ("less than 36" in s)


def looks_like_unknown_order(err: Exception) -> bool:
    s = repr(err).lower()
    return (
        ("-2011" in s)
        or ("unknown order" in s)
        or ("order does not exist" in s)
        or ("order not found" in s)
        or ("order_not_found" in s)
        or ("invalid order" in s and "id" in s)
        or ("cancel" in s and "already" in s and "order" in s)
    )


def classify_order_error(
    err: Exception,
    *,
    ex: Any = None,
    sym_raw: Optional[str] = None,
    map_reason: Optional[Callable[[str], str]] = None,
) -> tuple[bool, str, str]:
    msg = _error_text(err)
    ex_id = ""
    try:
        if ex is not None:
            ex_id = str(getattr(ex, "id", "") or "").strip().lower()
            if not ex_id:
                inner = getattr(ex, "exchange", None)
                ex_id = str(getattr(inner, "id", "") or "").strip().lower()
    except Exception:
        ex_id = ""
    is_futures = _is_futures_symbol_ex(ex, sym_raw)

    bin_filter = _binance_filter_reason(msg)
    if bin_filter:
        return False, bin_filter, _map_code(bin_filter, map_reason, "ERR_ROUTER_BLOCK")
    if "position side" in msg or "dual side position" in msg or "hedge mode" in msg:
        return False, "position_side", _map_code("position_side", map_reason, "ERR_ROUTER_BLOCK")
    if "reduceonly" in msg and ("rejected" in msg or "not required" in msg):
        return True, "reduceonly", _map_code("reduceonly", map_reason, "ERR_ROUTER_BLOCK")

    bin_code = _extract_binance_code(msg)
    if bin_code is not None:
        if bin_code in (-1000, -1001, -1003, -1006, -1007, -1008):
            return True, "exchange_busy", _map_code("exchange_busy", map_reason, "ERR_UNKNOWN")
        if bin_code in (-1021,):
            return True, "timestamp", _map_code("timestamp", map_reason, "ERR_UNKNOWN")
        if bin_code in (-1022,):
            return False, "auth", _map_code("auth", map_reason, "ERR_ROUTER_BLOCK")
        if bin_code in (-1100, -1101, -1102, -1103):
            return False, "invalid_params", _map_code("invalid_params", map_reason, "ERR_ROUTER_BLOCK")
        if bin_code in (-1111, -1112):
            return False, "price_filter", _map_code("price", map_reason, "ERR_ROUTER_BLOCK")
        if bin_code in (-1121,):
            return False, "invalid_symbol", _map_code("symbol", map_reason, "ERR_ROUTER_BLOCK")
        if bin_code in (-2019,):
            return False, "margin_insufficient", _map_code("margin", map_reason, "ERR_MARGIN")
        if bin_code in (-2021,):
            return False, "stop_price_invalid", _map_code("stop", map_reason, "ERR_ROUTER_BLOCK")
        if bin_code in (-2011,) and (not is_futures):
            return False, "unknown_order", _map_code("unknown_order", map_reason, "ERR_ROUTER_BLOCK")

    if any(x in msg for x in ("insufficient", "margin", "balance")):
        return False, "margin_insufficient", _map_code("margin", map_reason, "ERR_MARGIN")
    if any(x in msg for x in ("invalid symbol", "symbol not found", "unknown symbol")):
        return False, "invalid_symbol", _map_code("symbol", map_reason, "ERR_ROUTER_BLOCK")
    if any(x in msg for x in ("min notional", "min amount", "notional", "lot size")):
        return False, "min_notional", _map_code("min_notional", map_reason, "ERR_MIN_NOTIONAL")
    if any(x in msg for x in ("price filter", "precision", "invalid price", "bad price")):
        return False, "price_filter", _map_code("price", map_reason, "ERR_ROUTER_BLOCK")
    if any(x in msg for x in ("order would trigger immediately", "stop price")):
        return False, "stop_price_invalid", _map_code("stop", map_reason, "ERR_ROUTER_BLOCK")
    if any(x in msg for x in ("timeout", "timed out", "temporarily unavailable", "connection", "econnreset", "network")):
        return True, "network", _map_code("network", map_reason, "ERR_UNKNOWN")

    if ex_id == "coinbase":
        if any(
            x in msg
            for x in ("rate limit", "too many requests", "service unavailable", "internal server error", "gateway timeout", "engine is overloaded")
        ):
            return True, "exchange_busy", _map_code("exchange_busy", map_reason, "ERR_UNKNOWN")
        if any(x in msg for x in ("request timestamp expired", "timestamp")):
            return True, "timestamp", _map_code("timestamp", map_reason, "ERR_UNKNOWN")
        if any(x in msg for x in ("insufficient funds", "insufficient balance")):
            return False, "margin_insufficient", _map_code("margin", map_reason, "ERR_MARGIN")
        if any(
            x in msg
            for x in ("size is too small", "minimum size", "minimum order size", "invalid product", "product not found", "invalid side", "post only would execute")
        ):
            return False, "invalid_params", _map_code("invalid_params", map_reason, "ERR_ROUTER_BLOCK")

    # Conservative fallback: unknown classes should not fan out retries blindly.
    return False, "unknown", _map_code("unknown", map_reason, "ERR_UNKNOWN")


def classify_order_error_policy(
    err: Exception,
    *,
    ex: Any = None,
    sym_raw: Optional[str] = None,
    map_reason: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    retryable, reason, code = classify_order_error(err, ex=ex, sym_raw=sym_raw, map_reason=map_reason)
    rs = str(reason or "").strip().lower()
    if looks_like_binance_client_id_duplicate(err) or looks_like_binance_client_id_too_long(err):
        err_class = ERROR_CLASS_RETRYABLE_MOD
    elif looks_like_binance_reduceonly_not_required(err) or rs == "reduceonly":
        err_class = ERROR_CLASS_RETRYABLE_MOD
    elif rs == "unknown_order" or looks_like_unknown_order(err):
        err_class = ERROR_CLASS_IDEMPOTENT
    elif retryable:
        err_class = ERROR_CLASS_RETRYABLE
    else:
        err_class = ERROR_CLASS_FATAL
    retryable_out = bool(retryable) or (err_class == ERROR_CLASS_RETRYABLE_MOD)
    return {
        "retryable": bool(retryable_out),
        "reason": str(reason),
        "code": str(code),
        "error_class": err_class,
        "retry_with_modification": err_class == ERROR_CLASS_RETRYABLE_MOD,
        "idempotent_safe": err_class == ERROR_CLASS_IDEMPOTENT,
        "fatal": err_class == ERROR_CLASS_FATAL,
    }
