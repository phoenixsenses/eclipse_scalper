# execution/order_classification.py — Error classification functions extracted from order_router.py
# Guardian-safe: all imports wrapped in try/except

from __future__ import annotations

import re
from typing import Any, Optional

try:
    from execution import error_policy as _error_policy  # type: ignore
except Exception:
    _error_policy = None

try:
    from execution.error_codes import map_reason  # type: ignore
except Exception:
    map_reason = None

# ── Error class constants ──────────────────────────────────────────
_ERROR_CLASS_RETRYABLE = "retryable"
_ERROR_CLASS_RETRYABLE_MOD = "retryable_with_modification"
_ERROR_CLASS_IDEMPOTENT = "idempotent_safe"
_ERROR_CLASS_FATAL = "fatal"


# ── Helper functions ───────────────────────────────────────────────

def _error_text(err: Exception) -> str:
    try:
        return f"{repr(err)} {str(err)}".lower()
    except Exception:
        return str(err or "").lower()


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


def _classify_order_error(err: Exception, *, ex=None, sym_raw: Optional[str] = None) -> tuple[bool, str, str]:
    if _error_policy is None:
        msg = _error_text(err)
        binance_code = _extract_binance_code(msg)
        filter_reason = _binance_filter_reason(msg)
        if filter_reason is not None:
            return False, filter_reason, (map_reason(filter_reason) if callable(map_reason) else "ERR_UNKNOWN")
        if binance_code == -2019 or any(x in msg for x in ("margin is insufficient", "insufficient margin", "insufficient balance")):
            return False, "margin_insufficient", (map_reason("margin_insufficient") if callable(map_reason) else "ERR_UNKNOWN")
        if binance_code == -1021 or "recvwindow" in msg or "timestamp for this request is outside" in msg:
            return True, "timestamp", (map_reason("timestamp") if callable(map_reason) else "ERR_UNKNOWN")
        if any(x in msg for x in ("timeout", "timed out", "temporarily unavailable", "connection", "econnreset", "network")):
            return True, "network", (map_reason("network") if callable(map_reason) else "ERR_UNKNOWN")
        if "symbol not found" in msg or "invalid symbol" in msg:
            return False, "invalid_symbol", (map_reason("invalid_symbol") if callable(map_reason) else "ERR_UNKNOWN")
        return True, "unknown", (map_reason(msg) if callable(map_reason) else "ERR_UNKNOWN")
    return _error_policy.classify_order_error(err, ex=ex, sym_raw=sym_raw, map_reason=map_reason)


def _looks_like_binance_reduceonly_not_required(err: Exception) -> bool:
    if _error_policy is not None:
        return bool(_error_policy.looks_like_binance_reduceonly_not_required(err))
    s = repr(err).lower()
    return ("reduceonly" in s) and ("not required" in s or "sent when not required" in s or "parameter 'reduceonly'" in s)


def _looks_like_binance_client_id_duplicate(err: Exception) -> bool:
    if _error_policy is not None:
        return bool(_error_policy.looks_like_binance_client_id_duplicate(err))
    s = repr(err).lower()
    return ("-4116" in s) or ("clientorderid is duplicated" in s) or ("client order id is duplicated" in s)


def _looks_like_binance_client_id_too_long(err: Exception) -> bool:
    if _error_policy is not None:
        return bool(_error_policy.looks_like_binance_client_id_too_long(err))
    s = repr(err).lower()
    return ("-4015" in s) or ("client order id length" in s) or ("less than 36" in s)


def _looks_like_unknown_order(err: Exception) -> bool:
    """
    Binance/CCXT "already canceled / unknown order / order not found" patterns.
    Treat as idempotent success for cancel.
    """
    if _error_policy is not None:
        return bool(_error_policy.looks_like_unknown_order(err))
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


def _error_class_from_reason(err: Exception, *, retryable: bool, reason: str) -> str:
    if _error_policy is None:
        rs = str(reason or "").strip().lower()
        if _looks_like_binance_client_id_duplicate(err) or _looks_like_binance_client_id_too_long(err):
            return _ERROR_CLASS_RETRYABLE_MOD
        if _looks_like_binance_reduceonly_not_required(err) or rs == "reduceonly":
            return _ERROR_CLASS_RETRYABLE_MOD
        if rs == "unknown_order" or _looks_like_unknown_order(err):
            return _ERROR_CLASS_IDEMPOTENT
        if retryable:
            return _ERROR_CLASS_RETRYABLE
        return _ERROR_CLASS_FATAL
    policy = _error_policy.classify_order_error_policy(err, ex=None, sym_raw=None, map_reason=map_reason)
    return str(policy.get("error_class") or _ERROR_CLASS_FATAL)


def _classify_order_error_policy(err: Exception, *, ex=None, sym_raw: Optional[str] = None) -> dict[str, Any]:
    if _error_policy is None:
        retryable, reason, code = _classify_order_error(err, ex=ex, sym_raw=sym_raw)
        err_class = _error_class_from_reason(err, retryable=retryable, reason=reason)
        return {
            "retryable": bool(retryable),
            "reason": str(reason),
            "code": str(code),
            "error_class": err_class,
            "retry_with_modification": err_class == _ERROR_CLASS_RETRYABLE_MOD,
            "idempotent_safe": err_class == _ERROR_CLASS_IDEMPOTENT,
            "fatal": err_class == _ERROR_CLASS_FATAL,
        }
    return _error_policy.classify_order_error_policy(err, ex=ex, sym_raw=sym_raw, map_reason=map_reason)
