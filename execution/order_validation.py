# execution/order_validation.py — Pure validation/normalization helpers extracted from order_router.py
# No side effects, no I/O, no exchange calls (except precision wrappers that delegate to ccxt).

from __future__ import annotations

import re
import hashlib
from typing import Any, Optional, Tuple

from execution.runtime_helpers import safe_float as _safe_float, truthy as _truthy, symkey as _symkey

# Binance: clientOrderId MUST be < 36 chars (so max 35)
_BINANCE_CLIENT_ID_MAX = 35


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


def _normalize_pct(val: Any, default: float) -> float:
    """
    Accept percent as 0.5 or 0.5% (0.5) or 0.005.
    If value > 1, assume it's a percent (e.g., 0.5 => 0.5%, 2 => 2%).
    """
    v = _safe_float(val, default)
    if v <= 0:
        return 0.0
    if v > 1.0:
        return v / 100.0
    return v


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


def _is_futures_symbol(sym_raw: str) -> bool:
    s = sym_raw or ""
    return (":USDT" in s) or (":USD" in s) or ("PERP" in s.upper())


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
