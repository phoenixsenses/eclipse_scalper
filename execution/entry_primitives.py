from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def resolve_raw_symbol(bot: Any, canonical_symbol: str, fallback_symbol: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict):
            val = raw_map.get(canonical_symbol)
            if val:
                return str(val)
    except Exception:
        pass
    return fallback_symbol


def order_filled(order: dict) -> float:
    if not isinstance(order, dict):
        return 0.0
    filled = safe_float(order.get("filled"), 0.0)
    if filled > 0:
        return filled
    info = order.get("info") or {}
    filled2 = safe_float(info.get("executedQty"), 0.0)
    if filled2 > 0:
        return filled2
    return 0.0


def order_avg_price(order: dict, fallback: float) -> float:
    if not isinstance(order, dict):
        return float(fallback)
    avg = safe_float(order.get("average"), 0.0)
    if avg > 0:
        return avg
    price = safe_float(order.get("price"), 0.0)
    if price > 0:
        return price
    info = order.get("info") or {}
    avg2 = safe_float(info.get("avgPrice"), 0.0)
    if avg2 > 0:
        return avg2
    quote = safe_float(info.get("cummulativeQuoteQty"), 0.0)
    qty = safe_float(info.get("executedQty"), 0.0)
    if quote > 0 and qty > 0:
        return quote / qty
    return float(fallback)


def build_staged_protection_plan(
    *,
    requested_qty: float,
    filled_qty: float,
    min_fill_ratio: float = 0.5,
    trailing_enabled: bool = False,
) -> dict[str, Any]:
    req = abs(safe_float(requested_qty, 0.0))
    filled = abs(safe_float(filled_qty, 0.0))
    ratio = (filled / req) if req > 0 else 0.0
    min_ratio = max(0.0, safe_float(min_fill_ratio, 0.5))
    partial = (req > 0.0) and (filled > 0.0) and (filled < req)
    stage = "none"
    stage1 = False
    stage2 = False
    stage3 = False
    flatten_required = False
    if filled > 0.0:
        stage1 = True
        stage = "stage1_emergency"
        if ratio >= min_ratio:
            stage2 = True
            stage = "stage2_core"
            if bool(trailing_enabled):
                stage3 = True
                stage = "stage3_trailing"
        else:
            flatten_required = True
            stage = "stage1_emergency_partial_underfill"
    return {
        "stage": stage,
        "stage1_required": bool(stage1),
        "stage2_required": bool(stage2),
        "stage3_required": bool(stage3),
        "partial_fill": bool(partial),
        "flatten_required": bool(flatten_required),
        "requested_qty": float(req),
        "filled_qty": float(filled),
        "fill_ratio": float(ratio),
        "coverage_required_qty": float(filled),
        "min_fill_ratio": float(min_ratio),
    }
