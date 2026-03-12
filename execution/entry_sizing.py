# execution/entry_sizing.py — Entry sizing helpers (pure functions)
# Extracted from entry_loop.py (Phase 7.2) — guardian-safe, never raises

from __future__ import annotations

import os
from typing import Any, Optional

from execution.runtime_helpers import (
    cfg_env_bool as _cfg_env_bool,
    cfg_env_float as _cfg_env_float,
    symkey as _symkey,
)


def _env_float(*names: str, default: float = 0.0) -> float:
    for n in names:
        try:
            v = os.getenv(n, "")
            s = str(v).strip()
            if s != "":
                return float(s)
        except Exception:
            pass
    return float(default)


def _cfg_float(cfg_obj, *names: str, default: float = 0.0) -> float:
    for n in names:
        try:
            if cfg_obj is None:
                continue
            v = getattr(cfg_obj, n, None)
            if v is None:
                continue
            s = float(v)
            if s != 0.0:
                return s
        except Exception:
            pass
    return float(default)


def _resolve_sizing(bot) -> tuple[float, float]:
    """
    Returns (fixed_qty, fixed_notional_usdt)
    Priority:
      1) ENV (supports aliases)
      2) cfg (supports aliases)
    """
    cfg_obj = getattr(bot, "cfg", None)

    fixed_qty = _env_float("FIXED_QTY", "ORDER_QTY", "QTY", default=0.0)
    fixed_notional = _env_float(
        "FIXED_NOTIONAL_USDT",
        "ORDER_NOTIONAL_USDT",
        "FIXED_NOTIONAL",
        "NOTIONAL_USDT",
        "FIXED_USDT",
        "BASE_NOTIONAL_USDT",
        default=0.0,
    )

    if fixed_qty <= 0:
        fixed_qty = _cfg_float(cfg_obj, "FIXED_QTY", "ORDER_QTY", "QTY", default=0.0)

    if fixed_notional <= 0:
        fixed_notional = _cfg_float(
            cfg_obj,
            "FIXED_NOTIONAL_USDT",
            "ORDER_NOTIONAL_USDT",
            "FIXED_NOTIONAL",
            "NOTIONAL_USDT",
            "FIXED_USDT",
            "BASE_NOTIONAL_USDT",
            default=0.0,
        )

    return float(fixed_qty), float(fixed_notional)


def _resolve_symbol_sizing(bot, symbol: str) -> tuple[float, float]:
    """
    Per-symbol overrides:
      FIXED_QTY_<SYM>, FIXED_NOTIONAL_USDT_<SYM>
    Falls back to global sizing if not set.
    """
    base_qty, base_notional = _resolve_sizing(bot)
    sym = _symkey(symbol)
    if not sym:
        return base_qty, base_notional
    base = sym[:-4] if sym.endswith("USDT") and len(sym) > 4 else sym

    fixed_qty = _env_float(f"FIXED_QTY_{base}", f"FIXED_QTY_{sym}", default=0.0)
    fixed_notional = _env_float(
        f"FIXED_NOTIONAL_USDT_{base}",
        f"FIXED_NOTIONAL_USDT_{sym}",
        f"FIXED_NOTIONAL_{base}",
        f"FIXED_NOTIONAL_{sym}",
        default=0.0,
    )

    if fixed_qty <= 0:
        fixed_qty = base_qty
    if fixed_notional <= 0:
        fixed_notional = base_notional
    return float(fixed_qty), float(fixed_notional)


def _confidence_notional_scale(bot, confidence: float) -> tuple[float, str]:
    try:
        enabled = _cfg_env_bool(bot, "ENTRY_CONF_SCALE_ENABLED", False)
        if not enabled:
            return 1.0, ""
        min_conf = float(_cfg_env_float(bot, "ENTRY_CONF_SCALE_MIN_CONF", 0.0) or 0.0)
        max_conf = float(_cfg_env_float(bot, "ENTRY_CONF_SCALE_MAX_CONF", 1.0) or 1.0)
        min_scale = float(_cfg_env_float(bot, "ENTRY_CONF_SCALE_MIN", 0.5) or 0.5)
        max_scale = float(_cfg_env_float(bot, "ENTRY_CONF_SCALE_MAX", 1.0) or 1.0)
        conf = float(confidence or 0.0)
        if max_conf <= min_conf:
            return max_scale, ""
        if conf <= min_conf:
            return max(0.0, min_scale), f"confidence<{min_conf:.2f}"
        if conf >= max_conf:
            return max_scale, ""
        ratio = (conf - min_conf) / (max_conf - min_conf)
        scale = min_scale + ratio * (max_scale - min_scale)
        return float(scale), f"confidence={conf:.2f}"
    except Exception:
        return 1.0, ""


def _get_price(bot, symbol: str) -> float:
    """
    Best-effort last price getter for qty-from-notional conversion.
    """
    k = _symkey(symbol)
    px = 0.0
    try:
        data = getattr(bot, "data", None)
        gp = getattr(data, "get_price", None) if data is not None else None
        if callable(gp):
            try:
                px = float(gp(k, in_position=False) or 0.0)
            except TypeError:
                px = float(gp(k) or 0.0)
    except Exception:
        px = 0.0

    if px <= 0:
        try:
            price_map = getattr(getattr(bot, "data", None), "price", {}) or {}
            if isinstance(price_map, dict):
                px = float(price_map.get(k, 0.0) or 0.0)
        except Exception:
            px = 0.0

    return float(px) if px > 0 else 0.0
