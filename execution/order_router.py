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
import re
from typing import Any, Dict, Optional, Tuple, List

from utils.logging import log_entry

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit_order_create, emit_order_cancel, emit, emit_throttled, bump  # type: ignore
except Exception:
    emit_order_create = None
    emit_order_cancel = None
    emit = None
    emit_throttled = None
    bump = None

try:
    from execution.adaptive_guard import get_leverage_scale as get_adaptive_leverage_scale  # type: ignore
except Exception:
    get_adaptive_leverage_scale = None

try:
    from execution.error_codes import map_reason  # type: ignore
except Exception:
    map_reason = None

try:
    from execution import error_policy as _error_policy  # type: ignore
except Exception:
    _error_policy = None

try:
    from execution import replace_manager as _replace_manager  # type: ignore
except Exception:
    _replace_manager = None

try:
    from execution import state_machine as _state_machine  # type: ignore
except Exception:
    _state_machine = None

try:
    from execution import event_journal as _event_journal  # type: ignore
except Exception:
    _event_journal = None

try:
    from execution import intent_ledger as _intent_ledger  # type: ignore
except Exception:
    _intent_ledger = None


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


def _parse_symbol_overrides(raw: Any) -> dict:
    """
    Parse "BTCUSDT=0.2,ETHUSDT=0.3" into { "BTCUSDT": 0.2, ... }.
    Accepts ; or , separators.
    """
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    out: dict = {}
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = _symkey(k)
        if not k:
            continue
        out[k] = _safe_float(v, 0.0)
    return out


def _parse_group_kv(raw: Any) -> dict:
    """
    Parse "MEME=5,MAJOR=10" into {"MEME": 5, "MAJOR": 10}.
    """
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    out: dict = {}
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = str(k).strip()
        if not k:
            continue
        out[k] = _safe_float(v, 0.0)
    return out


_DEFAULT_RETRY_POLICIES: dict[str, dict[str, Any]] = {
    "network": {"extra_attempts": 2, "base_delay": 0.35, "max_delay": 3.0},
    "exchange_busy": {"extra_attempts": 2, "base_delay": 0.45, "max_delay": 4.0},
    "timestamp": {"extra_attempts": 1, "base_delay": 0.3, "max_delay": 2.5},
}

_ERROR_CLASS_RETRYABLE = "retryable"
_ERROR_CLASS_RETRYABLE_MOD = "retryable_with_modification"
_ERROR_CLASS_IDEMPOTENT = "idempotent_safe"
_ERROR_CLASS_FATAL = "fatal"

_EXCHANGE_RETRY_POLICIES: dict[str, dict[str, dict[str, Any]]] = {
    "binance": {
        "exchange_busy": {"extra_attempts": 2, "base_delay": 0.45, "max_delay": 4.0},
        "timestamp": {"extra_attempts": 1, "base_delay": 0.3, "max_delay": 2.5},
    },
    "coinbase": {
        "exchange_busy": {"extra_attempts": 3, "base_delay": 0.6, "max_delay": 6.0},
        "network": {"extra_attempts": 3, "base_delay": 0.45, "max_delay": 5.0},
        "timestamp": {"extra_attempts": 1, "base_delay": 0.35, "max_delay": 3.0},
    },
}


def _parse_retry_policy_entry(value: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    parts = [p.strip() for p in value.split(":") if p.strip()]
    if not parts:
        return out
    try:
        out["max_attempts"] = max(1, int(float(parts[0])))
    except Exception:
        pass
    if len(parts) >= 2:
        try:
            out["base_delay"] = float(parts[1])
        except Exception:
            pass
    if len(parts) >= 3:
        try:
            out["max_delay"] = float(parts[2])
        except Exception:
            pass
    if len(parts) >= 4:
        try:
            out["extra_attempts"] = max(0, int(float(parts[3])))
        except Exception:
            pass
    return out


def _collect_retry_policy_overrides(bot) -> dict[str, dict[str, Any]]:
    raw = str(_cfg_env(bot, "ROUTER_RETRY_POLICY", "") or "")
    if not raw:
        return {}
    out: dict[str, dict[str, Any]] = {}
    entries = [e.strip() for e in raw.replace(";", ",").split(",") if e.strip()]
    for entry in entries:
        if "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        key = str(name).strip().lower()
        if not key:
            continue
        parsed = _parse_retry_policy_entry(value)
        if parsed:
            out[key] = parsed
    return out


def _exchange_id(bot) -> str:
    try:
        ex = _get_ex(bot)
        if ex is None:
            return ""
        eid = getattr(ex, "id", None)
        if eid:
            return str(eid).strip().lower()
        inner = getattr(ex, "exchange", None)
        eid2 = getattr(inner, "id", None) if inner is not None else None
        if eid2:
            return str(eid2).strip().lower()
    except Exception:
        pass
    return ""


def _collect_exchange_retry_policy_overrides(bot, ex_id: str) -> dict[str, dict[str, Any]]:
    if not ex_id:
        return {}
    raw = str(_cfg_env(bot, f"ROUTER_RETRY_POLICY_{str(ex_id).upper()}", "") or "")
    if not raw:
        return {}
    out: dict[str, dict[str, Any]] = {}
    entries = [e.strip() for e in raw.replace(";", ",").split(",") if e.strip()]
    for entry in entries:
        if "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        key = str(name).strip().lower()
        if not key:
            continue
        parsed = _parse_retry_policy_entry(value)
        if parsed:
            out[key] = parsed
    return out


def _build_retry_policies(bot) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _DEFAULT_RETRY_POLICIES.items()}
    ex_id = _exchange_id(bot)
    ex_defaults = _EXCHANGE_RETRY_POLICIES.get(ex_id, {})
    for key, data in ex_defaults.items():
        merged = dict(policies.get(key, {}))
        merged.update(data)
        policies[key] = merged
    overrides = _collect_retry_policy_overrides(bot)
    ex_overrides = _collect_exchange_retry_policy_overrides(bot, ex_id)
    if ex_overrides:
        overrides.update(ex_overrides)
    for key, data in overrides.items():
        existing = policies.get(key, {})
        merged = dict(existing)
        merged.update(data)
        policies[key] = merged
    policies.setdefault("default", {})
    return policies


def _parse_groups(raw: Any) -> dict:
    """
    Parse "MEME:BTCUSDT,SHIBUSDT;MAJOR:BTCUSDT,ETHUSDT" into dict.
    """
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    out: dict = {}
    blocks = [b.strip() for b in s.split(";") if b.strip()]
    for b in blocks:
        if ":" not in b:
            continue
        g, syms = b.split(":", 1)
        g = str(g).strip()
        if not g:
            continue
        items = [x.strip() for x in syms.split(",") if x.strip()]
        out[g] = [_symkey(x) for x in items if _symkey(x)]
    return out


def _groups_for_symbol(k: str, groups: dict) -> list[str]:
    out: list[str] = []
    if not groups:
        return out
    for g, syms in groups.items():
        try:
            if k in syms:
                out.append(g)
        except Exception:
            continue
    return out


def _group_open_count(bot, groups: dict, group_name: str, *, exclude: Optional[str] = None) -> int:
    try:
        st = getattr(bot, "state", None)
        pos_map = getattr(st, "positions", None) if st is not None else None
        if not isinstance(pos_map, dict):
            return 0
        syms = groups.get(group_name) or []
        if not syms:
            return 0
        count = 0
        excl = _symkey(exclude) if exclude else None
        for s in syms:
            if excl and _symkey(s) == excl:
                continue
            if s in pos_map and getattr(pos_map.get(s), "size", 0.0) not in (0, None):
                count += 1
        return count
    except Exception:
        return 0


def _group_notional(bot, groups: dict, group_name: str, *, exclude: Optional[str] = None) -> float:
    try:
        st = getattr(bot, "state", None)
        pos_map = getattr(st, "positions", None) if st is not None else None
        if not isinstance(pos_map, dict):
            return 0.0
        syms = groups.get(group_name) or []
        if not syms:
            return 0.0
        total = 0.0
        excl = _symkey(exclude) if exclude else None
        for s in syms:
            if excl and _symkey(s) == excl:
                continue
            pos = pos_map.get(s)
            if pos is None:
                continue
            size = _safe_float(getattr(pos, "size", 0.0), 0.0)
            entry_px = _safe_float(getattr(pos, "entry_price", 0.0), 0.0)
            if size and entry_px > 0:
                total += abs(size) * entry_px
        return float(total)
    except Exception:
        return 0.0


def _resolve_leverage(bot, k: str, sym_raw: str, *, is_exit: bool = False) -> int:
    base = _safe_float(_cfg_env(bot, "LEVERAGE", _cfg(bot, "LEVERAGE", 1)), 1)
    lev = base

    sym_key = _symkey(k or sym_raw)
    try:
        ev = os.getenv(f"LEVERAGE_{sym_key}", None)
        if ev is not None and str(ev).strip() != "":
            lev = _safe_float(ev, lev)
    except Exception:
        pass

    per_sym = _parse_symbol_overrides(_cfg_env(bot, "LEVERAGE_BY_SYMBOL", None))
    if sym_key in per_sym:
        lev = _safe_float(per_sym.get(sym_key), lev)

    groups = _parse_groups(_cfg_env(bot, "CORR_GROUPS", ""))
    group_lev = _parse_group_kv(_cfg_env(bot, "LEVERAGE_BY_GROUP", ""))
    if groups and group_lev:
        hits = _groups_for_symbol(sym_key, groups)
        if hits:
            vals = [group_lev.get(h) for h in hits if group_lev.get(h) is not None]
            if vals:
                lev = min(float(lev), float(min(vals)))

    # Dynamic group scaling (reduce leverage as group exposure grows)
    if (not is_exit) and _truthy(_cfg_env(bot, "LEVERAGE_GROUP_DYNAMIC", True)):
        scale = _safe_float(_cfg_env(bot, "LEVERAGE_GROUP_SCALE", 0.7), 0.7)
        scale_min = _safe_float(_cfg_env(bot, "LEVERAGE_GROUP_SCALE_MIN", 1), 1.0)
        if groups and scale > 0 and scale < 1.0:
            hits = _groups_for_symbol(sym_key, groups)
            if hits:
                exclude_self = _truthy(_cfg_env(bot, "LEVERAGE_GROUP_EXCLUDE_SELF", True))
                use_exposure = _truthy(_cfg_env(bot, "LEVERAGE_GROUP_EXPOSURE", False))
                if use_exposure:
                    ref_pct = _safe_float(_cfg_env(bot, "LEVERAGE_GROUP_EXPOSURE_REF_PCT", 0.10), 0.10)
                    equity = _safe_float(getattr(getattr(bot, "state", None), "current_equity", 0.0), 0.0)
                    if equity > 0 and ref_pct > 0:
                        notional_vals = [
                            _group_notional(bot, groups, h, exclude=(sym_key if exclude_self else None))
                            for h in hits
                        ]
                        if notional_vals:
                            exposure_pct = max(0.0, max(notional_vals) / equity)
                            step = exposure_pct / float(ref_pct)
                            lev = max(float(scale_min), float(lev) * (float(scale) ** float(step)))
                else:
                    counts = [
                        _group_open_count(bot, groups, h, exclude=(sym_key if exclude_self else None))
                        for h in hits
                    ]
                    if counts:
                        exponent = max(0, min(counts))
                        lev = max(float(scale_min), float(lev) * (float(scale) ** float(exponent)))

    if (not is_exit) and callable(get_adaptive_leverage_scale) and _truthy(
        _cfg_env(bot, "ADAPTIVE_GUARD_LEVERAGE_SCALE", True)
    ):
        try:
            scale, reason = get_adaptive_leverage_scale(sym_key)
            if scale and scale < 1.0:
                lev = float(lev) * float(scale)
                if callable(emit_throttled):
                    try:
                        _telemetry_task(
                            emit_throttled(
                                bot,
                                "order.leverage_scaled",
                                key=f"{sym_key}:{reason or 'guard_history'}",
                                cooldown_sec=120.0,
                                data={
                                    "symbol": sym_key,
                                    "base_leverage": base,
                                    "scaled_leverage": lev,
                                    "scale": scale,
                                    "reason": reason or "guard_history",
                                },
                                symbol=sym_key,
                                level="warning",
                            )
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    # Belief controller is the single writer for entry guard knobs.
    if not is_exit:
        guard_cap = 0.0
        try:
            st = getattr(bot, "state", None)
            raw_knobs = getattr(st, "guard_knobs", None) if st is not None else None
            if isinstance(raw_knobs, dict):
                guard_cap = _safe_float(raw_knobs.get("max_leverage"), 0.0)
            elif raw_knobs is not None:
                guard_cap = _safe_float(getattr(raw_knobs, "max_leverage", 0.0), 0.0)
        except Exception:
            guard_cap = 0.0
        if guard_cap > 0 and lev > guard_cap:
            lev = float(guard_cap)
            if callable(emit_throttled):
                try:
                    _telemetry_task(
                        emit_throttled(
                            bot,
                            "order.leverage_scaled",
                            key=f"{sym_key}:belief_controller",
                            cooldown_sec=60.0,
                            data={
                                "symbol": sym_key,
                                "scaled_leverage": lev,
                                "cap": guard_cap,
                                "reason": "belief_controller_cap",
                            },
                            symbol=sym_key,
                            level="warning",
                        )
                    )
                except Exception:
                    pass

    lev_min = _safe_float(_cfg_env(bot, "LEVERAGE_MIN", 1), 1)
    lev_max = _safe_float(_cfg_env(bot, "LEVERAGE_MAX", 125), 125)
    lev = _clamp(float(lev), float(lev_min), float(lev_max))
    return max(1, int(round(lev)))


def _symbol_override_pct(bot, name: str, k: str, default: float) -> float:
    raw = _cfg_env(bot, name, None)
    if raw is not None:
        mp = _parse_symbol_overrides(raw)
        if k in mp:
            return _normalize_pct(mp[k], default)
    return _normalize_pct(_cfg_env(bot, name.replace("_BY_SYMBOL", ""), default), default)


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


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
        if any(x in msg for x in ("timeout", "timed out", "temporarily unavailable", "connection", "econnreset", "network")):
            return True, "network", (map_reason("network") if callable(map_reason) else "ERR_UNKNOWN")
        return True, "unknown", (map_reason(msg) if callable(map_reason) else "ERR_UNKNOWN")
    return _error_policy.classify_order_error(err, ex=ex, sym_raw=sym_raw, map_reason=map_reason)


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
    bucket: int = 0,
) -> str:
    pref = _sanitize_client_order_id(prefix, max_len=8) or "SE"
    sym = _symkey(sym_raw)[:6] or "SYM"
    blob = f"{sym_raw}|{type_norm}|{side_l}|{amount}|{price}|{stop_price}|{int(bucket)}"
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    out = f"{pref}_{sym}_{h}"
    return _sanitize_client_order_id(out) or "SE"


def _intent_client_order_prefix(*, type_u: str, is_exit: bool, fallback: str = "SE") -> str:
    t = str(type_u or "").upper().strip()
    if is_exit:
        if t.startswith("TAKE_PROFIT") or t == "TP_MARKET":
            return "TP"
        if t.startswith("STOP") or t == "TRAILING_STOP_MARKET":
            return "SL"
        return "EXIT"
    if t.startswith("TAKE_PROFIT") or t == "TP_MARKET":
        return "TP"
    if t.startswith("STOP") or t == "TRAILING_STOP_MARKET":
        return "SL"
    if t in ("MARKET", "LIMIT"):
        return "ENTRY"
    return _sanitize_client_order_id(fallback, max_len=8) or "SE"


def _make_correlation_id(
    *,
    intent_prefix: str,
    sym_raw: str,
    side_l: str,
    type_norm: str,
    bucket: int,
) -> str:
    pref = _sanitize_client_order_id(intent_prefix, max_len=8) or "SE"
    sym = _symkey(sym_raw)[:6] or "SYM"
    blob = f"{pref}|{sym_raw}|{side_l}|{type_norm}|{int(bucket)}"
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"{pref}-{sym}-{h}"


def _telemetry_task(coro) -> None:
    try:
        asyncio.create_task(coro)
    except Exception:
        try:
            coro.close()
        except Exception:
            pass


def _request_reconcile_hint(bot, *, symbol: str, reason: str, correlation_id: str = "") -> None:
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return
        rc = getattr(st, "run_context", None)
        if not isinstance(rc, dict):
            st.run_context = {}
            rc = st.run_context
        hints = rc.get("reconcile_hints")
        if not isinstance(hints, dict):
            hints = {}
            rc["reconcile_hints"] = hints
        k = _symkey(symbol)
        hints[k or str(symbol or "").upper()] = {
            "ts": float(time.time()),
            "reason": str(reason or "router_replace_fail_closed"),
            "correlation_id": str(correlation_id or ""),
        }
    except Exception:
        pass


def _record_reconcile_first_pressure(bot, *, symbol: str, severity: float, reason: str = "") -> None:
    """
    Router-side escalation hook for ambiguity storms (e.g., replace-race loops).
    Reconcile and belief-controller consume this via kill_metrics to clamp entries.
    """
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return
        km = getattr(st, "kill_metrics", None)
        if not isinstance(km, dict):
            st.kill_metrics = {}
            km = st.kill_metrics
        events = km.get("reconcile_first_gate_events")
        if not isinstance(events, list):
            events = []
            km["reconcile_first_gate_events"] = events
        now_ts = float(time.time())
        max_events = max(20, int(_safe_float(_cfg_env(bot, "BELIEF_RECONCILE_FIRST_EVENTS_MAX", 160), 160.0)))
        window_sec = max(10.0, _safe_float(_cfg_env(bot, "BELIEF_RECONCILE_FIRST_WINDOW_SEC", 120), 120.0))
        cutoff = now_ts - window_sec
        events[:] = [ev for ev in events if isinstance(ev, dict) and _safe_float(ev.get("ts"), 0.0) >= cutoff]
        sev = max(0.0, min(1.0, float(severity or 0.0)))
        events.append({"ts": now_ts, "symbol": _symkey(symbol), "severity": sev, "reason": str(reason or "")})
        if len(events) > max_events:
            del events[:-max_events]
        km["reconcile_first_gate_count"] = int(km.get("reconcile_first_gate_count", 0) or 0) + 1
        km["reconcile_first_gate_last_ts"] = float(now_ts)
        km["reconcile_first_gate_last_severity"] = float(sev)
        km["reconcile_first_gate_last_reason"] = str(reason or "")
    except Exception:
        pass


def _replace_budget_state(bot) -> tuple[dict, dict]:
    st = getattr(bot, "state", None)
    if st is None:
        return {}, {}
    rc = getattr(st, "run_context", None)
    if not isinstance(rc, dict):
        st.run_context = {}
        rc = st.run_context
    budget = rc.get("replace_budget")
    if not isinstance(budget, dict):
        budget = {"global": [], "by_symbol": {}}
        rc["replace_budget"] = budget
    by_symbol = budget.get("by_symbol")
    if not isinstance(by_symbol, dict):
        by_symbol = {}
        budget["by_symbol"] = by_symbol
    return budget, by_symbol


def _replace_budget_over_limit(bot, *, symbol: str) -> tuple[bool, dict]:
    window_sec = max(10.0, _safe_float(_cfg_env(bot, "ROUTER_REPLACE_BUDGET_WINDOW_SEC", 300), 300.0))
    max_global = max(1, int(_safe_float(_cfg_env(bot, "ROUTER_REPLACE_BUDGET_MAX_GLOBAL", 20), 20.0)))
    max_symbol = max(1, int(_safe_float(_cfg_env(bot, "ROUTER_REPLACE_BUDGET_MAX_PER_SYMBOL", 6), 6.0)))
    now_ts = float(time.time())
    cutoff = now_ts - window_sec
    budget, by_symbol = _replace_budget_state(bot)
    glob = budget.get("global")
    if not isinstance(glob, list):
        glob = []
        budget["global"] = glob
    glob[:] = [float(ts) for ts in glob if _safe_float(ts, 0.0) >= cutoff]
    k = _symkey(symbol)
    sym_list = by_symbol.get(k)
    if not isinstance(sym_list, list):
        sym_list = []
        by_symbol[k] = sym_list
    sym_list[:] = [float(ts) for ts in sym_list if _safe_float(ts, 0.0) >= cutoff]
    over_global = len(glob) >= max_global
    over_symbol = len(sym_list) >= max_symbol
    return bool(over_global or over_symbol), {
        "window_sec": float(window_sec),
        "max_global": int(max_global),
        "max_symbol": int(max_symbol),
        "count_global": int(len(glob)),
        "count_symbol": int(len(sym_list)),
        "over_global": bool(over_global),
        "over_symbol": bool(over_symbol),
        "symbol": str(k),
    }


def _replace_budget_mark(bot, *, symbol: str) -> None:
    now_ts = float(time.time())
    budget, by_symbol = _replace_budget_state(bot)
    glob = budget.get("global")
    if not isinstance(glob, list):
        glob = []
        budget["global"] = glob
    glob.append(now_ts)
    k = _symkey(symbol)
    sym_list = by_symbol.get(k)
    if not isinstance(sym_list, list):
        sym_list = []
        by_symbol[k] = sym_list
    sym_list.append(now_ts)


def _intent_ledger_record(
    bot,
    *,
    intent_id: str,
    stage: str,
    symbol: str = "",
    side: str = "",
    order_type: str = "",
    is_exit: bool = False,
    client_order_id: str = "",
    order_id: str = "",
    status: str = "",
    reason: str = "",
    meta: Optional[dict] = None,
) -> None:
    if _intent_ledger is None:
        return
    try:
        _intent_ledger.record(
            bot,
            intent_id=intent_id,
            stage=stage,
            symbol=symbol,
            side=side,
            order_type=order_type,
            is_exit=is_exit,
            client_order_id=client_order_id,
            order_id=order_id,
            status=status,
            reason=reason,
            meta=meta,
        )
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


def _freshen_client_order_id(existing: Any, *, salt: Optional[str] = None) -> str:
    """
    Keep <36 chars; rotate hash tail without growing the identifier on retries.
    """
    base = _sanitize_client_order_id(existing) or "SE"
    stem = re.sub(r"_[0-9a-f]{6,10}$", "", base, flags=re.IGNORECASE)
    room = 1 + 8
    stem = stem[: max(1, _BINANCE_CLIENT_ID_MAX - room)]
    payload = f"{stem}|{salt if salt is not None else f'{time.time()}|{random.random()}'}"
    suffix = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    out = f"{stem}_{suffix}"
    return _sanitize_client_order_id(out) or out


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


async def _fetch_ticker_bid_ask_mid(ex, sym_raw: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        fn = getattr(ex, "fetch_ticker", None)
        if not callable(fn):
            return None, None, None
        t = await fn(sym_raw)
        bid = _safe_float(t.get("bid"), 0.0)
        ask = _safe_float(t.get("ask"), 0.0)
        if bid <= 0 or ask <= 0:
            return None, None, None
        mid = (bid + ask) / 2.0
        return bid, ask, mid if mid > 0 else None
    except Exception:
        return None, None, None


async def _estimate_market_impact_pct(ex, sym_raw: str, side_l: str, amount: float) -> Optional[float]:
    """
    Best-effort impact estimate using top-of-book depth.
    Returns impact percentage vs mid, or None if insufficient data.
    """
    try:
        fn = getattr(ex, "fetch_order_book", None)
        if not callable(fn):
            return None
        ob = await fn(sym_raw)
        bids = (ob or {}).get("bids") or []
        asks = (ob or {}).get("asks") or []
        if not bids or not asks:
            return None
        bid = _safe_float(bids[0][0], 0.0)
        ask = _safe_float(asks[0][0], 0.0)
        if bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None

        remaining = float(amount)
        vwap = 0.0
        notional = 0.0
        book = asks if side_l == "buy" else bids
        for price, size in book:
            p = _safe_float(price, 0.0)
            s = _safe_float(size, 0.0)
            if p <= 0 or s <= 0:
                continue
            take = min(remaining, s)
            notional += take * p
            vwap += take
            remaining -= take
            if remaining <= 0:
                break

        if remaining > 0 or vwap <= 0:
            return None

        avg = notional / vwap
        if side_l == "buy":
            return max(0.0, (avg - mid) / mid)
        return max(0.0, (mid - avg) / mid)
    except Exception:
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


async def _fetch_order_status_best_effort(ex, order_id: str, sym_raw: str) -> Optional[str]:
    try:
        fn = getattr(ex, "fetch_order", None)
        if callable(fn):
            order = await fn(order_id, sym_raw)
            if isinstance(order, dict):
                st = str(order.get("status") or "").strip().lower()
                if st:
                    return st
    except Exception:
        return None
    return None


async def cancel_order(bot, order_id: str, symbol: str, *, correlation_id: Optional[str] = None) -> bool:
    """
    Idempotent cancel:
    - If order already gone/unknown -> return True (goal achieved: it's not live anymore)
    """
    ex = _get_ex(bot)
    if ex is None or not order_id:
        return False

    k = _symkey(symbol)
    corr_id = str(correlation_id or "").strip()
    if (not corr_id) and (_intent_ledger is not None):
        try:
            corr_id = str(_intent_ledger.resolve_intent_id(bot, order_id=str(order_id or "")) or "").strip()
        except Exception:
            corr_id = ""
    sym_raw = _resolve_raw_symbol(bot, k, symbol)

    candidates = [sym_raw]
    if symbol and symbol != sym_raw:
        candidates.append(symbol)
    if k and k not in (symbol, sym_raw):
        candidates.append(k)

    last_err: Optional[Exception] = None
    saw_unknown = False
    unknown_conflict = False
    unknown_status: Optional[str] = None
    def _emit_cancel_compat(ok_flag: bool, why_text: str, status_text: Optional[str] = None) -> None:
        if not callable(emit_order_cancel):
            return
        try:
            _telemetry_task(
                emit_order_cancel(
                    bot,
                    k,
                    order_id,
                    bool(ok_flag),
                    why=str(why_text),
                    correlation_id=corr_id,
                    status=status_text,
                )
            )
            return
        except TypeError:
            pass
        try:
            _telemetry_task(
                emit_order_cancel(
                    bot,
                    k,
                    order_id,
                    bool(ok_flag),
                    why=str(why_text),
                    status=status_text,
                )
            )
            return
        except TypeError:
            pass
        try:
            _telemetry_task(
                emit_order_cancel(
                    bot,
                    k,
                    order_id,
                    bool(ok_flag),
                    why=str(why_text),
                )
            )
        except Exception:
            pass
    _intent_ledger_record(
        bot,
        intent_id=(corr_id or f"CANCEL-{order_id}"),
        stage="CANCEL_SENT",
        symbol=k,
        order_id=str(order_id or ""),
        reason="cancel_request",
    )

    for sym_try in candidates:
        ok, err = await _cancel_order_raw(ex, order_id, sym_try)
        if ok:
            _emit_cancel_compat(True, "router")
            _intent_ledger_record(
                bot,
                intent_id=(corr_id or f"CANCEL-{order_id}"),
                stage="DONE",
                symbol=k,
                order_id=str(order_id or ""),
                status="canceled",
                reason="cancel_success",
            )
            return True
        if err is None:
            continue
        if _looks_like_unknown_order(err):
            saw_unknown = True
            st = await _fetch_order_status_best_effort(ex, order_id, sym_try)
            if st:
                unknown_status = st
            if st in ("open", "new", "partially_filled"):
                unknown_conflict = True
            continue
        last_err = err

    if saw_unknown and (not unknown_conflict):
        # ✅ idempotent success after exhausting symbol candidates
        log_entry.info(f"[router] cancel idempotent success (already gone) | k={k} id={order_id} st={unknown_status or 'na'}")
        _emit_cancel_compat(True, "already_gone", status_text=unknown_status)
        _intent_ledger_record(
            bot,
            intent_id=(corr_id or f"CANCEL-{order_id}"),
            stage="DONE",
            symbol=k,
            order_id=str(order_id or ""),
            status=(unknown_status or "already_gone"),
            reason="cancel_idempotent",
        )
        return True

    _emit_cancel_compat(False, (repr(last_err)[:120] if last_err else "unknown"), status_text=unknown_status)
    _intent_ledger_record(
        bot,
        intent_id=(corr_id or f"CANCEL-{order_id}"),
        stage="CANCEL_FAILED",
        symbol=k,
        order_id=str(order_id or ""),
        status=(unknown_status or ""),
        reason=(repr(last_err)[:120] if last_err else "unknown"),
    )

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
    correlation_id: Optional[str] = None,
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
    corr_id = str(correlation_id or "").strip()
    if (not corr_id) and (_intent_ledger is not None):
        try:
            corr_id = str(_intent_ledger.resolve_intent_id(bot, order_id=str(order_id or "")) or "").strip()
        except Exception:
            corr_id = ""
    sym_raw = _resolve_raw_symbol(bot, k, symbol)
    coid_bucket_sec = max(1, int(_safe_float(_cfg_env(bot, "ROUTER_CLIENT_ID_BUCKET_SEC", 30), 30)))
    coid_bucket = int(time.time() // coid_bucket_sec)

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
                            data={
                                "k": k,
                                "why": "missing_stopPrice",
                                "type": type_u,
                                "code": (map_reason("missing_stopPrice") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                            },
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
                            data={
                                "k": k,
                                "why": "missing_activationPrice",
                                "type": type_u,
                                "code": (map_reason("missing_activationPrice") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                            },
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
                            data={
                                "k": k,
                                "why": "missing_callbackRate",
                                "type": type_u,
                                "code": (map_reason("missing_callbackRate") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                            },
                            symbol=k,
                            level="critical",
                        )
                    )
                return None

    # ----------------------------
    # Entry safety: spread guard (best-effort)
    # ----------------------------
    if not is_exit:
        max_spread = _symbol_override_pct(bot, "MAX_SPREAD_PCT_BY_SYMBOL", k, 0.5)
        if max_spread > 0:
            bid, ask, mid = await _fetch_ticker_bid_ask_mid(ex, sym_raw)
            if bid and ask and mid:
                spread_pct = max(0.0, (ask - bid) / mid)
                if spread_pct > max_spread:
                    log_entry.critical(
                        f"ROUTER BLOCKED → spread too wide | k={k} raw={sym_raw} spread={spread_pct:.4f} max={max_spread:.4f}"
                    )
                    if callable(emit):
                        _telemetry_task(
                            emit(
                                bot,
                                "order.blocked",
                                data={
                                    "k": k,
                                    "why": "spread_too_wide",
                                    "spread": spread_pct,
                                    "max": max_spread,
                                    "code": (map_reason("spread_too_wide") if callable(map_reason) else "ERR_SPREAD"),
                                },
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
                    emit(
                        bot,
                        "order.blocked",
                        data={
                            "k": k,
                            "why": "first_live_symbol_not_allowed",
                            "code": (map_reason("first_live_symbol_not_allowed") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                        },
                        symbol=k,
                        level="critical",
                    )
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
                        emit(
                            bot,
                            "order.blocked",
                            data={
                                "k": k,
                                "why": "missing_hedge_side_hint_for_exit",
                                "code": (map_reason("missing_hedge_side_hint_for_exit") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                            },
                            symbol=k,
                            level="critical",
                        )
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
    intent_prefix = _intent_client_order_prefix(type_u=type_u, is_exit=is_exit, fallback=client_order_id_prefix)
    corr_id = str(correlation_id or "").strip()
    if not corr_id:
        corr_id = _make_correlation_id(
            intent_prefix=intent_prefix,
            sym_raw=sym_raw,
            side_l=side_l,
            type_norm=type_norm,
            bucket=coid_bucket,
        )

    client_order_id_live = ""
    order_id_live = ""
    intent_state = "INTENT_CREATED"

    def _journal_intent_transition(next_state: str, reason: str, *, meta: Optional[dict] = None) -> None:
        nonlocal intent_state
        nxt = str(next_state or "").strip().upper()
        if not nxt:
            return
        prev = intent_state
        try:
            if _state_machine is not None:
                _state_machine.transition(_state_machine.MachineKind.ORDER_INTENT, prev, nxt, reason)
            intent_state = nxt
        except Exception:
            intent_state = nxt
        _intent_ledger_record(
            bot,
            intent_id=str(corr_id or ""),
            stage=nxt,
            symbol=k,
            side=side_l,
            order_type=type_u,
            is_exit=bool(is_exit),
            client_order_id=client_order_id_live,
            order_id=order_id_live,
            status=str((meta or {}).get("status") or ""),
            reason=reason,
            meta=dict(meta or {}),
        )
        try:
            if _event_journal is not None:
                _event_journal.journal_transition(
                    bot,
                    machine="order_intent",
                    entity=str(corr_id or ""),
                    state_from=prev,
                    state_to=nxt,
                    reason=reason,
                    correlation_id=corr_id,
                    meta=dict(meta or {}),
                )
        except Exception:
            pass

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
                prefix=intent_prefix,
                sym_raw=sym_raw,
                type_norm=type_norm,
                side_l=side_l,
                amount=amount,
                price=price,
                stop_price=stop_for_id,
                bucket=coid_bucket,
            )
            p["clientOrderId"] = _sanitize_client_order_id(s) or "SE"

    # final hard sanitize (fail-safe)
    p = _sanitize_client_id_fields(p)
    client_order_id_live = str(p.get("clientOrderId") or "").strip()
    if _intent_ledger is not None and _truthy(_cfg_env(bot, "INTENT_LEDGER_REUSE_ENABLED", False)):
        try:
            pending_unknown = _intent_ledger.find_pending_unknown_intent(
                bot,
                intent_id=str(corr_id or ""),
                client_order_id=client_order_id_live,
            )
        except Exception:
            pending_unknown = None
        if isinstance(pending_unknown, dict):
            stage_unknown = str(pending_unknown.get("stage") or "SUBMITTED_UNKNOWN")
            status_unknown = str(pending_unknown.get("status") or "unknown").lower().strip() or "unknown"
            oid_unknown = str(pending_unknown.get("order_id") or "").strip()
            log_entry.warning(
                f"[router] pending unknown intent -> block duplicate submit | k={k} corr={corr_id} coid={client_order_id_live} stage={stage_unknown}"
            )
            _request_reconcile_hint(
                bot,
                symbol=k,
                reason=f"intent_pending_unknown:{stage_unknown}",
                correlation_id=str(corr_id or ""),
            )
            if callable(emit):
                _telemetry_task(
                    emit(
                        bot,
                        "order.intent_pending_unknown",
                        data={
                            "k": k,
                            "correlation_id": corr_id,
                            "client_order_id": client_order_id_live,
                            "order_id": oid_unknown,
                            "stage": stage_unknown,
                            "status": status_unknown,
                        },
                        symbol=k,
                        level="warning",
                    )
                )
            return {
                "id": (oid_unknown or None),
                "symbol": sym_raw,
                "type": type_norm,
                "side": side_l,
                "amount": _safe_float(amount, 0.0),
                "filled": 0.0,
                "status": status_unknown,
                "info": {
                    "intent_pending_unknown": True,
                    "correlation_id": corr_id,
                    "clientOrderId": client_order_id_live,
                    "stage": stage_unknown,
                },
            }
    if _intent_ledger is not None and _truthy(_cfg_env(bot, "INTENT_LEDGER_REUSE_ENABLED", False)):
        try:
            seen = _intent_ledger.find_reusable_intent(
                bot,
                intent_id=str(corr_id or ""),
                client_order_id=client_order_id_live,
            )
        except Exception:
            seen = None
        if isinstance(seen, dict):
            status_seen = str(seen.get("status") or "open").lower().strip() or "open"
            oid_seen = str(seen.get("order_id") or "").strip()
            log_entry.warning(
                f"[router] intent ledger reuse -> skip duplicate submit | k={k} corr={corr_id} coid={client_order_id_live} stage={seen.get('stage')}"
            )
            if callable(emit):
                _telemetry_task(
                    emit(
                        bot,
                        "order.intent_reused",
                        data={
                            "k": k,
                            "correlation_id": corr_id,
                            "client_order_id": client_order_id_live,
                            "order_id": oid_seen,
                            "stage": str(seen.get("stage") or ""),
                            "status": status_seen,
                        },
                        symbol=k,
                        level="warning",
                    )
                )
            return {
                "id": (oid_seen or None),
                "symbol": sym_raw,
                "type": type_norm,
                "side": side_l,
                "amount": _safe_float(amount, 0.0),
                "filled": 0.0,
                "status": status_seen,
                "info": {"intent_ledger_reused": True, "correlation_id": corr_id, "clientOrderId": client_order_id_live},
            }
    _intent_ledger_record(
        bot,
        intent_id=str(corr_id or ""),
        stage="INTENT_CREATED",
        symbol=k,
        side=side_l,
        order_type=type_u,
        is_exit=bool(is_exit),
        client_order_id=client_order_id_live,
        order_id=order_id_live,
        reason="router_prepare",
    )

    # log
    try:
        log_entry.info(
            f"[router] SEND k={k} raw={sym_raw} type={type_norm} side={side_l} amt={amount} px={price} "
            f"is_exit={is_exit} reduceOnly={p.get('reduceOnly')} closePosition={p.get('closePosition')} "
            f"positionSide={p.get('positionSide')} clientOrderId={p.get('clientOrderId')} corr={corr_id} params_keys={sorted(list(p.keys()))}"
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
    leverage = _resolve_leverage(bot, k, sym_raw, is_exit=is_exit)

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
                    data={
                        "k": k,
                        "raw": sym_raw,
                        "type": type_u,
                        "side": side_l,
                        "amount": amount_for_validation,
                        "price": price,
                        "why": why,
                        "code": (map_reason(why) if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                    },
                    symbol=k,
                    level="critical",
                )
            )
        return None

    # ----------------------------
    # Entry safety: market impact guard (best-effort)
    # ----------------------------
    if (not is_exit) and type_norm == "market":
        max_impact = _symbol_override_pct(bot, "MAX_IMPACT_PCT_BY_SYMBOL", k, 1.0)
        if max_impact > 0 and amt_norm is not None:
            impact = await _estimate_market_impact_pct(ex, sym_raw, side_l, _safe_float(amt_norm, 0.0))
            if impact is not None and impact > max_impact:
                log_entry.critical(
                    f"ROUTER BLOCKED → market impact too high | k={k} raw={sym_raw} impact={impact:.4f} max={max_impact:.4f}"
                )
                if callable(emit):
                    _telemetry_task(
                        emit(
                            bot,
                            "order.blocked",
                            data={
                                "k": k,
                                "why": "market_impact_too_high",
                                "impact": impact,
                                "max": max_impact,
                                "code": (map_reason("market_impact_too_high") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                            },
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

    # Router notional cap (entries only)
    route_cap = _safe_float(_cfg_env(bot, "ROUTER_MAX_NOTIONAL_USDT", "ROUTER_NOTIONAL_CAP"), 0.0)
    if route_cap > 0 and (not is_exit):
        try:
            px_for_cap: Optional[float] = None
            if px_prec is not None and _is_number_like(px_prec):
                px_for_cap = _safe_float(px_prec, 0.0) or None
            if px_for_cap is None:
                px_for_cap = await _fetch_last_price(ex, sym_raw)

            if amt_prec is not None and px_for_cap and px_for_cap > 0:
                notion = _safe_float(amt_prec, 0.0) * float(px_for_cap)
                if notion > route_cap:
                    log_entry.warning(
                        f"ROUTER NOTIONAL BLOCK → cap exceeded: k={k} raw={sym_raw} notional={notion:.4f} cap={route_cap}"
                    )
                    if callable(emit):
                        _telemetry_task(
                            emit(
                                bot,
                                "order.blocked",
                                data={
                                    "k": k,
                                    "why": "router_notional_cap",
                                    "notional": notion,
                                    "cap": route_cap,
                                    "code": (map_reason("router_notional_cap") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                                },
                                symbol=k,
                                level="warning",
                            )
                        )
                    return None
        except Exception:
            pass

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
                                data={
                                    "k": k,
                                    "why": "first_live_notional_cap",
                                    "notional": notion,
                                    "cap": cap,
                                    "code": (map_reason("first_live_notional_cap") if callable(map_reason) else "ERR_ROUTER_BLOCK"),
                                },
                                symbol=k,
                                level="critical",
                            )
                        )
                    return None
        except Exception:
            pass

    max_attempts = 6
    base_delay = float(_safe_float(_cfg_env(bot, "ROUTER_RETRY_BASE_SEC", 0.25), 0.25))
    max_delay = float(_safe_float(_cfg_env(bot, "ROUTER_RETRY_MAX_DELAY_SEC", 5.0), 5.0))
    jitter_pct = float(_safe_float(_cfg_env(bot, "ROUTER_RETRY_JITTER_PCT", 0.25), 0.25))
    max_elapsed = float(_safe_float(_cfg_env(bot, "ROUTER_RETRY_MAX_ELAPSED_SEC", 30.0), 30.0))
    max_total_tries = int(_safe_float(_cfg_env(bot, "ROUTER_RETRY_MAX_TOTAL_TRIES", 0), 0))
    if retries is not None:
        max_attempts = max(1, int(retries))

    last_err: Optional[Exception] = None
    last_err_code: str = "ERR_UNKNOWN"
    last_err_reason: str = "unknown"
    last_err_class: str = _ERROR_CLASS_FATAL
    retry_alert_tries = int(_safe_float(_cfg_env(bot, "ROUTER_RETRY_ALERT_TRIES", 6), 6))
    retry_alert_cd = float(_safe_float(_cfg_env(bot, "ROUTER_RETRY_ALERT_COOLDOWN_SEC", 60.0), 60.0))

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

    retry_policies = _build_retry_policies(bot)
    policy_delay_override: Optional[float] = None
    policy_max_delay_override: Optional[float] = None
    start_ts = time.monotonic()
    tries = 0
    attempt = 0
    abort_retries = False
    if max_total_tries <= 0:
        # Keep retry storms bounded even if retry variants expand on repeated errors.
        max_total_tries = max(1, (max_attempts * max(1, len(variants)) * 2))
    while attempt < max_attempts:
        if max_elapsed > 0 and (time.monotonic() - start_ts) >= max_elapsed:
            break
        for (raw_sym, amt_try, px_try, p_try) in list(variants):
            if tries >= max_total_tries:
                abort_retries = True
                break
            tries += 1
            try:
                _journal_intent_transition(
                    "SUBMITTED",
                    "router_send",
                    meta={"k": k, "attempt": attempt + 1, "tries": tries, "raw": raw_sym, "type": type_u, "side": side_l},
                )
                res = await _attempt(raw_sym, amt_try, px_try, p_try)
                order_id_live = str((res or {}).get("id") or "").strip()
                status = str((res or {}).get("status") or "").lower().strip()
                if status in ("partially_filled",):
                    _journal_intent_transition("PARTIAL", "exchange_status", meta={"status": status})
                elif status in ("closed", "filled"):
                    _journal_intent_transition("FILLED", "exchange_status", meta={"status": status})
                    _journal_intent_transition("DONE", "terminal_fill", meta={"status": status})
                elif status in ("open", "new"):
                    _journal_intent_transition("OPEN", "exchange_status", meta={"status": status})
                else:
                    _journal_intent_transition("ACKED", "exchange_ack", meta={"status": status})
                if callable(emit_order_create):
                    try:
                        _telemetry_task(
                            emit_order_create(
                                bot,
                                k,
                                res,
                                intent=f"{type_u}:{side_l}",
                                correlation_id=corr_id,
                            )
                        )
                    except TypeError:
                        # Backward compatibility for older telemetry helpers that
                        # do not accept correlation_id yet.
                        _telemetry_task(
                            emit_order_create(
                                bot,
                                k,
                                res,
                                intent=f"{type_u}:{side_l}",
                            )
                        )
                return res
            except Exception as e:
                last_err = e
                policy_meta = _classify_order_error_policy(e, ex=ex, sym_raw=sym_raw)
                retryable = bool(policy_meta.get("retryable"))
                reason = str(policy_meta.get("reason") or "unknown")
                code = str(policy_meta.get("code") or "ERR_UNKNOWN")
                err_class = str(policy_meta.get("error_class") or _ERROR_CLASS_FATAL)
                if reason in ("unknown_order", "unknown"):
                    unknown_state = "SUBMITTED_UNKNOWN"
                    try:
                        if _state_machine is not None:
                            unknown_state = str(_state_machine.map_unknown_order_state(intent_state))
                    except Exception:
                        unknown_state = "SUBMITTED_UNKNOWN"
                    _journal_intent_transition(unknown_state, f"error:{reason}", meta={"error_class": err_class})
                last_err_code = code
                last_err_reason = reason
                last_err_class = err_class
                if bump is not None:
                    bump(bot, "order.create.retry", 1)
                if callable(emit):
                    _telemetry_task(
                        emit(
                            bot,
                            "order.retry",
                            data={
                                "k": k,
                                "raw": raw_sym,
                                "tries": tries,
                                "attempt": attempt + 1,
                                "type": type_u,
                                "side": side_l,
                                "error_class": err_class,
                                "reason": reason,
                                "code": code,
                                "retryable": retryable,
                                "correlation_id": corr_id,
                                "err": (repr(e)[:220] if e else "unknown"),
                            },
                            level="warning",
                            symbol=k,
                        )
                    )
                if callable(emit_throttled) and tries >= retry_alert_tries:
                    _telemetry_task(
                        emit_throttled(
                            bot,
                            "order.create.retry_alert",
                            key=f"{k}:{type_u}:{side_l}",
                            cooldown_sec=retry_alert_cd,
                            data={
                                "k": k,
                                "tries": tries,
                                "variants": len(variants),
                                "type": type_u,
                                "side": side_l,
                                "err": (repr(e)[:200] if e else "unknown"),
                                "reason": reason,
                                "code": code,
                                "error_class": err_class,
                                "correlation_id": corr_id,
                            },
                            level="warning",
                            symbol=k,
                        )
                    )

                policy = retry_policies.get(str(reason).strip().lower()) or retry_policies.get("default")
                if policy:
                    max_val = policy.get("max_attempts")
                    if max_val is not None:
                        max_attempts = max(max_attempts, int(max_val))
                    extra = int(_safe_float(policy.get("extra_attempts", 0), 0))
                    if extra > 0:
                        max_attempts = max(max_attempts, attempt + 1 + extra)
                    bd = policy.get("base_delay")
                    if isinstance(bd, (int, float)):
                        policy_delay_override = float(bd)
                    md = policy.get("max_delay")
                    if isinstance(md, (int, float)):
                        policy_max_delay_override = float(md)

                # -1106 reduceOnly not required -> add stripped variant
                if _looks_like_binance_reduceonly_not_required(e):
                    if "reduceOnly" in p_try:
                        p3 = dict(p_try)
                        p3.pop("reduceOnly", None)
                        _push_variant(variants, seen, raw_sym, amt_try, px_try, p3)

                # -4015 id too long -> sanitize again (hash/compact) and retry
                if _looks_like_binance_client_id_too_long(e):
                    coid = (p_try or {}).get("clientOrderId")
                    if coid:
                        p5 = dict(p_try)
                        p5["clientOrderId"] = _sanitize_client_order_id(coid)
                        _push_variant(variants, seen, raw_sym, amt_try, px_try, p5)
                        log_entry.warning(
                            f"[router] BINANCE CLIENT_ID TOO LONG -> sanitized clientOrderId {coid} -> {p5.get('clientOrderId')} | k={k}"
                        )

                # -4116 clientOrderId duplicated -> add fresh clientOrderId variant (ALWAYS short)
                if _looks_like_binance_client_id_duplicate(e):
                    coid = (p_try or {}).get("clientOrderId")
                    if coid:
                        p4 = dict(p_try)
                        p4["clientOrderId"] = _freshen_client_order_id(coid, salt=f"{corr_id}:{tries}")
                        _push_variant(variants, seen, raw_sym, amt_try, px_try, p4)
                        log_entry.warning(
                            f"[router] BINANCE DUP CLIENT_ID -> freshened clientOrderId {coid} -> {p4.get('clientOrderId')} | k={k}"
                        )

                if err_class in (_ERROR_CLASS_FATAL, _ERROR_CLASS_IDEMPOTENT):
                    abort_retries = True
                    break
                if not retryable:
                    abort_retries = True
                    break

        if abort_retries:
            break

        delay_base = policy_delay_override if policy_delay_override is not None else base_delay
        delay_max = policy_max_delay_override if policy_max_delay_override is not None else max_delay
        policy_delay_override = None
        policy_max_delay_override = None
        delay = delay_base * (2 ** attempt)
        if delay_max > 0:
            delay = min(delay, delay_max)
        if jitter_pct and jitter_pct > 0:
            jitter_pct = abs(jitter_pct)
            delay = random.uniform(delay * (1.0 - jitter_pct), delay * (1.0 + jitter_pct))
        delay = max(0.0, delay)
        if max_elapsed > 0:
            remaining = max_elapsed - (time.monotonic() - start_ts)
            if remaining <= 0:
                break
            delay = min(delay, remaining)
        await asyncio.sleep(delay)
        attempt += 1

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
                    "reason": last_err_reason,
                    "code": last_err_code,
                    "error_class": last_err_class,
                    "correlation_id": corr_id,
                },
                symbol=k,
                level="critical",
            )
        )

    log_entry.error(
        f"ORDER ROUTER FAILED → k={k} raw={sym_raw} {type_norm} {side_l} amount={amount} price={price} corr={corr_id} err={last_err}"
    )
    _journal_intent_transition(
        "DONE",
        "terminal_error",
        meta={"reason": last_err_reason, "code": last_err_code, "error_class": last_err_class},
    )
    return None


# ----------------------------
# Cancel / Replace helper
# ----------------------------

async def cancel_replace_order(
    bot,
    *,
    cancel_order_id: str,
    symbol: str,
    type: str,
    side: str,
    amount: Any,
    price: Optional[Any] = None,
    stop_price: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    correlation_id: Optional[str] = None,
) -> Optional[dict]:
    over_budget, budget_meta = _replace_budget_over_limit(bot, symbol=symbol)
    if over_budget:
        _record_reconcile_first_pressure(
            bot,
            symbol=_symkey(symbol),
            severity=_safe_float(_cfg_env(bot, "ROUTER_REPLACE_BUDGET_GATE_SEVERITY", 0.95), 0.95),
            reason="replace_budget_exceeded",
        )
        if callable(emit):
            _telemetry_task(
                emit(
                    bot,
                    "order.replace_budget_block",
                    data={
                        "k": _symkey(symbol),
                        "cancel_order_id": str(cancel_order_id),
                        "reason": "replace_budget_exceeded",
                        "correlation_id": str(correlation_id or ""),
                        **dict(budget_meta),
                    },
                    symbol=_symkey(symbol),
                    level="critical",
                )
            )
        _request_reconcile_hint(
            bot,
            symbol=str(symbol),
            reason="replace_budget_exceeded",
            correlation_id=str(correlation_id or ""),
        )
        return None

    def _estimate_symbol_exposure_notional(sym_key: str) -> float:
        total = 0.0
        try:
            pos_map = getattr(getattr(bot, "state", None), "positions", None)
            if not isinstance(pos_map, dict):
                return 0.0
            for kk, pos in pos_map.items():
                if _symkey(kk) != _symkey(sym_key):
                    continue
                try:
                    sz = abs(float(getattr(pos, "size", 0.0) or 0.0))
                    px = float(getattr(pos, "entry_price", 0.0) or 0.0)
                    if sz > 0 and px > 0:
                        total += (sz * px)
                except Exception:
                    continue
        except Exception:
            return 0.0
        return float(max(0.0, total))

    max_attempts = max(1, int(retries or 1))
    corr = str(correlation_id or "").strip()
    if not corr:
        corr = _make_correlation_id(
            intent_prefix="REPLACE",
            sym_raw=symbol,
            side_l=str(side or "").lower().strip(),
            type_norm=_normalize_type_for_ccxt(str(type or "").upper().strip()),
            bucket=int(time.time() // max(1, int(_safe_float(_cfg_env(bot, "ROUTER_CLIENT_ID_BUCKET_SEC", 30), 30)))),
        )
    _intent_ledger_record(
        bot,
        intent_id=str(corr or ""),
        stage="INTENT_CREATED",
        symbol=_symkey(symbol),
        side=str(side or "").lower().strip(),
        order_type=str(type or "").upper().strip(),
        reason="cancel_replace_start",
        order_id=str(cancel_order_id or ""),
    )
    new_notional = 0.0
    try:
        amt_f = abs(float(amount or 0.0))
        px_f = float(price or 0.0)
        if amt_f > 0 and px_f > 0:
            new_notional = float(amt_f * px_f)
    except Exception:
        new_notional = 0.0
    cur_notional = _estimate_symbol_exposure_notional(_symkey(symbol))
    max_worst_case_notional = max(0.0, float(_safe_float(_cfg_env(bot, "ROUTER_REPLACE_MAX_WORST_CASE_NOTIONAL", 0.0), 0.0)))
    max_ambiguity_attempts = max(0, int(_safe_float(_cfg_env(bot, "ROUTER_REPLACE_MAX_AMBIGUITY_ATTEMPTS", 0), 0.0)))
    if _replace_manager is not None:
        async def _cancel_fn(order_id: str, sym: str) -> bool:
            return bool(await cancel_order(bot, order_id, sym, correlation_id=corr))

        async def _create_fn() -> Optional[dict]:
            return await create_order(
                bot,
                symbol=symbol,
                type=type,
                side=side,
                amount=amount,
                price=price,
                stop_price=stop_price,
                params=params or {},
                retries=1,
                correlation_id=corr,
            )

        status_cb = None
        if _truthy(_cfg_env(bot, "ROUTER_REPLACE_STATUS_CHECK", False)):
            async def _status_fn(order_id: str, sym: str) -> Optional[str]:
                ex = _get_ex(bot)
                if ex is None:
                    return None
                return await _fetch_order_status_best_effort(ex, order_id, _resolve_raw_symbol(bot, _symkey(sym), sym))
            status_cb = _status_fn

        strict_replace = _truthy(_cfg_env(bot, "ROUTER_REPLACE_STRICT_TRANSITIONS", True))
        _replace_budget_mark(bot, symbol=symbol)
        try:
            outcome = await _replace_manager.run_cancel_replace(
                cancel_order_id=str(cancel_order_id),
                symbol=str(symbol),
                max_attempts=max_attempts,
                cancel_fn=_cancel_fn,
                create_fn=_create_fn,
                status_fn=status_cb,
                strict_transitions=strict_replace,
                current_exposure_notional=float(cur_notional),
                new_order_notional=float(new_notional),
                max_worst_case_notional=float(max_worst_case_notional),
                max_ambiguity_attempts=int(max_ambiguity_attempts),
            )
        except Exception as exc:
            exc_name = str(getattr(getattr(exc, "__class__", None), "__name__", "Exception") or "Exception")
            _request_reconcile_hint(
                bot,
                symbol=str(symbol),
                reason=f"replace_transition_error:{exc_name}",
                correlation_id=corr,
            )
            if callable(emit):
                _telemetry_task(
                    emit(
                        bot,
                        "order.cancel_replace_transition_error",
                        data={
                            "k": _symkey(symbol),
                            "cancel_order_id": str(cancel_order_id),
                            "max_attempts": max_attempts,
                            "strict_transitions": bool(strict_replace),
                            "err": repr(exc)[:240],
                            "correlation_id": corr,
                        },
                        symbol=_symkey(symbol),
                        level="critical",
                    )
                )
            _intent_ledger_record(
                bot,
                intent_id=str(corr or ""),
                stage="DONE",
                symbol=_symkey(symbol),
                side=str(side or "").lower().strip(),
                order_type=str(type or "").upper().strip(),
                order_id=str(cancel_order_id or ""),
                status="transition_error",
                reason=f"replace_transition_error:{exc_name}",
            )
            return None
        if bool(getattr(outcome, "success", False)):
            new_order = getattr(outcome, "order", None)
            oid = str((new_order or {}).get("id") or "").strip() if isinstance(new_order, dict) else ""
            _intent_ledger_record(
                bot,
                intent_id=str(corr or ""),
                stage="DONE",
                symbol=_symkey(symbol),
                side=str(side or "").lower().strip(),
                order_type=str(type or "").upper().strip(),
                order_id=oid,
                status="replaced",
                reason="cancel_replace_success",
            )
            return getattr(outcome, "order", None)
        outcome_reason = str(getattr(outcome, "reason", "") or "")
        if outcome_reason in ("replace_envelope_block", "replace_ambiguity_cap", "replace_reconcile_required"):
            if outcome_reason in ("replace_ambiguity_cap", "replace_reconcile_required"):
                race_base = _safe_float(_cfg_env(bot, "ROUTER_REPLACE_RACE_GATE_SEVERITY", 0.90), 0.90)
                ambiguity_bump = _safe_float(getattr(outcome, "ambiguity_count", 0), 0.0) * 0.05
                attempt_bump = _safe_float(getattr(outcome, "attempts", 0), 0.0) * 0.02
                race_sev = max(0.0, min(1.0, float(race_base + ambiguity_bump + attempt_bump)))
                _record_reconcile_first_pressure(
                    bot,
                    symbol=_symkey(symbol),
                    severity=race_sev,
                    reason=f"replace_race:{outcome_reason}",
                )
            if callable(emit):
                _telemetry_task(
                    emit(
                        bot,
                        ("order.replace_reconcile_required" if outcome_reason == "replace_reconcile_required" else "order.replace_envelope_block"),
                        data={
                            "k": _symkey(symbol),
                            "cancel_order_id": str(cancel_order_id),
                            "reason": outcome_reason,
                            "state": str(getattr(outcome, "state", "")),
                            "attempts": int(getattr(outcome, "attempts", 0) or 0),
                            "last_status": str(getattr(outcome, "last_status", "")),
                            "current_exposure_notional": float(cur_notional),
                            "new_order_notional": float(new_notional),
                            "max_worst_case_notional": float(max_worst_case_notional),
                            "max_ambiguity_attempts": int(max_ambiguity_attempts),
                            "ambiguity_count": int(getattr(outcome, "ambiguity_count", 0) or 0),
                            "cancel_attempts": int(getattr(outcome, "cancel_attempts", 0) or 0),
                            "create_attempts": int(getattr(outcome, "create_attempts", 0) or 0),
                            "status_checks": int(getattr(outcome, "status_checks", 0) or 0),
                            "correlation_id": corr,
                        },
                        symbol=_symkey(symbol),
                        level="critical",
                    )
                )
        if callable(emit):
            _telemetry_task(
                emit(
                    bot,
                    "order.cancel_replace_giveup",
                    data={
                        "k": _symkey(symbol),
                        "cancel_order_id": str(cancel_order_id),
                        "max_attempts": max_attempts,
                        "state": str(getattr(outcome, "state", "")),
                        "reason": str(getattr(outcome, "reason", "")),
                        "attempts": int(getattr(outcome, "attempts", 0) or 0),
                        "last_status": str(getattr(outcome, "last_status", "")),
                        "current_exposure_notional": float(cur_notional),
                        "new_order_notional": float(new_notional),
                        "max_worst_case_notional": float(max_worst_case_notional),
                        "max_ambiguity_attempts": int(max_ambiguity_attempts),
                        "ambiguity_count": int(getattr(outcome, "ambiguity_count", 0) or 0),
                        "cancel_attempts": int(getattr(outcome, "cancel_attempts", 0) or 0),
                        "create_attempts": int(getattr(outcome, "create_attempts", 0) or 0),
                        "status_checks": int(getattr(outcome, "status_checks", 0) or 0),
                        "correlation_id": corr,
                    },
                    symbol=_symkey(symbol),
                    level="critical",
                )
            )
        _request_reconcile_hint(
            bot,
            symbol=str(symbol),
            reason=f"replace_giveup:{str(getattr(outcome, 'reason', '') or 'unknown')}",
            correlation_id=corr,
        )
        _intent_ledger_record(
            bot,
            intent_id=str(corr or ""),
            stage="DONE",
            symbol=_symkey(symbol),
            side=str(side or "").lower().strip(),
            order_type=str(type or "").upper().strip(),
            order_id=str(cancel_order_id or ""),
            status="giveup",
            reason="cancel_replace_giveup",
            meta={
                "attempts": int(getattr(outcome, "attempts", 0) or 0),
                "last_status": str(getattr(outcome, "last_status", "")),
                "ambiguity_count": int(getattr(outcome, "ambiguity_count", 0) or 0),
                "cancel_attempts": int(getattr(outcome, "cancel_attempts", 0) or 0),
                "create_attempts": int(getattr(outcome, "create_attempts", 0) or 0),
                "status_checks": int(getattr(outcome, "status_checks", 0) or 0),
            },
        )
        return None

    for idx in range(max_attempts):
        ok = await cancel_order(bot, cancel_order_id, symbol, correlation_id=corr)
        if not ok:
            if callable(emit):
                _telemetry_task(
                    emit(
                        bot,
                        "order.cancel_replace_failed",
                        data={
                            "k": _symkey(symbol),
                            "cancel_order_id": str(cancel_order_id),
                            "attempt": idx + 1,
                            "reason": "cancel_failed",
                            "correlation_id": corr,
                        },
                        symbol=_symkey(symbol),
                        level="warning",
                    )
                )
            continue
        res = await create_order(
            bot,
            symbol=symbol,
            type=type,
            side=side,
            amount=amount,
            price=price,
            stop_price=stop_price,
            params=params or {},
            retries=1,
            correlation_id=corr,
        )
        if res is not None:
            oid = str((res or {}).get("id") or "").strip() if isinstance(res, dict) else ""
            _intent_ledger_record(
                bot,
                intent_id=str(corr or ""),
                stage="DONE",
                symbol=_symkey(symbol),
                side=str(side or "").lower().strip(),
                order_type=str(type or "").upper().strip(),
                order_id=oid,
                status="replaced",
                reason="cancel_replace_success",
            )
            return res
    if callable(emit):
        _telemetry_task(
            emit(
                bot,
                "order.cancel_replace_giveup",
                data={
                    "k": _symkey(symbol),
                    "cancel_order_id": str(cancel_order_id),
                    "max_attempts": max_attempts,
                    "correlation_id": corr,
                },
                symbol=_symkey(symbol),
                level="critical",
            )
        )
    _request_reconcile_hint(
        bot,
        symbol=str(symbol),
        reason="replace_giveup:fallback_loop",
        correlation_id=corr,
    )
    _intent_ledger_record(
        bot,
        intent_id=str(corr or ""),
        stage="DONE",
        symbol=_symkey(symbol),
        side=str(side or "").lower().strip(),
        order_type=str(type or "").upper().strip(),
        order_id=str(cancel_order_id or ""),
        status="giveup",
        reason="cancel_replace_giveup",
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
