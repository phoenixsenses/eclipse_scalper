from __future__ import annotations

import time
from typing import Any, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on", "t")
    return False


def _is_reduce_only(order: dict[str, Any]) -> bool:
    info = order.get("info") or {}
    params = order.get("params") or {}
    return any(
        (
            _truthy(order.get("reduceOnly")),
            _truthy(order.get("closePosition")),
            _truthy(info.get("reduceOnly")),
            _truthy(info.get("closePosition")),
            _truthy(params.get("reduceOnly")),
            _truthy(params.get("closePosition")),
        )
    )


def _is_stop_like(order: dict[str, Any]) -> bool:
    t = str(order.get("type") or "").upper()
    if "STOP" in t:
        return True
    info = order.get("info") or {}
    it = str(info.get("type") or info.get("orderType") or "").upper()
    if "STOP" in it:
        return True
    if order.get("stopPrice") is not None or info.get("stopPrice") is not None:
        return True
    return False


def _is_tp_like(order: dict[str, Any]) -> bool:
    t = str(order.get("type") or "").upper()
    if "TAKE_PROFIT" in t or t == "TP_MARKET" or "TP" in t:
        return True
    info = order.get("info") or {}
    it = str(info.get("type") or info.get("orderType") or "").upper()
    if "TAKE_PROFIT" in it:
        return True
    return False


def _extract_qty(order: dict[str, Any]) -> float:
    info = order.get("info") or {}
    vals = [
        order.get("amount"),
        order.get("remaining"),
        info.get("origQty"),
        info.get("quantity"),
        info.get("orig_quantity"),
    ]
    best = 0.0
    for v in vals:
        q = abs(_safe_float(v, 0.0))
        if q > best:
            best = q
    return best


def assess_stop_coverage(
    open_orders: list[dict[str, Any]],
    *,
    required_qty: float,
    min_coverage_ratio: float = 0.98,
) -> dict[str, Any]:
    req = abs(_safe_float(required_qty, 0.0))
    best: Optional[dict[str, Any]] = None
    best_qty = 0.0
    for order in (open_orders or []):
        if not isinstance(order, dict):
            continue
        if not _is_reduce_only(order):
            continue
        if not _is_stop_like(order):
            continue
        if (
            _truthy(order.get("closePosition"))
            or _truthy((order.get("info") or {}).get("closePosition"))
            or _truthy((order.get("params") or {}).get("closePosition"))
        ):
            return {
                "covered": True,
                "reason": "close_position_stop",
                "order_id": str(order.get("id") or ""),
                "existing_qty": req,
                "needs_refresh": False,
            }
        qty = _extract_qty(order)
        if qty >= best_qty:
            best = order
            best_qty = qty
    if best is None:
        return {
            "covered": False,
            "reason": "missing_stop",
            "order_id": "",
            "existing_qty": 0.0,
            "needs_refresh": True,
        }
    ratio = (best_qty / req) if req > 0 else 1.0
    covered = ratio >= max(0.0, float(min_coverage_ratio))
    return {
        "covered": bool(covered),
        "reason": ("covered" if covered else "qty_under_covered"),
        "order_id": str(best.get("id") or ""),
        "existing_qty": float(best_qty),
        "coverage_ratio": float(ratio),
        "coverage_shortfall_ratio": float(max(0.0, 1.0 - float(ratio))),
        "needs_refresh": (not covered),
    }


def assess_tp_coverage(
    open_orders: list[dict[str, Any]],
    *,
    required_qty: float,
    min_coverage_ratio: float = 0.98,
) -> dict[str, Any]:
    req = abs(_safe_float(required_qty, 0.0))
    best: Optional[dict[str, Any]] = None
    best_qty = 0.0
    for order in (open_orders or []):
        if not isinstance(order, dict):
            continue
        if not _is_reduce_only(order):
            continue
        if not _is_tp_like(order):
            continue
        if (
            _truthy(order.get("closePosition"))
            or _truthy((order.get("info") or {}).get("closePosition"))
            or _truthy((order.get("params") or {}).get("closePosition"))
        ):
            return {
                "covered": True,
                "reason": "close_position_tp",
                "order_id": str(order.get("id") or ""),
                "existing_qty": req,
                "needs_refresh": False,
            }
        qty = _extract_qty(order)
        if qty >= best_qty:
            best = order
            best_qty = qty
    if best is None:
        return {
            "covered": False,
            "reason": "missing_tp",
            "order_id": "",
            "existing_qty": 0.0,
            "needs_refresh": True,
        }
    ratio = (best_qty / req) if req > 0 else 1.0
    covered = ratio >= max(0.0, float(min_coverage_ratio))
    return {
        "covered": bool(covered),
        "reason": ("covered" if covered else "qty_under_covered"),
        "order_id": str(best.get("id") or ""),
        "existing_qty": float(best_qty),
        "coverage_ratio": float(ratio),
        "coverage_shortfall_ratio": float(max(0.0, 1.0 - float(ratio))),
        "needs_refresh": (not covered),
    }


def should_refresh_protection(
    *,
    previous_qty: float,
    new_qty: float,
    last_refresh_ts: float,
    min_delta_ratio: float = 0.10,
    min_delta_abs: float = 0.0,
    max_refresh_interval_sec: float = 45.0,
    now_ts: Optional[float] = None,
) -> bool:
    prev = abs(_safe_float(previous_qty, 0.0))
    new = abs(_safe_float(new_qty, 0.0))
    if new <= 0:
        return False
    if prev <= 0:
        return True
    now = float(now_ts if now_ts is not None else time.time())
    last = _safe_float(last_refresh_ts, 0.0)
    delta_abs = abs(new - prev)
    delta_ratio = abs(new - prev) / max(1e-9, prev)
    if delta_ratio >= max(0.0, float(min_delta_ratio)):
        return True
    min_abs = max(0.0, float(min_delta_abs))
    if min_abs > 0 and delta_abs >= min_abs:
        return True
    if max_refresh_interval_sec > 0 and (now - last) >= float(max_refresh_interval_sec):
        return True
    return False


def should_allow_refresh_budget(
    state: dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    window_sec: float = 60.0,
    max_refresh_per_window: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    """
    Gate protection refresh churn by limiting refresh count per window.

    The caller can set force=True to bypass the budget for safety-critical cases
    (for example, severe under-coverage or a TTL-breached protection gap).
    """
    now = float(now_ts if now_ts is not None else time.time())
    if not isinstance(state, dict):
        state = {}
    win = max(0.0, _safe_float(window_sec, 0.0))
    limit = int(max(0, int(_safe_float(max_refresh_per_window, 0.0))))
    if force or limit <= 0 or win <= 0.0:
        return {"allowed": True, "count": 0, "limit": int(limit), "force": bool(force)}

    entries = state.get("refresh_events")
    if not isinstance(entries, list):
        entries = []
    cutoff = float(now - win)
    kept: list[float] = []
    for ts in entries:
        t = _safe_float(ts, -1.0)
        if t >= cutoff:
            kept.append(float(t))
    state["refresh_events"] = kept
    count = len(kept)
    if count >= limit:
        return {"allowed": False, "count": int(count), "limit": int(limit), "force": False}
    return {"allowed": True, "count": int(count), "limit": int(limit), "force": False}


def record_refresh_budget_event(
    state: dict[str, Any],
    *,
    now_ts: Optional[float] = None,
) -> None:
    now = float(now_ts if now_ts is not None else time.time())
    if not isinstance(state, dict):
        return
    entries = state.get("refresh_events")
    if not isinstance(entries, list):
        entries = []
        state["refresh_events"] = entries
    entries.append(float(now))


def update_coverage_gap_state(
    state: dict[str, Any],
    *,
    required_qty: float,
    covered: bool,
    ttl_sec: float = 90.0,
    now_ts: Optional[float] = None,
    reason: str = "",
    coverage_ratio: float = 1.0,
) -> dict[str, Any]:
    """
    Track protection coverage gaps across ticks and surface TTL breaches.
    """
    now = float(now_ts if now_ts is not None else time.time())
    req = abs(_safe_float(required_qty, 0.0))
    ttl = max(0.0, _safe_float(ttl_sec, 0.0))
    ratio = max(0.0, _safe_float(coverage_ratio, 1.0))
    if not isinstance(state, dict):
        state = {}
    prev_ttl_breached = bool(state.get("ttl_breached", False))

    # No open exposure (or coverage restored) clears active gap state.
    if req <= 0.0 or bool(covered):
        if _safe_float(state.get("gap_first_ts", 0.0), 0.0) > 0.0:
            state["last_resolved_ts"] = float(now)
        state["active"] = False
        state["covered"] = True
        state["required_qty"] = float(req)
        state["coverage_ratio"] = float(max(1.0, ratio if req > 0.0 else 1.0))
        state["gap_first_ts"] = 0.0
        state["gap_last_ts"] = float(now)
        state["gap_seconds"] = 0.0
        state["ttl_breached"] = False
        state["reason"] = "covered"
        return {
            "active": False,
            "gap_seconds": 0.0,
            "ttl_breached": False,
            "new_ttl_breach": False,
            "breach_count": int(_safe_float(state.get("breach_count", 0), 0.0)),
            "required_qty": float(req),
            "coverage_ratio": float(state["coverage_ratio"]),
            "reason": "covered",
        }

    first_ts = _safe_float(state.get("gap_first_ts", 0.0), 0.0)
    if first_ts <= 0.0:
        first_ts = float(now)
    gap_seconds = max(0.0, float(now - first_ts))
    ttl_breached = bool(ttl > 0.0 and gap_seconds >= ttl)
    new_ttl_breach = bool(ttl_breached and not prev_ttl_breached)
    if new_ttl_breach:
        state["breach_count"] = int(_safe_float(state.get("breach_count", 0), 0.0)) + 1

    state["active"] = True
    state["covered"] = False
    state["required_qty"] = float(req)
    state["coverage_ratio"] = float(ratio)
    state["gap_first_ts"] = float(first_ts)
    state["gap_last_ts"] = float(now)
    state["gap_seconds"] = float(gap_seconds)
    state["ttl_breached"] = bool(ttl_breached)
    state["ttl_sec"] = float(ttl)
    state["reason"] = str(reason or "uncovered")

    return {
        "active": True,
        "gap_seconds": float(gap_seconds),
        "ttl_breached": bool(ttl_breached),
        "new_ttl_breach": bool(new_ttl_breach),
        "breach_count": int(_safe_float(state.get("breach_count", 0), 0.0)),
        "required_qty": float(req),
        "coverage_ratio": float(ratio),
        "reason": str(state["reason"]),
    }


def _looks_like_dup_client_order_id(err: Exception) -> bool:
    s = repr(err)
    return ("-4116" in s) or ("ClientOrderId is duplicated" in s) or ("clientorderid is duplicated" in s.lower())


def _extract_order_id(order: dict[str, Any]) -> str:
    if not isinstance(order, dict):
        return ""
    for k in ("id", "orderId", "order_id"):
        v = order.get(k)
        if v:
            return str(v)
    info = order.get("info") or {}
    for k in ("orderId", "id"):
        v = info.get(k)
        if v:
            return str(v)
    return ""


def _order_client_id_any(order: dict[str, Any]) -> str:
    if not isinstance(order, dict):
        return ""
    info = order.get("info") or {}
    for kk in ("clientOrderId", "clientOrderID", "origClientOrderId", "origClientOrderID"):
        v = order.get(kk) or info.get(kk)
        if v:
            return str(v)
    return ""


async def _fetch_open_orders_safe(bot, sym_raw: str) -> list[dict[str, Any]]:
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return []
        fn = getattr(ex, "fetch_open_orders", None)
        if not callable(fn):
            return []
        res = await fn(sym_raw)
        return res if isinstance(res, list) else []
    except Exception:
        return []


async def _adopt_open_order_id_by_client_id(bot, sym_raw: str, client_id: str) -> Optional[str]:
    if not client_id or not sym_raw:
        return None
    oo = await _fetch_open_orders_safe(bot, sym_raw)
    for order in oo:
        try:
            if _order_client_id_any(order) == str(client_id):
                oid = _extract_order_id(order)
                if oid:
                    return oid
        except Exception:
            continue
    return None


async def place_stop_ladder_router(
    bot,
    *,
    sym_raw: str,
    side: str,
    qty: float,
    stop_price: float,
    hedge_side_hint: Optional[str],
    stop_client_id_base: str,
    intent_component: str = "entry",
    intent_kind: str = "STOP_RESTORE",
) -> Optional[str]:
    from execution.order_router import create_order  # local import to keep dependency surface narrow

    stop_side = "sell" if str(side).lower().strip() == "long" else "buy"
    cid_a = f"{stop_client_id_base}_A"
    cid_b = f"{stop_client_id_base}_B"

    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=float(abs(_safe_float(qty, 0.0))),
            price=None,
            params={},
            intent_reduce_only=True,
            intent_close_position=False,
            stop_price=float(stop_price),
            hedge_side_hint=hedge_side_hint,
            client_order_id=cid_a,
            retries=6,
            intent_component=intent_component,
            intent_kind=intent_kind,
        )
        oid = _extract_order_id(order if isinstance(order, dict) else {})
        if oid:
            return oid
    except Exception as e1:
        if _looks_like_dup_client_order_id(e1):
            adopted = await _adopt_open_order_id_by_client_id(bot, sym_raw, cid_a)
            if adopted:
                return adopted

    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=None,
            price=None,
            params={"closePosition": True},
            intent_reduce_only=False,
            intent_close_position=True,
            stop_price=float(stop_price),
            hedge_side_hint=hedge_side_hint,
            client_order_id=cid_b,
            retries=6,
            intent_component=intent_component,
            intent_kind=intent_kind,
        )
        oid = _extract_order_id(order if isinstance(order, dict) else {})
        if oid:
            return oid
    except Exception as e2:
        if _looks_like_dup_client_order_id(e2):
            adopted = await _adopt_open_order_id_by_client_id(bot, sym_raw, cid_b)
            if adopted:
                return adopted
    return None


async def place_trailing_router(
    bot,
    *,
    sym_raw: str,
    side: str,
    qty: float,
    activation_price: float,
    callback_rate: float,
    hedge_side_hint: Optional[str],
    intent_component: str = "entry",
    intent_kind: str = "TRAILING_RESTORE",
) -> Optional[str]:
    from execution.order_router import create_order  # local import to keep dependency surface narrow

    close_side = "sell" if str(side).lower().strip() == "long" else "buy"
    cb = max(0.1, min(5.0, float(_safe_float(callback_rate, 1.0))))
    qty_abs = float(abs(_safe_float(qty, 0.0)))

    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="TRAILING_STOP_MARKET",
            side=close_side,
            amount=qty_abs,
            price=None,
            params={},
            intent_reduce_only=True,
            activation_price=float(activation_price),
            callback_rate=float(cb),
            hedge_side_hint=hedge_side_hint,
            retries=6,
            intent_component=intent_component,
            intent_kind=intent_kind,
        )
        oid = _extract_order_id(order if isinstance(order, dict) else {})
        if oid:
            return oid
    except Exception:
        pass

    try:
        order = await create_order(
            bot,
            symbol=sym_raw,
            type="TRAILING_STOP_MARKET",
            side=close_side,
            amount=qty_abs,
            price=None,
            params={},
            intent_reduce_only=True,
            activation_price=float(activation_price),
            hedge_side_hint=hedge_side_hint,
            retries=6,
            intent_component=intent_component,
            intent_kind=intent_kind,
        )
        oid = _extract_order_id(order if isinstance(order, dict) else {})
        if oid:
            return oid
    except Exception:
        pass
    return None


def record_entry_stage_hint(
    run_context: dict[str, Any],
    *,
    symbol: str,
    plan: dict[str, Any],
    requested_qty: float,
    filled_qty: float,
    ttl_sec: float = 120.0,
    now_ts: Optional[float] = None,
) -> dict[str, Any]:
    """
    Persist a short-lived staged-protection hint emitted by entry execution.
    Position manager can consume this to prioritize immediate protection checks.
    """
    now = float(now_ts if now_ts is not None else time.time())
    if not isinstance(run_context, dict):
        return {}
    store = run_context.get("entry_stage_hints")
    if not isinstance(store, dict):
        store = {}
        run_context["entry_stage_hints"] = store
    sym = str(symbol or "").upper().strip()
    if not sym:
        return {}
    entry = {
        "ts": float(now),
        "expires_at": float(now + max(1.0, _safe_float(ttl_sec, 120.0))),
        "stage": str((plan or {}).get("stage") or ""),
        "stage1_required": bool((plan or {}).get("stage1_required", False)),
        "stage2_required": bool((plan or {}).get("stage2_required", False)),
        "stage3_required": bool((plan or {}).get("stage3_required", False)),
        "flatten_required": bool((plan or {}).get("flatten_required", False)),
        "requested_qty": float(abs(_safe_float(requested_qty, 0.0))),
        "filled_qty": float(abs(_safe_float(filled_qty, 0.0))),
        "fill_ratio": float(_safe_float((plan or {}).get("fill_ratio", 0.0), 0.0)),
    }
    store[sym] = entry
    return dict(entry)


def get_entry_stage_hint(
    run_context: dict[str, Any],
    *,
    symbol: str,
    now_ts: Optional[float] = None,
    consume: bool = False,
) -> Optional[dict[str, Any]]:
    if not isinstance(run_context, dict):
        return None
    store = run_context.get("entry_stage_hints")
    if not isinstance(store, dict):
        return None
    sym = str(symbol or "").upper().strip()
    if not sym:
        return None
    hint = store.get(sym)
    if not isinstance(hint, dict):
        return None
    now = float(now_ts if now_ts is not None else time.time())
    expires_at = _safe_float(hint.get("expires_at", 0.0), 0.0)
    if expires_at > 0 and now > expires_at:
        store.pop(sym, None)
        return None
    if consume:
        store.pop(sym, None)
    return dict(hint)
