from __future__ import annotations

import time
import hashlib
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from execution import event_journal as _event_journal  # type: ignore
except Exception:
    _event_journal = None

try:
    from execution import state_machine as _state_machine  # type: ignore
except Exception:
    _state_machine = None

try:
    from execution import intent_ledger as _intent_ledger  # type: ignore
except Exception:
    _intent_ledger = None

try:
    from execution import intent_allocator as _intent_allocator  # type: ignore
except Exception:
    _intent_allocator = None


def _now() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _truthy(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on", "t")
    return False


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _ensure_run_context(bot) -> dict:
    st = getattr(bot, "state", None)
    if st is None:
        return {}
    rc = getattr(st, "run_context", None)
    if not isinstance(rc, dict):
        try:
            st.run_context = {}
            rc = st.run_context
        except Exception:
            rc = {}
    return rc if isinstance(rc, dict) else {}


def _position_cls():
    try:
        from brain.state import Position  # type: ignore

        return Position
    except Exception:
        return None


def _build_position(*, symbol: str, side: str, size: float, entry_price: float, entry_ts: float):
    Pos = _position_cls()
    if Pos is not None:
        try:
            p = Pos(side=side, size=float(size), entry_price=float(entry_price), entry_ts=float(entry_ts), symbol=symbol)
            return p
        except Exception:
            pass
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        size=float(size),
        entry_price=float(entry_price),
        entry_ts=float(entry_ts),
        atr=0.0,
        leverage=0,
    )


def _extract_pos_size_side(pos: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    contracts = _safe_float(pos.get("contracts"), 0.0)
    if contracts:
        side = str(pos.get("side") or "").lower().strip()
        if side in ("long", "short"):
            return abs(contracts), side
    info = pos.get("info") or {}
    amt = info.get("positionAmt")
    if amt is not None:
        signed = _safe_float(amt, 0.0)
        if signed != 0:
            return abs(signed), ("long" if signed > 0 else "short")
    amt2 = _safe_float(pos.get("amount"), 0.0)
    if amt2 != 0:
        return abs(amt2), None
    return 0.0, None


def _extract_entry_price(pos: Dict[str, Any]) -> float:
    px = _safe_float(pos.get("entryPrice"), 0.0)
    if px > 0:
        return px
    info = pos.get("info") or {}
    for key in ("entryPrice", "avgPrice", "markPrice"):
        px2 = _safe_float(info.get(key), 0.0)
        if px2 > 0:
            return px2
    return 0.0


def _extract_trade_ts(tr: Dict[str, Any]) -> float:
    ts = _safe_float(tr.get("timestamp"), 0.0)
    if ts > 0:
        if ts > 1e12:
            return ts / 1000.0
        return ts
    info = tr.get("info") or {}
    for k in ("time", "timestamp", "T"):
        t2 = _safe_float(info.get(k), 0.0)
        if t2 > 0:
            if t2 > 1e12:
                return t2 / 1000.0
            return t2
    return 0.0


def _is_reduce_only_order(order: Dict[str, Any]) -> bool:
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


def _order_client_order_id(order: Dict[str, Any]) -> str:
    coid = str(order.get("clientOrderId") or "").strip()
    if coid:
        return coid
    info = order.get("info") or {}
    return str(info.get("clientOrderId") or info.get("origClientOrderId") or "").strip()


def _order_type(order: Dict[str, Any]) -> str:
    try:
        t = str(order.get("type") or "").strip().upper()
        if t:
            return t
        info = order.get("info") or {}
        t2 = str(info.get("type") or info.get("orderType") or "").strip().upper()
        return t2
    except Exception:
        return ""


def _order_is_protective(order: Dict[str, Any]) -> bool:
    t = _order_type(order)
    if "STOP" in t or "TAKE_PROFIT" in t or "TP" in t:
        return True
    info = order.get("info") or {}
    if info.get("stopPrice") is not None:
        return True
    if order.get("stopPrice") is not None:
        return True
    return False


def _classify_orphan(
    *,
    has_position: bool,
    reduce_only: bool,
    protective: bool,
    freeze_on_orphans: bool,
) -> tuple[str, str, str]:
    if has_position and reduce_only and protective:
        return "orphan_protective_order", "ADOPT", "position_has_protection_candidate"
    if not has_position and reduce_only:
        return "orphan_reduce_only_exit", "CANCEL", "no_position_for_reduce_only_order"
    if has_position and not reduce_only:
        if freeze_on_orphans:
            return "unknown_position_exposure", "FREEZE", "non_reduce_order_with_position"
        return "orphan_entry_order", "CANCEL", "non_reduce_order_with_position"
    if not has_position and not reduce_only:
        if freeze_on_orphans:
            return "unknown_position_exposure", "FREEZE", "no_position_non_reduce_order"
        return "orphan_entry_order", "CANCEL", "no_position_non_reduce_order"
    return "unknown_position_exposure", "FREEZE", "unclassified_orphan"


def _parse_orphan_policy(raw: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = str(raw or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        klass = str(k or "").strip().lower()
        action = str(v or "").strip().upper()
        if not klass:
            continue
        if action not in ("ADOPT", "CANCEL", "FREEZE"):
            continue
        out[klass] = action
    return out


def _orphan_policy_map(bot) -> Dict[str, str]:
    policy: Dict[str, str] = {
        "orphan_protective_order": "ADOPT",
        "orphan_reduce_only_exit": "CANCEL",
        "orphan_entry_order": "CANCEL",
        "unknown_position_exposure": "FREEZE",
    }
    raw = ""
    try:
        cfg = getattr(bot, "cfg", None)
        raw = str(getattr(cfg, "REBUILD_ORPHAN_POLICY", "") or "").strip()
    except Exception:
        raw = ""
    if not raw:
        try:
            import os

            raw = str(os.getenv("REBUILD_ORPHAN_POLICY", "") or "").strip()
        except Exception:
            raw = ""
    override = _parse_orphan_policy(raw)
    if override:
        policy.update(override)
    return policy


def _rebuild_intent_id(*, symbol: str, order_id: str, client_order_id: str) -> str:
    base = f"{symbol}|{order_id}|{client_order_id}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    sym = _symkey(symbol)[:8] or "UNK"
    return f"REBUILD-{sym}-{digest}"


def _rebuild_position_intent_id(*, symbol: str, side: str, entry_ts: float) -> str:
    base = f"{symbol}|{side}|{int(float(entry_ts or 0.0))}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    sym = _symkey(symbol)[:8] or "UNK"
    return f"REBUILD-POS-{sym}-{digest}"


def _rebuild_allocate_intent_id(
    bot,
    *,
    symbol: str,
    intent_kind: str,
    fallback: str,
    is_exit: bool = False,
) -> str:
    if _intent_allocator is not None:
        try:
            out = str(
                _intent_allocator.allocate_intent_id(
                    bot,
                    component="rebuild",
                    intent_kind=str(intent_kind or "REBUILD"),
                    symbol=str(symbol or ""),
                    is_exit=bool(is_exit),
                )
                or ""
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return str(fallback or "").strip()


async def _fetch_positions(ex, symbols: Optional[List[str]] = None) -> List[dict]:
    fp = getattr(ex, "fetch_positions", None)
    if not callable(fp):
        return []
    try:
        if symbols:
            out = await fp(symbols)
        else:
            out = await fp()
        return list(out or [])
    except Exception:
        try:
            out = await fp()
            return list(out or [])
        except Exception:
            return []


async def _fetch_open_orders(ex) -> List[dict]:
    fn = getattr(ex, "fetch_open_orders", None)
    if not callable(fn):
        return []
    try:
        out = await fn()
        return list(out or [])
    except Exception:
        return []


async def _fetch_recent_trades(ex, symbols: Iterable[str], since_ms: Optional[int]) -> List[dict]:
    fn = getattr(ex, "fetch_my_trades", None)
    if not callable(fn):
        return []
    out: List[dict] = []
    for sym in symbols:
        try:
            part = await fn(sym, since_ms) if since_ms is not None else await fn(sym)
            if isinstance(part, list):
                out.extend(part)
        except Exception:
            continue
    return out


async def _cancel_order_best_effort(ex, order_id: str, symbol: str) -> bool:
    fn = getattr(ex, "cancel_order", None)
    if not callable(fn):
        return False
    try:
        await fn(order_id, symbol)
        return True
    except Exception:
        try:
            await fn(order_id)
            return True
        except Exception:
            return False


async def rebuild_local_state(
    bot,
    *,
    symbols: Optional[List[str]] = None,
    fill_window_sec: float = 3600.0,
    adopt_orphans: bool = True,
    freeze_on_orphans: bool = False,
) -> dict:
    ex = getattr(bot, "ex", None)
    st = getattr(bot, "state", None)
    if ex is None or st is None:
        return {"ok": False, "reason": "missing_bot_state_or_exchange"}

    if not isinstance(getattr(st, "positions", None), dict):
        st.positions = {}
    old_positions = dict(getattr(st, "positions", {}) or {})

    positions_raw = await _fetch_positions(ex, symbols)
    open_orders = await _fetch_open_orders(ex)
    tracked_symbols = set(symbols or [])
    for p in positions_raw:
        if isinstance(p, dict):
            tracked_symbols.add(str(p.get("symbol") or ""))
    if not tracked_symbols and isinstance(old_positions, dict):
        tracked_symbols.update(list(old_positions.keys()))
    since_ms = int((_now() - max(60.0, float(fill_window_sec or 3600.0))) * 1000.0)
    fills = await _fetch_recent_trades(ex, tracked_symbols, since_ms)

    fill_ts: Dict[str, float] = {}
    for tr in fills:
        if not isinstance(tr, dict):
            continue
        k = _symkey(str(tr.get("symbol") or ""))
        if not k:
            continue
        t = _extract_trade_ts(tr)
        if t > fill_ts.get(k, 0.0):
            fill_ts[k] = t

    rebuilt: Dict[str, Any] = {}
    for p in positions_raw:
        if not isinstance(p, dict):
            continue
        k = _symkey(str(p.get("symbol") or ""))
        if not k:
            continue
        size, side_hint = _extract_pos_size_side(p)
        if size <= 0:
            continue
        side = side_hint if side_hint in ("long", "short") else str((p.get("side") or "")).lower().strip()
        if side not in ("long", "short"):
            side = "long"
        entry_price = _extract_entry_price(p)
        entry_ts = fill_ts.get(k, _now())
        rebuilt[k] = _build_position(symbol=k, side=side, size=float(size), entry_price=float(entry_price), entry_ts=float(entry_ts))
        if _intent_ledger is not None:
            try:
                pos_intent_id = _rebuild_position_intent_id(symbol=k, side=side, entry_ts=entry_ts)
                pos_intent_id = _rebuild_allocate_intent_id(
                    bot,
                    symbol=k,
                    intent_kind="REBUILD_POSITION",
                    fallback=pos_intent_id,
                    is_exit=False,
                )
                _intent_ledger.record(
                    bot,
                    intent_id=pos_intent_id,
                    stage="OPEN",
                    symbol=k,
                    side=side,
                    order_type="POSITION",
                    is_exit=False,
                    client_order_id="",
                    order_id="",
                    status="open",
                    reason="rebuild_position_seen",
                    meta={"size": float(size), "entry_price": float(entry_price)},
                )
            except Exception:
                pass

    orphan_decisions: List[dict] = []
    orphan_policy = _orphan_policy_map(bot)
    duplicate_open_orders_skipped = 0
    intent_collision_count = 0
    intent_collision_components: Dict[str, int] = {}
    seen_open_order_keys: set[str] = set()
    for o in open_orders:
        if not isinstance(o, dict):
            continue
        status = str(o.get("status") or "").lower().strip()
        if status not in ("open", "new", "partially_filled", ""):
            continue
        k = _symkey(str(o.get("symbol") or ""))
        if not k:
            continue
        order_id = str(o.get("id") or "").strip()
        client_order_id = _order_client_order_id(o)
        if order_id:
            dedupe_key = f"oid:{order_id}"
        else:
            dedupe_key = f"coid:{k}:{client_order_id}:{str(o.get('side') or '').lower().strip()}:{_order_type(o)}"
        if dedupe_key in seen_open_order_keys:
            duplicate_open_orders_skipped += 1
            continue
        seen_open_order_keys.add(dedupe_key)
        has_pos = k in rebuilt
        reduce_only = _is_reduce_only_order(o)
        protective = _order_is_protective(o)
        klass, action, reason = _classify_orphan(
            has_position=has_pos,
            reduce_only=reduce_only,
            protective=protective,
            freeze_on_orphans=freeze_on_orphans,
        )
        policy_action = str(orphan_policy.get(str(klass or "").lower()) or action).upper()
        policy_source = "default"
        if policy_action != str(action or "").upper():
            policy_source = "policy_override"
            action = policy_action
            reason = f"{reason}|policy:{policy_action.lower()}"
        if freeze_on_orphans and str(klass or "").lower() == "unknown_position_exposure":
            if str(action or "").upper() != "FREEZE":
                reason = f"{reason}|policy:forced_freeze"
            action = "FREEZE"
            policy_source = "forced_freeze"
        intent_id = ""
        current_intent_collision = False
        collision_existing_intent_id = ""
        collision_existing_order_id = ""
        collision_component = ""
        if _intent_ledger is not None:
            try:
                intent_id = str(
                    _intent_ledger.resolve_intent_id(
                        bot,
                        client_order_id=client_order_id,
                        order_id=order_id,
                    )
                    or ""
                ).strip()
            except Exception:
                intent_id = ""
        if intent_id and _intent_ledger is not None and order_id:
            try:
                existing = _intent_ledger.get_intent(bot, intent_id)
            except Exception:
                existing = None
            if isinstance(existing, dict):
                existing_oid = str(existing.get("order_id") or "").strip()
                if existing_oid and existing_oid != order_id:
                    intent_collision_count += 1
                    current_intent_collision = True
                    collision_existing_intent_id = str(intent_id or "")
                    collision_existing_order_id = str(existing_oid or "")
                    collision_component = str((existing.get("meta") or {}).get("component") or "unknown")
                    key = collision_component or "unknown"
                    intent_collision_components[key] = int(intent_collision_components.get(key, 0) or 0) + 1
                    intent_id = ""
        if not intent_id:
            intent_id = _rebuild_allocate_intent_id(
                bot,
                symbol=k,
                intent_kind="REBUILD_ORDER",
                fallback=_rebuild_intent_id(symbol=k, order_id=order_id, client_order_id=client_order_id),
                is_exit=bool(reduce_only),
            )
        if _intent_ledger is not None:
            try:
                _intent_ledger.record(
                    bot,
                    intent_id=intent_id,
                    stage="OPEN",
                    symbol=k,
                    side=str(o.get("side") or "").lower().strip(),
                    order_type=_order_type(o),
                    is_exit=bool(reduce_only),
                    client_order_id=client_order_id,
                    order_id=order_id,
                    status=(status or "open"),
                    reason="rebuild_open_order_seen",
                    meta={
                        "class": klass,
                        "action": action,
                        "protective": bool(protective),
                        "component": "rebuild",
                    },
                )
            except Exception:
                pass
        ent = {
            "symbol": k,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "intent_id": intent_id,
            "status": status or "open",
            "class": klass,
            "action": action,
            "reason": reason,
            "reduce_only": bool(reduce_only),
            "protective": bool(protective),
            "cancel_ok": False,
            "policy_source": str(policy_source),
            "intent_collision": bool(current_intent_collision),
            "intent_collision_component": str(collision_component or ""),
            "intent_collision_existing_intent_id": str(collision_existing_intent_id or ""),
            "intent_collision_existing_order_id": str(collision_existing_order_id or ""),
        }
        if action == "ADOPT":
            if _intent_ledger is not None:
                try:
                    _intent_ledger.record(
                        bot,
                        intent_id=intent_id,
                        stage="OPEN",
                        symbol=k,
                        side=str(o.get("side") or "").lower().strip(),
                        order_type=_order_type(o),
                        is_exit=bool(reduce_only),
                        client_order_id=client_order_id,
                        order_id=order_id,
                        status=(status or "open"),
                        reason="rebuild_orphan_adopted",
                        meta={"class": klass, "action": action, "policy_source": policy_source},
                    )
                except Exception:
                    pass
        elif action == "CANCEL":
            ent["cancel_ok"] = bool(await _cancel_order_best_effort(ex, str(ent["order_id"]), str(o.get("symbol") or k)))
            if _intent_ledger is not None:
                try:
                    _intent_ledger.record(
                        bot,
                        intent_id=intent_id,
                        stage=("DONE" if ent["cancel_ok"] else "CANCEL_SENT_UNKNOWN"),
                        symbol=k,
                        side=str(o.get("side") or "").lower().strip(),
                        order_type=_order_type(o),
                        is_exit=bool(reduce_only),
                        client_order_id=client_order_id,
                        order_id=order_id,
                        status=("canceled" if ent["cancel_ok"] else "open"),
                        reason=("rebuild_orphan_canceled" if ent["cancel_ok"] else "rebuild_orphan_cancel_failed"),
                        meta={"class": klass, "action": action, "policy_source": policy_source},
                    )
                except Exception:
                    pass
        elif action == "FREEZE" and _intent_ledger is not None:
            try:
                _intent_ledger.record(
                    bot,
                    intent_id=intent_id,
                    stage="OPEN_UNKNOWN",
                    symbol=k,
                    side=str(o.get("side") or "").lower().strip(),
                    order_type=_order_type(o),
                    is_exit=bool(reduce_only),
                    client_order_id=client_order_id,
                    order_id=order_id,
                    status="open",
                    reason="rebuild_orphan_frozen",
                    meta={"class": klass, "action": action, "policy_source": policy_source},
                )
            except Exception:
                pass
        orphan_decisions.append(ent)
        if _event_journal is not None:
            try:
                _event_journal.append_event(
                    bot,
                    "rebuild.orphan_decision",
                    {
                        "symbol": str(ent.get("symbol") or ""),
                        "order_id": str(ent.get("order_id") or ""),
                        "class": str(ent.get("class") or ""),
                        "action": str(ent.get("action") or ""),
                        "reason": str(ent.get("reason") or ""),
                        "reduce_only": bool(ent.get("reduce_only", False)),
                        "protective": bool(ent.get("protective", False)),
                        "cancel_ok": bool(ent.get("cancel_ok", False)),
                        "policy_source": str(ent.get("policy_source") or "default"),
                        "intent_collision": bool(ent.get("intent_collision", False)),
                        "intent_collision_component": str(ent.get("intent_collision_component") or ""),
                        "intent_collision_existing_intent_id": str(ent.get("intent_collision_existing_intent_id") or ""),
                        "intent_collision_existing_order_id": str(ent.get("intent_collision_existing_order_id") or ""),
                    },
                )
            except Exception:
                pass

    if adopt_orphans:
        for ent in orphan_decisions:
            k = _symkey(str(ent.get("symbol") or ""))
            if not k or k in rebuilt:
                continue
            prev = old_positions.get(k)
            if prev is not None:
                rebuilt[k] = prev

    orphan_decisions = sorted(
        list(orphan_decisions),
        key=lambda d: (
            str(d.get("symbol") or ""),
            str(d.get("action") or ""),
            str(d.get("order_id") or ""),
        ),
    )

    st.positions = rebuilt

    rc = _ensure_run_context(bot)
    actionable_orphans = [x for x in orphan_decisions if str(x.get("action") or "").upper() != "ADOPT"]
    summary = {
        "ok": True,
        "ts": _now(),
        "positions_rebuilt": int(len(rebuilt)),
        "positions_prev": int(len(old_positions)),
        "open_orders": int(len(open_orders)),
        "fills_seen": int(len(fills)),
        "orphans": int(len(actionable_orphans)),
        "orphans_list": actionable_orphans[:50],
        "orphan_decisions_total": int(len(orphan_decisions)),
        "orphan_decisions_list": orphan_decisions[:50],
        "orphan_action_counts": {
            "ADOPT": int(sum(1 for x in orphan_decisions if str(x.get("action") or "").upper() == "ADOPT")),
            "CANCEL": int(sum(1 for x in orphan_decisions if str(x.get("action") or "").upper() == "CANCEL")),
            "FREEZE": int(sum(1 for x in orphan_decisions if str(x.get("action") or "").upper() == "FREEZE")),
        },
        "duplicate_open_orders_skipped": int(duplicate_open_orders_skipped),
        "intent_collision_count": int(intent_collision_count),
        "intent_collision_components": dict(intent_collision_components),
        "orphan_policy": dict(orphan_policy),
    }
    rc["rebuild"] = summary
    try:
        st.rebuild_summary = summary
    except Exception:
        pass

    freeze_count = int(sum(1 for x in orphan_decisions if str(x.get("action") or "").upper() == "FREEZE"))
    if (freeze_on_orphans and len(orphan_decisions) > 0) or freeze_count > 0:
        try:
            st.halt = True
            st.shutdown_reason = "rebuild_orphans_detected" if freeze_count <= 0 else "rebuild_orphan_freeze"
            st.shutdown_source = "execution.rebuild"
            st.shutdown_ts = _now()
        except Exception:
            pass
        summary["halted"] = True
    else:
        summary["halted"] = False

    if _event_journal is not None:
        try:
            _event_journal.append_event(
                bot,
                "rebuild.summary",
                {
                    "positions_rebuilt": int(summary.get("positions_rebuilt", 0) or 0),
                    "positions_prev": int(summary.get("positions_prev", 0) or 0),
                    "orphans": int(summary.get("orphans", 0) or 0),
                    "orphan_action_counts": dict(summary.get("orphan_action_counts", {}) or {}),
                    "duplicate_open_orders_skipped": int(summary.get("duplicate_open_orders_skipped", 0) or 0),
                    "intent_collision_count": int(summary.get("intent_collision_count", 0) or 0),
                    "intent_collision_components": dict(summary.get("intent_collision_components", {}) or {}),
                    "halted": bool(summary.get("halted", False)),
                },
            )
        except Exception:
            pass
    if _event_journal is not None and _state_machine is not None:
        try:
            for sym in rebuilt.keys():
                _event_journal.journal_transition(
                    bot,
                    machine=_state_machine.MachineKind.POSITION_BELIEF.value,
                    entity=str(sym),
                    state_from=_state_machine.PositionBeliefState.FLAT.value,
                    state_to=_state_machine.PositionBeliefState.OPEN_CONFIRMED.value,
                    reason="rebuild_position",
                    meta={},
                )
        except Exception:
            pass

    return summary
