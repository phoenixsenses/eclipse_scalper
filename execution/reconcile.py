# execution/reconcile.py — SCALPER ETERNAL — REALITY RECONCILER — 2026 v1.8 (STOP SPAM FIX + RAW SYMBOL FALLBACK + THROTTLE)
# Patch vs v1.7:
# - ✅ FIX: Prevents "STOP_MARKET placed 30x" by ensuring open-orders fetch uses correct raw futures symbol (DOGE/USDT:USDT)
# - ✅ HARDEN: Adds per-symbol stop placement throttle (default 60s) even if open-orders fetch fails
# - ✅ HARDEN: Ensures bot.state.positions dict is persistent and written back after orphan adoption
# - ✅ Keeps: orphan adoption, hedge-aware ex_map, optional imports, router-integrated stop ladder

import asyncio
import os
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple, List

from utils.logging import log_entry, log_core
from execution.order_router import create_order, cancel_order, cancel_replace_order  # ✅ ROUTER


# ─────────────────────────────────────────────────────────────────────
# Diagnostics wiring (never fatal)
# ─────────────────────────────────────────────────────────────────────

_RECONCILE_ONCE = False


def _banner_once() -> None:
    global _RECONCILE_ONCE
    if _RECONCILE_ONCE:
        return
    _RECONCILE_ONCE = True
    log_core.info("RECONCILE ONLINE — reality reconciling is armed")


def _optional_import(module: str, attr: Optional[str] = None):
    """
    Best-effort optional import that logs explicitly what's missing.
    Returns module or attribute; returns None if unavailable.
    """
    try:
        mod = __import__(module, fromlist=[attr] if attr else [])
        if attr is None:
            return mod
        return getattr(mod, attr)
    except Exception:
        what = f"{module}.{attr}" if attr else module
        log_core.warning(f"OPTIONAL MISSING — {what}")
        return None


# Optional diagnostics dump (if present)
_print_diagnostics = _optional_import("execution.diagnostics", "print_diagnostics")
_tel_emit = _optional_import("execution.telemetry", "emit")
_BeliefController = _optional_import("execution.belief_controller", "BeliefController")
_compute_belief_evidence = _optional_import("execution.belief_evidence", "compute_belief_evidence")
_intent_ledger_summary = _optional_import("execution.intent_ledger", "summary")
_runtime_reliability_gate = _optional_import("execution.reliability_gate_runtime", "get_runtime_gate")
_kill_request_halt = _optional_import("risk.kill_switch", "request_halt")
_assess_stop_coverage = _optional_import("execution.protection_manager", "assess_stop_coverage")
_assess_tp_coverage = _optional_import("execution.protection_manager", "assess_tp_coverage")
_should_refresh_protection = _optional_import("execution.protection_manager", "should_refresh_protection")
_update_coverage_gap_state = _optional_import("execution.protection_manager", "update_coverage_gap_state")
_state_machine = _optional_import("execution.state_machine")
_journal_transition = _optional_import("execution.event_journal", "journal_transition")


def _diag_dump(bot, note: str) -> None:
    try:
        if callable(_print_diagnostics):
            log_core.warning(f"DIAG DUMP — {note}")
            _print_diagnostics(bot)
    except Exception:
        pass


# Optional kill-switch gate (used only to skip creating new protective stops)
is_halted = _optional_import("risk.kill_switch", "is_halted")

# Optional Position model (if present)
Position = _optional_import("brain.state", "Position")


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


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


def _runtime_reliability_coupling_enabled(bot) -> bool:
    # Default-on so existing safety posture remains unchanged unless explicitly disabled.
    env_raw = str(os.getenv("RUNTIME_RELIABILITY_COUPLING", "") or "").strip().lower()
    if env_raw:
        return env_raw in ("1", "true", "yes", "on")
    try:
        raw = str(getattr(getattr(bot, "cfg", None), "RUNTIME_RELIABILITY_COUPLING", "1") or "").strip().lower()
    except Exception:
        raw = "1"
    return raw in ("1", "true", "yes", "on")


def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
    except Exception:
        return default


def _ensure_shutdown_event(bot) -> asyncio.Event:
    ev = getattr(bot, "_shutdown", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot._shutdown = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


def _ensure_run_context(bot) -> dict:
    rc = getattr(getattr(bot, "state", None), "run_context", None)
    if not isinstance(rc, dict):
        try:
            bot.state.run_context = {}
        except Exception:
            pass
        rc = getattr(getattr(bot, "state", None), "run_context", None)
    return rc if isinstance(rc, dict) else {}


def _phantom_store(bot) -> dict:
    rc = _ensure_run_context(bot)
    store = rc.get("phantom")
    if not isinstance(store, dict):
        store = {}
        rc["phantom"] = store
    return store


def _position_belief_store(bot) -> dict:
    rc = _ensure_run_context(bot)
    store = rc.get("position_belief_state")
    if not isinstance(store, dict):
        store = {}
        rc["position_belief_state"] = store
    return store


def _set_position_belief_state(bot, symbol: str, next_state: str, reason: str) -> None:
    if not symbol:
        return
    k = _symkey(symbol)
    if not k:
        return
    store = _position_belief_store(bot)
    prev = str(store.get(k) or "FLAT").upper()
    nxt = str(next_state or "").upper().strip()
    if not nxt or prev == nxt:
        return
    strict = _truthy(_cfg(bot, "RECONCILE_STRICT_POSITION_TRANSITIONS", False))
    if _state_machine is not None:
        try:
            mk = getattr(_state_machine, "MachineKind", None)
            if mk is not None:
                _state_machine.transition(mk.POSITION_BELIEF, prev, nxt, reason)
        except Exception:
            if strict:
                raise
            pass
    store[k] = nxt
    if callable(_journal_transition):
        try:
            _journal_transition(
                bot,
                machine="position_belief",
                entity=k,
                state_from=prev,
                state_to=nxt,
                reason=reason,
                meta={},
            )
        except Exception:
            pass


def _is_binance_futures(bot) -> bool:
    """
    Best-effort check for binance futures mode so we can build raw symbols like DOGE/USDT:USDT.
    """
    ex = getattr(bot, "ex", None)
    if ex is None:
        return False
    # wrapper path: ex.exchange.options
    try:
        inner = getattr(ex, "exchange", None)
        opt = getattr(inner, "options", None) if inner is not None else None
        if isinstance(opt, dict):
            dt = str(opt.get("defaultType") or "").lower().strip()
            if dt in ("future", "futures", "swap"):
                return True
    except Exception:
        pass
    # ccxt wrappers sometimes expose options directly
    try:
        opt2 = getattr(ex, "options", None)
        if isinstance(opt2, dict):
            dt2 = str(opt2.get("defaultType") or "").lower().strip()
            if dt2 in ("future", "futures", "swap"):
                return True
    except Exception:
        pass
    # If we can't confirm futures, default to False to avoid bad symbol format on spot.
    return False


def _ensure_reconcile_metrics(bot) -> Dict[str, Any]:
    st = getattr(bot, "state", None)
    if st is None:
        return {}
    rm = getattr(st, "reconcile_metrics", None)
    if not isinstance(rm, dict):
        rm = {}
        try:
            st.reconcile_metrics = rm
        except Exception:
            pass
    rm.setdefault("mismatch_streak", 0)
    rm.setdefault("last_mismatch_ts", 0.0)
    rm.setdefault("repair_last_ts", {})
    rm.setdefault("mismatch_by_symbol", {})
    rm.setdefault("belief_debt_sec", 0.0)
    rm.setdefault("belief_debt_symbols", 0)
    rm.setdefault("belief_confidence", 1.0)
    rm.setdefault("evidence_confidence", 1.0)
    rm.setdefault("evidence_ws_score", 1.0)
    rm.setdefault("evidence_rest_score", 1.0)
    rm.setdefault("evidence_fill_score", 1.0)
    rm.setdefault("evidence_ws_last_seen_ts", 0.0)
    rm.setdefault("evidence_rest_last_seen_ts", 0.0)
    rm.setdefault("evidence_fill_last_seen_ts", 0.0)
    rm.setdefault("evidence_ws_gap_rate", 0.0)
    rm.setdefault("evidence_rest_gap_rate", 0.0)
    rm.setdefault("evidence_fill_gap_rate", 0.0)
    rm.setdefault("evidence_ws_error_rate", 0.0)
    rm.setdefault("evidence_rest_error_rate", 0.0)
    rm.setdefault("evidence_fill_error_rate", 0.0)
    rm.setdefault("evidence_contradiction_count", 0)
    rm.setdefault("evidence_degraded_sources", 0)
    rm.setdefault("evidence_contradiction_burn_rate", 0.0)
    rm.setdefault("intent_unknown_count", 0)
    rm.setdefault("intent_unknown_oldest_sec", 0.0)
    rm.setdefault("intent_unknown_mean_resolve_sec", 0.0)
    rm.setdefault("runtime_gate_available", False)
    rm.setdefault("runtime_gate_degraded", False)
    rm.setdefault("runtime_gate_reason", "")
    rm.setdefault("runtime_gate_replay_mismatch_count", 0)
    rm.setdefault("runtime_gate_invalid_transition_count", 0)
    rm.setdefault("runtime_gate_journal_coverage_ratio", 1.0)
    rm.setdefault("runtime_gate_mismatch_severity", 0.0)
    rm.setdefault("runtime_gate_invalid_severity", 0.0)
    rm.setdefault("runtime_gate_coverage_severity", 0.0)
    rm.setdefault("runtime_gate_mismatch_category_score", 0.0)
    rm.setdefault("runtime_gate_cat_ledger", 0)
    rm.setdefault("runtime_gate_cat_transition", 0)
    rm.setdefault("runtime_gate_cat_belief", 0)
    rm.setdefault("runtime_gate_cat_position", 0)
    rm.setdefault("runtime_gate_cat_orphan", 0)
    rm.setdefault("runtime_gate_cat_coverage_gap", 0)
    rm.setdefault("runtime_gate_cat_replace_race", 0)
    rm.setdefault("runtime_gate_cat_contradiction", 0)
    rm.setdefault("runtime_gate_cat_unknown", 0)
    rm.setdefault("runtime_gate_position_mismatch_count", 0)
    rm.setdefault("runtime_gate_orphan_count", 0)
    rm.setdefault("runtime_gate_protection_coverage_gap_seconds", 0.0)
    rm.setdefault("runtime_gate_replace_race_count", 0)
    rm.setdefault("runtime_gate_evidence_contradiction_count", 0)
    rm.setdefault("runtime_gate_degrade_score", 0.0)
    rm.setdefault("runtime_gate_replay_mismatch_ids", [])
    rm.setdefault("protection_coverage_gap_seconds", 0.0)
    rm.setdefault("protection_coverage_gap_symbols", 0)
    rm.setdefault("protection_coverage_ttl_breaches", 0)
    rm.setdefault("reconcile_first_gate_count", 0)
    rm.setdefault("reconcile_first_gate_max_severity", 0.0)
    rm.setdefault("reconcile_first_gate_max_streak", 0)
    rm.setdefault("reconcile_first_gate_last_reason", "")
    return rm


def _recent_reconcile_first_gate_metrics(bot) -> Dict[str, Any]:
    out = {
        "count": 0,
        "max_severity": 0.0,
        "max_streak": 0,
        "last_reason": "",
    }
    try:
        st = getattr(bot, "state", None)
        km = getattr(st, "kill_metrics", None) if st is not None else None
        if not isinstance(km, dict):
            return out
        events = km.get("reconcile_first_gate_events")
        if not isinstance(events, list):
            return out
        now_ts = _now()
        window_sec = max(10.0, float(_cfg(bot, "BELIEF_RECONCILE_FIRST_WINDOW_SEC", 120.0) or 120.0))
        severity_threshold = max(
            0.0, float(_cfg(bot, "BELIEF_RECONCILE_FIRST_SEVERITY_THRESHOLD", 0.85) or 0.85)
        )
        max_events = int(_cfg(bot, "BELIEF_RECONCILE_FIRST_EVENTS_MAX", 160) or 160)
        if max_events < 20:
            max_events = 20
        recent: list[dict] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ts = _safe_float(ev.get("ts", 0.0), 0.0)
            if ts <= 0 or (now_ts - ts) > window_sec:
                continue
            recent.append(ev)
        if len(recent) > max_events:
            recent = recent[-max_events:]
        # Trim stored list to avoid unbounded growth.
        km["reconcile_first_gate_events"] = recent
        if not recent:
            km["reconcile_first_gate_current_streak"] = 0
            return out
        out["count"] = int(len(recent))
        out["max_severity"] = float(
            max(_safe_float(ev.get("severity", 0.0), 0.0) for ev in recent)
        )
        out["last_reason"] = str((recent[-1] or {}).get("reason", "") or "")
        streak = 0
        max_streak = 0
        for ev in recent:
            sev = _safe_float(ev.get("severity", 0.0), 0.0)
            if sev >= severity_threshold:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0
        out["max_streak"] = int(max_streak)
        km["reconcile_first_gate_current_streak"] = int(streak)
        km["reconcile_first_gate_max_streak"] = int(
            max(int(km.get("reconcile_first_gate_max_streak", 0) or 0), int(max_streak))
        )
    except Exception:
        return out
    return out


def _record_symbol_mismatch(metrics: Dict[str, Any], symbol: str) -> None:
    if not isinstance(metrics, dict):
        return
    by_sym = metrics.get("mismatch_by_symbol")
    if not isinstance(by_sym, dict):
        by_sym = {}
        metrics["mismatch_by_symbol"] = by_sym
    k = _symkey(symbol)
    if not k:
        return
    now = _now()
    ent = by_sym.get(k)
    if not isinstance(ent, dict):
        ent = {"first_ts": now, "last_ts": now, "count": 0}
        by_sym[k] = ent
    ent["last_ts"] = now
    ent["count"] = int(ent.get("count", 0) or 0) + 1
    if _safe_float(ent.get("first_ts", 0.0), 0.0) <= 0:
        ent["first_ts"] = now


def _compute_belief_debt(metrics: Dict[str, Any], *, clear: Optional[set[str]] = None) -> Tuple[float, int]:
    if not isinstance(metrics, dict):
        return 0.0, 0
    by_sym = metrics.get("mismatch_by_symbol")
    if not isinstance(by_sym, dict):
        return 0.0, 0
    clear_set = set(_symkey(x) for x in (clear or set()) if _symkey(x))
    now = _now()
    max_age = 0.0
    active = 0
    stale_ttl = 3600.0
    for k in list(by_sym.keys()):
        ent = by_sym.get(k)
        if not isinstance(ent, dict):
            by_sym.pop(k, None)
            continue
        last_ts = _safe_float(ent.get("last_ts", 0.0), 0.0)
        first_ts = _safe_float(ent.get("first_ts", 0.0), 0.0)
        if k in clear_set:
            by_sym.pop(k, None)
            continue
        if last_ts > 0 and (now - last_ts) > stale_ttl:
            by_sym.pop(k, None)
            continue
        if first_ts <= 0:
            first_ts = last_ts if last_ts > 0 else now
            ent["first_ts"] = first_ts
        age = max(0.0, now - first_ts)
        if age > 0:
            active += 1
            if age > max_age:
                max_age = age
    metrics["belief_debt_sec"] = float(max_age)
    metrics["belief_debt_symbols"] = int(active)
    return float(max_age), int(active)


def _collect_symbol_belief_debt(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    by_sym = metrics.get("mismatch_by_symbol")
    if not isinstance(by_sym, dict):
        return out
    now = _now()
    for k, ent in by_sym.items():
        if not isinstance(ent, dict):
            continue
        kk = _symkey(str(k or ""))
        if not kk:
            continue
        first_ts = _safe_float(ent.get("first_ts", 0.0), 0.0)
        if first_ts <= 0:
            continue
        age = max(0.0, now - first_ts)
        if age > 0:
            out[kk] = float(age)
    return out


def _stop_outcome_is_covered(outcome: str) -> bool:
    return str(outcome or "").strip().lower() in ("present", "restored")


def _ensure_belief_controller(bot):
    st = getattr(bot, "state", None)
    if st is None or _BeliefController is None:
        return None
    ctl = getattr(st, "belief_controller", None)
    if ctl is not None:
        return ctl
    try:
        ctl = _BeliefController()
        st.belief_controller = ctl
        return ctl
    except Exception:
        return None


def _canonical_to_binance_linear_raw(k: str) -> str:
    """
    Convert DOGEUSDT -> DOGE/USDT:USDT
    Only safe for USDT linear futures.
    """
    kk = _symkey(k)
    if not kk.endswith("USDT"):
        return kk
    base = kk[:-4]
    if not base:
        return kk
    return f"{base}/USDT:USDT"


def _resolve_raw_symbol(bot, k: str) -> str:
    """
    Prefer bot.data.raw_symbol map.
    If missing and we're in futures, build binance-style raw symbol fallback so fetch_open_orders works.
    """
    kk = _symkey(k)

    # 1) data.raw_symbol (best)
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(kk):
            return str(raw_map[kk])
    except Exception:
        pass

    # 2) heuristic for binance linear futures (critical fix for your stop spam)
    try:
        if _is_binance_futures(bot):
            return _canonical_to_binance_linear_raw(kk)
    except Exception:
        pass

    # 3) fallback canonical
    return kk


def _extract_pos_size_side(pos: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Returns (abs_size, side_hint) where side_hint in {"long","short"} when known.
    """
    side = None

    contracts = _safe_float(pos.get("contracts"), 0.0)
    if contracts:
        side = pos.get("side")
        if isinstance(side, str):
            side = side.lower().strip()
        return abs(contracts), side if side in ("long", "short") else None

    info = pos.get("info") or {}
    amt = info.get("positionAmt")
    if amt is not None:
        signed = _safe_float(amt, 0.0)
        if signed != 0:
            side = "long" if signed > 0 else "short"
            return abs(signed), side

    amt2 = _safe_float(pos.get("amount"), 0.0)
    if amt2 != 0:
        return abs(amt2), None

    return 0.0, None


def _extract_entry_price(pos: Dict[str, Any]) -> float:
    for kk in ("entryPrice", "entry_price", "average"):
        v = _safe_float(pos.get(kk), 0.0)
        if v > 0:
            return v
    info = pos.get("info") or {}
    for kk in ("entryPrice", "avgPrice"):
        v = _safe_float(info.get(kk), 0.0)
        if v > 0:
            return v
    return 0.0


def _is_reduce_only(order: Dict[str, Any]) -> bool:
    try:
        if _truthy(order.get("reduceOnly")):
            return True
        info = order.get("info") or {}
        if _truthy(info.get("reduceOnly")):
            return True
        if _truthy(info.get("closePosition")):
            return True
        params = order.get("params") or {}
        if _truthy(params.get("reduceOnly")):
            return True
        if _truthy(params.get("closePosition")):
            return True
    except Exception:
        pass
    return False


def _is_stop_like(order: Dict[str, Any]) -> bool:
    try:
        t = str(order.get("type") or "").upper()
        if "STOP" in t:
            return True
        info = order.get("info") or {}
        it = str(info.get("type") or info.get("orderType") or "").upper()
        if "STOP" in it:
            return True
        if (order.get("stopPrice") is not None) or (info.get("stopPrice") is not None):
            return True
    except Exception:
        pass
    return False


def _is_tp_like(order: Dict[str, Any]) -> bool:
    try:
        t = str(order.get("type") or "").upper()
        if "TAKE_PROFIT" in t or "TP" in t:
            return True
        info = order.get("info") or {}
        it = str(info.get("type") or info.get("orderType") or "").upper()
        if "TAKE_PROFIT" in it:
            return True
    except Exception:
        pass
    return False


async def _safe_speak(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _alert_ok(bot, key: str, msg: str, cooldown_sec: float) -> bool:
    """
    Dedup + cooldown by 'key'. Stores last ts and last msg.
    """
    rc = _ensure_run_context(bot)
    store = rc.get("reconcile_alerts")
    if not isinstance(store, dict):
        store = {}
        rc["reconcile_alerts"] = store

    now = _now()
    prev = store.get(key)
    if not isinstance(prev, dict):
        prev = {"ts": 0.0, "msg": ""}
        store[key] = prev

    last_ts = _safe_float(prev.get("ts"), 0.0)
    last_msg = str(prev.get("msg") or "")

    if msg == last_msg and (now - last_ts) < cooldown_sec:
        return False

    if (now - last_ts) < cooldown_sec and msg != last_msg:
        return False

    prev["ts"] = now
    prev["msg"] = msg
    return True


def _make_pos_obj(k: str, *, side: Optional[str], size_abs: float, entry_price: float):
    """
    Create a Position-like object with attribute access.
    Prefers brain.state.Position if available, else SimpleNamespace.
    """
    try:
        if callable(Position):
            # try common constructor patterns
            try:
                return Position(symbol=k, side=side, size=float(size_abs), entry_price=float(entry_price))  # type: ignore
            except Exception:
                pass
            try:
                p = Position()  # type: ignore
                setattr(p, "symbol", k)
                setattr(p, "side", side)
                setattr(p, "size", float(size_abs))
                setattr(p, "entry_price", float(entry_price))
                return p
            except Exception:
                pass
    except Exception:
        pass

    p = SimpleNamespace()
    p.symbol = k
    p.side = side
    p.size = float(size_abs)
    p.entry_price = float(entry_price)
    # safety defaults expected by other modules
    p.atr = float(getattr(p, "atr", 0.0) or 0.0)
    return p


# ----------------------------
# Exchange fetch helpers
# ----------------------------

async def _fetch_open_orders_best_effort(bot, sym_raw: str) -> List[dict]:
    ex = getattr(bot, "ex", None)
    if ex is None:
        return []
    fn = getattr(ex, "fetch_open_orders", None)
    if not callable(fn):
        return []
    try:
        res = await fn(sym_raw)
        return res if isinstance(res, list) else []
    except Exception:
        return []


async def _fetch_positions_best_effort(bot, symbols: Optional[List[str]] = None) -> Tuple[List[dict], bool]:
    """
    Some ccxt wrappers accept fetch_positions([symbols]); some don't.
    If symbols is None -> fetch all positions (for orphan scan).
    """
    ex = getattr(bot, "ex", None)
    if ex is None:
        return [], False
    fp = getattr(ex, "fetch_positions", None)
    if not callable(fp):
        return [], False

    if symbols is None:
        try:
            res = await ex.fetch_positions()
            return (res if isinstance(res, list) else []), True
        except Exception:
            return [], False

    try:
        res = await ex.fetch_positions(symbols)
        if isinstance(res, list):
            return res, True
    except Exception:
        pass

    try:
        res = await ex.fetch_positions()
        if not isinstance(res, list):
            return [], False
        want = set(_symkey(s) for s in symbols)
        out = []
        for p in res:
            if isinstance(p, dict) and _symkey(p.get("symbol") or "") in want:
                out.append(p)
        return out, True
    except Exception:
        return [], False


async def _cancel_reduce_only_open_orders(bot, sym_raw: str, *, cancel_stops: bool = True, cancel_tps: bool = True):
    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        for o in oo:
            if not isinstance(o, dict):
                continue
            if not _is_reduce_only(o):
                continue

            if (not cancel_stops) and _is_stop_like(o):
                continue
            if (not cancel_tps) and _is_tp_like(o):
                continue

            oid = o.get("id")
            if oid:
                try:
                    await cancel_order(bot, str(oid), sym_raw)
                except Exception:
                    pass
    except Exception:
        pass


# ----------------------------
# Protective stop (ladder)
# ----------------------------

async def _place_stop_ladder_router(
    bot,
    *,
    sym_raw: str,
    side: str,             # "long"/"short"
    qty: float,
    stop_price: float,
    hedge_side_hint: Optional[str],
    k: str,
) -> Optional[str]:
    """
    Ladder stop restore:
    A) amount + reduceOnly
    B) closePosition fallbacks
    """
    stop_side = "sell" if side == "long" else "buy"

    try:
        o = await create_order(
            bot,
            symbol=sym_raw,
            type="STOP_MARKET",
            side=stop_side,
            amount=float(qty),
            price=None,
            params={},
            intent_reduce_only=True,
            intent_close_position=False,
            stop_price=float(stop_price),
            hedge_side_hint=hedge_side_hint,
            retries=6,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e1:
        log_entry.warning(f"{k} reconcile stop A failed: {e1}")

    last = None
    for amt_try in (float(qty), 0.0, 0, "0", "0.0"):
        try:
            o = await create_order(
                bot,
                symbol=sym_raw,
                type="STOP_MARKET",
                side=stop_side,
                amount=amt_try,
                price=None,
                params={"closePosition": True},
                intent_reduce_only=True,
                intent_close_position=True,
                stop_price=float(stop_price),
                hedge_side_hint=hedge_side_hint,
                retries=6,
            )
            if isinstance(o, dict) and o.get("id"):
                return str(o.get("id"))
        except Exception as e2:
            last = e2
            continue

    log_entry.critical(f"{k} reconcile stop B failed (all fallbacks): {last}")
    return None


async def _place_tp_ladder_router(
    bot,
    *,
    sym_raw: str,
    side: str,             # "long"/"short"
    qty: float,
    tp_price: float,
    hedge_side_hint: Optional[str],
    k: str,
) -> Optional[str]:
    tp_side = "sell" if side == "long" else "buy"
    try:
        o = await create_order(
            bot,
            symbol=sym_raw,
            type="TAKE_PROFIT_MARKET",
            side=tp_side,
            amount=float(qty),
            price=None,
            params={},
            intent_reduce_only=True,
            intent_close_position=False,
            stop_price=float(tp_price),
            hedge_side_hint=hedge_side_hint,
            retries=4,
        )
        if isinstance(o, dict) and o.get("id"):
            return str(o.get("id"))
    except Exception as e1:
        log_entry.warning(f"{k} reconcile tp A failed: {e1}")

    last = None
    for amt_try in (float(qty), 0.0, 0, "0", "0.0"):
        try:
            o = await create_order(
                bot,
                symbol=sym_raw,
                type="TAKE_PROFIT_MARKET",
                side=tp_side,
                amount=amt_try,
                price=None,
                params={"closePosition": True},
                intent_reduce_only=True,
                intent_close_position=True,
                stop_price=float(tp_price),
                hedge_side_hint=hedge_side_hint,
                retries=4,
            )
            if isinstance(o, dict) and o.get("id"):
                return str(o.get("id"))
        except Exception as e2:
            last = e2
            continue
    log_entry.warning(f"{k} reconcile tp B failed (all fallbacks): {last}")
    return None


def _stop_throttle_ok(bot, k: str) -> bool:
    """
    Hard guard to prevent repeated stop placement attempts.
    Default: 60 seconds per symbol.
    """
    rc = _ensure_run_context(bot)
    store = rc.get("stop_throttle")
    if not isinstance(store, dict):
        store = {}
        rc["stop_throttle"] = store

    now = _now()
    last = _safe_float(store.get(k), 0.0)
    throttle = float(_cfg(bot, "RECONCILE_STOP_THROTTLE_SEC", 60.0))

    if throttle <= 0:
        store[k] = now
        return True

    if (now - last) < throttle:
        return False

    store[k] = now
    return True


async def _ensure_protective_stop(bot, k: str, pos_obj, ex_side_hint: Optional[str], ex_size: float) -> str:
    if not _cfg(bot, "GUARDIAN_ENSURE_STOP", True):
        return "disabled"

    if _truthy(_cfg(bot, "GUARDIAN_RESPECT_KILL_SWITCH", True)) and callable(is_halted):
        try:
            if is_halted(bot):
                return "halted"
        except Exception:
            pass

    # Throttle first: even if open-orders fetch fails, we refuse to spam stops.
    if not _stop_throttle_ok(bot, k):
        return "throttled"

    sym_raw = _resolve_raw_symbol(bot, k)
    rm = _ensure_reconcile_metrics(bot)
    repair_last = rm.get("repair_last_ts")
    if not isinstance(repair_last, dict):
        repair_last = {}
        rm["repair_last_ts"] = repair_last
    repair_cd = float(_cfg(bot, "RECONCILE_REPAIR_COOLDOWN_SEC", 90.0))
    if repair_cd > 0:
        last_rep = _safe_float(repair_last.get(k), 0.0)
        if (_now() - last_rep) < repair_cd:
            return "repair_cooldown"

    existing_stop_id = ""
    existing_stop_qty = 0.0
    coverage_ratio = 1.0
    needs_refresh = False
    refresh_reason = ""
    # if stop exists and coverage is adequate, we're good
    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        if callable(_assess_stop_coverage):
            cov = _assess_stop_coverage(
                oo if isinstance(oo, list) else [],
                required_qty=float(max(0.0, ex_size)),
                min_coverage_ratio=float(_cfg(bot, "RECONCILE_STOP_MIN_COVERAGE_RATIO", 0.98) or 0.98),
            )
            existing_stop_id = str((cov or {}).get("order_id") or "")
            existing_stop_qty = _safe_float((cov or {}).get("existing_qty"), 0.0)
            coverage_ratio = _safe_float((cov or {}).get("coverage_ratio"), 1.0)
            needs_refresh = bool((cov or {}).get("needs_refresh", False))
            refresh_reason = str((cov or {}).get("reason") or "")
            if bool((cov or {}).get("covered", False)):
                try:
                    if existing_stop_id:
                        setattr(pos_obj, "stop_order_id", existing_stop_id)
                except Exception:
                    pass
                return "present"
        else:
            for o in oo:
                if isinstance(o, dict) and _is_reduce_only(o) and _is_stop_like(o):
                    try:
                        if o.get("id"):
                            setattr(pos_obj, "stop_order_id", str(o.get("id")))
                    except Exception:
                        pass
                    return "present"
    except Exception:
        pass

    side = None
    try:
        side = str(getattr(pos_obj, "side", "") or "").lower().strip()
    except Exception:
        side = None
    if side not in ("long", "short"):
        side = ex_side_hint if ex_side_hint in ("long", "short") else None
    if side not in ("long", "short"):
        msg = f"RECONCILE: {k} no side detected — stop not placed"
        if _alert_ok(bot, f"{k}:no_side", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
        return "no_side"

    hedge_side_hint = side  # router maps to positionSide LONG/SHORT if hedge enabled

    entry_px = _safe_float(getattr(pos_obj, "entry_price", 0.0), 0.0)
    atr = _safe_float(getattr(pos_obj, "atr", 0.0), 0.0)
    if entry_px <= 0:
        msg = f"RECONCILE: {k} missing entry_price — stop not placed"
        if _alert_ok(bot, f"{k}:no_entry", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
        return "no_entry"

    stop_atr_mult = float(_cfg(bot, "STOP_ATR_MULT", 2.0))
    buffer_atr_mult = float(_cfg(bot, "GUARDIAN_STOP_BUFFER_ATR_MULT", 0.0))

    if atr > 0:
        dist = atr * (stop_atr_mult + buffer_atr_mult)
    else:
        dist = entry_px * float(_cfg(bot, "GUARDIAN_STOP_FALLBACK_PCT", 0.0035))

    stop_price = entry_px - dist if side == "long" else entry_px + dist
    if stop_price <= 0:
        return "invalid_stop_price"

    oid = None
    if needs_refresh and existing_stop_id:
        rc = _ensure_run_context(bot)
        pstore = rc.get("protection_refresh")
        if not isinstance(pstore, dict):
            pstore = {}
            rc["protection_refresh"] = pstore
        prev = pstore.get(k)
        if not isinstance(prev, dict):
            prev = {}
            pstore[k] = prev
        prev_qty = _safe_float(prev.get("qty"), 0.0)
        prev_ts = _safe_float(prev.get("ts"), 0.0)
        allow_refresh = True
        force_ratio = _safe_float(_cfg(bot, "RECONCILE_STOP_REFRESH_FORCE_COVERAGE_RATIO", 0.80), 0.80)
        if force_ratio > 0 and coverage_ratio < force_ratio:
            allow_refresh = True
        elif force_ratio <= 0:
            allow_refresh = True
        if callable(_should_refresh_protection):
            allow_refresh = bool(
                _should_refresh_protection(
                    previous_qty=float(prev_qty if prev_qty > 0 else existing_stop_qty),
                    new_qty=float(ex_size),
                    last_refresh_ts=float(prev_ts),
                    min_delta_ratio=float(_cfg(bot, "RECONCILE_STOP_REFRESH_MIN_DELTA_RATIO", 0.10) or 0.10),
                    min_delta_abs=float(_cfg(bot, "RECONCILE_STOP_REFRESH_MIN_DELTA_ABS", 0.0) or 0.0),
                    max_refresh_interval_sec=float(_cfg(bot, "RECONCILE_STOP_REFRESH_MAX_INTERVAL_SEC", 45.0) or 45.0),
                    now_ts=_now(),
                )
            )
            if force_ratio > 0 and coverage_ratio < force_ratio:
                allow_refresh = True
        if not allow_refresh:
            return "refresh_deferred"
        if allow_refresh:
            stop_side = "sell" if side == "long" else "buy"
            try:
                rep = await cancel_replace_order(
                    bot,
                    cancel_order_id=str(existing_stop_id),
                    symbol=sym_raw,
                    type="STOP_MARKET",
                    side=stop_side,
                    amount=float(ex_size),
                    price=None,
                    stop_price=float(stop_price),
                    params={},
                    retries=int(_cfg(bot, "RECONCILE_STOP_REPLACE_RETRIES", 2) or 2),
                    correlation_id=f"RECON_STOP_{k}",
                )
                if isinstance(rep, dict):
                    oid = str(rep.get("id") or "") or None
                    prev["qty"] = float(ex_size)
                    prev["ts"] = _now()
            except Exception:
                oid = None
            if not oid:
                return "refresh_failed"

    if not oid:
        oid = await _place_stop_ladder_router(
            bot,
            sym_raw=sym_raw,
            side=side,
            qty=float(ex_size),
            stop_price=float(stop_price),
            hedge_side_hint=hedge_side_hint,
            k=k,
        )

    if oid:
        repair_last[k] = _now()
        try:
            setattr(pos_obj, "stop_order_id", str(oid))
        except Exception:
            pass
        msg = f"RECONCILE: PROTECTIVE STOP RESTORED {k} (id={oid})"
        if refresh_reason:
            msg = f"{msg} [{refresh_reason}]"
        if _alert_ok(bot, f"{k}:stop_restored", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "critical")
        return "restored"
    else:
        repair_last[k] = _now()
        msg = f"RECONCILE: STOP RESTORE FAILED {k}"
        if _alert_ok(bot, f"{k}:stop_failed", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "critical")
        return "failed"


async def _ensure_protective_tp(bot, k: str, pos_obj, ex_side_hint: Optional[str], ex_size: float) -> str:
    if not _cfg(bot, "GUARDIAN_ENSURE_TP", False):
        return "tp_disabled"

    sym_raw = _resolve_raw_symbol(bot, k)
    side = None
    try:
        side = str(getattr(pos_obj, "side", "") or "").lower().strip()
    except Exception:
        side = None
    if side not in ("long", "short"):
        side = ex_side_hint if ex_side_hint in ("long", "short") else None
    if side not in ("long", "short"):
        return "no_side"

    entry_px = _safe_float(getattr(pos_obj, "entry_price", 0.0), 0.0)
    atr = _safe_float(getattr(pos_obj, "atr", 0.0), 0.0)
    if entry_px <= 0:
        return "no_entry"

    tp_atr_mult = float(_cfg(bot, "TP_ATR_MULT", 1.8))
    if atr > 0:
        dist = atr * tp_atr_mult
    else:
        dist = entry_px * float(_cfg(bot, "RECONCILE_TP_FALLBACK_PCT", 0.0050))
    tp_price = entry_px + dist if side == "long" else entry_px - dist
    if tp_price <= 0:
        return "invalid_tp_price"

    existing_tp_id = ""
    existing_tp_qty = 0.0
    coverage_ratio = 1.0
    needs_refresh = False
    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        if callable(_assess_tp_coverage):
            cov = _assess_tp_coverage(
                oo if isinstance(oo, list) else [],
                required_qty=float(max(0.0, ex_size)),
                min_coverage_ratio=float(_cfg(bot, "RECONCILE_TP_MIN_COVERAGE_RATIO", 0.98) or 0.98),
            )
            existing_tp_id = str((cov or {}).get("order_id") or "")
            existing_tp_qty = _safe_float((cov or {}).get("existing_qty"), 0.0)
            coverage_ratio = _safe_float((cov or {}).get("coverage_ratio"), 1.0)
            needs_refresh = bool((cov or {}).get("needs_refresh", False))
            if bool((cov or {}).get("covered", False)):
                return "tp_present"
    except Exception:
        pass

    oid = None
    if needs_refresh and existing_tp_id:
        rc = _ensure_run_context(bot)
        pstore = rc.get("protection_refresh_tp")
        if not isinstance(pstore, dict):
            pstore = {}
            rc["protection_refresh_tp"] = pstore
        prev = pstore.get(k)
        if not isinstance(prev, dict):
            prev = {}
            pstore[k] = prev
        prev_qty = _safe_float(prev.get("qty"), 0.0)
        prev_ts = _safe_float(prev.get("ts"), 0.0)
        allow_refresh = True
        force_ratio = _safe_float(_cfg(bot, "RECONCILE_TP_REFRESH_FORCE_COVERAGE_RATIO", 0.80), 0.80)
        if callable(_should_refresh_protection):
            allow_refresh = bool(
                _should_refresh_protection(
                    previous_qty=float(prev_qty if prev_qty > 0 else existing_tp_qty),
                    new_qty=float(ex_size),
                    last_refresh_ts=float(prev_ts),
                    min_delta_ratio=float(_cfg(bot, "RECONCILE_TP_REFRESH_MIN_DELTA_RATIO", 0.10) or 0.10),
                    min_delta_abs=float(_cfg(bot, "RECONCILE_TP_REFRESH_MIN_DELTA_ABS", 0.0) or 0.0),
                    max_refresh_interval_sec=float(_cfg(bot, "RECONCILE_TP_REFRESH_MAX_INTERVAL_SEC", 45.0) or 45.0),
                    now_ts=_now(),
                )
            )
            if force_ratio > 0 and coverage_ratio < force_ratio:
                allow_refresh = True
        if not allow_refresh:
            return "tp_refresh_deferred"
        tp_side = "sell" if side == "long" else "buy"
        try:
            rep = await cancel_replace_order(
                bot,
                cancel_order_id=str(existing_tp_id),
                symbol=sym_raw,
                type="TAKE_PROFIT_MARKET",
                side=tp_side,
                amount=float(ex_size),
                price=None,
                stop_price=float(tp_price),
                params={},
                retries=int(_cfg(bot, "RECONCILE_TP_REPLACE_RETRIES", 2) or 2),
                correlation_id=f"RECON_TP_{k}",
            )
            if isinstance(rep, dict):
                oid = str(rep.get("id") or "") or None
                prev["qty"] = float(ex_size)
                prev["ts"] = _now()
        except Exception:
            oid = None
        if not oid:
            return "tp_refresh_failed"

    if not oid:
        oid = await _place_tp_ladder_router(
            bot,
            sym_raw=sym_raw,
            side=side,
            qty=float(ex_size),
            tp_price=float(tp_price),
            hedge_side_hint=side,
            k=k,
        )
    return "tp_restored" if oid else "tp_failed"


# ----------------------------
# Orphan reduceOnly order cleanup (optional)
# ----------------------------

async def _cancel_orphan_reduce_only_orders_if_no_position(bot, k: str):
    if not _truthy(_cfg(bot, "RECONCILE_CANCEL_ORPHAN_REDUCE_ONLY_ORDERS", True)):
        return

    sym_raw = _resolve_raw_symbol(bot, k)
    try:
        oo = await _fetch_open_orders_best_effort(bot, sym_raw)
        if not oo:
            return
        any_ro = any(isinstance(o, dict) and _is_reduce_only(o) for o in oo)
        if not any_ro:
            return

        await _cancel_reduce_only_open_orders(bot, sym_raw, cancel_stops=True, cancel_tps=True)

        msg = f"RECONCILE: ORPHAN reduceOnly orders canceled {k}"
        if _alert_ok(bot, f"{k}:orphan_ro_canceled", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
    except Exception:
        pass


# ----------------------------
# Main reconcile tick
# ----------------------------

async def reconcile_tick(bot):
    """
    Single pass: reconcile bot.state.positions vs exchange positions.
    Called by execution/guardian.guardian_loop().
    """
    _banner_once()
    metrics = _ensure_reconcile_metrics(bot)
    mismatch_events = 0
    repair_actions = 0
    repair_skipped = 0
    mismatch_symbols: set[str] = set()
    coverage_ttl_breaches = 0

    drift_abs = float(_cfg(bot, "GUARDIAN_SIZE_DRIFT_ABS", 0.0))
    drift_pct = float(_cfg(bot, "GUARDIAN_SIZE_DRIFT_PCT", 0.05))
    auto_flat_orphans = bool(_cfg(bot, "GUARDIAN_AUTO_FLAT_ORPHANS", False))
    full_scan = _truthy(_cfg(bot, "RECONCILE_FULL_SCAN_ORPHANS", True))

    adopt_orphans = _truthy(_cfg(bot, "RECONCILE_ADOPT_ORPHANS", True))

    _ensure_shutdown_event(bot)
    rc = _ensure_run_context(bot)
    protection_gap_state = rc.get("protection_gap_state")
    if not isinstance(protection_gap_state, dict):
        protection_gap_state = {}
        rc["protection_gap_state"] = protection_gap_state

    # Ensure a persistent dict reference lives on bot.state
    try:
        if not hasattr(bot.state, "positions") or not isinstance(getattr(bot.state, "positions", None), dict):
            bot.state.positions = {}
    except Exception:
        try:
            bot.state.positions = {}
        except Exception:
            pass

    state_positions: Dict[str, Any] = getattr(bot.state, "positions", {})  # must be the actual dict on state
    if not isinstance(state_positions, dict):
        state_positions = {}
        try:
            bot.state.positions = state_positions
        except Exception:
            pass

    active = getattr(bot, "active_symbols", set()) or set()

    tracked_syms = set(_symkey(s) for s in state_positions.keys())
    tracked_syms |= set(_symkey(s) for s in active if s)

    if full_scan:
        ex_positions, ok = await _fetch_positions_best_effort(bot, None)
    else:
        if not tracked_syms:
            return
        ex_positions, ok = await _fetch_positions_best_effort(bot, list(tracked_syms))

    if not ok:
        log_entry.warning("RECONCILE: fetch_positions failed — skipping phantom/orphan checks this cycle")
        mismatch_events += 1
        for s in tracked_syms:
            if s:
                mismatch_symbols.add(_symkey(s))
                _record_symbol_mismatch(metrics, s)
        km = getattr(getattr(bot, "state", None), "kill_metrics", None)
        if isinstance(km, dict):
            km["reconcile_mismatch_streak"] = int(km.get("reconcile_mismatch_streak", 0) or 0) + 1
            km["reconcile_last_mismatch_ts"] = _now()
            debt_sec, debt_syms = _compute_belief_debt(metrics)
            km["reconcile_belief_debt_sec"] = float(debt_sec)
            km["reconcile_belief_debt_symbols"] = int(debt_syms)
        return

    # Build exchange map: symbol -> list of position dicts (hedge-mode can return multiple legs)
    ex_map: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(ex_positions, list):
        for p in ex_positions:
            if not isinstance(p, dict):
                continue
            sym = p.get("symbol")
            if not sym:
                continue
            kk = _symkey(sym)
            size, _ = _extract_pos_size_side(p)
            if size > 0:
                ex_map.setdefault(kk, []).append(p)
    for kx in ex_map.keys():
        _set_position_belief_state(bot, kx, "OPEN_CONFIRMED", "exchange_position_seen")

    # 1) ORPHAN EXCHANGE POSITIONS (adopt)
    for k, plist in ex_map.items():
        if k in state_positions:
            _set_position_belief_state(bot, k, "OPEN_CONFIRMED", "position_exists_both")
            continue
        mismatch_events += 1
        mismatch_symbols.add(k)
        _record_symbol_mismatch(metrics, k)

        legs: List[Tuple[float, Optional[str], float]] = []
        for p in plist:
            sz, sd = _extract_pos_size_side(p)
            ep = _extract_entry_price(p)
            if sz > 0:
                legs.append((sz, sd, ep))

        adopted_side: Optional[str] = None
        adopted_size: float = 0.0
        adopted_entry: float = 0.0
        multi = False

        if len(legs) == 1 and legs[0][1] in ("long", "short"):
            adopted_size, adopted_side, adopted_entry = legs[0]
        else:
            multi = True
            adopted_size = sum(x[0] for x in legs)
            adopted_entry = 0.0
            adopted_side = None

        if legs:
            for (sz, sd, ep) in legs:
                msg = f"RECONCILE: ORPHAN EXCHANGE POSITION {k} | side={sd} | size={sz:.6f} | entry={ep:.6f}"
                log_core.critical(msg)
                if _alert_ok(bot, f"{k}:orphan_pos:{sd}:{int(sz*1e6)}", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                    await _safe_speak(bot, msg, "critical")

        if adopt_orphans and adopted_size > 0:
            try:
                pos_obj = _make_pos_obj(k, side=adopted_side, size_abs=adopted_size, entry_price=adopted_entry)
                state_positions[k] = pos_obj
                _set_position_belief_state(bot, k, "OPEN_CONFIRMED", "orphan_adopted")
                msgA = f"RECONCILE: ORPHAN ADOPTED {k} | side={adopted_side} | size={adopted_size:.6f}"
                if multi:
                    msgA += " | NOTE=multi-leg/unknown-side (entry blocked; stop not auto-placed)"
                log_core.critical(msgA)
                if _alert_ok(bot, f"{k}:orphan_adopted", msgA, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                    await _safe_speak(bot, msgA, "critical")
            except Exception as e:
                log_entry.error(f"RECONCILE: orphan adopt failed {k}: {e}")

        if auto_flat_orphans:
            if len(legs) == 1 and legs[0][1] in ("long", "short"):
                size, side, _ = legs[0]
                try:
                    sym_raw = _resolve_raw_symbol(bot, k)
                    close_side = "sell" if side == "long" else "buy"
                    await create_order(
                        bot,
                        symbol=sym_raw,
                        type="MARKET",
                        side=close_side,
                        amount=float(size),
                        params={},
                        intent_reduce_only=True,
                        hedge_side_hint=side,
                        retries=6,
                    )
                    msg2 = f"RECONCILE: ORPHAN FLATTENED {k}"
                    if _alert_ok(bot, f"{k}:orphan_flat", msg2, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                        await _safe_speak(bot, msg2, "critical")
                except Exception as e:
                    log_entry.error(f"Reconcile orphan flatten failed {k}: {e}")
                    msg3 = f"RECONCILE: ORPHAN FLATTEN FAILED {k} — {e}"
                    if _alert_ok(bot, f"{k}:orphan_flat_fail", msg3, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                        await _safe_speak(bot, msg3, "critical")
                    mismatch_events += 1
                    mismatch_symbols.add(k)
                    _record_symbol_mismatch(metrics, k)

    # 2) PHANTOM STATE POSITIONS
    phantom = _phantom_store(bot)
    phantom_miss = int(_cfg(bot, "RECONCILE_PHANTOM_MISS_COUNT", 3))
    phantom_grace = float(_cfg(bot, "RECONCILE_PHANTOM_GRACE_SEC", 45.0))

    for k in list(state_positions.keys()):
        if k in ex_map:
            phantom.pop(k, None)
            _set_position_belief_state(bot, k, "OPEN_CONFIRMED", "phantom_cleared_by_exchange")
            continue

        now = _now()
        ent = phantom.get(k) if isinstance(phantom.get(k), dict) else {}
        first_ts = _safe_float(ent.get("first_ts", 0.0), 0.0) or now
        misses = int(ent.get("misses", 0)) + 1
        phantom[k] = {"first_ts": first_ts, "misses": misses}
        _set_position_belief_state(bot, k, "CLOSED_UNKNOWN", "phantom_missing")

        if misses < max(1, phantom_miss) or (now - first_ts) < max(0.0, phantom_grace):
            continue

        sym_raw = _resolve_raw_symbol(bot, k)
        log_core.warning(f"RECONCILE: PHANTOM STATE POSITION {k} — clearing + cancel reduceOnly orders")
        mismatch_events += 1
        mismatch_symbols.add(k)
        _record_symbol_mismatch(metrics, k)
        msg = f"RECONCILE: PHANTOM STATE POSITION {k} — cleared"
        if _alert_ok(bot, f"{k}:phantom_cleared", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
            await _safe_speak(bot, msg, "info")
        await _cancel_reduce_only_open_orders(bot, sym_raw)
        try:
            state_positions.pop(k, None)
        except Exception:
            pass
        _set_position_belief_state(bot, k, "FLAT", "phantom_removed")
        phantom.pop(k, None)

    # 2b) ORPHAN reduceOnly orders (no position exists)
    for k in list(tracked_syms):
        if k and k not in ex_map and k not in state_positions:
            await _cancel_orphan_reduce_only_orders_if_no_position(bot, k)

    # 3) DRIFT reconcile + entry_price patch + protective stop
    for k, pos_obj in list(state_positions.items()):
        plist = ex_map.get(k)
        if not plist:
            _set_position_belief_state(bot, k, "CLOSED_UNKNOWN", "state_without_exchange_position")
            continue

        best_p = None
        best_size = 0.0
        best_side = None
        for p in plist:
            ex_size, ex_side = _extract_pos_size_side(p)
            if ex_size > best_size:
                best_size, best_side, best_p = ex_size, ex_side, p

        if not isinstance(best_p, dict) or best_size <= 0:
            continue

        st_size = abs(_safe_float(getattr(pos_obj, "size", 0.0), 0.0))
        thresh = max(drift_abs, st_size * drift_pct) if st_size > 0 else drift_abs

        if st_size <= 0 or (thresh > 0 and abs(st_size - best_size) > thresh):
            mismatch_events += 1
            mismatch_symbols.add(k)
            _record_symbol_mismatch(metrics, k)
            try:
                setattr(pos_obj, "size", float(best_size))
            except Exception:
                pass
            msg = f"RECONCILE: SIZE SYNC {k} state={st_size:.6f} -> ex={best_size:.6f}"
            if _alert_ok(bot, f"{k}:size_sync", msg, float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0))):
                await _safe_speak(bot, msg, "info")

        if _safe_float(getattr(pos_obj, "entry_price", 0.0), 0.0) <= 0:
            ep = _extract_entry_price(best_p)
            if ep > 0:
                try:
                    setattr(pos_obj, "entry_price", float(ep))
                except Exception:
                    pass

        try:
            st_side = str(getattr(pos_obj, "side", "") or "").lower().strip()
        except Exception:
            st_side = ""
        if st_side not in ("long", "short") and best_side in ("long", "short"):
            try:
                setattr(pos_obj, "side", best_side)
            except Exception:
                pass

        if best_side in ("long", "short"):
            _set_position_belief_state(bot, k, "OPEN_CONFIRMED", "drift_reconciled")
            stop_outcome = await _ensure_protective_stop(bot, k, pos_obj, best_side, best_size)
            if stop_outcome in ("restored", "failed", "refresh_failed"):
                repair_actions += 1
            if stop_outcome in ("throttled", "repair_cooldown", "refresh_deferred"):
                repair_skipped += 1
            if stop_outcome in ("failed", "refresh_failed"):
                mismatch_events += 1
                mismatch_symbols.add(k)
                _record_symbol_mismatch(metrics, k)
            tp_outcome = await _ensure_protective_tp(bot, k, pos_obj, best_side, best_size)
            if tp_outcome in ("tp_restored", "tp_failed", "tp_refresh_failed"):
                repair_actions += 1
            if tp_outcome in ("tp_refresh_deferred",):
                repair_skipped += 1
            if tp_outcome in ("tp_failed", "tp_refresh_failed"):
                mismatch_events += 1
                mismatch_symbols.add(k)
                _record_symbol_mismatch(metrics, k)

            if callable(_update_coverage_gap_state):
                s = protection_gap_state.get(k)
                if not isinstance(s, dict):
                    s = {}
                    protection_gap_state[k] = s
                cov = _update_coverage_gap_state(
                    s,
                    required_qty=float(best_size),
                    covered=bool(_stop_outcome_is_covered(stop_outcome)),
                    ttl_sec=float(_cfg(bot, "RECONCILE_PROTECTION_GAP_TTL_SEC", 90.0) or 90.0),
                    now_ts=_now(),
                    reason=str(stop_outcome),
                    coverage_ratio=(1.0 if _stop_outcome_is_covered(stop_outcome) else 0.0),
                )
                if bool((cov or {}).get("new_ttl_breach", False)):
                    coverage_ttl_breaches += 1
                    mismatch_events += 1
                    mismatch_symbols.add(k)
                    _record_symbol_mismatch(metrics, k)
                    gap_sec = float(_safe_float((cov or {}).get("gap_seconds", 0.0), 0.0))
                    msg = (
                        f"RECONCILE: PROTECTION COVERAGE GAP TTL BREACH {k} "
                        f"gap={gap_sec:.1f}s outcome={stop_outcome}"
                    )
                    if _alert_ok(
                        bot,
                        f"{k}:protection_gap_ttl",
                        msg,
                        float(_cfg(bot, "RECONCILE_ALERT_COOLDOWN_SEC", 120.0)),
                    ):
                        await _safe_speak(bot, msg, "critical")

    # ensure state dict stays attached
    try:
        bot.state.positions = state_positions
    except Exception:
        pass

    protection_gap_seconds = 0.0
    protection_gap_symbols = 0
    if isinstance(protection_gap_state, dict):
        for key in list(protection_gap_state.keys()):
            ent = protection_gap_state.get(key)
            if not isinstance(ent, dict):
                protection_gap_state.pop(key, None)
                continue
            active_gap = bool(ent.get("active", False))
            gap_sec = max(0.0, _safe_float(ent.get("gap_seconds", 0.0), 0.0))
            req_qty = max(0.0, _safe_float(ent.get("required_qty", 0.0), 0.0))
            if active_gap and req_qty > 0.0 and gap_sec > 0.0:
                protection_gap_symbols += 1
                if gap_sec > protection_gap_seconds:
                    protection_gap_seconds = gap_sec
            elif (not active_gap) and req_qty <= 0.0:
                protection_gap_state.pop(key, None)
    metrics["protection_coverage_gap_seconds"] = float(protection_gap_seconds)
    metrics["protection_coverage_gap_symbols"] = int(protection_gap_symbols)
    metrics["protection_coverage_ttl_breaches"] = int(coverage_ttl_breaches)

    if mismatch_events > 0:
        metrics["mismatch_streak"] = int(metrics.get("mismatch_streak", 0) or 0) + int(mismatch_events)
        metrics["last_mismatch_ts"] = _now()
    else:
        metrics["mismatch_streak"] = max(0, int(metrics.get("mismatch_streak", 0) or 0) - 1)
    metrics["last_summary"] = {
        "mismatch_events": int(mismatch_events),
        "repair_actions": int(repair_actions),
        "repair_skipped": int(repair_skipped),
        "protection_coverage_gap_seconds": float(protection_gap_seconds),
        "protection_coverage_gap_symbols": int(protection_gap_symbols),
        "protection_coverage_ttl_breaches": int(coverage_ttl_breaches),
        "ts": _now(),
    }
    healthy_syms = set(_symkey(s) for s in tracked_syms if _symkey(s) and _symkey(s) not in mismatch_symbols)
    debt_sec, debt_syms = _compute_belief_debt(metrics, clear=healthy_syms)
    symbol_belief_debt_sec = _collect_symbol_belief_debt(metrics)
    worst_symbols = sorted(symbol_belief_debt_sec.items(), key=lambda kv: kv[1], reverse=True)[:5]
    debt_ref = max(1.0, float(_cfg(bot, "RECONCILE_BELIEF_DEBT_REF_SEC", 300.0) or 300.0))
    conf = max(0.0, min(1.0, 1.0 - (float(debt_sec) / debt_ref)))
    metrics["belief_confidence"] = float(conf)
    evidence = {
        "evidence_confidence": 1.0,
        "evidence_ws_score": 1.0,
        "evidence_rest_score": 1.0,
        "evidence_fill_score": 1.0,
        "evidence_ws_last_seen_ts": 0.0,
        "evidence_rest_last_seen_ts": 0.0,
        "evidence_fill_last_seen_ts": 0.0,
        "evidence_ws_gap_rate": 0.0,
        "evidence_rest_gap_rate": 0.0,
        "evidence_fill_gap_rate": 0.0,
        "evidence_ws_error_rate": 0.0,
        "evidence_rest_error_rate": 0.0,
        "evidence_fill_error_rate": 0.0,
        "evidence_contradiction_score": 0.0,
        "evidence_contradiction_count": 0,
        "evidence_contradiction_streak": 0,
        "evidence_contradiction_burn_rate": 0.0,
        "evidence_degraded_sources": 0,
    }
    if callable(_compute_belief_evidence):
        try:
            got = _compute_belief_evidence(bot, getattr(bot, "cfg", None))
            if isinstance(got, dict):
                evidence.update(got)
        except Exception:
            pass
    metrics["evidence_confidence"] = float(_safe_float(evidence.get("evidence_confidence", 1.0), 1.0))
    metrics["evidence_ws_score"] = float(_safe_float(evidence.get("evidence_ws_score", 1.0), 1.0))
    metrics["evidence_rest_score"] = float(_safe_float(evidence.get("evidence_rest_score", 1.0), 1.0))
    metrics["evidence_fill_score"] = float(_safe_float(evidence.get("evidence_fill_score", 1.0), 1.0))
    metrics["evidence_ws_last_seen_ts"] = float(_safe_float(evidence.get("evidence_ws_last_seen_ts", 0.0), 0.0))
    metrics["evidence_rest_last_seen_ts"] = float(_safe_float(evidence.get("evidence_rest_last_seen_ts", 0.0), 0.0))
    metrics["evidence_fill_last_seen_ts"] = float(_safe_float(evidence.get("evidence_fill_last_seen_ts", 0.0), 0.0))
    metrics["evidence_ws_gap_rate"] = float(_safe_float(evidence.get("evidence_ws_gap_rate", 0.0), 0.0))
    metrics["evidence_rest_gap_rate"] = float(_safe_float(evidence.get("evidence_rest_gap_rate", 0.0), 0.0))
    metrics["evidence_fill_gap_rate"] = float(_safe_float(evidence.get("evidence_fill_gap_rate", 0.0), 0.0))
    metrics["evidence_ws_error_rate"] = float(_safe_float(evidence.get("evidence_ws_error_rate", 0.0), 0.0))
    metrics["evidence_rest_error_rate"] = float(_safe_float(evidence.get("evidence_rest_error_rate", 0.0), 0.0))
    metrics["evidence_fill_error_rate"] = float(_safe_float(evidence.get("evidence_fill_error_rate", 0.0), 0.0))
    metrics["evidence_contradiction_score"] = float(_safe_float(evidence.get("evidence_contradiction_score", 0.0), 0.0))
    metrics["evidence_contradiction_count"] = int(_safe_float(evidence.get("evidence_contradiction_count", 0), 0.0))
    metrics["evidence_contradiction_streak"] = int(_safe_float(evidence.get("evidence_contradiction_streak", 0), 0.0))
    metrics["evidence_contradiction_burn_rate"] = float(
        _safe_float(evidence.get("evidence_contradiction_burn_rate", 0.0), 0.0)
    )
    metrics["evidence_degraded_sources"] = int(_safe_float(evidence.get("evidence_degraded_sources", 0), 0.0))
    intent_stats = {
        "intent_unknown_count": 0,
        "intent_unknown_oldest_sec": 0.0,
        "intent_unknown_mean_resolve_sec": 0.0,
    }
    if callable(_intent_ledger_summary):
        try:
            got_stats = _intent_ledger_summary(bot)
            if isinstance(got_stats, dict):
                intent_stats.update(got_stats)
        except Exception:
            pass
    metrics["intent_unknown_count"] = int(_safe_float(intent_stats.get("intent_unknown_count", 0), 0.0))
    metrics["intent_unknown_oldest_sec"] = float(_safe_float(intent_stats.get("intent_unknown_oldest_sec", 0.0), 0.0))
    metrics["intent_unknown_mean_resolve_sec"] = float(
        _safe_float(intent_stats.get("intent_unknown_mean_resolve_sec", 0.0), 0.0)
    )
    gate = {
        "available": False,
        "degraded": False,
        "reason": "",
        "replay_mismatch_count": 0,
        "invalid_transition_count": 0,
        "journal_coverage_ratio": 1.0,
        "path": "",
    }
    if _runtime_reliability_coupling_enabled(bot) and callable(_runtime_reliability_gate):
        try:
            got_gate = _runtime_reliability_gate(getattr(bot, "cfg", None))
            if isinstance(got_gate, dict):
                gate.update(got_gate)
        except Exception:
            pass
    elif not _runtime_reliability_coupling_enabled(bot):
        gate["reason"] = "runtime_reliability_coupling_disabled"
    metrics["runtime_gate_available"] = bool(gate.get("available", False))
    metrics["runtime_gate_degraded"] = bool(gate.get("degraded", False))
    metrics["runtime_gate_reason"] = str(gate.get("reason", "") or "")
    metrics["runtime_gate_replay_mismatch_count"] = int(_safe_float(gate.get("replay_mismatch_count", 0), 0.0))
    metrics["runtime_gate_invalid_transition_count"] = int(_safe_float(gate.get("invalid_transition_count", 0), 0.0))
    metrics["runtime_gate_journal_coverage_ratio"] = float(_safe_float(gate.get("journal_coverage_ratio", 1.0), 1.0))
    metrics["runtime_gate_mismatch_severity"] = float(_safe_float(gate.get("mismatch_severity", 0.0), 0.0))
    metrics["runtime_gate_invalid_severity"] = float(_safe_float(gate.get("invalid_severity", 0.0), 0.0))
    metrics["runtime_gate_coverage_severity"] = float(_safe_float(gate.get("coverage_severity", 0.0), 0.0))
    metrics["runtime_gate_mismatch_category_score"] = float(
        _safe_float(gate.get("mismatch_category_score", 0.0), 0.0)
    )
    cats = gate.get("replay_mismatch_categories")
    if isinstance(cats, dict):
        metrics["runtime_gate_cat_ledger"] = int(_safe_float(cats.get("ledger", 0), 0.0))
        metrics["runtime_gate_cat_transition"] = int(_safe_float(cats.get("transition", 0), 0.0))
        metrics["runtime_gate_cat_belief"] = int(_safe_float(cats.get("belief", 0), 0.0))
        metrics["runtime_gate_cat_position"] = int(_safe_float(cats.get("position", 0), 0.0))
        metrics["runtime_gate_cat_orphan"] = int(_safe_float(cats.get("orphan", 0), 0.0))
        metrics["runtime_gate_cat_coverage_gap"] = int(_safe_float(cats.get("coverage_gap", 0), 0.0))
        metrics["runtime_gate_cat_replace_race"] = int(_safe_float(cats.get("replace_race", 0), 0.0))
        metrics["runtime_gate_cat_contradiction"] = int(_safe_float(cats.get("contradiction", 0), 0.0))
        metrics["runtime_gate_cat_unknown"] = int(_safe_float(cats.get("unknown", 0), 0.0))
    else:
        metrics["runtime_gate_cat_ledger"] = 0
        metrics["runtime_gate_cat_transition"] = 0
        metrics["runtime_gate_cat_belief"] = 0
        metrics["runtime_gate_cat_position"] = 0
        metrics["runtime_gate_cat_orphan"] = 0
        metrics["runtime_gate_cat_coverage_gap"] = 0
        metrics["runtime_gate_cat_replace_race"] = 0
        metrics["runtime_gate_cat_contradiction"] = 0
        metrics["runtime_gate_cat_unknown"] = 0
    metrics["runtime_gate_position_mismatch_count"] = int(_safe_float(gate.get("position_mismatch_count", 0), 0.0))
    metrics["runtime_gate_orphan_count"] = int(_safe_float(gate.get("orphan_count", 0), 0.0))
    metrics["runtime_gate_protection_coverage_gap_seconds"] = float(
        _safe_float(gate.get("protection_coverage_gap_seconds", 0.0), 0.0)
    )
    metrics["runtime_gate_replace_race_count"] = int(_safe_float(gate.get("replace_race_count", 0), 0.0))
    metrics["runtime_gate_evidence_contradiction_count"] = int(
        _safe_float(gate.get("evidence_contradiction_count", 0), 0.0)
    )
    metrics["runtime_gate_degrade_score"] = float(_safe_float(gate.get("degrade_score", 0.0), 0.0))
    gate_ids = gate.get("replay_mismatch_ids")
    metrics["runtime_gate_replay_mismatch_ids"] = list(gate_ids) if isinstance(gate_ids, list) else []
    reconcile_gate_recent = _recent_reconcile_first_gate_metrics(bot)
    metrics["reconcile_first_gate_count"] = int(
        _safe_float(reconcile_gate_recent.get("count", 0), 0.0)
    )
    metrics["reconcile_first_gate_max_severity"] = float(
        _safe_float(reconcile_gate_recent.get("max_severity", 0.0), 0.0)
    )
    metrics["reconcile_first_gate_max_streak"] = int(
        _safe_float(reconcile_gate_recent.get("max_streak", 0), 0.0)
    )
    metrics["reconcile_first_gate_last_reason"] = str(
        reconcile_gate_recent.get("last_reason", "") or ""
    )
    metrics["last_summary"]["belief_debt_sec"] = float(debt_sec)
    metrics["last_summary"]["belief_debt_symbols"] = int(debt_syms)
    metrics["last_summary"]["belief_confidence"] = float(conf)
    metrics["last_summary"]["evidence_confidence"] = float(metrics["evidence_confidence"])
    metrics["last_summary"]["evidence_contradiction_score"] = float(metrics["evidence_contradiction_score"])
    metrics["last_summary"]["evidence_contradiction_count"] = int(metrics["evidence_contradiction_count"])
    metrics["last_summary"]["evidence_contradiction_streak"] = int(metrics["evidence_contradiction_streak"])
    metrics["last_summary"]["evidence_contradiction_burn_rate"] = float(
        metrics["evidence_contradiction_burn_rate"]
    )
    metrics["last_summary"]["evidence_degraded_sources"] = int(metrics["evidence_degraded_sources"])
    metrics["last_summary"]["intent_unknown_count"] = int(metrics["intent_unknown_count"])
    km = getattr(getattr(bot, "state", None), "kill_metrics", None)
    if isinstance(km, dict):
        km["reconcile_mismatch_streak"] = int(metrics.get("mismatch_streak", 0) or 0)
        km["reconcile_last_mismatch_ts"] = _safe_float(metrics.get("last_mismatch_ts", 0.0), 0.0)
        km["reconcile_belief_debt_sec"] = float(debt_sec)
        km["reconcile_belief_debt_symbols"] = int(debt_syms)
        km["reconcile_belief_confidence"] = float(conf)
        km["reconcile_evidence_confidence"] = float(metrics["evidence_confidence"])
        km["reconcile_evidence_degraded_sources"] = int(metrics["evidence_degraded_sources"])
        km["reconcile_intent_unknown_count"] = int(metrics["intent_unknown_count"])
        km["reconcile_intent_unknown_oldest_sec"] = float(metrics["intent_unknown_oldest_sec"])
        km["reconcile_intent_unknown_mean_resolve_sec"] = float(metrics["intent_unknown_mean_resolve_sec"])
        km["reconcile_protection_coverage_gap_seconds"] = float(
            metrics.get("protection_coverage_gap_seconds", 0.0) or 0.0
        )
        km["reconcile_protection_coverage_gap_symbols"] = int(
            metrics.get("protection_coverage_gap_symbols", 0) or 0
        )
        km["reconcile_protection_coverage_ttl_breaches"] = int(
            metrics.get("protection_coverage_ttl_breaches", 0) or 0
        )

    belief_trace = {}
    guard_knobs = None
    prev_guard_mode = ""
    try:
        prev_guard = getattr(getattr(bot, "state", None), "guard_knobs", None)
        if isinstance(prev_guard, dict):
            prev_guard_mode = str(prev_guard.get("mode", "") or "").upper()
    except Exception:
        prev_guard_mode = ""
    ctl = _ensure_belief_controller(bot)
    if ctl is not None:
        try:
            guard_knobs = ctl.update(
                {
                    "belief_debt_sec": float(debt_sec),
                    "belief_debt_symbols": int(debt_syms),
                    "symbol_belief_debt_sec": dict(symbol_belief_debt_sec),
                    "mismatch_streak": int(metrics.get("mismatch_streak", 0) or 0),
                    "protection_coverage_gap_seconds": float(
                        metrics.get("protection_coverage_gap_seconds", 0.0) or 0.0
                    ),
                    "protection_coverage_gap_symbols": int(
                        metrics.get("protection_coverage_gap_symbols", 0) or 0
                    ),
                    "protection_coverage_ttl_breaches": int(
                        metrics.get("protection_coverage_ttl_breaches", 0) or 0
                    ),
                    "evidence_confidence": float(metrics.get("evidence_confidence", 1.0) or 1.0),
                    "evidence_degraded_sources": int(metrics.get("evidence_degraded_sources", 0) or 0),
                    "evidence_ws_score": float(metrics.get("evidence_ws_score", 1.0) or 1.0),
                    "evidence_rest_score": float(metrics.get("evidence_rest_score", 1.0) or 1.0),
                    "evidence_fill_score": float(metrics.get("evidence_fill_score", 1.0) or 1.0),
                    "evidence_ws_gap_rate": float(metrics.get("evidence_ws_gap_rate", 0.0) or 0.0),
                    "evidence_rest_gap_rate": float(metrics.get("evidence_rest_gap_rate", 0.0) or 0.0),
                    "evidence_fill_gap_rate": float(metrics.get("evidence_fill_gap_rate", 0.0) or 0.0),
                    "evidence_ws_error_rate": float(metrics.get("evidence_ws_error_rate", 0.0) or 0.0),
                    "evidence_rest_error_rate": float(metrics.get("evidence_rest_error_rate", 0.0) or 0.0),
                    "evidence_fill_error_rate": float(metrics.get("evidence_fill_error_rate", 0.0) or 0.0),
                    "evidence_contradiction_score": float(metrics.get("evidence_contradiction_score", 0.0) or 0.0),
                    "evidence_contradiction_count": int(metrics.get("evidence_contradiction_count", 0) or 0),
                    "evidence_contradiction_streak": int(metrics.get("evidence_contradiction_streak", 0) or 0),
                    "evidence_contradiction_burn_rate": float(
                        metrics.get("evidence_contradiction_burn_rate", 0.0) or 0.0
                    ),
                    "runtime_gate_degraded": bool(metrics.get("runtime_gate_degraded", False)),
                    "runtime_gate_reason": str(metrics.get("runtime_gate_reason", "") or ""),
                    "runtime_gate_replay_mismatch_count": int(metrics.get("runtime_gate_replay_mismatch_count", 0) or 0),
                    "runtime_gate_invalid_transition_count": int(
                        metrics.get("runtime_gate_invalid_transition_count", 0) or 0
                    ),
                    "runtime_gate_journal_coverage_ratio": float(
                        metrics.get("runtime_gate_journal_coverage_ratio", 1.0) or 1.0
                    ),
                    "runtime_gate_mismatch_severity": float(
                        metrics.get("runtime_gate_mismatch_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_invalid_severity": float(
                        metrics.get("runtime_gate_invalid_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_coverage_severity": float(
                        metrics.get("runtime_gate_coverage_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_degrade_score": float(
                        metrics.get("runtime_gate_degrade_score", 0.0) or 0.0
                    ),
                    "runtime_gate_cat_ledger": int(metrics.get("runtime_gate_cat_ledger", 0) or 0),
                    "runtime_gate_cat_transition": int(metrics.get("runtime_gate_cat_transition", 0) or 0),
                    "runtime_gate_cat_belief": int(metrics.get("runtime_gate_cat_belief", 0) or 0),
                    "runtime_gate_cat_position": int(metrics.get("runtime_gate_cat_position", 0) or 0),
                    "runtime_gate_cat_orphan": int(metrics.get("runtime_gate_cat_orphan", 0) or 0),
                    "runtime_gate_cat_coverage_gap": int(metrics.get("runtime_gate_cat_coverage_gap", 0) or 0),
                    "runtime_gate_cat_replace_race": int(metrics.get("runtime_gate_cat_replace_race", 0) or 0),
                    "runtime_gate_cat_contradiction": int(metrics.get("runtime_gate_cat_contradiction", 0) or 0),
                    "runtime_gate_cat_unknown": int(metrics.get("runtime_gate_cat_unknown", 0) or 0),
                    "runtime_gate_position_mismatch_count": int(
                        metrics.get("runtime_gate_position_mismatch_count", 0) or 0
                    ),
                    "runtime_gate_orphan_count": int(metrics.get("runtime_gate_orphan_count", 0) or 0),
                    "runtime_gate_protection_coverage_gap_seconds": float(
                        metrics.get("runtime_gate_protection_coverage_gap_seconds", 0.0) or 0.0
                    ),
                    "runtime_gate_replace_race_count": int(metrics.get("runtime_gate_replace_race_count", 0) or 0),
                    "runtime_gate_evidence_contradiction_count": int(
                        metrics.get("runtime_gate_evidence_contradiction_count", 0) or 0
                    ),
                    "runtime_gate_replay_mismatch_ids": list(
                        metrics.get("runtime_gate_replay_mismatch_ids", []) or []
                    )[:5],
                    "reconcile_first_gate_count": int(metrics.get("reconcile_first_gate_count", 0) or 0),
                    "reconcile_first_gate_max_severity": float(
                        metrics.get("reconcile_first_gate_max_severity", 0.0) or 0.0
                    ),
                    "reconcile_first_gate_max_streak": int(
                        metrics.get("reconcile_first_gate_max_streak", 0) or 0
                    ),
                    "reconcile_first_gate_last_reason": str(
                        metrics.get("reconcile_first_gate_last_reason", "") or ""
                    ),
                },
                getattr(bot, "cfg", None),
            )
            trace = ctl.explain()
            belief_trace = trace.to_dict() if hasattr(trace, "to_dict") else {"mode": str(getattr(trace, "mode", ""))}
            transition_label = str(belief_trace.get("transition", "") or "").strip()
            if transition_label and callable(_tel_emit):
                try:
                    new_mode = str(getattr(guard_knobs, "mode", "") or "").upper()
                    from_mode = prev_guard_mode
                    if "->" in transition_label:
                        from_mode = str(transition_label.split("->", 1)[0] or "").strip().upper() or from_mode
                        new_mode = str(transition_label.split("->", 1)[1] or "").strip().upper() or new_mode
                    await _tel_emit(
                        bot,
                        "execution.posture_transition",
                        data={
                            "previous_mode": str(from_mode),
                            "new_mode": str(new_mode),
                            "transition": str(transition_label),
                            "cause_tags": str(belief_trace.get("cause_tags", "") or ""),
                            "dominant_contributors": str(
                                belief_trace.get("dominant_contributors", "") or ""
                            ),
                            "unlock_requirements": str(
                                belief_trace.get("unlock_requirements", "") or ""
                            ),
                            "reason": str(belief_trace.get("reason", "") or ""),
                        },
                        level="warning",
                    )
                except Exception:
                    pass
            try:
                if hasattr(bot, "state"):
                    bot.state.guard_knobs = (guard_knobs.to_dict() if hasattr(guard_knobs, "to_dict") else dict(guard_knobs))
            except Exception:
                pass
            # Read halt duration from cfg first, then env fallback, then default.
            # This keeps run-profile env overrides effective even when cfg omits the field.
            _belief_halt_raw = _cfg(bot, "BELIEF_CONTROLLER_HALT_SEC", None)
            if _belief_halt_raw in (None, ""):
                _belief_halt_raw = os.getenv("BELIEF_CONTROLLER_HALT_SEC", "60")
            belief_halt_sec = _safe_float(_belief_halt_raw, 60.0)
            if (
                bool(getattr(guard_knobs, "kill_switch_trip", False))
                and callable(_kill_request_halt)
                and belief_halt_sec > 0.0
            ):
                try:
                    await _kill_request_halt(
                        bot,
                        float(belief_halt_sec),
                        f"belief_controller mode={getattr(guard_knobs, 'mode', 'RED')} debt={debt_sec:.0f}s",
                        "critical",
                    )
                except Exception:
                    pass
        except Exception:
            guard_knobs = None

    if callable(_tel_emit) and (mismatch_events > 0 or repair_actions > 0 or repair_skipped > 0):
        try:
            await _tel_emit(
                bot,
                "reconcile.summary",
                data={
                    "mismatch_events": int(mismatch_events),
                    "repair_actions": int(repair_actions),
                    "repair_skipped": int(repair_skipped),
                    "protection_coverage_gap_seconds": float(
                        metrics.get("protection_coverage_gap_seconds", 0.0) or 0.0
                    ),
                    "protection_coverage_gap_symbols": int(
                        metrics.get("protection_coverage_gap_symbols", 0) or 0
                    ),
                    "protection_coverage_ttl_breaches": int(
                        metrics.get("protection_coverage_ttl_breaches", 0) or 0
                    ),
                    "mismatch_streak": int(metrics.get("mismatch_streak", 0) or 0),
                    "belief_debt_sec": float(debt_sec),
                    "belief_debt_symbols": int(debt_syms),
                    "worst_symbols": list(worst_symbols),
                    "belief_confidence": float(conf),
                    "evidence_confidence": float(metrics.get("evidence_confidence", 1.0) or 1.0),
                    "evidence_ws_score": float(metrics.get("evidence_ws_score", 1.0) or 1.0),
                    "evidence_rest_score": float(metrics.get("evidence_rest_score", 1.0) or 1.0),
                    "evidence_fill_score": float(metrics.get("evidence_fill_score", 1.0) or 1.0),
                    "evidence_ws_last_seen_ts": float(metrics.get("evidence_ws_last_seen_ts", 0.0) or 0.0),
                    "evidence_rest_last_seen_ts": float(metrics.get("evidence_rest_last_seen_ts", 0.0) or 0.0),
                    "evidence_fill_last_seen_ts": float(metrics.get("evidence_fill_last_seen_ts", 0.0) or 0.0),
                    "evidence_ws_gap_rate": float(metrics.get("evidence_ws_gap_rate", 0.0) or 0.0),
                    "evidence_rest_gap_rate": float(metrics.get("evidence_rest_gap_rate", 0.0) or 0.0),
                    "evidence_fill_gap_rate": float(metrics.get("evidence_fill_gap_rate", 0.0) or 0.0),
                    "evidence_ws_error_rate": float(metrics.get("evidence_ws_error_rate", 0.0) or 0.0),
                    "evidence_rest_error_rate": float(metrics.get("evidence_rest_error_rate", 0.0) or 0.0),
                    "evidence_fill_error_rate": float(metrics.get("evidence_fill_error_rate", 0.0) or 0.0),
                    "evidence_contradiction_count": int(metrics.get("evidence_contradiction_count", 0) or 0),
                    "evidence_degraded_sources": int(metrics.get("evidence_degraded_sources", 0) or 0),
                    "evidence_contradiction_burn_rate": float(
                        metrics.get("evidence_contradiction_burn_rate", 0.0) or 0.0
                    ),
                    "runtime_gate_available": bool(metrics.get("runtime_gate_available", False)),
                    "runtime_gate_degraded": bool(metrics.get("runtime_gate_degraded", False)),
                    "runtime_gate_reason": str(metrics.get("runtime_gate_reason", "") or ""),
                    "runtime_gate_replay_mismatch_count": int(metrics.get("runtime_gate_replay_mismatch_count", 0) or 0),
                    "runtime_gate_invalid_transition_count": int(
                        metrics.get("runtime_gate_invalid_transition_count", 0) or 0
                    ),
                    "runtime_gate_journal_coverage_ratio": float(
                        metrics.get("runtime_gate_journal_coverage_ratio", 1.0) or 1.0
                    ),
                    "runtime_gate_mismatch_severity": float(
                        metrics.get("runtime_gate_mismatch_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_invalid_severity": float(
                        metrics.get("runtime_gate_invalid_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_coverage_severity": float(
                        metrics.get("runtime_gate_coverage_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_degrade_score": float(
                        metrics.get("runtime_gate_degrade_score", 0.0) or 0.0
                    ),
                    "runtime_gate_cat_ledger": int(metrics.get("runtime_gate_cat_ledger", 0) or 0),
                    "runtime_gate_cat_transition": int(metrics.get("runtime_gate_cat_transition", 0) or 0),
                    "runtime_gate_cat_belief": int(metrics.get("runtime_gate_cat_belief", 0) or 0),
                    "runtime_gate_cat_position": int(metrics.get("runtime_gate_cat_position", 0) or 0),
                    "runtime_gate_cat_orphan": int(metrics.get("runtime_gate_cat_orphan", 0) or 0),
                    "runtime_gate_cat_coverage_gap": int(metrics.get("runtime_gate_cat_coverage_gap", 0) or 0),
                    "runtime_gate_cat_replace_race": int(metrics.get("runtime_gate_cat_replace_race", 0) or 0),
                    "runtime_gate_cat_contradiction": int(metrics.get("runtime_gate_cat_contradiction", 0) or 0),
                    "runtime_gate_cat_unknown": int(metrics.get("runtime_gate_cat_unknown", 0) or 0),
                    "runtime_gate_position_mismatch_count": int(
                        metrics.get("runtime_gate_position_mismatch_count", 0) or 0
                    ),
                    "runtime_gate_orphan_count": int(metrics.get("runtime_gate_orphan_count", 0) or 0),
                    "runtime_gate_protection_coverage_gap_seconds": float(
                        metrics.get("runtime_gate_protection_coverage_gap_seconds", 0.0) or 0.0
                    ),
                    "runtime_gate_replace_race_count": int(metrics.get("runtime_gate_replace_race_count", 0) or 0),
                    "runtime_gate_evidence_contradiction_count": int(
                        metrics.get("runtime_gate_evidence_contradiction_count", 0) or 0
                    ),
                    "runtime_gate_replay_mismatch_ids": list(
                        metrics.get("runtime_gate_replay_mismatch_ids", []) or []
                    )[:5],
                    "reconcile_first_gate_count": int(metrics.get("reconcile_first_gate_count", 0) or 0),
                    "reconcile_first_gate_max_severity": float(
                        metrics.get("reconcile_first_gate_max_severity", 0.0) or 0.0
                    ),
                    "reconcile_first_gate_max_streak": int(
                        metrics.get("reconcile_first_gate_max_streak", 0) or 0
                    ),
                    "reconcile_first_gate_last_reason": str(
                        metrics.get("reconcile_first_gate_last_reason", "") or ""
                    ),
                    "intent_unknown_count": int(metrics.get("intent_unknown_count", 0) or 0),
                    "intent_unknown_oldest_sec": float(metrics.get("intent_unknown_oldest_sec", 0.0) or 0.0),
                    "intent_unknown_mean_resolve_sec": float(
                        metrics.get("intent_unknown_mean_resolve_sec", 0.0) or 0.0
                    ),
                    "guard_mode": (str(getattr(guard_knobs, "mode", "")) if guard_knobs is not None else ""),
                    "guard_recovery_stage": (
                        str(getattr(guard_knobs, "recovery_stage", "")) if guard_knobs is not None else ""
                    ),
                    "guard_unlock_conditions": (
                        str(getattr(guard_knobs, "unlock_conditions", "")) if guard_knobs is not None else ""
                    ),
                    "guard_next_unlock_sec": (
                        float(getattr(guard_knobs, "next_unlock_sec", 0.0) or 0.0) if guard_knobs is not None else 0.0
                    ),
                },
                level=("warning" if mismatch_events > 0 else "info"),
            )
        except Exception:
            pass

    if callable(_tel_emit):
        try:
            await _tel_emit(
                bot,
                "execution.belief_state",
                data={
                    "mismatch_streak": int(metrics.get("mismatch_streak", 0) or 0),
                    "belief_debt_sec": float(debt_sec),
                    "belief_debt_symbols": int(debt_syms),
                    "worst_symbols": list(worst_symbols),
                    "belief_confidence": float(conf),
                    "evidence_confidence": float(metrics.get("evidence_confidence", 1.0) or 1.0),
                    "evidence_ws_score": float(metrics.get("evidence_ws_score", 1.0) or 1.0),
                    "evidence_rest_score": float(metrics.get("evidence_rest_score", 1.0) or 1.0),
                    "evidence_fill_score": float(metrics.get("evidence_fill_score", 1.0) or 1.0),
                    "evidence_ws_last_seen_ts": float(metrics.get("evidence_ws_last_seen_ts", 0.0) or 0.0),
                    "evidence_rest_last_seen_ts": float(metrics.get("evidence_rest_last_seen_ts", 0.0) or 0.0),
                    "evidence_fill_last_seen_ts": float(metrics.get("evidence_fill_last_seen_ts", 0.0) or 0.0),
                    "evidence_ws_gap_rate": float(metrics.get("evidence_ws_gap_rate", 0.0) or 0.0),
                    "evidence_rest_gap_rate": float(metrics.get("evidence_rest_gap_rate", 0.0) or 0.0),
                    "evidence_fill_gap_rate": float(metrics.get("evidence_fill_gap_rate", 0.0) or 0.0),
                    "evidence_ws_error_rate": float(metrics.get("evidence_ws_error_rate", 0.0) or 0.0),
                    "evidence_rest_error_rate": float(metrics.get("evidence_rest_error_rate", 0.0) or 0.0),
                    "evidence_fill_error_rate": float(metrics.get("evidence_fill_error_rate", 0.0) or 0.0),
                    "evidence_contradiction_count": int(metrics.get("evidence_contradiction_count", 0) or 0),
                    "evidence_degraded_sources": int(metrics.get("evidence_degraded_sources", 0) or 0),
                    "evidence_contradiction_burn_rate": float(
                        metrics.get("evidence_contradiction_burn_rate", 0.0) or 0.0
                    ),
                    "runtime_gate_degraded": bool(metrics.get("runtime_gate_degraded", False)),
                    "runtime_gate_reason": str(metrics.get("runtime_gate_reason", "") or ""),
                    "runtime_gate_replay_mismatch_count": int(metrics.get("runtime_gate_replay_mismatch_count", 0) or 0),
                    "runtime_gate_invalid_transition_count": int(
                        metrics.get("runtime_gate_invalid_transition_count", 0) or 0
                    ),
                    "runtime_gate_journal_coverage_ratio": float(
                        metrics.get("runtime_gate_journal_coverage_ratio", 1.0) or 1.0
                    ),
                    "runtime_gate_mismatch_severity": float(
                        metrics.get("runtime_gate_mismatch_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_invalid_severity": float(
                        metrics.get("runtime_gate_invalid_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_coverage_severity": float(
                        metrics.get("runtime_gate_coverage_severity", 0.0) or 0.0
                    ),
                    "runtime_gate_degrade_score": float(
                        metrics.get("runtime_gate_degrade_score", 0.0) or 0.0
                    ),
                    "runtime_gate_cat_ledger": int(metrics.get("runtime_gate_cat_ledger", 0) or 0),
                    "runtime_gate_cat_transition": int(metrics.get("runtime_gate_cat_transition", 0) or 0),
                    "runtime_gate_cat_belief": int(metrics.get("runtime_gate_cat_belief", 0) or 0),
                    "runtime_gate_cat_position": int(metrics.get("runtime_gate_cat_position", 0) or 0),
                    "runtime_gate_cat_orphan": int(metrics.get("runtime_gate_cat_orphan", 0) or 0),
                    "runtime_gate_cat_coverage_gap": int(metrics.get("runtime_gate_cat_coverage_gap", 0) or 0),
                    "runtime_gate_cat_replace_race": int(metrics.get("runtime_gate_cat_replace_race", 0) or 0),
                    "runtime_gate_cat_contradiction": int(metrics.get("runtime_gate_cat_contradiction", 0) or 0),
                    "runtime_gate_cat_unknown": int(metrics.get("runtime_gate_cat_unknown", 0) or 0),
                    "runtime_gate_position_mismatch_count": int(
                        metrics.get("runtime_gate_position_mismatch_count", 0) or 0
                    ),
                    "runtime_gate_orphan_count": int(metrics.get("runtime_gate_orphan_count", 0) or 0),
                    "runtime_gate_protection_coverage_gap_seconds": float(
                        metrics.get("runtime_gate_protection_coverage_gap_seconds", 0.0) or 0.0
                    ),
                    "runtime_gate_replace_race_count": int(metrics.get("runtime_gate_replace_race_count", 0) or 0),
                    "runtime_gate_evidence_contradiction_count": int(
                        metrics.get("runtime_gate_evidence_contradiction_count", 0) or 0
                    ),
                    "runtime_gate_replay_mismatch_ids": list(
                        metrics.get("runtime_gate_replay_mismatch_ids", []) or []
                    )[:5],
                    "reconcile_first_gate_count": int(metrics.get("reconcile_first_gate_count", 0) or 0),
                    "reconcile_first_gate_max_severity": float(
                        metrics.get("reconcile_first_gate_max_severity", 0.0) or 0.0
                    ),
                    "reconcile_first_gate_max_streak": int(
                        metrics.get("reconcile_first_gate_max_streak", 0) or 0
                    ),
                    "reconcile_first_gate_last_reason": str(
                        metrics.get("reconcile_first_gate_last_reason", "") or ""
                    ),
                    "intent_unknown_count": int(metrics.get("intent_unknown_count", 0) or 0),
                    "intent_unknown_oldest_sec": float(metrics.get("intent_unknown_oldest_sec", 0.0) or 0.0),
                    "intent_unknown_mean_resolve_sec": float(
                        metrics.get("intent_unknown_mean_resolve_sec", 0.0) or 0.0
                    ),
                    "repair_actions": int(repair_actions),
                    "repair_skipped": int(repair_skipped),
                    "protection_coverage_gap_seconds": float(
                        metrics.get("protection_coverage_gap_seconds", 0.0) or 0.0
                    ),
                    "protection_coverage_gap_symbols": int(
                        metrics.get("protection_coverage_gap_symbols", 0) or 0
                    ),
                    "protection_coverage_ttl_breaches": int(
                        metrics.get("protection_coverage_ttl_breaches", 0) or 0
                    ),
                    "guard_mode": (str(getattr(guard_knobs, "mode", "")) if guard_knobs is not None else ""),
                    "allow_entries": (bool(getattr(guard_knobs, "allow_entries", True)) if guard_knobs is not None else True),
                    "guard_recovery_stage": (
                        str(getattr(guard_knobs, "recovery_stage", "")) if guard_knobs is not None else ""
                    ),
                    "guard_unlock_conditions": (
                        str(getattr(guard_knobs, "unlock_conditions", "")) if guard_knobs is not None else ""
                    ),
                    "guard_next_unlock_sec": (
                        float(getattr(guard_knobs, "next_unlock_sec", 0.0) or 0.0) if guard_knobs is not None else 0.0
                    ),
                    "trace": belief_trace,
                },
                level=("warning" if conf < 0.75 else "info"),
            )
        except Exception:
            pass


# Backward compatibility for older code that still imports guardian_loop from reconcile.py
async def guardian_loop(bot):
    poll_sec = float(_cfg(bot, "GUARDIAN_POLL_SEC", 15.0))
    shutdown_ev = _ensure_shutdown_event(bot)

    log_core.critical("RECONCILE LEGACY LOOP ONLINE — (guardian.py should own the loop)")

    while not shutdown_ev.is_set():
        try:
            await reconcile_tick(bot)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"Reconcile legacy loop error: {e}")
        await asyncio.sleep(poll_sec)
