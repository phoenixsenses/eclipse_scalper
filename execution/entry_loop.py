# execution/entry_loop.py — SCALPER ETERNAL — ENTRY LOOP — 2026 v1.6 (PENDING-LOCK + COOLDOWN + OPEN-ORDERS ADOPT)
# Patch vs v1.5:
# - ✅ FIX: Per-symbol "pending entry" lock so you cannot machine-gun entries (even if reconcile is lagging)
# - ✅ FIX: Cooldown after ANY submitted entry attempt (success OR fail) to avoid rapid re-fire loops
# - ✅ HARDEN: Optional open-orders / open-position probe (best-effort) to detect real exposure even if brain-state is stale
# - ✅ SAFETY: backoff on margin-insufficient (-2019) retained
# - ✅ Keeps: ENV-first overrides, sizing resolver, tuple adapter, throttled logs, guardian-safe (never raises)

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple

from utils.logging import log_entry, log_core
from execution.order_router import create_order, cancel_order
from execution.anomaly_guard import should_pause as anomaly_should_pause
from execution.entry_decision import (
    EntryDecisionRecord,
    compute_entry_decision,
    commit_entry_intent,
)

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit, emit_throttled, count_recent  # type: ignore
except Exception:
    emit = None
    emit_throttled = None
    count_recent = None

try:
    from execution.data_quality import staleness_check, update_quality_state  # type: ignore
    from execution.error_codes import (
        ERR_STALE_DATA,
        ERR_DATA_QUALITY,
        ERR_ROUTER_BLOCK,
        ERR_RISK,
        ERR_RELIABILITY_GATE,
        ERR_PARTIAL_FILL,
        ERR_UNKNOWN,
        map_reason,
    )  # type: ignore
except Exception:
    staleness_check = None
    update_quality_state = None
    ERR_STALE_DATA = "ERR_STALE_DATA"
    ERR_DATA_QUALITY = "ERR_DATA_QUALITY"
    ERR_ROUTER_BLOCK = "ERR_ROUTER_BLOCK"
    ERR_RISK = "ERR_RISK"
    ERR_RELIABILITY_GATE = "ERR_RELIABILITY_GATE"
    ERR_PARTIAL_FILL = "ERR_PARTIAL_FILL"
    ERR_UNKNOWN = "ERR_UNKNOWN"
    map_reason = None

# Optional entry_watch (never fatal)
try:
    from execution.entry_watch import register_entry_watch  # type: ignore
except Exception:
    register_entry_watch = None

# Optional kill-switch gate (never fatal)
try:
    from risk.kill_switch import trade_allowed  # type: ignore
except Exception:
    trade_allowed = None

try:
    from execution.adaptive_guard import refresh_state as refresh_adaptive_guard, get_override as get_adaptive_override  # type: ignore
except Exception:
    refresh_adaptive_guard = None
    get_adaptive_override = None

try:
    from execution.adaptive_guard import get_notional_scale as get_adaptive_notional_scale  # type: ignore
except Exception:
    get_adaptive_notional_scale = None


# ----------------------------
# Helpers
# ----------------------------

def _now() -> float:
    return time.time()


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        return getattr(getattr(bot, "cfg", None), name, default)
    except Exception:
        return default


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "on"):
        return True
    return False


def _env_get(name: str) -> str:
    try:
        return str(os.getenv(name, "")).strip()
    except Exception:
        return ""


def _cfg_env_float(bot, name: str, default: float) -> float:
    """
    ENV wins, then cfg, then default.
    """
    v = _env_get(name)
    if v != "":
        try:
            return float(v)
        except Exception:
            pass
    try:
        return float(_cfg(bot, name, default) or default)
    except Exception:
        return float(default)


def _cfg_env_bool(bot, name: str, default: Any = False) -> bool:
    """
    ENV wins, then cfg.
    Accepts strings like "1/true/on".
    """
    v = _env_get(name)
    if v != "":
        return _truthy(v)
    return _truthy(_cfg(bot, name, default))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _parse_group_kv(raw: str) -> dict:
    """
    Parse "MEME=1,MAJOR=2" into dict.
    """
    out: dict = {}
    try:
        s = str(raw or "").strip()
        if not s:
            return out
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            kk = str(k or "").strip().upper()
            if not kk:
                continue
            try:
                out[kk] = float(v)
            except Exception:
                continue
    except Exception:
        return out
    return out


def _parse_groups(raw: str) -> dict:
    """
    Parse "MEME:BTCUSDT,SHIBUSDT;MAJOR:BTCUSDT,ETHUSDT" into dict.
    """
    out: dict = {}
    try:
        s = str(raw or "").strip()
        if not s:
            return out
        groups = [g.strip() for g in s.split(";") if g.strip()]
        for g in groups:
            if ":" not in g:
                continue
            name, syms = g.split(":", 1)
            gn = str(name or "").strip().upper()
            if not gn:
                continue
            members = []
            for p in syms.replace(" ", "").split(","):
                if not p:
                    continue
                members.append(_symkey(p))
            if members:
                out[gn] = members
    except Exception:
        return out
    return out


def _get_corr_groups(bot) -> dict:
    try:
        cfg = getattr(bot, "cfg", None)
        g = getattr(cfg, "CORRELATION_GROUPS", None)
        if isinstance(g, dict) and g:
            return {str(k).upper(): [_symkey(x) for x in v] for k, v in g.items()}
    except Exception:
        pass
    try:
        g2 = getattr(bot, "CORRELATION_GROUPS", None)
        if isinstance(g2, dict) and g2:
            return {str(k).upper(): [_symkey(x) for x in v] for k, v in g2.items()}
    except Exception:
        pass
    try:
        raw = os.getenv("CORR_GROUPS", "").strip()
        g3 = _parse_groups(raw)
        if g3:
            return g3
    except Exception:
        pass
    return {}


def _check_corr_group(bot, k: str, planned_notional: float) -> tuple[Optional[str], dict]:
    """
    Returns reason string if blocked, else None.
    """
    group_name = ""
    group_syms = []
    group_count = 0
    group_notional = 0.0
    try:
        groups = _get_corr_groups(bot)
        for gname, members in (groups or {}).items():
            if k in members:
                group_name = str(gname).upper()
                group_syms = [_symkey(x) for x in members]
                break
        if not group_name:
            return None, {}
        pos_map = getattr(getattr(bot, "state", None), "positions", None)
        if isinstance(pos_map, dict):
            for pk, pos in pos_map.items():
                if _symkey(pk) not in group_syms:
                    continue
                try:
                    sz = float(getattr(pos, "size", 0.0) or 0.0)
                    if abs(sz) <= 0:
                        continue
                    group_count += 1
                    group_notional += abs(sz) * float(getattr(pos, "entry_price", 0.0) or 0.0)
                except Exception:
                    continue
        group_max_pos = int(_cfg_env_float(bot, "CORR_GROUP_MAX_POSITIONS", 0) or 0)
        group_max_notional = float(_cfg_env_float(bot, "CORR_GROUP_MAX_NOTIONAL_USDT", 0.0) or 0.0)
        per_group_pos = _parse_group_kv(os.getenv("CORR_GROUP_LIMITS", ""))
        per_group_not = _parse_group_kv(os.getenv("CORR_GROUP_NOTIONAL", ""))
        if group_name in per_group_pos:
            group_max_pos = int(per_group_pos.get(group_name) or group_max_pos)
        if group_name in per_group_not:
            group_max_notional = float(per_group_not.get(group_name) or group_max_notional)
        meta = {
            "group": group_name,
            "group_count": group_count,
            "group_max_positions": group_max_pos,
            "group_notional": group_notional,
            "group_max_notional": group_max_notional,
        }
        if group_max_pos > 0 and group_count >= group_max_pos:
            return f"group {group_name} max positions {group_max_pos} reached", meta
        if group_max_notional > 0 and planned_notional > 0:
            if (group_notional + planned_notional) > group_max_notional:
                meta.update(
                    {
                        "planned_notional": planned_notional,
                        "group_projected_notional": group_notional + planned_notional,
                    }
                )
                return (
                    f"group {group_name} notional {(group_notional + planned_notional):.2f} > {group_max_notional:.2f}",
                    meta,
                )
        if planned_notional > 0:
            meta.update(
                {
                    "planned_notional": planned_notional,
                    "group_projected_notional": group_notional + planned_notional,
                }
            )
        return None, meta
    except Exception:
        return None, {}
    return None, {}


def _corr_group_scale(bot, meta: dict) -> tuple[float, str]:
    """
    Returns (scale, reason). Scale=1.0 means no scaling.
    """
    try:
        if not meta or not meta.get("group"):
            return 1.0, ""
        enabled = _cfg_env_bool(bot, "CORR_GROUP_SCALE_ENABLED", True)
        if not enabled:
            return 1.0, ""
        group = str(meta.get("group") or "").upper()
        count = int(meta.get("group_count") or 0)
        if count <= 0:
            return 1.0, ""
        scale_base = float(_cfg_env_float(bot, "CORR_GROUP_SCALE", 0.7) or 0.7)
        scale_min = float(_cfg_env_float(bot, "CORR_GROUP_SCALE_MIN", 0.25) or 0.25)
        per_group = _parse_group_kv(os.getenv("CORR_GROUP_SCALE_BY_GROUP", ""))
        if group in per_group:
            try:
                scale_base = float(per_group.get(group) or scale_base)
            except Exception:
                pass
        if scale_base <= 0 or scale_base >= 1.0:
            return 1.0, ""
        scale = max(scale_min, float(scale_base) ** float(count))
        return float(scale), f"corr_group_scale {group} count={count}"
    except Exception:
        return 1.0, ""


def _corr_group_exposure_scale(bot, meta: dict, planned_notional: float) -> tuple[float, str]:
    """
    Scale notional as group exposure grows.
    Scale <= 1.0. If disabled or missing data, returns 1.0.
    """
    try:
        if not meta or not meta.get("group"):
            return 1.0, ""
        enabled = _cfg_env_bool(bot, "CORR_GROUP_EXPOSURE_SCALE_ENABLED", False)
        if not enabled:
            return 1.0, ""
        group = str(meta.get("group") or "").upper()
        group_notional = float(meta.get("group_notional") or 0.0)
        if planned_notional <= 0 and group_notional <= 0:
            return 1.0, ""

        base_scale = float(_cfg_env_float(bot, "CORR_GROUP_EXPOSURE_SCALE", 0.7) or 0.7)
        min_scale = float(_cfg_env_float(bot, "CORR_GROUP_EXPOSURE_SCALE_MIN", 0.25) or 0.25)
        ref_notional = float(_cfg_env_float(bot, "CORR_GROUP_EXPOSURE_REF_NOTIONAL", 0.0) or 0.0)
        per_group_scale = _parse_group_kv(os.getenv("CORR_GROUP_EXPOSURE_SCALE_BY_GROUP", ""))
        per_group_min = _parse_group_kv(os.getenv("CORR_GROUP_EXPOSURE_SCALE_MIN_BY_GROUP", ""))
        per_group_ref = _parse_group_kv(os.getenv("CORR_GROUP_EXPOSURE_REF_NOTIONAL_BY_GROUP", ""))
        if group in per_group_scale:
            base_scale = float(per_group_scale.get(group) or base_scale)
        if group in per_group_min:
            min_scale = float(per_group_min.get(group) or min_scale)
        if group in per_group_ref:
            ref_notional = float(per_group_ref.get(group) or ref_notional)

        total = group_notional + max(0.0, planned_notional)
        if ref_notional <= 0:
            ref_notional = float(_cfg_env_float(bot, "CORR_GROUP_MAX_NOTIONAL_USDT", 0.0) or 0.0)
        if ref_notional <= 0:
            return 1.0, ""

        if base_scale <= 0 or base_scale >= 1.0:
            return 1.0, ""

        factor = total / max(1e-9, ref_notional)
        scale = max(min_scale, base_scale ** max(1.0, factor))
        return float(scale), f"corr_group_exposure {group} notional={total:.2f}"
    except Exception:
        return 1.0, ""


_SIGNAL_FEEDBACK_STATE: Dict[str, float] = {"offset": 0.0, "last_ts": 0.0}
_SIGNAL_FEEDBACK_LOG_THROTTLE_SEC = 120.0


def _signal_feedback_path() -> Path:
    path_str = _env_get("TELEMETRY_PATH")
    if not path_str:
        path_str = "logs/telemetry.jsonl"
    return Path(path_str)


def _collect_signal_feedback_events() -> list[dict[str, Any]]:
    path = _signal_feedback_path()
    if not path.exists():
        return []

    state = _SIGNAL_FEEDBACK_STATE
    try:
        size = path.stat().st_size
    except Exception:
        return []

    if size < float(state.get("offset") or 0.0):
        state["offset"] = 0.0

    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(float(state.get("offset") or 0.0))
            for line in fh:
                state["offset"] = fh.tell()
                raw = line.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                if str(ev.get("event")) != "exit.signal_issue":
                    continue
                ts = float(ev.get("ts") or 0.0)
                if ts <= 0 or ts <= float(state.get("last_ts") or 0.0):
                    continue
                state["last_ts"] = ts
                data = ev.get("data") or {}
                out.append({"ts": ts, "data": data})
    except Exception:
        return out
    return out


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


def _ensure_data_ready_event(bot) -> asyncio.Event:
    ev = getattr(bot, "data_ready", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot.data_ready = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


async def _safe_speak(bot, text: str, priority: str = "info") -> None:
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _pick_symbols(bot) -> list[str]:
    """
    Best-effort symbol universe:
    1) bot.active_symbols (set/list)
    2) cfg.ACTIVE_SYMBOLS (list)
    3) fallback ["BTCUSDT"]
    """
    try:
        s = getattr(bot, "active_symbols", None)
        if isinstance(s, set) and s:
            return sorted(list(s))
        if isinstance(s, (list, tuple)) and s:
            return [str(x) for x in s if str(x).strip()]
    except Exception:
        pass

    try:
        s2 = getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(s2, (list, tuple)) and s2:
            return [str(x) for x in s2 if str(x).strip()]
    except Exception:
        pass

    return ["BTCUSDT"]


def _in_position_brain(bot, k: str) -> bool:
    """
    Cheap check: state.positions contains k with nonzero size.
    Note: if reconcile never adopts exchange positions, this may remain false.
    """
    try:
        st = getattr(bot, "state", None)
        posd = getattr(st, "positions", None)
        if not isinstance(posd, dict):
            return False
        p = posd.get(k)
        if p is None:
            return False
        sz = getattr(p, "size", 0.0)
        return abs(float(sz)) > 0.0
    except Exception:
        return False


def _resolve_raw_symbol(bot, k: str, fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return fallback


def _order_filled(order: dict) -> float:
    try:
        if order is None:
            return 0.0
        if "filled" in order:
            return float(order.get("filled") or 0.0)
        info = order.get("info") or {}
        return float(info.get("executedQty") or 0.0)
    except Exception:
        return 0.0


def _recent_router_blocks(bot, k: str, window_sec: float, *, now_ts: Optional[float] = None) -> int:
    try:
        st = getattr(bot, "state", None)
        tel = getattr(st, "telemetry", None) if st is not None else None
        recent = getattr(tel, "recent", None)
        if recent is None:
            return 0
        now = _now() if now_ts is None else float(now_ts)
        kk = _symkey(k)
        count = 0
        for ev in list(recent):
            try:
                if str(ev.get("event")) != "order.blocked":
                    continue
                ts = float(ev.get("ts") or 0.0)
                if ts <= 0 or (now - ts) > window_sec:
                    continue
                sym = _symkey(ev.get("symbol") or ev.get("data", {}).get("k") or "")
                if sym and sym == kk:
                    count += 1
            except Exception:
                continue
        return count
    except Exception:
        return 0


def _corr_snapshot(bot, guard_knobs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Correlation context for entry decision/blocked telemetry.
    Prefer guard-level values when present; fallback to reconcile metrics.
    """
    out: Dict[str, Any] = {}
    try:
        src = dict(guard_knobs or {})
    except Exception:
        src = {}
    if not src:
        try:
            st = getattr(bot, "state", None)
            rm = getattr(st, "reconcile_metrics", None) if st is not None else None
            if isinstance(rm, dict):
                src = rm
        except Exception:
            src = {}
    try:
        out["corr_pressure"] = float(src.get("corr_pressure", 0.0) or 0.0)
    except Exception:
        out["corr_pressure"] = 0.0
    out["corr_regime"] = str(src.get("corr_regime", "NORMAL") or "NORMAL")
    try:
        out["corr_confidence"] = float(src.get("corr_confidence", 1.0) or 1.0)
    except Exception:
        out["corr_confidence"] = 1.0
    out["corr_reason_tags"] = str(src.get("corr_reason_tags", "stable") or "stable")
    out["corr_worst_group"] = str(src.get("corr_worst_group", "") or "")
    return out




# ----------------------------
# ENV + CFG sizing resolver
# ----------------------------

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


def _get_guard_knobs(bot) -> dict[str, Any]:
    st = getattr(bot, "state", None)
    if st is None:
        return {}
    raw = getattr(st, "guard_knobs", None)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "to_dict") and callable(getattr(raw, "to_dict", None)):
        try:
            data = raw.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _guard_block_reason_code(guard_knobs: dict[str, Any]) -> tuple[str, str]:
    reason = str(guard_knobs.get("reason") or "")
    runtime_gate_degraded = bool(guard_knobs.get("runtime_gate_degraded", False))
    reconcile_first_gate_degraded = bool(guard_knobs.get("reconcile_first_gate_degraded", False))
    if (not runtime_gate_degraded) and ("runtime_gate_degraded" in reason.lower()):
        runtime_gate_degraded = True
    if runtime_gate_degraded:
        return "runtime_gate_reconcile_first", ERR_RELIABILITY_GATE
    if reconcile_first_gate_degraded:
        return "reconcile_first_pressure", ERR_RELIABILITY_GATE
    return "belief_controller_block", ERR_ROUTER_BLOCK


def _resolve_symbol_guard(guard_knobs: dict[str, Any], symbol: str) -> dict[str, Any]:
    base = dict(guard_knobs or {})
    per_symbol = base.get("per_symbol")
    if not isinstance(per_symbol, dict):
        return base
    sym = _symkey(symbol)
    if not sym:
        return base
    override = per_symbol.get(sym)
    if not isinstance(override, dict):
        return base
    merged = dict(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def _record_reconcile_first_gate(
    bot,
    symbol: str,
    severity: float,
    reason: str = "",
    corr: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Track reconcile-first pressure in state.kill_metrics so reconcile/belief_controller
    can consume recent spike pressure as runtime policy input.
    """
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return
        km = getattr(st, "kill_metrics", None)
        if not isinstance(km, dict):
            km = {}
            st.kill_metrics = km
        now_ts = _now()
        sev = max(0.0, min(1.0, float(severity or 0.0)))
        threshold = float(_cfg_env_float(bot, "ENTRY_RECONCILE_FIRST_SEVERITY_THRESHOLD", 0.85) or 0.85)
        events = km.get("reconcile_first_gate_events")
        if not isinstance(events, list):
            events = []
            km["reconcile_first_gate_events"] = events
        events.append(
            {
                "ts": float(now_ts),
                "severity": float(sev),
                "symbol": _symkey(symbol),
                "reason": str(reason or ""),
                "corr_pressure": float(_safe_float((corr or {}).get("corr_pressure", 0.0), 0.0)),
                "corr_regime": str((corr or {}).get("corr_regime", "") or ""),
                "corr_reason_tags": str((corr or {}).get("corr_reason_tags", "") or ""),
                "corr_worst_group": str((corr or {}).get("corr_worst_group", "") or ""),
            }
        )
        max_events = int(_cfg_env_float(bot, "ENTRY_RECONCILE_FIRST_EVENTS_MAX", 160.0) or 160)
        if max_events < 20:
            max_events = 20
        if len(events) > max_events:
            del events[:-max_events]
        current_streak = int(km.get("reconcile_first_gate_current_streak", 0) or 0)
        if sev >= threshold:
            current_streak += 1
        else:
            current_streak = 0
        km["reconcile_first_gate_current_streak"] = int(current_streak)
        km["reconcile_first_gate_max_streak"] = max(
            int(km.get("reconcile_first_gate_max_streak", 0) or 0),
            int(current_streak),
        )
        km["reconcile_first_gate_count"] = int(km.get("reconcile_first_gate_count", 0) or 0) + 1
        km["reconcile_first_gate_last_ts"] = float(now_ts)
        km["reconcile_first_gate_last_severity"] = float(sev)
        km["reconcile_first_gate_last_reason"] = str(reason or "")
        km["reconcile_first_gate_last_corr_pressure"] = float(
            _safe_float((corr or {}).get("corr_pressure", 0.0), 0.0)
        )
        km["reconcile_first_gate_last_corr_regime"] = str((corr or {}).get("corr_regime", "") or "")
    except Exception:
        return


def _estimate_open_exposure_usdt(bot) -> float:
    total = 0.0
    try:
        pos_map = getattr(getattr(bot, "state", None), "positions", None)
        if not isinstance(pos_map, dict):
            return 0.0
        for _k, pos in pos_map.items():
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


def _entry_budget_snapshot(bot, guard_knobs: dict[str, Any]) -> tuple[bool, float, float, str]:
    enabled = _cfg_env_bool(bot, "ENTRY_BUDGET_ENABLED", True)
    if not enabled:
        return False, 0.0, 0.0, ""
    total = float(_cfg_env_float(bot, "ENTRY_GLOBAL_BUDGET_USDT", 0.0) or 0.0)
    if total <= 0:
        total = float(_cfg(bot, "ENTRY_GLOBAL_BUDGET_USDT", 0.0) or 0.0)
    if total <= 0:
        total = float(guard_knobs.get("global_entry_budget_usdt", 0.0) or 0.0)
    if total <= 0:
        return False, 0.0, 0.0, ""
    include_open = _cfg_env_bool(bot, "ENTRY_BUDGET_INCLUDE_OPEN_EXPOSURE", True)
    open_exposure = _estimate_open_exposure_usdt(bot) if include_open else 0.0
    remaining = max(0.0, total - open_exposure)
    return True, float(total), float(remaining), f"entry_budget total={total:.2f} open={open_exposure:.2f}"


def _entry_budget_symbol_cap(bot, confidence: float, min_conf: float, remaining: float) -> float:
    rem = max(0.0, float(remaining or 0.0))
    if rem <= 0:
        return 0.0
    conf = max(0.0, min(1.0, float(confidence or 0.0)))
    mn = max(0.0, min(1.0, float(min_conf or 0.0)))
    span = max(1e-6, 1.0 - mn)
    score = max(0.0, min(1.0, (conf - mn) / span))
    min_share = max(0.01, min(1.0, float(_cfg_env_float(bot, "ENTRY_BUDGET_MIN_SHARE", 0.10) or 0.10)))
    max_share = max(min_share, min(1.0, float(_cfg_env_float(bot, "ENTRY_BUDGET_MAX_SHARE", 0.60) or 0.60)))
    share = min_share + ((max_share - min_share) * score)
    return float(max(0.0, rem * share))


# ----------------------------
# Throttled logging (no-trade spam killer)
# ----------------------------

_LAST_LOG_TS: Dict[str, float] = {}


def _throttled_log(key: str, every_sec: float, fn: Callable[[str], None], msg: str) -> None:
    """
    Log msg at most once per `every_sec` per key.
    `fn` is e.g. log_entry.info / log_entry.warning.
    """
    try:
        every_sec = float(every_sec)
        if every_sec <= 0:
            fn(msg)
            return
        now = _now()
        last = float(_LAST_LOG_TS.get(key, 0.0) or 0.0)
        if (now - last) >= every_sec:
            _LAST_LOG_TS[key] = now
            fn(msg)
    except Exception:
        pass


# ----------------------------
# Strategy signal adapter
# ----------------------------

def _load_signal_fn() -> Optional[Callable]:
    """
    Locate a signal function without hard dependency.
    Preferred: strategies.eclipse_scalper.scalper_signal
    """
    try:
        from strategies.eclipse_scalper import scalper_signal as fn  # type: ignore
        if callable(fn):
            return fn
    except Exception:
        pass
    return None


def _tuple_to_sig_dict(
    sym: str,
    out: Any,
    diag: bool = False,
    no_trade_log_every: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """
    Accepts tuple/list like (long_bool, short_bool, confidence)
    Returns a dict compatible with the rest of this entry loop.
    """
    try:
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            return None

        long_sig = bool(out[0])
        short_sig = bool(out[1])
        conf = float(out[2]) if out[2] is not None else 0.0

        if long_sig and not short_sig:
            return {"action": "buy", "confidence": conf, "type": "market", "symbol": sym}
        if short_sig and not long_sig:
            return {"action": "sell", "confidence": conf, "type": "market", "symbol": sym}

        if diag:
            _throttled_log(
                key=f"tuple_no_trade:{sym}",
                every_sec=no_trade_log_every,
                fn=log_entry.info,
                msg=f"ENTRY_LOOP tuple no-trade {sym}: long={long_sig} short={short_sig} conf={conf}",
            )
        return None
    except Exception:
        return None


async def _maybe_call_signal(fn: Callable, bot, symbol: str, diag: bool = False) -> Optional[Dict[str, Any]]:
    """
    Supports:
    - dict-returning strategies
    - tuple-returning scalper_signal() -> (long, short, confidence)
    Tries multiple signatures, including canonical: fn(sym, data=bot.data, cfg=bot.cfg)
    """
    no_trade_log_every = float(_cfg_env_float(bot, "ENTRY_NO_TRADE_LOG_EVERY_SEC", 5.0) or 5.0)

    # 1) Preferred: fn(symbol, data=bot.data, cfg=bot.cfg)
    try:
        out = fn(symbol, data=getattr(bot, "data", None), cfg=getattr(bot, "cfg", None))
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, dict):
            return out
        t = _tuple_to_sig_dict(symbol, out, diag=diag, no_trade_log_every=no_trade_log_every)
        if t:
            return t
    except TypeError:
        pass
    except Exception as e:
        log_entry.warning(f"ENTRY_LOOP signal failed {symbol}: {e}")
        return None

    # 2) fn(bot, symbol)
    try:
        out = fn(bot, symbol)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, dict):
            return out
        t = _tuple_to_sig_dict(symbol, out, diag=diag, no_trade_log_every=no_trade_log_every)
        if t:
            return t
    except TypeError:
        pass
    except Exception as e:
        log_entry.warning(f"ENTRY_LOOP signal failed {symbol}: {e}")
        return None

    # 3) fn(symbol, bot=bot)
    try:
        out = fn(symbol, bot=bot)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, dict):
            return out
        t = _tuple_to_sig_dict(symbol, out, diag=diag, no_trade_log_every=no_trade_log_every)
        if t:
            return t
    except Exception as e:
        log_entry.warning(f"ENTRY_LOOP signal failed {symbol}: {e}")
        return None

    return None


def _parse_action(sig: Dict[str, Any]) -> Optional[str]:
    a = str(sig.get("action") or sig.get("side") or "").strip().lower()
    if a in ("long", "buy"):
        return "buy"
    if a in ("short", "sell"):
        return "sell"
    return None


def _parse_order_type(sig: Dict[str, Any]) -> str:
    t = str(sig.get("type") or sig.get("order_type") or "market").strip().lower()
    return "limit" if t in ("limit",) else "market"


def _parse_amount(sig: Dict[str, Any]) -> Optional[float]:
    for key in ("amount", "qty", "size"):
        if key in sig:
            try:
                v = float(sig.get(key))
                return v if v > 0 else None
            except Exception:
                return None
    return None


def _parse_price(sig: Dict[str, Any]) -> Optional[float]:
    for key in ("price", "limit_price"):
        if key in sig:
            try:
                v = float(sig.get(key))
                return v if v > 0 else None
            except Exception:
                return None
    return None


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


def _sizing_fallback_amount(bot, symbol: str) -> Optional[float]:
    """
    Minimal fallback sizing:
    - If (ENV/CFG) FIXED_QTY is set => use that
    - Else if (ENV/CFG) FIXED_NOTIONAL_USDT and we can get price => qty = notional/price
    - Else None (skip)
    """
    fixed_qty, notional = _resolve_symbol_sizing(bot, symbol)
    base_notional = notional
    base_qty = fixed_qty
    if notional > 0 and callable(get_adaptive_notional_scale) and _cfg_env_bool(
        bot, "ADAPTIVE_GUARD_NOTIONAL_SCALE", True
    ):
        try:
            scale, reason = get_adaptive_notional_scale(symbol)
            if scale and scale < 1.0:
                notional = float(notional) * float(scale)
                if callable(emit_throttled):
                    try:
                        asyncio.create_task(
                            emit_throttled(
                                bot,
                                "entry.notional_scaled",
                                key=f"{_symkey(symbol)}:{reason or 'guard_history'}",
                                cooldown_sec=120.0,
                                data={
                                    "symbol": _symkey(symbol),
                                    "base_notional": base_notional,
                                    "scaled_notional": notional,
                                    "scale": scale,
                                    "reason": reason or "guard_history",
                                },
                                symbol=_symkey(symbol),
                                level="warning",
                            )
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        if fixed_qty > 0 and callable(get_adaptive_notional_scale) and _cfg_env_bool(
            bot, "ADAPTIVE_GUARD_QTY_SCALE", True
        ):
            try:
                scale, reason = get_adaptive_notional_scale(symbol)
                if scale and scale < 1.0:
                    fixed_qty = float(fixed_qty) * float(scale)
                    if callable(emit_throttled):
                        try:
                            asyncio.create_task(
                                emit_throttled(
                                    bot,
                                    "entry.qty_scaled",
                                    key=f"{_symkey(symbol)}:{reason or 'guard_history'}",
                                    cooldown_sec=120.0,
                                    data={
                                        "symbol": _symkey(symbol),
                                        "base_qty": base_qty,
                                        "scaled_qty": fixed_qty,
                                        "scale": scale,
                                        "reason": reason or "guard_history",
                                    },
                                    symbol=_symkey(symbol),
                                    level="warning",
                                )
                            )
                        except Exception:
                            pass
            except Exception:
                pass
        if fixed_qty > 0:
            return float(fixed_qty)
    except Exception:
        pass

    if notional <= 0:
        return None

    px = _get_price(bot, symbol)
    if px <= 0:
        return None

    qty = float(notional) / float(px)

    try:
        min_qty = float(_cfg(bot, "MIN_ENTRY_QTY", 0.0) or 0.0)
        if min_qty > 0 and qty < min_qty:
            qty = min_qty
    except Exception:
        pass

    return qty if qty > 0 else None


# ----------------------------
# Anti-spam: pending entry tracking (locks + cooldown)
# ----------------------------

_ENTRY_LOCKS: Dict[str, asyncio.Lock] = {}
_PENDING_UNTIL: Dict[str, float] = {}          # k -> ts until which entries are blocked
_PENDING_ORDER_ID: Dict[str, str] = {}         # k -> last submitted order id (best-effort)
_QUALITY_LOW_START: Dict[str, float] = {}
_QUALITY_LOW_LAST: Dict[str, float] = {}
_PARTIAL_FILL_STREAK: Dict[str, dict[str, float]] = {}


def _record_partial_fill_hit(bot, symbol: str) -> float:
    window = max(1.0, float(_cfg_env_float(bot, "ENTRY_PARTIAL_ESCALATE_WINDOW_SEC", 600.0) or 600.0))
    threshold = max(1, int(_cfg_env_float(bot, "ENTRY_PARTIAL_ESCALATE_COUNT", 3) or 3))
    penalty = float(_cfg_env_float(bot, "ENTRY_PARTIAL_ESCALATE_BACKOFF_SEC", 120.0) or 120.0)
    entry = _PARTIAL_FILL_STREAK.setdefault(symbol, {"count": 0, "last_ts": 0.0})
    now = _now()
    if (now - float(entry.get("last_ts") or 0.0)) > window:
        entry["count"] = 0
    entry["count"] = int(entry.get("count", 0) or 0) + 1
    entry["last_ts"] = now
    if threshold > 0 and int(entry.get("count", 0) or 0) >= threshold:
        entry["count"] = 0
        return penalty
    return 0.0


def _reset_partial_fill_hits(symbol: str) -> None:
    _PARTIAL_FILL_STREAK.pop(symbol, None)


async def _resolve_partial_fill_state(
    bot,
    *,
    symbol: str,
    sym_raw: str,
    action: str,
    otype: str,
    order_id: Optional[str],
    requested: float,
    filled: float,
    min_ratio: float,
    hedge_side_hint: Optional[str],
) -> Dict[str, Any]:
    """
    Resolve a below-threshold partial fill.
    Returns metadata with a concrete outcome:
      - partial_forced_flatten
      - partial_stuck
    """
    ratio = (filled / requested) if (requested > 0 and filled > 0) else 0.0
    out: Dict[str, Any] = {
        "requested": float(requested),
        "filled": float(filled),
        "ratio": float(ratio),
        "min_ratio": float(min_ratio),
        "cancel_attempted": False,
        "cancel_ok": False,
        "flatten_attempted": False,
        "flatten_ok": False,
        "outcome": "partial_stuck",
    }

    cancel_enabled = _cfg_env_bool(bot, "ENTRY_PARTIAL_CANCEL", True)
    if cancel_enabled and order_id and str(otype).lower().strip() == "limit":
        out["cancel_attempted"] = True
        cancel_retries = max(1, int(_cfg_env_float(bot, "ENTRY_PARTIAL_CANCEL_RETRIES", 2) or 2))
        cancel_delay = max(0.0, float(_cfg_env_float(bot, "ENTRY_PARTIAL_CANCEL_DELAY_SEC", 0.25) or 0.25))
        for _ in range(cancel_retries):
            try:
                ok = await cancel_order(bot, str(order_id), sym_raw)
                if ok is not False:
                    out["cancel_ok"] = True
                    break
            except Exception:
                pass
            if cancel_delay > 0:
                await asyncio.sleep(cancel_delay)

    force_flatten = _cfg_env_bool(bot, "ENTRY_PARTIAL_FORCE_FLATTEN", True)
    if force_flatten and filled > 0:
        out["flatten_attempted"] = True
        flatten_side = "sell" if str(action).lower().strip() == "buy" else "buy"
        flatten_retries = max(1, int(_cfg_env_float(bot, "ENTRY_PARTIAL_FLATTEN_RETRIES", 2) or 2))
        for _ in range(flatten_retries):
            try:
                res = await create_order(
                    bot,
                    symbol=sym_raw,
                    type="MARKET",
                    side=flatten_side,
                    amount=float(filled),
                    price=None,
                    params={},
                    intent_reduce_only=True,
                    intent_close_position=False,
                    hedge_side_hint=hedge_side_hint,
                    retries=int(_cfg_env_float(bot, "ENTRY_ROUTER_RETRIES", 4) or 4),
                    intent_component="entry_loop_partial_flatten",
                    intent_kind="ENTRY_PARTIAL_FLATTEN",
                )
                if isinstance(res, dict):
                    if res.get("id") or _order_filled(res) > 0:
                        out["flatten_ok"] = True
                        break
            except Exception:
                pass
        if out["flatten_ok"]:
            out["outcome"] = "partial_forced_flatten"

    if callable(emit):
        try:
            await emit(
                bot,
                "entry.partial_fill_state",
                data={"symbol": symbol, **out, "code": ERR_PARTIAL_FILL},
                symbol=symbol,
                level=("warning" if out.get("flatten_ok") else "critical"),
            )
        except Exception:
            pass
    return out


async def _emit_latency(
    bot,
    *,
    symbol: str,
    stage: str,
    duration_ms: float,
    result: str = "success",
) -> None:
    if duration_ms <= 0:
        return
    if not callable(emit):
        return
    data = {
        "symbol": symbol,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        "result": result,
    }
    try:
        await emit(bot, "telemetry.latency", data=data, symbol=symbol, level="info")
    except Exception:
        pass


async def _emit_entry_blocked(
    bot,
    symbol: str,
    reason: str,
    *,
    level: str = "warning",
    data: Optional[Dict[str, Any]] = None,
    throttle_sec: float = 0.0,
    throttle_key: Optional[str] = None,
) -> None:
    if not callable(emit):
        return
    data_payload: Dict[str, Any] = dict(data or {})
    data_payload.setdefault("symbol", symbol)
    data_payload.setdefault("reason", reason)
    for ck, cv in _corr_snapshot(bot).items():
        data_payload.setdefault(ck, cv)
    if "code" not in data_payload:
        try:
            code = map_reason(reason) if callable(map_reason) else ERR_UNKNOWN
        except Exception:
            code = ERR_UNKNOWN
        data_payload["code"] = code
    try:
        if throttle_sec > 0 and callable(emit_throttled):
            key = throttle_key or f"{symbol}:{reason}"
            await emit_throttled(
                bot,
                "entry.blocked",
                key=key,
                cooldown_sec=throttle_sec,
                data=data_payload,
                symbol=symbol,
                level=level,
            )
        else:
            await emit(bot, "entry.blocked", data=data_payload, symbol=symbol, level=level)
    except Exception:
        pass


async def _emit_entry_decision(bot, rec: EntryDecisionRecord, *, level: str = "info", throttle_sec: float = 0.0) -> None:
    data = rec.to_dict()
    try:
        guard_meta = data.get("guard_knobs")
        guard_knobs = guard_meta if isinstance(guard_meta, dict) else None
    except Exception:
        guard_knobs = None
    for ck, cv in _corr_snapshot(bot, guard_knobs=guard_knobs).items():
        data.setdefault(ck, cv)
    symbol = str(data.get("symbol") or "")
    if throttle_sec > 0 and callable(emit_throttled):
        try:
            await emit_throttled(
                bot,
                "entry.decision",
                key=f"{symbol}:{rec.stage}:{rec.action}:{rec.reason_primary or 'ok'}",
                cooldown_sec=float(throttle_sec),
                data=data,
                symbol=symbol or None,
                level=level,
            )
        except Exception:
            pass
        return
    if callable(emit):
        try:
            await emit(
                bot,
                "entry.decision",
                data=data,
                symbol=symbol or None,
                level=level,
            )
        except Exception:
            pass


async def _report_signal_feedback(bot) -> None:
    events = _collect_signal_feedback_events()
    if not events:
        return
    for ev in events:
        data = ev.get("data") or {}
        ratio = float(data.get("low_confidence_ratio") or 0.0)
        count = int(data.get("low_confidence_exits") or 0)
        guard_hits = int(data.get("guard_hits") or 0)
        severity = str(data.get("severity") or "warning")
        min_conf = float(data.get("min_confidence_threshold") or 0.0)
        reason = str(data.get("reason") or "telemetry_signal_feedback")
        msg = (
            f"Telemetry signal feedback: ratio={ratio:.1%}, count={count}, "
            f"guard_hits={guard_hits}, min_conf={min_conf:.2f}, reason={reason}"
        )
        _throttled_log("signal_feedback", _SIGNAL_FEEDBACK_LOG_THROTTLE_SEC, log_entry.warning, msg)
        await _emit_entry_blocked(
            bot,
            "exit_signal_feedback",
            "signal_feedback",
            data={
                "ratio": ratio,
                "low_confidence_count": count,
                "guard_hits": guard_hits,
                "min_confidence_threshold": min_conf,
            },
            level=severity,
            throttle_sec=60.0,
        )


def _get_entry_lock(k: str) -> asyncio.Lock:
    lk = _ENTRY_LOCKS.get(k)
    if isinstance(lk, asyncio.Lock):
        return lk
    lk = asyncio.Lock()
    _ENTRY_LOCKS[k] = lk
    return lk


def _set_pending(k: str, *, sec: float, order_id: Optional[str] = None) -> None:
    try:
        _PENDING_UNTIL[k] = _now() + max(0.0, float(sec))
        if order_id:
            _PENDING_ORDER_ID[k] = str(order_id)
    except Exception:
        pass


def _pending_active(k: str) -> bool:
    try:
        until = float(_PENDING_UNTIL.get(k, 0.0) or 0.0)
        return _now() < until
    except Exception:
        return False


async def _has_open_entry_order(bot, k: str, sym_raw: str, *, max_open: int = 1) -> bool:
    """
    Best-effort: detect open orders from exchange to avoid stacking while reconcile lags.
    This is intentionally conservative and never fatal.
    """
    try:
        enabled = _cfg_env_bool(bot, "ENTRY_PROBE_OPEN_ORDERS", True)
        if not enabled:
            return False

        ex = getattr(bot, "ex", None)
        if ex is None:
            return False

        fn = getattr(ex, "fetch_open_orders", None)
        if not callable(fn):
            return False

        # Some exchanges want raw symbol; we try sym_raw first, then k.
        try:
            orders = await fn(sym_raw)
        except Exception:
            try:
                orders = await fn(k)
            except Exception:
                return False

        if not isinstance(orders, (list, tuple)) or not orders:
            return False

        # Ignore reduceOnly/closePosition orders (exits); entry loop only cares about entry-ish orders.
        open_count = 0
        for o in orders:
            if not isinstance(o, dict):
                continue
            info = o.get("info") or {}
            params = o.get("params") or {}

            ro = info.get("reduceOnly", params.get("reduceOnly"))
            cp = info.get("closePosition", params.get("closePosition"))
            if _truthy(ro) or _truthy(cp):
                continue

            status = str(o.get("status") or "").lower()
            if status in ("open", "new", "partially_filled", ""):
                open_count += 1
                if open_count >= max(1, int(max_open or 1)):
                    return True

        return False
    except Exception:
        return False


async def _has_open_position_exchange(bot, k: str, sym_raw: str) -> bool:
    """
    Best-effort: detect if exchange reports an open position.
    This helps when brain-state isn't adopted yet.
    """
    try:
        enabled = _cfg_env_bool(bot, "ENTRY_PROBE_EXCHANGE_POSITIONS", False)
        if not enabled:
            return False

        ex = getattr(bot, "ex", None)
        if ex is None:
            return False

        fn = getattr(ex, "fetch_positions", None)
        if not callable(fn):
            return False

        try:
            poss = await fn([sym_raw])
        except Exception:
            try:
                poss = await fn([k])
            except Exception:
                return False

        if not isinstance(poss, (list, tuple)) or not poss:
            return False

        for p in poss:
            if not isinstance(p, dict):
                continue
            # ccxt position formats vary; try common fields:
            sz = p.get("contracts")
            if sz is None:
                sz = p.get("contractSize")
            if sz is None:
                sz = p.get("size")
            if sz is None:
                sz = p.get("positionAmt")
            try:
                if sz is not None and abs(float(sz)) > 0.0:
                    return True
            except Exception:
                continue

        return False
    except Exception:
        return False


# ----------------------------
# Entry loop
# ----------------------------

async def entry_loop(bot) -> None:
    """
    Main entry loop.
    Places NEW entries only. Exits/stop management handled elsewhere (exit/posmgr).
    """
    shutdown_ev = _ensure_shutdown_event(bot)
    data_ready_ev = _ensure_data_ready_event(bot)

    # ENV overrides for safety knobs
    poll_sec = _cfg_env_float(bot, "ENTRY_POLL_SEC", 1.0)
    wait_data_sec = _cfg_env_float(bot, "ENTRY_WAIT_FOR_DATA_READY_SEC", 8.0)
    per_symbol_gap_sec = _cfg_env_float(bot, "ENTRY_PER_SYMBOL_GAP_SEC", 2.5)
    base_local_cooldown_sec = _cfg_env_float(bot, "ENTRY_LOCAL_COOLDOWN_SEC", 8.0)
    base_min_conf = _cfg_env_float(bot, "ENTRY_MIN_CONFIDENCE", 0.0)

    # NEW: pending-block window after submit to stop stacking while reconcile adopts
    pending_block_sec = _cfg_env_float(bot, "ENTRY_PENDING_BLOCK_SEC", 30.0)

    respect_kill = bool(_cfg_env_bool(bot, "ENTRY_RESPECT_KILL_SWITCH", True))

    # Hedge hint mode can be set via ENV too
    hedge_hint_mode = bool(
        _cfg_env_bool(bot, "HEDGE_MODE", False)
        or _cfg_env_bool(bot, "HEDGE_SAFE", False)
        or _truthy(_cfg(bot, "HEDGE_MODE", False) or _cfg(bot, "HEDGE_SAFE", False))
    )

    # diag can be controlled via ENV, else cfg
    diag = _cfg_env_bool(bot, "SCALPER_SIGNAL_DIAG", _cfg(bot, "SCALPER_SIGNAL_DIAG", "0"))

    sizing_warn_every = _cfg_env_float(bot, "ENTRY_SIZING_WARN_EVERY_SEC", 30.0)
    last_sizing_warn_ts = 0.0

    # Backoff on margin insufficient to prevent spam
    margin_backoff_sec = _cfg_env_float(bot, "ENTRY_MARGIN_INSUFFICIENT_BACKOFF_SEC", 900.0)  # 15m default
    backoff_until_by_sym: Dict[str, float] = {}

    # Local cooldown memory
    last_attempt_by_sym: Dict[str, float] = {}
    last_symbol_tick = 0.0

    sig_fn = _load_signal_fn()
    if not callable(sig_fn):
        log_core.warning("ENTRY_LOOP: strategy signal missing (strategies.eclipse_scalper.scalper_signal). Loop will idle.")

    log_core.info("ENTRY_LOOP ONLINE — scanning for new entries")

    # initial data-ready wait (best-effort)
    if wait_data_sec > 0 and not data_ready_ev.is_set():
        try:
            await asyncio.wait_for(data_ready_ev.wait(), timeout=wait_data_sec)
        except Exception:
            pass

    while not shutdown_ev.is_set():
        try:
            now = _now()

            if wait_data_sec > 0 and not data_ready_ev.is_set():
                await asyncio.sleep(max(0.25, poll_sec))
                continue

            # optional kill-switch gate
            if respect_kill and callable(trade_allowed):
                try:
                    ok = await trade_allowed(bot)
                    if not ok:
                        await asyncio.sleep(max(0.25, poll_sec))
                        continue
                except Exception:
                    await asyncio.sleep(max(0.25, poll_sec))
                    continue

            paused, remaining = anomaly_should_pause()
            if paused:
                await _emit_entry_blocked(bot, "ANOMALY", "anomaly_pause", throttle_sec=60.0)
                wait = remaining if remaining > 0 else poll_sec
                await asyncio.sleep(max(poll_sec, wait))
                continue

            if _cfg_env_bool(bot, "ENTRY_SIGNAL_FEEDBACK_ENABLED", True):
                await _report_signal_feedback(bot)

            if callable(refresh_adaptive_guard):
                try:
                    refresh_adaptive_guard()
                except Exception:
                    pass

            # avoid super tight spin if poll_sec tiny
            if poll_sec > 0 and (now - last_symbol_tick) < poll_sec:
                await asyncio.sleep(max(0.05, poll_sec - (now - last_symbol_tick)))
            last_symbol_tick = _now()

            syms = _pick_symbols(bot)
            if not syms:
                await asyncio.sleep(max(0.25, poll_sec))
                continue
            global_guard_knobs = _get_guard_knobs(bot)
            budget_enabled, budget_total, budget_remaining, budget_reason = _entry_budget_snapshot(bot, global_guard_knobs)
            budget_spent = 0.0

            for sym in syms:
                if shutdown_ev.is_set():
                    break

                k = _symkey(sym)
                if not k:
                    continue

                guard_knobs = _resolve_symbol_guard(global_guard_knobs, k)
                guard_mode = str(guard_knobs.get("mode") or "").upper()
                allow_entries = bool(guard_knobs.get("allow_entries", True))
                guard_min_conf = float(guard_knobs.get("min_entry_conf", 0.0) or 0.0)
                guard_cooldown = float(guard_knobs.get("entry_cooldown_seconds", 0.0) or 0.0)
                guard_max_notional = float(guard_knobs.get("max_notional_usdt", 0.0) or 0.0)
                guard_max_open = int(guard_knobs.get("max_open_orders_per_symbol", 1) or 1)
                runtime_gate_degraded = bool(guard_knobs.get("runtime_gate_degraded", False))
                reconcile_first_gate_degraded = bool(guard_knobs.get("reconcile_first_gate_degraded", False))
                runtime_gate_reason = str(guard_knobs.get("runtime_gate_reason") or "")
                runtime_gate_degrade_score = float(guard_knobs.get("runtime_gate_degrade_score", 0.0) or 0.0)
                symbol_debt_score = float(guard_knobs.get("debt_score", 0.0) or 0.0)
                symbol_severity = max(0.0, min(1.0, symbol_debt_score))
                reconcile_first_severity = max(float(runtime_gate_degrade_score), float(symbol_severity))
                local_cooldown_sec = max(float(base_local_cooldown_sec), max(0.0, guard_cooldown))
                current_min_conf = max(float(base_min_conf), max(0.0, guard_min_conf))

                if not allow_entries:
                    block_reason, block_code = _guard_block_reason_code(guard_knobs)
                    corr_ctx = _corr_snapshot(bot, guard_knobs)
                    _record_reconcile_first_gate(
                        bot,
                        k,
                        reconcile_first_severity,
                        runtime_gate_reason or str(guard_knobs.get("reason") or ""),
                        corr=corr_ctx,
                    )
                    await _emit_entry_blocked(
                        bot,
                        k,
                        block_reason,
                        data={
                            "mode": guard_mode or "UNKNOWN",
                            "reason": str(guard_knobs.get("reason") or ""),
                            "debt_score": float(guard_knobs.get("debt_score", 0.0) or 0.0),
                            "debt_growth_per_min": float(guard_knobs.get("debt_growth_per_min", 0.0) or 0.0),
                            "runtime_gate_degraded": runtime_gate_degraded,
                            "runtime_gate_reason": runtime_gate_reason,
                            "runtime_gate_degrade_score": runtime_gate_degrade_score,
                            "symbol_debt_score": symbol_debt_score,
                            "symbol_severity": symbol_severity,
                            "reconcile_first_severity": reconcile_first_severity,
                            **corr_ctx,
                            "code": block_code,
                        },
                        throttle_sec=15.0,
                    )
                    if (runtime_gate_degraded or reconcile_first_gate_degraded) and callable(emit_throttled):
                        try:
                            await emit_throttled(
                                bot,
                                "entry.reconcile_first_gate",
                                key=f"{k}:reconcile_first_gate",
                                cooldown_sec=15.0,
                                data={
                                    "symbol": k,
                                    "mode": guard_mode or "UNKNOWN",
                                    "reason": runtime_gate_reason or str(guard_knobs.get("reason") or ""),
                                    "runtime_gate_degrade_score": runtime_gate_degrade_score,
                                    "reconcile_first_gate_degraded": reconcile_first_gate_degraded,
                                    "reconcile_first_gate_count": int(
                                        guard_knobs.get("reconcile_first_gate_count", 0) or 0
                                    ),
                                    "reconcile_first_gate_max_severity": float(
                                        guard_knobs.get("reconcile_first_gate_max_severity", 0.0) or 0.0
                                    ),
                                    "reconcile_first_gate_max_streak": int(
                                        guard_knobs.get("reconcile_first_gate_max_streak", 0) or 0
                                    ),
                                    "symbol_debt_score": symbol_debt_score,
                                    "symbol_severity": symbol_severity,
                                    "reconcile_first_severity": reconcile_first_severity,
                                    **corr_ctx,
                                    "code": ERR_RELIABILITY_GATE,
                                },
                                symbol=k,
                                level="warning",
                            )
                        except Exception:
                            pass
                    continue
                if budget_enabled:
                    rem = max(0.0, float(budget_remaining) - float(budget_spent))
                    if rem <= 0:
                        await _emit_entry_blocked(
                            bot,
                            k,
                            "entry_budget_depleted",
                            data={
                                "reason": budget_reason,
                                "budget_total_usdt": float(budget_total),
                                "budget_remaining_usdt": float(rem),
                                "code": ERR_RISK,
                            },
                            throttle_sec=15.0,
                        )
                        continue

                guard_reason = ""
                if callable(get_adaptive_override):
                    try:
                        current_min_conf, guard_reason = get_adaptive_override(k, current_min_conf)
                    except Exception:
                        current_min_conf = current_min_conf
                        guard_reason = ""
                if guard_reason and diag:
                    _throttled_log(
                        key=f"adaptive_guard:{k}:{guard_reason}",
                        every_sec=60.0,
                        fn=log_entry.info,
                        msg=f"ENTRY_LOOP adaptive guard {k} raised min_conf to {current_min_conf:.2f} ({guard_reason})",
                    )

                # margin-insufficient backoff gate
                bo = float(backoff_until_by_sym.get(k, 0.0) or 0.0)
                if bo > 0 and _now() < bo:
                    continue

                # hard pending gate (prevents stacking during adopt/reconcile lag)
                if _pending_active(k):
                    continue

                # skip if already in position (brain-state)
                if _in_position_brain(bot, k):
                    continue

                # local cooldown
                la = float(last_attempt_by_sym.get(k, 0.0) or 0.0)
                if local_cooldown_sec > 0 and (_now() - la) < local_cooldown_sec:
                    await _emit_entry_blocked(bot, k, "cooldown_local", throttle_sec=5.0)
                    continue

                # must have signal function to do anything
                if not callable(sig_fn):
                    await _emit_entry_blocked(bot, k, "signal_missing", throttle_sec=60.0)
                    continue

                # resolve raw symbol once (needed for exchange probes)
                sym_raw = _resolve_raw_symbol(bot, k, k)

                # best-effort exchange probes (optional)
                try:
                    if await _has_open_entry_order(bot, k, sym_raw, max_open=max(1, guard_max_open)):
                        _set_pending(k, sec=max(5.0, pending_block_sec * 0.5))
                        await _emit_entry_blocked(bot, k, "router_open_order", throttle_sec=15.0)
                        continue
                except Exception:
                    pass

                try:
                    if await _has_open_position_exchange(bot, k, sym_raw):
                        _set_pending(k, sec=max(5.0, pending_block_sec * 0.5))
                        await _emit_entry_blocked(bot, k, "risk_exchange_position", throttle_sec=15.0)
                        continue
                except Exception:
                    pass

                # Acquire per-symbol entry lock to prevent concurrent submit storms
                lk = _get_entry_lock(k)
                if lk.locked():
                    continue

                async with lk:
                    # re-check gates after lock (race-safe)
                    if shutdown_ev.is_set():
                        break
                    if _pending_active(k):
                        await _emit_entry_blocked(bot, k, "cooldown_pending", throttle_sec=10.0)
                        continue
                    if _in_position_brain(bot, k):
                        await _emit_entry_blocked(bot, k, "risk_in_position", throttle_sec=10.0)
                        continue
                    bo = float(backoff_until_by_sym.get(k, 0.0) or 0.0)
                    if bo > 0 and _now() < bo:
                        continue

                    # mark attempt NOW to enforce cooldown even if we error later
                    last_attempt_by_sym[k] = _now()

                    # Unified staleness + data-quality guard
                    if callable(staleness_check):
                        max_sec = float(_cfg_env_float(bot, "ENTRY_DATA_MAX_STALE_SEC", 180.0) or 180.0)
                        ok, age_sec, _src = staleness_check(bot, k, tf="1m", max_sec=max_sec)
                        if not ok:
                            if callable(emit):
                                try:
                                    await emit(
                                        bot,
                                        "entry.blocked",
                                        data={"symbol": k, "reason": "stale_data", "age_sec": age_sec, "max_sec": max_sec, "code": ERR_STALE_DATA},
                                        symbol=k,
                                        level="warning",
                                    )
                                except Exception:
                                    pass
                            await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                            continue

                    if callable(update_quality_state):
                        q_min = float(_cfg_env_float(bot, "ENTRY_DATA_QUALITY_MIN", 60.0) or 60.0)
                        q_tf = str(_cfg(bot, "ENTRY_DATA_QUALITY_TF", os.getenv("ENTRY_DATA_QUALITY_TF", "1m")) or "1m")
                        q_window = int(_cfg_env_float(bot, "ENTRY_DATA_QUALITY_WINDOW", 120) or 120)
                        q_emit = float(_cfg_env_float(bot, "ENTRY_DATA_QUALITY_EMIT_SEC", 60.0) or 60.0)
                        roll_min = float(_cfg_env_float(bot, "ENTRY_DATA_QUALITY_ROLL_MIN", q_min) or q_min)
                        kill_sec = float(_cfg_env_float(bot, "ENTRY_DATA_QUALITY_KILL_SEC", 120.0) or 120.0)
                        score = update_quality_state(bot, k, tf=q_tf, max_sec=float(_cfg_env_float(bot, "ENTRY_DATA_MAX_STALE_SEC", 180.0) or 180.0), window=q_window, emit_sec=q_emit)
                        dq = getattr(getattr(bot, "state", None), "data_quality", {}) or {}
                        info = dq.get(k) or {}
                        roll_score = float(info.get("roll", score))
                        now_ts = _now()
                        if roll_score < roll_min and kill_sec > 0:
                            last = float(_QUALITY_LOW_START.get(k, 0.0) or 0.0)
                            if last == 0:
                                _QUALITY_LOW_START[k] = now_ts
                            elif (now_ts - last) >= kill_sec:
                                _QUALITY_LOW_LAST[k] = now_ts
                        if callable(emit):
                            try:
                                await emit(
                                    bot,
                                    "entry.blocked",
                                    data={
                                        "symbol": k,
                                        "reason": "data_quality_roll",
                                        "roll": roll_score,
                                        "min": roll_min,
                                        "history_n": int(info.get("n", 0) or 0),
                                        "code": ERR_DATA_QUALITY,
                                    },
                                    symbol=k,
                                    level="warning",
                                )
                            except Exception:
                                pass
                        if callable(emit_throttled):
                            try:
                                await emit_throttled(
                                    bot,
                                    "data.quality.roll_alert",
                                    key=f"{k}:quality_roll",
                                    cooldown_sec=max(15.0, kill_sec),
                                    data={
                                        "symbol": k,
                                        "roll": roll_score,
                                        "min": roll_min,
                                        "history_n": int(info.get("n", 0) or 0),
                                    },
                                    symbol=k,
                                    level="warning",
                                )
                            except Exception:
                                pass
                                await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                continue
                        else:
                            _QUALITY_LOW_START.pop(k, None)
                            _QUALITY_LOW_LAST.pop(k, None)
                        if q_min > 0 and score < q_min:
                            if callable(emit):
                                try:
                                    await emit(
                                        bot,
                                        "entry.blocked",
                                        data={"symbol": k, "reason": "data_quality", "score": score, "min": q_min, "code": ERR_DATA_QUALITY},
                                        symbol=k,
                                        level="warning",
                                    )
                                except Exception:
                                    pass
                            await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                            continue
                    sig_start = _now()
                    sig = await _maybe_call_signal(sig_fn, bot, k, diag=diag)
                    sig_duration_ms = (_now() - sig_start) * 1000.0
                    await _emit_latency(bot, symbol=k, stage="signal", duration_ms=sig_duration_ms, result="success" if sig else "no_signal")
                    if not isinstance(sig, dict) or not sig:
                        await _emit_entry_blocked(bot, k, "signal_missing", throttle_sec=30.0)
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    action = _parse_action(sig)
                    # confidence snapshot used by EDR + notional scaling
                    try:
                        conf = float(sig.get("confidence", sig.get("conf", 0.0)) or 0.0)
                    except Exception:
                        conf = 0.0

                    otype = _parse_order_type(sig)
                    price = _parse_price(sig) if otype == "limit" else None
                    amt = _parse_amount(sig)
                    if amt is None:
                        amt = _sizing_fallback_amount(bot, k)

                    propose_rec = compute_entry_decision(
                        symbol=k,
                        signal=sig if isinstance(sig, dict) else None,
                        guard_knobs=guard_knobs,
                        min_confidence=float(current_min_conf),
                        amount=amt,
                        order_type=otype,
                        price=price,
                        planned_notional=0.0,
                        stage="propose",
                        meta={
                            "runtime_gate_degraded": runtime_gate_degraded,
                            "reconcile_first_gate_degraded": reconcile_first_gate_degraded,
                            **_corr_snapshot(bot, guard_knobs),
                        },
                    )
                    if propose_rec.action in ("DENY", "DEFER"):
                        reason = propose_rec.reason_primary or "signal_missing"
                        await _emit_entry_decision(bot, propose_rec, level="warning", throttle_sec=15.0)
                        await _emit_entry_blocked(bot, k, reason, throttle_sec=30.0)
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    if amt is None or amt <= 0:
                        if sizing_warn_every > 0 and (_now() - last_sizing_warn_ts) >= sizing_warn_every:
                            last_sizing_warn_ts = _now()
                            fixed_qty, fixed_notional = _resolve_symbol_sizing(bot, k)
                            log_entry.warning(
                                "ENTRY_LOOP: sizing missing; set FIXED_QTY or FIXED_NOTIONAL_USDT. "
                                f"(FIXED_QTY={fixed_qty}, FIXED_NOTIONAL_USDT={fixed_notional})"
                            )
                        await _emit_entry_blocked(bot, k, "sizing_missing", throttle_sec=60.0)
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    # Hedge hint: router accepts "long"/"short" for entries
                    hedge_side_hint = None
                    if hedge_hint_mode:
                        hedge_side_hint = "long" if action == "buy" else "short"

                    conf_scale, conf_reason = _confidence_notional_scale(bot, float(conf or 0.0))
                    if conf_scale and conf_scale < 1.0 and amt is not None and amt > 0:
                        try:
                            base_amt = float(amt)
                            amt = float(amt) * float(conf_scale)
                            if callable(emit_throttled):
                                asyncio.create_task(
                                    emit_throttled(
                                        bot,
                                        "entry.notional_scaled",
                                        key=f"{k}:{conf_reason or 'confidence'}",
                                        cooldown_sec=120.0,
                                        data={
                                            "symbol": k,
                                            "base_qty": base_amt,
                                            "scaled_qty": amt,
                                            "scale": conf_scale,
                                            "reason": conf_reason or "confidence",
                                            "confidence": float(conf or 0.0),
                                        },
                                        symbol=k,
                                        level="info",
                                    )
                                )
                        except Exception:
                            pass

                    planned_notional = 0.0
                    try:
                        ref_px = float(price or 0.0)
                        if ref_px <= 0:
                            ref_px = float(_get_price(bot, k) or 0.0)
                        if ref_px > 0:
                            planned_notional = float(amt) * ref_px
                    except Exception:
                        planned_notional = 0.0
                    corr_reason, corr_meta = _check_corr_group(bot, k, planned_notional)
                    scale, scale_reason = _corr_group_scale(bot, corr_meta or {})
                    if scale and scale < 1.0 and planned_notional > 0:
                        try:
                            amt = float(amt) * float(scale)
                            planned_notional = float(planned_notional) * float(scale)
                            if callable(emit_throttled):
                                asyncio.create_task(
                                    emit_throttled(
                                        bot,
                                        "entry.notional_scaled",
                                        key=f"{k}:{scale_reason or 'corr_group'}",
                                        cooldown_sec=120.0,
                                        data={
                                            "symbol": k,
                                            "planned_notional": planned_notional,
                                            "scale": scale,
                                            "reason": scale_reason or "corr_group",
                                            **(corr_meta or {}),
                                        },
                                        symbol=k,
                                        level="warning",
                                    )
                                )
                        except Exception:
                            pass
                    exp_scale, exp_reason = _corr_group_exposure_scale(bot, corr_meta or {}, planned_notional)
                    if exp_scale and exp_scale < 1.0 and planned_notional > 0:
                        try:
                            amt = float(amt) * float(exp_scale)
                            planned_notional = float(planned_notional) * float(exp_scale)
                            if callable(emit_throttled):
                                asyncio.create_task(
                                    emit_throttled(
                                        bot,
                                        "entry.notional_scaled",
                                        key=f"{k}:{exp_reason or 'corr_group_exposure'}",
                                        cooldown_sec=120.0,
                                        data={
                                            "symbol": k,
                                            "planned_notional": planned_notional,
                                            "scale": exp_scale,
                                            "reason": exp_reason or "corr_group_exposure",
                                            **(corr_meta or {}),
                                        },
                                        symbol=k,
                                        level="warning",
                                    )
                                )
                        except Exception:
                            pass
                    if corr_reason:
                        await _emit_entry_blocked(
                            bot,
                            k,
                            "corr_group_cap",
                            data={
                                "reason": corr_reason,
                                "planned_notional": planned_notional,
                                **(corr_meta or {}),
                            },
                            throttle_sec=30.0,
                        )
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    if guard_max_notional > 0 and planned_notional > guard_max_notional:
                        try:
                            scale = float(guard_max_notional) / max(1e-9, float(planned_notional))
                            if scale <= 0:
                                await _emit_entry_blocked(
                                    bot,
                                    k,
                                    "belief_notional_cap",
                                    data={"cap": guard_max_notional, "planned_notional": planned_notional, "mode": guard_mode},
                                    throttle_sec=15.0,
                                )
                                await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                continue
                            amt = float(amt) * float(scale)
                            planned_notional = float(planned_notional) * float(scale)
                            if callable(emit_throttled):
                                asyncio.create_task(
                                    emit_throttled(
                                        bot,
                                        "entry.notional_scaled",
                                        key=f"{k}:belief_cap",
                                        cooldown_sec=60.0,
                                        data={
                                            "symbol": k,
                                            "scale": scale,
                                            "planned_notional": planned_notional,
                                            "cap": guard_max_notional,
                                            "reason": "belief_controller_cap",
                                            "mode": guard_mode,
                                        },
                                        symbol=k,
                                        level="warning",
                                    )
                                )
                        except Exception:
                            pass

                    if budget_enabled and planned_notional > 0:
                        rem_budget = max(0.0, float(budget_remaining) - float(budget_spent))
                        sym_budget_cap = _entry_budget_symbol_cap(bot, conf, current_min_conf, rem_budget)
                        if sym_budget_cap <= 0:
                            await _emit_entry_blocked(
                                bot,
                                k,
                                "entry_budget_symbol_cap",
                                data={
                                    "budget_total_usdt": float(budget_total),
                                    "budget_remaining_usdt": float(rem_budget),
                                    "planned_notional": float(planned_notional),
                                    "code": ERR_RISK,
                                },
                                throttle_sec=15.0,
                            )
                            await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                            continue
                        if planned_notional > sym_budget_cap:
                            try:
                                bscale = float(sym_budget_cap) / max(1e-9, float(planned_notional))
                                amt = float(amt) * float(bscale)
                                planned_notional = float(planned_notional) * float(bscale)
                                if callable(emit_throttled):
                                    asyncio.create_task(
                                        emit_throttled(
                                            bot,
                                            "entry.notional_scaled",
                                            key=f"{k}:entry_budget_allocator",
                                            cooldown_sec=60.0,
                                            data={
                                                "symbol": k,
                                                "scale": bscale,
                                                "planned_notional": planned_notional,
                                                "cap": sym_budget_cap,
                                                "budget_total_usdt": float(budget_total),
                                                "budget_remaining_usdt": float(rem_budget),
                                                "reason": "entry_budget_allocator",
                                            },
                                            symbol=k,
                                            level="warning",
                                        )
                                )
                            except Exception:
                                pass

                    propose_rec = compute_entry_decision(
                        symbol=k,
                        signal=sig if isinstance(sig, dict) else None,
                        guard_knobs=guard_knobs,
                        min_confidence=float(current_min_conf),
                        amount=amt,
                        order_type=otype,
                        price=price,
                        planned_notional=float(planned_notional),
                        stage="propose",
                        meta={
                            "runtime_gate_degraded": runtime_gate_degraded,
                            "reconcile_first_gate_degraded": reconcile_first_gate_degraded,
                            "budget_enabled": bool(budget_enabled),
                            **_corr_snapshot(bot, guard_knobs),
                        },
                    )
                    if propose_rec.action in ("DENY", "DEFER"):
                        reason = propose_rec.reason_primary or "signal_missing"
                        await _emit_entry_decision(bot, propose_rec, level="warning", throttle_sec=15.0)
                        await _emit_entry_blocked(bot, k, reason, throttle_sec=30.0)
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    # Safety: block entries after recent router blocks
                    if _cfg_env_bool(bot, "ENTRY_BLOCK_ON_ROUTER_BLOCK", True):
                        window_sec = float(_cfg_env_float(bot, "ENTRY_BLOCK_ROUTER_WINDOW_SEC", 60.0) or 60.0)
                        threshold = int(_cfg_env_float(bot, "ENTRY_BLOCK_ROUTER_THRESHOLD", 1) or 1)
                        backoff_sec = float(_cfg_env_float(bot, "ENTRY_BLOCK_ROUTER_BACKOFF_SEC", 10.0) or 10.0)
                        if window_sec > 0 and threshold > 0:
                            if callable(count_recent):
                                blocked = int(
                                    count_recent(bot, event="order.blocked", symbol=k, window_sec=window_sec)
                                )
                            else:
                                blocked = _recent_router_blocks(bot, k, window_sec)
                            if blocked >= threshold:
                                log_entry.warning(
                                    f"ENTRY_LOOP: router blocks={blocked} within {window_sec:.0f}s → backoff {k}"
                                )
                                await _emit_entry_blocked(
                                    bot,
                                    k,
                                    "router_block",
                                    data={"count": blocked, "window_sec": window_sec},
                                    throttle_sec=5.0,
                                )
                                _set_pending(k, sec=max(3.0, backoff_sec))
                                await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                continue

                    # Unified safety throttle: recent entry.blocked events
                    if _cfg_env_bool(bot, "ENTRY_BLOCK_ON_ERRORS", False) and callable(count_recent):
                        err_window = float(_cfg_env_float(bot, "ENTRY_BLOCK_ERRORS_WINDOW_SEC", 60.0) or 60.0)
                        err_thresh = int(_cfg_env_float(bot, "ENTRY_BLOCK_ERRORS_THRESHOLD", 3) or 3)
                        err_backoff = float(_cfg_env_float(bot, "ENTRY_BLOCK_ERRORS_BACKOFF_SEC", 15.0) or 15.0)
                        if err_window > 0 and err_thresh > 0:
                            err_count = int(count_recent(bot, event="entry.blocked", symbol=k, window_sec=err_window))
                            if err_count >= err_thresh:
                                log_entry.warning(
                                    f"ENTRY_LOOP: entry.blocked={err_count} within {err_window:.0f}s → backoff {k}"
                                )
                                await _emit_entry_blocked(
                                    bot,
                                    k,
                                    "error_flood",
                                    data={"count": err_count, "window_sec": err_window},
                                    throttle_sec=30.0,
                                )
                                _set_pending(k, sec=max(3.0, err_backoff))
                                await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                continue

                    # Commit phase: re-check fast-moving constraints before submit.
                    commit_ok, commit_rec = commit_entry_intent(
                        propose_rec,
                        current_guard_knobs=_resolve_symbol_guard(_get_guard_knobs(bot), k),
                        in_position_fn=lambda: _in_position_brain(bot, k),
                        pending_fn=lambda: _pending_active(k),
                    )
                    await _emit_entry_decision(
                        bot,
                        commit_rec,
                        level=("warning" if not commit_ok else "info"),
                        throttle_sec=(10.0 if not commit_ok else 0.0),
                    )
                    if not commit_ok:
                        await _emit_entry_blocked(bot, k, commit_rec.reason_primary or "posture_changed", throttle_sec=20.0)
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    # Submit order
                    order_start = _now()
                    order_result = "success"
                    try:
                        if otype == "limit":
                            if price is None or price <= 0:
                                log_entry.warning(f"ENTRY_LOOP: limit signal missing price for {k}")
                                await _emit_entry_blocked(bot, k, "signal_limit_price", throttle_sec=30.0)
                                await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                continue

                            res = await create_order(
                                bot,
                                symbol=sym_raw,
                                type="LIMIT",
                                side=action,
                                amount=float(amt),
                                price=float(price),
                                params={},
                                intent_reduce_only=False,
                                intent_close_position=False,
                                hedge_side_hint=hedge_side_hint,
                                retries=int(_cfg_env_float(bot, "ENTRY_ROUTER_RETRIES", 6) or 6),
                                intent_component="entry_loop",
                                intent_kind="ENTRY",
                            )
                        else:
                            res = await create_order(
                                bot,
                                symbol=sym_raw,
                                type="MARKET",
                                side=action,
                                amount=float(amt),
                                price=None,
                                params={},
                                intent_reduce_only=False,
                                intent_close_position=False,
                                hedge_side_hint=hedge_side_hint,
                                retries=int(_cfg_env_float(bot, "ENTRY_ROUTER_RETRIES", 4) or 4),
                                intent_component="entry_loop",
                                intent_kind="ENTRY",
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        order_result = "error"
                        # try to detect margin insufficient and backoff
                        msg = str(e)
                        if "Margin is insufficient" in msg or '"code":-2019' in msg or "code': -2019" in msg:
                            until = _now() + max(60.0, float(margin_backoff_sec))
                            backoff_until_by_sym[k] = until
                            log_entry.critical(f"ENTRY_LOOP: margin insufficient → backing off {k} for {int(margin_backoff_sec)}s")
                            await _emit_entry_blocked(
                                bot,
                                k,
                                "margin_insufficient",
                                level="critical",
                                throttle_sec=30.0,
                            )
                            _set_pending(k, sec=max(10.0, pending_block_sec * 0.5))
                        else:
                            log_entry.error(f"ENTRY_LOOP: order submit failed {k}: {e}")
                            _set_pending(k, sec=max(3.0, pending_block_sec * 0.25))

                        if callable(emit):
                            try:
                                await emit(
                                    bot,
                                    "entry.exception",
                                    data={"symbol": k, "err": repr(e)[:300]},
                                    symbol=k,
                                    level="critical",
                                )
                            except Exception:
                                pass
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue
                    finally:
                        duration_ms = (_now() - order_start) * 1000.0
                        await _emit_latency(bot, symbol=k, stage="order_router", duration_ms=duration_ms, result=order_result)

                        oid = None
                        if isinstance(res, dict):
                            oid = res.get("id") or (res.get("info") or {}).get("orderId")

                        # Partial fill handling (optional)
                        if isinstance(res, dict):
                            min_ratio = float(_cfg_env_float(bot, "ENTRY_PARTIAL_MIN_FILL_RATIO", 0.5) or 0.5)
                            if min_ratio > 0:
                                filled = _order_filled(res)
                                req_amt = float(amt or 0.0)
                                ratio = (filled / req_amt) if (req_amt > 0 and filled > 0) else 0.0
                                if ratio > 0 and ratio < min_ratio:
                                    pf = await _resolve_partial_fill_state(
                                        bot,
                                        symbol=k,
                                        sym_raw=sym_raw,
                                        action=action,
                                        otype=otype,
                                        order_id=str(oid) if oid else None,
                                        requested=req_amt,
                                        filled=filled,
                                        min_ratio=min_ratio,
                                        hedge_side_hint=hedge_side_hint,
                                    )
                                    await _emit_entry_blocked(
                                        bot,
                                        k,
                                        "partial_fill",
                                        data={
                                            "filled": filled,
                                            "requested": req_amt,
                                            "ratio": ratio,
                                            "min_ratio": min_ratio,
                                            "outcome": pf.get("outcome"),
                                            "cancel_ok": bool(pf.get("cancel_ok")),
                                            "flatten_ok": bool(pf.get("flatten_ok")),
                                            "code": ERR_PARTIAL_FILL,
                                        },
                                        level=("warning" if str(pf.get("outcome")) == "partial_forced_flatten" else "critical"),
                                        throttle_sec=20.0,
                                    )
                                    base_backoff = max(3.0, float(_cfg_env_float(bot, "ENTRY_PARTIAL_BACKOFF_SEC", 10.0) or 10.0))
                                    extra_backoff = _record_partial_fill_hit(bot, k)
                                    total_backoff = base_backoff if extra_backoff <= 0 else max(base_backoff, extra_backoff)
                                    if str(pf.get("outcome")) == "partial_stuck":
                                        total_backoff = max(total_backoff, base_backoff * 2.0)
                                    _set_pending(k, sec=total_backoff)
                                    if extra_backoff > 0 and callable(emit_throttled):
                                        try:
                                            cooldown = max(30.0, float(_cfg_env_float(bot, "ENTRY_PARTIAL_ESCALATE_TELEM_CD_SEC", 300.0) or 300.0))
                                            await emit_throttled(
                                                bot,
                                                "entry.partial_fill_escalation",
                                                key=f"{k}:partial_fill",
                                                cooldown_sec=cooldown,
                                                data={
                                                    "ratio": ratio,
                                                    "requested": req_amt,
                                                    "filled": filled,
                                                    "backoff": extra_backoff,
                                                    "min_ratio": min_ratio,
                                                },
                                                symbol=k,
                                                level="warning",
                                            )
                                        except Exception:
                                            pass
                                    await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                                    continue

                        if res is None:
                            log_entry.warning(f"ENTRY_LOOP: create_order returned None for {k}")
                            if callable(emit):
                                try:
                                    await emit(
                                        bot,
                                        "entry.order_failed",
                                        data={"symbol": k, "action": action, "type": otype},
                                        symbol=k,
                                        level="critical",
                                    )
                                except Exception:
                                    pass
                            # even if failed, block briefly to prevent rapid spam while exchange is angry
                            _set_pending(k, sec=max(3.0, pending_block_sec * 0.25))
                            await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                            continue

                        if budget_enabled and planned_notional > 0:
                            budget_spent += max(0.0, float(planned_notional))

                        # ✅ key anti-stack: once we submitted ANY entry, block more entries for a while
                        _set_pending(k, sec=max(5.0, pending_block_sec), order_id=str(oid) if oid else None)

                        log_core.critical(f"ENTRY_LOOP: ORDER SUBMITTED {k} {action.upper()} type={otype} amt={amt} id={oid}")
                        _reset_partial_fill_hits(k)

                        # Register watch for limit/pending orders (optional)
                        if otype == "limit" and callable(register_entry_watch) and oid:
                            try:
                                await register_entry_watch(
                                    bot,
                                    symbol=sym_raw,
                                    order_id=str(oid),
                                    side=("long" if action == "buy" else "short"),
                                    amount=float(amt),
                                    price=float(price or 0.0),
                                    meta={"confidence": conf, "signal": {kk: vv for kk, vv in sig.items() if kk not in ("raw",)}},
                                )
                            except Exception:
                                pass

                        # Telemetry (optional)
                        if callable(emit):
                            try:
                                await emit(
                                    bot,
                                    "entry.submitted",
                                    data={
                                        "symbol": k,
                                        "action": action,
                                        "type": otype,
                                        "amount": float(amt),
                                        "price": float(price) if price is not None else None,
                                        "confidence": conf,
                                        "order_id": oid,
                                        "pending_block_sec": float(pending_block_sec),
                                    },
                                    symbol=k,
                                    level="info",
                                )
                            except Exception:
                                pass

                        if _truthy(_cfg(bot, "ENTRY_NOTIFY", False)):
                            await _safe_speak(bot, f"ENTRY {k} {action.upper()} {otype} amt={amt}", "info")

                await asyncio.sleep(max(0.01, per_symbol_gap_sec))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"ENTRY_LOOP outer error: {e}")
            await asyncio.sleep(1.0)

    log_core.critical("ENTRY_LOOP OFFLINE — shutdown flag set")
