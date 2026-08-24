# execution/entry_gates.py — Entry gate checks, budget caps, correlation groups, regime blocking
# Extracted from entry_loop.py (Phase 7.1) — guardian-safe, never raises

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

from execution.runtime_helpers import (
    cfg_env_bool as _cfg_env_bool,
    cfg_env_float as _cfg_env_float,
    cfg_value as _cfg,
    symkey as _symkey,
    parse_group_kv as _parse_group_kv,
)

try:
    from execution.error_codes import ERR_RELIABILITY_GATE, ERR_ROUTER_BLOCK  # type: ignore
except Exception:
    ERR_RELIABILITY_GATE = "ERR_RELIABILITY_GATE"
    ERR_ROUTER_BLOCK = "ERR_ROUTER_BLOCK"


def _now() -> float:
    return time.time()


# ----------------------------
# Adaptive guard / min confidence
# ----------------------------

def _adaptive_guard_enabled(bot) -> bool:
    return _cfg_env_bool(bot, "ENTRY_ADAPTIVE_GUARD_ENABLED", True)


def _effective_min_conf(base_min_conf: float, guard_min_conf: float, adaptive_enabled: bool) -> float:
    if not bool(adaptive_enabled):
        return float(max(0.0, float(base_min_conf)))
    return float(max(float(base_min_conf), max(0.0, float(guard_min_conf))))


# ----------------------------
# Guard knobs
# ----------------------------

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


# ----------------------------
# Reconcile-first gate recording
# ----------------------------

def _record_reconcile_first_gate(bot, symbol: str, severity: float, reason: str = "") -> None:
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
    except Exception:
        return


# ----------------------------
# Exposure estimation
# ----------------------------

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


# ----------------------------
# Entry budget
# ----------------------------

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
# Regime blocking
# ----------------------------

def _parse_regime_mode(raw: Any) -> str:
    v = str(raw or "").strip().lower()
    if v in ("up", "down", "none"):
        return v
    return "none"


def _should_block_regime(
    *,
    mode: str,
    current_regime: str,
    block_transition: bool,
    block_unknown: bool,
    allow_unknown: bool = False,
    warmup_active: bool = False,
) -> tuple[bool, str]:
    m = _parse_regime_mode(mode)
    cur = str(current_regime or "").strip().upper()
    unknown_allowed = bool(allow_unknown) or bool(warmup_active)
    if m == "none":
        if block_transition and cur == "TRANSITION":
            return True, "regime_transition"
        if block_unknown and cur == "UNKNOWN" and not unknown_allowed:
            return True, "regime_unknown"
        return False, ""
    if block_transition and cur == "TRANSITION":
        return True, "regime_transition"
    if block_unknown and cur == "UNKNOWN" and not unknown_allowed:
        return True, "regime_unknown"
    if cur == "UNKNOWN" and unknown_allowed:
        return False, ""
    if m == "up" and cur != "UP":
        return True, "regime_mismatch"
    if m == "down" and cur != "DOWN":
        return True, "regime_mismatch"
    return False, ""


# ----------------------------
# Correlation group gate
# ----------------------------

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
