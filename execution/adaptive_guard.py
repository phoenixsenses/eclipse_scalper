#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

TELEMETRY_PATH = Path(os.getenv("TELEMETRY_PATH", "logs/telemetry.jsonl"))
DRIFT_PATH = Path(os.getenv("TELEMETRY_DRIFT_PATH", "logs/telemetry_drift.jsonl"))
GUARD_HISTORY_EVENTS_PATH = Path(os.getenv("TELEMETRY_GUARD_HISTORY_EVENTS", "logs/telemetry_guard_history_events.jsonl"))
STATE_PATH = Path(os.getenv("ADAPTIVE_GUARD_STATE", "logs/telemetry_adaptive_guard.json"))
DEFAULT_DURATION = float(os.getenv("ADAPTIVE_GUARD_DURATION_SEC", 900))
PARTIAL_DELTA = float(os.getenv("ADAPTIVE_GUARD_PARTIAL_DELTA", 0.1))
PARTIAL_DURATION = float(os.getenv("ADAPTIVE_GUARD_PARTIAL_DURATION_SEC", 600))
PARTIAL_ESCALATE_DELTA = float(os.getenv("ADAPTIVE_GUARD_PARTIAL_ESCALATE_DELTA", 0.2))
PARTIAL_ESCALATE_DURATION = float(os.getenv("ADAPTIVE_GUARD_PARTIAL_ESCALATE_DURATION_SEC", 900))
RETRY_DELTA = float(os.getenv("ADAPTIVE_GUARD_RETRY_DELTA", 0.15))
RETRY_DURATION = float(os.getenv("ADAPTIVE_GUARD_RETRY_DURATION_SEC", 600))
GUARD_HISTORY_DELTA = float(os.getenv("ADAPTIVE_GUARD_GUARD_HISTORY_DELTA", 0.1))
GUARD_HISTORY_DURATION = float(os.getenv("ADAPTIVE_GUARD_GUARD_HISTORY_DURATION_SEC", 900))
GUARD_HISTORY_LEVERAGE_SCALE = float(os.getenv("ADAPTIVE_GUARD_GUARD_HISTORY_LEVERAGE_SCALE", 0.85))
GUARD_HISTORY_LEVERAGE_DURATION = float(os.getenv("ADAPTIVE_GUARD_GUARD_HISTORY_LEVERAGE_DURATION_SEC", 900))
GUARD_HISTORY_NOTIONAL_SCALE = float(os.getenv("ADAPTIVE_GUARD_GUARD_HISTORY_NOTIONAL_SCALE", 0.8))
GUARD_HISTORY_NOTIONAL_DURATION = float(os.getenv("ADAPTIVE_GUARD_GUARD_HISTORY_NOTIONAL_DURATION_SEC", 900))
_STATE_CACHE: Dict[str, Any] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _load_state() -> Dict[str, Any]:
    global _STATE_CACHE
    if _STATE_CACHE:
        return _STATE_CACHE
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data.setdefault("offset", 0.0)
    data.setdefault("drift_offset", 0.0)
    data.setdefault("guard_history_offset", 0.0)
    data.setdefault("symbols", {})
    data.setdefault("global", {})
    _STATE_CACHE = data
    return data


def _persist(state: Dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _cleanup(state: Dict[str, Any]) -> None:
    now = time.time()
    symbols = state.get("symbols") or {}
    for sym in list(symbols):
        if _safe_float(symbols[sym].get("expires"), 0.0) <= now:
            symbols.pop(sym, None)
    global_entry = state.get("global") or {}
    if global_entry:
        if _safe_float(global_entry.get("expires"), 0.0) <= now:
            global_entry.pop("delta", None)
            global_entry.pop("expires", None)
        if _safe_float(global_entry.get("leverage_expires"), 0.0) <= now:
            global_entry.pop("leverage_scale", None)
            global_entry.pop("leverage_expires", None)
        if _safe_float(global_entry.get("notional_expires"), 0.0) <= now:
            global_entry.pop("notional_scale", None)
            global_entry.pop("notional_expires", None)
        if not global_entry.get("delta") and not global_entry.get("leverage_scale") and not global_entry.get("notional_scale"):
            state.pop("global", None)


def _apply_override(state: Dict[str, Any], symbol: str, delta: float, duration: float, reason: str) -> None:
    symbols = state.setdefault("symbols", {})
    now = time.time()
    entry = symbols.get(symbol, {})
    existing_delta = _safe_float(entry.get("delta"))
    new_delta = max(existing_delta, min(delta, 1.0))
    symbols[symbol] = {
        "delta": new_delta,
        "expires": now + max(0.0, duration),
        "reason": reason,
        "last_ts": now,
    }


def _apply_global_override(
    state: Dict[str, Any],
    delta: float,
    duration: float,
    leverage_scale: float,
    leverage_duration: float,
    notional_scale: float,
    notional_duration: float,
    reason: str,
) -> None:
    now = time.time()
    entry = state.setdefault("global", {})
    existing_delta = _safe_float(entry.get("delta"))
    new_delta = max(existing_delta, min(delta, 1.0))
    if new_delta > 0:
        entry["delta"] = new_delta
        entry["expires"] = now + max(0.0, duration)
    scale = _safe_float(leverage_scale, 1.0)
    if scale > 0:
        existing_scale = _safe_float(entry.get("leverage_scale"), 1.0)
        entry["leverage_scale"] = min(existing_scale, scale)
        entry["leverage_expires"] = now + max(0.0, leverage_duration)
    notional = _safe_float(notional_scale, 1.0)
    if notional > 0:
        existing_notional = _safe_float(entry.get("notional_scale"), 1.0)
        entry["notional_scale"] = min(existing_notional, notional)
        entry["notional_expires"] = now + max(0.0, notional_duration)
    entry["reason"] = reason
    entry["last_ts"] = now


def _process_event(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    name = str(event.get("event") or event.get("type") or "")
    data = event.get("data") or {}
    if name == "telemetry.guard_history_spike":
        hit_count = int(_safe_float(data.get("hit_count"), 0))
        hit_rate = _safe_float(data.get("hit_rate"), 0.0)
        delta = _safe_float(data.get("confidence_delta"), GUARD_HISTORY_DELTA)
        duration = _safe_float(data.get("confidence_duration"), GUARD_HISTORY_DURATION)
        lev_scale = _safe_float(data.get("leverage_scale"), GUARD_HISTORY_LEVERAGE_SCALE)
        lev_duration = _safe_float(data.get("leverage_duration"), GUARD_HISTORY_LEVERAGE_DURATION)
        notional_scale = _safe_float(data.get("notional_scale"), GUARD_HISTORY_NOTIONAL_SCALE)
        notional_duration = _safe_float(data.get("notional_duration"), GUARD_HISTORY_NOTIONAL_DURATION)
        reason = f"guard_history hits={hit_count} rate={hit_rate:.2f}"
        _apply_global_override(
            state,
            delta,
            duration,
            lev_scale,
            lev_duration,
            notional_scale,
            notional_duration,
            reason,
        )
        return
    symbol = str(data.get("symbol") or event.get("symbol") or "").upper()
    if not symbol:
        return
    if name == "telemetry.confidence_drift":
        z = _safe_float(data.get("zscore") or 0.0)
        delta = min(0.4, 0.05 + z * 0.02)
        duration = float(os.getenv("ADAPTIVE_GUARD_DRIFT_DURATION_SEC", 900))
        _apply_override(state, symbol, delta, duration, f"drift z{z:.2f}")
    elif name == "exit.signal_issue":
        delta = 0.1
        duration = float(os.getenv("ADAPTIVE_GUARD_EXIT_DURATION_SEC", 600))
        _apply_override(state, symbol, delta, duration, "signal_issue")
    elif name == "exit.telemetry_guard":
        delta = 0.15
        duration = float(os.getenv("ADAPTIVE_GUARD_TELEMETRY_DURATION_SEC", 600))
        reason = str(data.get("reason") or "telemetry_guard")
        _apply_override(state, symbol, delta, duration, reason)
    elif name == "entry.blocked":
        reason_val = str(data.get("reason") or "").lower()
        if reason_val == "partial_fill":
            ratio = _safe_float(data.get("ratio"))
            duration = float(os.getenv("ADAPTIVE_GUARD_PARTIAL_DURATION_SEC", PARTIAL_DURATION))
            reason = f"partial_fill ratio {ratio:.2f}"
            _apply_override(state, symbol, PARTIAL_DELTA, duration, reason)
    elif name == "order.create.retry_alert":
        tries = int(_safe_float(data.get("tries"), 0))
        variant = int(_safe_float(data.get("variants"), 0))
        base_reason = str(data.get("reason") or "retry_alert")
        reason = f"{base_reason} tries={tries} variants={variant}"
        _apply_override(state, symbol, RETRY_DELTA, RETRY_DURATION, reason)
    elif name == "entry.partial_fill_escalation":
        ratio = _safe_float(data.get("ratio"))
        backoff = _safe_float(data.get("backoff"), 0.0)
        reason = f"partial_fill_escalation ratio {ratio:.2f} backoff {int(backoff)}"
        duration = float(os.getenv("ADAPTIVE_GUARD_PARTIAL_ESCALATE_DURATION_SEC", PARTIAL_ESCALATE_DURATION))
        _apply_override(state, symbol, PARTIAL_ESCALATE_DELTA, duration, reason)


def _tail_file(state: Dict[str, Any], path: Path, key: str) -> None:
    if not path.exists():
        return
    try:
        size = path.stat().st_size
    except Exception:
        return
    offset = _safe_float(state.get(key), 0.0)
    if size < offset:
        offset = 0.0
    try:
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(int(offset))
            chunk = fh.read()
            state[key] = float(fh.tell())
        if not chunk:
            return
        for line in chunk.splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except Exception:
                continue
            _process_event(state, event)
    except Exception:
        pass


def refresh_state() -> None:
    state = _load_state()
    _tail_file(state, TELEMETRY_PATH, "offset")
    _tail_file(state, DRIFT_PATH, "drift_offset")
    _tail_file(state, GUARD_HISTORY_EVENTS_PATH, "guard_history_offset")
    _cleanup(state)
    _persist(state)


def get_override(symbol: str, base_min_conf: float) -> tuple[float, str]:
    state = _load_state()
    now = time.time()
    global_entry = state.get("global") or {}
    global_delta = _safe_float(global_entry.get("delta"))
    global_reason = str(global_entry.get("reason") or "")
    global_expires = _safe_float(global_entry.get("expires"))
    if global_delta and global_expires and global_expires <= now:
        global_entry.pop("delta", None)
        global_entry.pop("expires", None)
        if not global_entry.get("leverage_scale") and not global_entry.get("notional_scale"):
            state.pop("global", None)
        _persist(state)
        global_delta = 0.0
        global_reason = ""
    entry = (state.get("symbols") or {}).get(symbol.upper())
    if not entry:
        if global_delta > 0:
            return min(1.0, base_min_conf + global_delta), global_reason
        return base_min_conf, ""
    expires = _safe_float(entry.get("expires"))
    if expires <= now:
        state.get("symbols", {}).pop(symbol.upper(), None)
        _persist(state)
        if global_delta > 0:
            return min(1.0, base_min_conf + global_delta), global_reason
        return base_min_conf, ""
    delta = _safe_float(entry.get("delta"))
    merged_delta = max(global_delta, delta)
    override = min(1.0, base_min_conf + merged_delta)
    reason = str(entry.get("reason") or "")
    if global_reason and reason and global_reason != reason:
        reason = f"{reason}; {global_reason}"
    elif global_reason and not reason:
        reason = global_reason
    return override, reason


def get_leverage_scale(symbol: str) -> tuple[float, str]:
    state = _load_state()
    global_entry = state.get("global") or {}
    scale = _safe_float(global_entry.get("leverage_scale"), 1.0)
    expires = _safe_float(global_entry.get("leverage_expires"))
    if scale <= 0:
        return 1.0, ""
    if expires and expires <= time.time():
        global_entry.pop("leverage_scale", None)
        global_entry.pop("leverage_expires", None)
        if not global_entry.get("delta") and not global_entry.get("notional_scale"):
            state.pop("global", None)
        _persist(state)
        return 1.0, ""
    reason = str(global_entry.get("reason") or "")
    return float(scale), reason


def get_notional_scale(symbol: str) -> tuple[float, str]:
    state = _load_state()
    global_entry = state.get("global") or {}
    scale = _safe_float(global_entry.get("notional_scale"), 1.0)
    expires = _safe_float(global_entry.get("notional_expires"))
    if scale <= 0:
        return 1.0, ""
    if expires and expires <= time.time():
        global_entry.pop("notional_scale", None)
        global_entry.pop("notional_expires", None)
        if not global_entry.get("delta") and not global_entry.get("leverage_scale"):
            state.pop("global", None)
        _persist(state)
        return 1.0, ""
    reason = str(global_entry.get("reason") or "")
    return float(scale), reason
