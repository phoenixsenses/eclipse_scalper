#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


STATE_PATH = Path(os.getenv("TELEMETRY_RECOVERY_STATE", "logs/telemetry_recovery_state.json"))
_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}}
_TTL = 5.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:  # NaN guard
            return default
        return v
    except Exception:
        return default


def _load_state() -> dict[str, Any]:
    now = time.time()
    cache = _CACHE
    if (now - _safe_float(cache.get("ts"), 0.0)) < _TTL and cache.get("data"):
        return cache["data"]
    try:
        text = STATE_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception:
        data = {}
    cache["ts"] = now
    cache["data"] = data
    return data


def get_active_state() -> dict[str, Any] | None:
    state = _load_state()
    expires = _safe_float(state.get("expires_at"), 0.0)
    if expires > time.time():
        return state
    return None


def write_recovery_state(
    min_confidence: float,
    duration_sec: float,
    reason: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    data = {
        "min_confidence_override": float(min_confidence),
        "expires_at": time.time() + max(0.0, float(duration_sec)),
        "reason": str(reason or "telemetry_recovery"),
        "severity": str(severity),
        "generated_at": time.time(),
    }
    if extra:
        data["extra"] = extra
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
