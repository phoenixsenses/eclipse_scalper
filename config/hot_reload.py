# config/hot_reload.py — Runtime config hot-reload from JSON override file
# Guardian-safe: never raises, never fatal

from __future__ import annotations

import json
import os
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Optional, Set

from config.contracts import IMMUTABLE, DANGEROUS, SAFE

_OVERRIDE_PATH = Path(os.environ.get("CONFIG_OVERRIDE_PATH", "config/runtime_overrides.json"))

# ── Three-tier classification from config.contracts (single source of truth) ──
# DANGEROUS: risk-critical, hot-reloadable with CRITICAL warning + audit trail.
# Set HOT_RELOAD_DANGEROUS_CONFIRM=0 to treat them as BLOCKED.
HOT_RELOAD_DANGEROUS: Set[str] = set(DANGEROUS)

_DANGEROUS_CONFIRM_ENABLED = os.environ.get(
    "HOT_RELOAD_DANGEROUS_CONFIRM", "1"
).strip().lower() not in ("0", "false", "no", "off")

# SAFE: freely hot-reloadable (thresholds, limits, timeouts)
HOT_RELOAD_ALLOWED: Set[str] = set(SAFE)

# BLOCKED: structural params that require restart
HOT_RELOAD_BLOCKED: Set[str] = set(IMMUTABLE)

_last_mtime: float = 0.0
_last_check_ts: float = 0.0
_CHECK_INTERVAL_SEC: float = 10.0  # don't stat() more than once per 10s

_CONFIG_CHANGES_PATH = Path("logs/config_changes.jsonl")


def _log(msg: str) -> None:
    try:
        from utils.logging import log_core
        log_core.info(msg)
    except Exception:
        pass


def _audit_config_change(key: str, old_value: Any, new_value: Any, *, dangerous: bool = False) -> None:
    """Append config change to JSONL audit trail. Guardian-safe: never raises."""
    try:
        _CONFIG_CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "ts": time.time(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "field": key,
            "old": old_value if isinstance(old_value, (int, float, bool, str, type(None))) else str(old_value),
            "new": new_value if isinstance(new_value, (int, float, bool, str, type(None))) else str(new_value),
        }
        if dangerous:
            record["tier"] = "DANGEROUS"
        entry = json.dumps(record, ensure_ascii=False)
        with open(_CONFIG_CHANGES_PATH, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def check_and_apply(cfg: Any) -> Dict[str, Any]:
    """Check override file and apply changes to cfg. Returns dict of changed fields.

    Guardian-safe: never raises. Returns empty dict on any error.
    """
    global _last_mtime, _last_check_ts
    changes: Dict[str, Any] = {}
    try:
        now = time.time()
        if (now - _last_check_ts) < _CHECK_INTERVAL_SEC:
            return changes
        _last_check_ts = now

        if not _OVERRIDE_PATH.exists():
            return changes

        mtime = _OVERRIDE_PATH.stat().st_mtime
        if mtime <= _last_mtime:
            return changes
        _last_mtime = mtime

        raw = _OVERRIDE_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return changes
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            _log(f"CONFIG HOT-RELOAD: override file is not a JSON object, skipping")
            return changes

        # Get valid field names from the dataclass
        valid_fields = {f.name: f for f in dc_fields(cfg)} if hasattr(cfg, "__dataclass_fields__") else {}

        for key, value in overrides.items():
            if key in HOT_RELOAD_BLOCKED:
                _log(f"CONFIG HOT-RELOAD: BLOCKED field '{key}' — requires restart")
                continue

            is_dangerous = key in HOT_RELOAD_DANGEROUS
            is_safe = key in HOT_RELOAD_ALLOWED

            if not is_dangerous and not is_safe:
                _log(f"CONFIG HOT-RELOAD: unknown/disallowed field '{key}' — skipping")
                continue

            # Dangerous params blocked when HOT_RELOAD_DANGEROUS_CONFIRM=0
            if is_dangerous and not _DANGEROUS_CONFIRM_ENABLED:
                _log(f"CONFIG HOT-RELOAD: DANGEROUS field '{key}' BLOCKED — "
                     "set HOT_RELOAD_DANGEROUS_CONFIRM=1 to allow")
                continue
            if key not in valid_fields:
                continue

            old_value = getattr(cfg, key, None)
            # Type coerce to match the dataclass field type
            field_type = valid_fields[key].type
            try:
                new_value = _coerce(value, field_type, old_value)
            except Exception as e:
                _log(f"CONFIG HOT-RELOAD: type error for '{key}': {e}")
                continue

            if new_value != old_value:
                setattr(cfg, key, new_value)
                changes[key] = {"old": old_value, "new": new_value}
                if is_dangerous:
                    _log(f"CONFIG HOT-RELOAD: ⚠️ DANGEROUS FIELD '{key}': {old_value} → {new_value} — "
                         "risk model may be affected during active trading")
                    _audit_config_change(key, old_value, new_value, dangerous=True)
                else:
                    _log(f"CONFIG HOT-RELOAD: {key}: {old_value} → {new_value}")
                    _audit_config_change(key, old_value, new_value)

        if changes:
            _log(f"CONFIG HOT-RELOAD: applied {len(changes)} change(s)")

    except json.JSONDecodeError as e:
        _log(f"CONFIG HOT-RELOAD: JSON parse error — {e}")
    except Exception:
        pass
    return changes


def _coerce(value: Any, type_hint: str, current: Any) -> Any:
    """Coerce a JSON value to match the expected config field type."""
    if current is None:
        return value
    t = type(current)
    if t == bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes", "on")
    if t == int:
        return int(float(value))
    if t == float:
        return float(value)
    if t == str:
        return str(value)
    return value


def get_override_path() -> Path:
    """Return the path to the override file (for dashboard/API)."""
    return _OVERRIDE_PATH
