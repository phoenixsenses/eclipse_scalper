# config/hot_reload.py — Runtime config hot-reload from JSON override file
# Guardian-safe: never raises, never fatal

from __future__ import annotations

import json
import os
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Optional, Set

_OVERRIDE_PATH = Path(os.environ.get("CONFIG_OVERRIDE_PATH", "config/runtime_overrides.json"))

# Fields that are SAFE to hot-reload at runtime (thresholds, limits, timeouts)
HOT_RELOAD_ALLOWED: Set[str] = {
    # Risk limits
    "MAX_RISK_PER_TRADE",
    "MAX_PORTFOLIO_HEAT",
    "MAX_CONCURRENT_POSITIONS",
    "MAX_DAILY_LOSS_PCT",
    "MIN_RISK_DOLLARS",
    "CORRELATION_HEAT_CAP",
    "SESSION_EQUITY_PEAK_PROTECTION_PCT",
    "VELOCITY_DRAWDOWN_PCT",
    "VELOCITY_MINUTES",
    # Entry thresholds
    "MIN_CONFIDENCE",
    "MIN_CONFIDENCE_HIGH_VOL",
    "ENTRY_MIN_CONFIDENCE",
    "ENTRY_POLL_SEC",
    "ENTRY_PER_SYMBOL_GAP_SEC",
    "ENTRY_LOCAL_COOLDOWN_SEC",
    "ENTRY_ROUTER_RETRIES",
    "MIN_ATR_PCT_FOR_ENTRY",
    # Sizing
    "FIXED_NOTIONAL_USDT",
    "FIXED_QTY",
    "MIN_ENTRY_QTY",
    "MIN_NOTIONAL_USDT",
    "MIN_MARGIN_USDT",
    # Stop/TP
    "STOP_ATR_MULT",
    "MAX_STOP_PCT",
    "BREAKEVEN_BUFFER_ATR_MULT",
    "TP1_RR_MULT",
    "TP2_RR_MULT",
    "TRAILING_ACTIVATION_RR",
    "TRAILING_CALLBACK_RATE",
    "TRAILING_TIGHT_PCT",
    "TRAILING_LOOSE_PCT",
    # Funding filters
    "MAX_FUNDING_LONG",
    "MIN_FUNDING_SHORT",
    # Kill switch
    "KILL_SWITCH_COOLDOWN_SEC",
    "KILL_MAX_DATA_STALENESS_SEC",
    "KILL_MAX_API_ERROR_RATE",
    "KILL_MAX_API_ERROR_BURST",
    # Notifications
    "NOTIFY_ON_ENTRY",
    "NOTIFY_ON_EXIT",
    "NOTIFY_ON_BREAKEVEN",
    "NOTIFY_ON_BLACKLIST",
    "ENTRY_NOTIFY",
    # Cooldowns
    "SYMBOL_COOLDOWN_MINUTES",
    "CONSECUTIVE_LOSS_BLACKLIST_COUNT",
    "SYMBOL_BLACKLIST_DURATION_HOURS",
    # Order retries
    "MAX_ORDER_RETRIES",
    "ORDER_RETRY_SLEEP_SEC",
    # Execution
    "SLIPPAGE_MAX_PCT",
    "MIN_FILL_RATIO",
}

# Fields that must NEVER be hot-reloaded (structural, require restart)
HOT_RELOAD_BLOCKED: Set[str] = {
    "ACTIVE_SYMBOLS",       # needs data loop restart
    "LEVERAGE",             # affects open positions
    "TIMEFRAME",            # needs data re-fetch
    "TIMEFRAME_5M",
    "TIMEFRAME_15M",
    "KILL_SWITCH_ENABLED",  # safety-critical toggle
    "CONFIG_VERSION",
    "CONFIG_FORGED_DATE",
    "TRADING_HOURS_UTC",    # needs entry loop restart
}

_last_mtime: float = 0.0
_last_check_ts: float = 0.0
_CHECK_INTERVAL_SEC: float = 10.0  # don't stat() more than once per 10s


def _log(msg: str) -> None:
    try:
        from utils.logging import log_core
        log_core.info(msg)
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
            if key not in HOT_RELOAD_ALLOWED:
                _log(f"CONFIG HOT-RELOAD: unknown/disallowed field '{key}' — skipping")
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
                _log(f"CONFIG HOT-RELOAD: {key}: {old_value} → {new_value}")

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
