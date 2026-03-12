"""
Cross-module health dashboard aggregator.

Collects status snapshots from every active module and writes them to
logs/health/dashboard.json every DASHBOARD_EXPORT_INTERVAL_SEC seconds
(default 30).  Never raises -- all errors are caught and surfaced as
partial data under an "errors" key.

Usage (from guardian.py)
-------------------------
    from execution.dashboard_aggregator import dashboard_export_tick
    await dashboard_export_tick(bot)

    # or inline with interval guard:
    if callable(dashboard_export_tick):
        await _safe_call("dashboard_aggregator.tick", dashboard_export_tick, bot)

ENV
---
    DASHBOARD_EXPORT_INTERVAL_SEC=30   (0 = every guardian cycle)
    DASHBOARD_EXPORT_PATH=logs/health/dashboard.json
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_INTERVAL_SEC = int(os.getenv("DASHBOARD_EXPORT_INTERVAL_SEC", "30") or "30")
_DEFAULT_PATH = os.getenv("DASHBOARD_EXPORT_PATH", "logs/health/dashboard.json")

_last_export_ts: float = 0.0


# ---------------------------------------------------------------------------
# Sub-collectors (each catches its own exceptions)
# ---------------------------------------------------------------------------

def _collect_health(bot) -> dict:
    try:
        from execution.health_monitor import HealthMonitor  # type: ignore
        hm = HealthMonitor.get(bot)
        return hm.get_summary()
    except Exception as exc:
        return {"error": str(exc)}


def _collect_metrics() -> dict:
    try:
        from execution.metrics_collector import MetricsCollector  # type: ignore
        mc = MetricsCollector.get()
        return mc.export()
    except Exception as exc:
        return {"error": str(exc)}


def _collect_risk(bot) -> dict:
    try:
        rc = getattr(getattr(bot, "state", None), "run_context", None) or {}
        risk_mgr = rc.get("regime_risk_manager")
        if risk_mgr is None:
            return {}
        return risk_mgr.state_dict()
    except Exception as exc:
        return {"error": str(exc)}


def _collect_event_gate(bot) -> dict:
    try:
        from execution.event_lane_gate import load_current_event_gate  # type: ignore
        db = os.getenv(
            "ENTRY_EVENT_LANE_GATE_DB",
            os.getenv("MICRO_SIGNAL_DB", str(getattr(getattr(bot, "cfg", None), "DB_PATH", "") or "")),
        )
        if not db:
            return {}
        gate = load_current_event_gate(db=db, symbol="ETHUSDT")
        return dict(gate) if gate else {}
    except Exception as exc:
        return {"error": str(exc)}


def _collect_regime_watch() -> dict:
    try:
        from tools.watch_regime_recovery import evaluate_status  # type: ignore
        return evaluate_status()
    except Exception as exc:
        return {"error": str(exc)}


def _collect_pocket_scheduler(bot) -> dict:
    try:
        rc = getattr(getattr(bot, "state", None), "run_context", None) or {}
        sched = rc.get("pocket_scheduler")
        if sched is None:
            return {}
        return sched.state_dict()
    except Exception as exc:
        return {"error": str(exc)}


def _collect_guardian_steps() -> dict:
    """Phase 5.1: Collect guardian step timings and failure counts."""
    result: Dict[str, Any] = {}
    try:
        from execution.guardian import get_step_timings, get_step_failures
        timings = get_step_timings()
        failures = get_step_failures()
        result["step_timings"] = timings
        result["step_failures"] = failures
        result["total_failures"] = sum(failures.values()) if failures else 0
        if timings:
            slowest = max(timings.items(), key=lambda x: x[1].get("avg_ms", 0))
            result["slowest_step"] = slowest[0]
            result["slowest_avg_ms"] = slowest[1].get("avg_ms", 0)
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_system_dashboard(bot) -> Dict[str, Any]:
    """Collect a snapshot from all modules. Never raises."""
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "ts_epoch": time.time(),
        "health":          _collect_health(bot),
        "metrics":         _collect_metrics(),
        "risk_state":      _collect_risk(bot),
        "event_gate":      _collect_event_gate(bot),
        "regime_watch":    _collect_regime_watch(),
        "pocket_scheduler": _collect_pocket_scheduler(bot),
        "guardian_steps":  _collect_guardian_steps(),
    }


# ---------------------------------------------------------------------------
# Export tick (async-safe, called from guardian)
# ---------------------------------------------------------------------------

async def dashboard_export_tick(bot) -> None:
    """
    Write the dashboard snapshot to disk if the interval has elapsed.
    Called from guardian._one_cycle() after metrics tick.
    """
    global _last_export_ts
    now = time.time()
    if _INTERVAL_SEC > 0 and (now - _last_export_ts) < _INTERVAL_SEC:
        return

    data = get_system_dashboard(bot)
    _write_dashboard(data)
    _last_export_ts = now


def _write_dashboard(data: dict) -> None:
    path = Path(_DEFAULT_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass  # never fatal
