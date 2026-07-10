"""Deterministic RED/DEGRADED/GREEN policy for the microstructure collector's
native WebSocket health, independent of REST-fallback-masked freshness.

Background: the 2026-07-06..2026-07-10 outage (see
reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md) stayed
"watchdog_overall=GREEN" for ~4 days because the existing health chain only
checked a blended "data_progressing" flag (native OR REST) and never looked at
native-only signals (last_message_ts_utc, liquidation_transport_available) or
per-table DB freshness at all. This module adds that missing evaluation as a
pure, unit-testable function; tools/heartbeat_watchdog.py merges its output
additively into the existing WATCHDOG_STATUS.json / logs/health/watchdog.json
payloads without changing any existing field.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Thresholds -- see reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md
# ("Monitoring follow-up") and the acceptance-review report for the reasoning
# behind each value; summarized inline below.
# ---------------------------------------------------------------------------

# Matches MicrostructureCollector.stall_timeout_sec default (data/microstructure_collector.py).
STALL_TIMEOUT_SEC = 45.0
# Matches MicrostructureCollector.heartbeat_write_interval_sec default. The collector's
# _stats_loop refreshes the heartbeat JSON on this cadence independent of the (much
# longer, 300s-by-default) console/log stats-print cadence -- see the "Heartbeat
# persistence design" note in this corrective batch. This is what bounds how stale
# last_message_ts_utc can read during genuinely healthy operation; without this
# decoupling that bound would be the 300s stats interval instead, which is why the
# threshold below is small rather than a multiple of the stats interval.
HEARTBEAT_WRITE_INTERVAL_SEC = 15.0
# Matches heartbeat_watchdog's own --interval-sec default.
WATCHDOG_POLL_INTERVAL_SEC = 30.0
# Matches --rest-poll-interval-sec.
REST_POLL_INTERVAL_SEC = 5.0
# Matches MicrostructureCollector._reconnect_base -- a backoff above this means the
# collector is actively cycling reconnects rather than sustaining a stable connection.
RECONNECT_BASE_SEC = 1.0

# Warning threshold = stall timeout + one heartbeat-write cadence of margin (jitter
# guard so a read landing exactly at a write boundary doesn't false-positive).
# Worst-case detection latency once frames genuinely stop: STALL_TIMEOUT_SEC (collector
# notices + writes immediately) + WATCHDOG_POLL_INTERVAL_SEC (next poll) ~= 75s, well
# under "several minutes" and within "one watchdog evaluation interval plus a small
# scheduling allowance" of the 45s stall firing.
NATIVE_WS_MESSAGE_WARNING_AGE_SEC = STALL_TIMEOUT_SEC + HEARTBEAT_WRITE_INTERVAL_SEC  # 60s
NATIVE_WS_MESSAGE_CRITICAL_AGE_SEC = 4 * STALL_TIMEOUT_SEC  # 180s

# Brief REST substitution during a single reconnect cycle should not itself be
# flagged; roughly one stall-timeout cycle.
REST_FALLBACK_GRACE_SEC = 60.0

AGG_TRADES_WARNING_AGE_SEC = 60.0
AGG_TRADES_CRITICAL_AGE_SEC = 180.0
MARK_PRICES_WARNING_AGE_SEC = 30.0
MARK_PRICES_CRITICAL_AGE_SEC = 120.0

# Liquidations are naturally sparser than markPrice/aggTrade -- a 45s threshold would
# be noisy. These values are derived from two independent bounded rowid-range reads
# (not full-table scans) of ~90-120 minutes each of known-healthy, global (all-symbol,
# matching the !forceOrder@arr subscription scope) liquidation activity immediately
# preceding the 2026-07-06 outage: observed max gap 77.1s / 75.4s, p99 ~30-34s, zero
# gaps over 300s in either sample. Thresholds below carry an ~8x and ~23x margin over
# that observed maximum rather than reusing the unrelated downstream
# s34_v_engine_v02_shadow_mirror.CLOSE_GRACE_SEC=3600 bucket-finalization constant.
# Deliberately global (not per-required-symbol): a single quiet symbol is normal and
# must not be flagged; only market-wide silence on the all-market stream is evidence
# of a transport problem, which is what read_source_freshness measures.
LIQUIDATIONS_WARNING_AGE_SEC = 600.0   # 10 min
LIQUIDATIONS_CRITICAL_AGE_SEC = 1800.0  # 30 min

STATUS_GREEN = "GREEN"
STATUS_DEGRADED = "DEGRADED"
STATUS_RED = "RED"
_SEVERITY_RANK = {STATUS_GREEN: 0, STATUS_DEGRADED: 1, STATUS_RED: 2}

NATIVE_WS_DISCONNECTED = "NATIVE_WS_DISCONNECTED"
NATIVE_WS_MESSAGE_STALE = "NATIVE_WS_MESSAGE_STALE"
NATIVE_WS_MESSAGE_CRITICALLY_STALE = "NATIVE_WS_MESSAGE_CRITICALLY_STALE"
RECONNECT_BACKOFF_ACTIVE = "RECONNECT_BACKOFF_ACTIVE"
REST_FALLBACK_ACTIVE = "REST_FALLBACK_ACTIVE"
LIQUIDATIONS_STALE = "LIQUIDATIONS_STALE"
LIQUIDATIONS_CRITICALLY_STALE = "LIQUIDATIONS_CRITICALLY_STALE"
AGG_TRADES_STALE = "AGG_TRADES_STALE"
AGG_TRADES_CRITICALLY_STALE = "AGG_TRADES_CRITICALLY_STALE"
MARK_PRICES_STALE = "MARK_PRICES_STALE"
MARK_PRICES_CRITICALLY_STALE = "MARK_PRICES_CRITICALLY_STALE"
COLLECTOR_PROCESS_MISSING = "COLLECTOR_PROCESS_MISSING"
HEALTH_STATE_UNREADABLE = "HEALTH_STATE_UNREADABLE"
ALL_COLLECTION_PATHS_STOPPED = "ALL_COLLECTION_PATHS_STOPPED"

_SOURCE_THRESHOLDS = {
    "agg_trades": (AGG_TRADES_WARNING_AGE_SEC, AGG_TRADES_CRITICAL_AGE_SEC, AGG_TRADES_STALE, AGG_TRADES_CRITICALLY_STALE),
    "mark_prices": (MARK_PRICES_WARNING_AGE_SEC, MARK_PRICES_CRITICAL_AGE_SEC, MARK_PRICES_STALE, MARK_PRICES_CRITICALLY_STALE),
    "liquidations": (LIQUIDATIONS_WARNING_AGE_SEC, LIQUIDATIONS_CRITICAL_AGE_SEC, LIQUIDATIONS_STALE, LIQUIDATIONS_CRITICALLY_STALE),
}


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _age_sec(raw: Any, now: datetime) -> Optional[float]:
    dt = _parse_ts(raw)
    if dt is None:
        return None
    return max(0.0, (now - dt).total_seconds())


def read_source_freshness(db_path: str | Path, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """Bounded, rowid-tail freshness read for the three collector-owned tables.

    Deliberately uses `ORDER BY rowid DESC LIMIT 1` (an O(log n) rowid-btree
    seek) instead of `SELECT MAX(ts_ms)` -- the latter defeated SQLite's rowid
    fast-path during this incident's own investigation and forced a full scan
    of a 400M+ row table. Never introduce a MAX(ts_ms)-style query here.
    """
    now_ts = float(now if now is not None else time.time())
    out: Dict[str, Dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except Exception as exc:
        for table in _SOURCE_THRESHOLDS:
            out[table] = {"last_ts_ms": None, "age_sec": None, "error": f"open_failed:{exc}"}
        return out
    try:
        cur = conn.cursor()
        for table in _SOURCE_THRESHOLDS:
            try:
                cur.execute(f"SELECT ts_ms FROM {table} ORDER BY rowid DESC LIMIT 1")
                row = cur.fetchone()
            except sqlite3.DatabaseError as exc:
                out[table] = {"last_ts_ms": None, "age_sec": None, "error": f"query_failed:{exc}"}
                continue
            if not row or row[0] is None:
                out[table] = {"last_ts_ms": None, "age_sec": None, "error": "no_rows"}
                continue
            last_ts_ms = int(row[0])
            age = max(0.0, now_ts - last_ts_ms / 1000.0)
            out[table] = {"last_ts_ms": last_ts_ms, "age_sec": age, "error": None}
    finally:
        conn.close()
    return out


def evaluate_policy(
    *,
    collector_heartbeat: Optional[Dict[str, Any]],
    collector_component: Optional[Dict[str, Any]],
    collector_process_alive: bool,
    source_freshness: Optional[Dict[str, Dict[str, Any]]],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Pure decision function -- no I/O. All inputs are plain dicts/bools so this
    is fully unit-testable with fixtures."""
    now = now or datetime.now(timezone.utc)
    reasons: List[str] = []

    if not collector_process_alive:
        return _result(STATUS_RED, [COLLECTOR_PROCESS_MISSING], collector_heartbeat, collector_component, source_freshness, now)

    hb = collector_heartbeat if isinstance(collector_heartbeat, dict) else None
    comp = collector_component if isinstance(collector_component, dict) else None
    if not hb or not comp:
        return _result(STATUS_RED, [HEALTH_STATE_UNREADABLE], collector_heartbeat, collector_component, source_freshness, now)

    connected = bool(hb.get("connected"))
    transport_connected = bool(comp.get("transport_connected"))
    rest_fallback_active = bool(hb.get("rest_fallback_active"))
    rest_last_progress_age = _age_sec(hb.get("rest_last_progress_ts_utc"), now)
    native_age = _age_sec(hb.get("last_message_ts_utc"), now)
    backoff = hb.get("current_backoff_seconds")
    try:
        backoff_f = float(backoff) if backoff is not None else None
    except Exception:
        backoff_f = None

    freshness = source_freshness if isinstance(source_freshness, dict) else {}

    # --- RED: all collection paths stopped -------------------------------------
    # ">=" throughout: the threshold value itself is already the first flagged
    # value (boundary tests assert DEGRADED/RED "at" the threshold, GREEN just
    # below it).
    native_dead = native_age is None or native_age >= NATIVE_WS_MESSAGE_CRITICAL_AGE_SEC
    rest_dead = (not rest_fallback_active) or rest_last_progress_age is None or rest_last_progress_age >= max(
        AGG_TRADES_CRITICAL_AGE_SEC, MARK_PRICES_CRITICAL_AGE_SEC
    )
    if native_dead and rest_dead:
        reasons.append(ALL_COLLECTION_PATHS_STOPPED)

    # --- RED: severe per-table staleness --------------------------------------
    for table, (warn_sec, crit_sec, warn_code, crit_code) in _SOURCE_THRESHOLDS.items():
        entry = freshness.get(table) or {}
        age = entry.get("age_sec")
        if age is not None and age >= crit_sec:
            reasons.append(crit_code)

    if reasons:
        return _result(STATUS_RED, reasons, collector_heartbeat, collector_component, source_freshness, now)

    # --- DEGRADED conditions ---------------------------------------------------
    if not connected:
        reasons.append(NATIVE_WS_DISCONNECTED)
    if not transport_connected and NATIVE_WS_DISCONNECTED not in reasons:
        reasons.append(NATIVE_WS_DISCONNECTED)
    if native_age is None or native_age >= NATIVE_WS_MESSAGE_WARNING_AGE_SEC:
        reasons.append(NATIVE_WS_MESSAGE_STALE)
    if rest_fallback_active and (native_age is None or native_age >= REST_FALLBACK_GRACE_SEC):
        reasons.append(REST_FALLBACK_ACTIVE)
    if backoff_f is not None and backoff_f > RECONNECT_BASE_SEC:
        # Strictly-greater here: RECONNECT_BASE_SEC itself is the normal,
        # non-backing-off value, not a staleness-style boundary.
        reasons.append(RECONNECT_BACKOFF_ACTIVE)

    for table, (warn_sec, crit_sec, warn_code, crit_code) in _SOURCE_THRESHOLDS.items():
        entry = freshness.get(table) or {}
        age = entry.get("age_sec")
        if age is not None and age >= warn_sec:
            reasons.append(warn_code)

    if reasons:
        return _result(STATUS_DEGRADED, reasons, collector_heartbeat, collector_component, source_freshness, now)

    return _result(STATUS_GREEN, [], collector_heartbeat, collector_component, source_freshness, now)


def _result(
    status: str,
    reasons: List[str],
    hb: Optional[Dict[str, Any]],
    comp: Optional[Dict[str, Any]],
    freshness: Optional[Dict[str, Dict[str, Any]]],
    now: datetime,
) -> Dict[str, Any]:
    hb = hb if isinstance(hb, dict) else {}
    comp = comp if isinstance(comp, dict) else {}
    freshness = freshness if isinstance(freshness, dict) else {}
    native_age = _age_sec(hb.get("last_message_ts_utc"), now)
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "native_websocket": {
            "connected": hb.get("connected"),
            "transport_connected": comp.get("transport_connected"),
            "last_message_ts_utc": hb.get("last_message_ts_utc"),
            "message_age_sec": native_age,
            "current_backoff_seconds": hb.get("current_backoff_seconds"),
            "last_error": hb.get("last_error"),
        },
        "rest_fallback": {
            "enabled": hb.get("rest_fallback_enabled"),
            "active": hb.get("rest_fallback_active"),
            "last_progress_ts_utc": hb.get("rest_last_progress_ts_utc"),
            "last_progress_age_sec": _age_sec(hb.get("rest_last_progress_ts_utc"), now),
        },
        "source_freshness": freshness,
        "thresholds": {
            "native_ws_message_warning_age_sec": NATIVE_WS_MESSAGE_WARNING_AGE_SEC,
            "native_ws_message_critical_age_sec": NATIVE_WS_MESSAGE_CRITICAL_AGE_SEC,
            "rest_fallback_grace_sec": REST_FALLBACK_GRACE_SEC,
            "agg_trades_warning_age_sec": AGG_TRADES_WARNING_AGE_SEC,
            "agg_trades_critical_age_sec": AGG_TRADES_CRITICAL_AGE_SEC,
            "mark_prices_warning_age_sec": MARK_PRICES_WARNING_AGE_SEC,
            "mark_prices_critical_age_sec": MARK_PRICES_CRITICAL_AGE_SEC,
            "liquidations_warning_age_sec": LIQUIDATIONS_WARNING_AGE_SEC,
            "liquidations_critical_age_sec": LIQUIDATIONS_CRITICAL_AGE_SEC,
        },
        "ts_utc": now.isoformat(),
    }
