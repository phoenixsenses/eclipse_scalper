"""Deterministic liquidation-silence classification policy.

Distinguishes a genuinely quiet liquidation market from transport-specific
failure, at both per-symbol and all-tracked-symbol granularity, by
cross-validating liquidation-stream silence against control-stream
(mark_prices/agg_trades) advancement and the existing native-WS status.

Background: a confirmed ~40-day-3-hour all-tracked-symbol liquidation
outage (2026-04-27T14:24:51Z .. 2026-06-06T17:47:05Z -- ETHUSDT/BTCUSDT/
SOLUSDT all stopped within seconds of each other and recovered within
seconds of each other) was never surfaced as a decisive canonical health
failure. A second, shorter (~4-day) all-market liquidation outage
(2026-07-06T10:06:39Z .. 2026-07-10T11:24:37Z, a routed-WS-endpoint
regression, mechanism confirmed independent of the April-June gap) is
covered separately by tools/native_ws_health_policy.py's global
(all-744-symbol-table) liquidations freshness check. That existing check
has no per-symbol granularity and does not itself branch to an
evidence-insufficient state when control streams are also stale -- this
module adds both. See
reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md for full
calibration evidence (inter-arrival distributions, threshold derivation,
historical replay) and
reports/research/s34/S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_FORENSIC_2026-07-11.md
for the incident's forensic timeline (exact per-symbol boundaries) this
policy was calibrated against.

Pure decision function, no I/O -- fully unit-testable with fixtures.
tools/liquidation_silence_detector.py performs the bounded read-only
snapshot acquisition and writes the result to its own dedicated component
file (logs/health/liquidation_silence.json) via
tools.health_state.write_component_health. This module never writes
anything and never reads logs/health/overall.json, which remains owned
solely by tools/heartbeat_watchdog.py.

Disabled-by-default: nothing in this repository currently calls this
module's output into tools/heartbeat_watchdog.py's severity composition.
compose_with_overall_severity() below defines the INTENDED future mapping
for a separate, explicit controlled-activation batch; it is not invoked by
any production code path today.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Policy version / thresholds -- FROZEN. See
# reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md ("Frozen
# Detector Policy") for the calibration evidence behind every value below.
# Do not edit these without bumping POLICY_VERSION and re-running the full
# replay + false-positive accounting; POLICY_FINGERPRINT is derived from
# them automatically and any edit changes it.
# ---------------------------------------------------------------------------
POLICY_VERSION = "liquidation_silence_policy_v1_2026-07-11"

# Per-symbol silence: never alone escalates past YELLOW (a single tracked
# symbol going quiet is a normal, frequent market condition). Calibrated
# from combined post-2026-06-06 (current all_market_arr architecture)
# healthy per-symbol inter-arrival gaps across BTCUSDT/ETHUSDT/SOLUSDT:
# observed max 6733.2s (SOLUSDT), p999 3705.0s. 9000s gives ~1.34x margin
# over the observed historical maximum.
SYMBOL_SILENCE_WARNING_AGE_SEC = 9000.0  # 150 min

# All-tracked-symbol silence age = time since the FRESHEST of the tracked
# symbols last posted a liquidation (i.e. every tracked symbol has been
# silent at least this long simultaneously). Calibrated from a
# chronological freshest-of-3 reconstruction over ~19 days of combined
# post-2026-06-06 healthy data (2026-06-10/17, 2026-06-28/07-03,
# 2026-07-10 12:00/07-11 07:00 -- the 2026-07-06..07-10 routed-endpoint
# outage excluded): observed max 2508.9s, p999 1141.4s, n=34526 transitions,
# zero occurrences above 3600s in the entire calibration population.
ALL_SYMBOL_SILENCE_WARNING_AGE_SEC = 3600.0  # 60 min, ~1.43x observed max
ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC = 7200.0  # 120 min, ~2.87x observed max

# Control-stream ("is the data layer fundamentally alive") freshness budget.
# Deliberately coarser than tools/native_ws_health_policy.py's own
# MARK_PRICES/AGG_TRADES warning/critical tiers (30/120s, 60/180s) -- this
# module does not re-grade native-WS health (that remains
# native_ws_health_policy's ownership); it only needs a single conservative
# "still fundamentally advancing" gate to decide whether liquidation
# silence can be attributed to a liquidation-specific transport problem
# rather than a general data-layer outage. 300s carries ~1.67x margin over
# native_ws_health_policy's own AGG_TRADES_CRITICAL_AGE_SEC=180s boundary,
# and is well above the isolated ~1737s/1674s mark/trade staleness blip
# observed at 2026-05-21T16:26Z *inside* the confirmed April-June outage
# window (see calibration report Part C) -- proving this gate correctly
# reclassifies that one moment as CONTROL_STREAMS_STALE rather than
# over-claiming LIQUIDATION_TRANSPORT_OUTAGE when the evidence does not
# support attributing silence specifically to the liquidation path.
CONTROL_STREAM_FRESH_AGE_SEC = 300.0  # 5 min


def _policy_frozen_params() -> Dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "symbol_silence_warning_age_sec": SYMBOL_SILENCE_WARNING_AGE_SEC,
        "all_symbol_silence_warning_age_sec": ALL_SYMBOL_SILENCE_WARNING_AGE_SEC,
        "all_symbol_silence_critical_age_sec": ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC,
        "control_stream_fresh_age_sec": CONTROL_STREAM_FRESH_AGE_SEC,
    }


def _compute_policy_fingerprint() -> str:
    canonical = json.dumps(_policy_frozen_params(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


POLICY_FINGERPRINT = _compute_policy_fingerprint()

# --- Classification enum ----------------------------------------------------
STATUS_HEALTHY = "HEALTHY"
STATUS_SYMBOL_SILENCE_WARNING = "SYMBOL_SILENCE_WARNING"
STATUS_ALL_SYMBOL_SILENCE_WARNING = "ALL_SYMBOL_SILENCE_WARNING"
STATUS_LIQUIDATION_TRANSPORT_OUTAGE = "LIQUIDATION_TRANSPORT_OUTAGE"
STATUS_CONTROL_STREAMS_STALE = "CONTROL_STREAMS_STALE"
STATUS_NATIVE_WS_UNHEALTHY = "NATIVE_WS_UNHEALTHY"
STATUS_UNKNOWN = "UNKNOWN_INSUFFICIENT_EVIDENCE"

# --- Severity (independent of tools/heartbeat_watchdog.py's own RED/
# DEGRADED/GREEN or ok/degraded/halted vocabularies -- this module owns its
# own severity axis, mapped for a future controlled activation by
# compose_with_overall_severity() below.) ------------------------------------
SEVERITY_GREEN = "GREEN"
SEVERITY_YELLOW = "YELLOW"
SEVERITY_RED = "RED"
SEVERITY_UNKNOWN = "UNKNOWN"

_STATUS_SEVERITY = {
    STATUS_HEALTHY: SEVERITY_GREEN,
    STATUS_SYMBOL_SILENCE_WARNING: SEVERITY_YELLOW,
    STATUS_ALL_SYMBOL_SILENCE_WARNING: SEVERITY_YELLOW,
    STATUS_LIQUIDATION_TRANSPORT_OUTAGE: SEVERITY_RED,
    STATUS_CONTROL_STREAMS_STALE: SEVERITY_UNKNOWN,
    STATUS_NATIVE_WS_UNHEALTHY: SEVERITY_RED,
    STATUS_UNKNOWN: SEVERITY_UNKNOWN,
}

# --- Reason codes ------------------------------------------------------------
REASON_NO_TRACKED_SYMBOLS = "NO_TRACKED_SYMBOLS"
REASON_ALL_SYMBOL_EVIDENCE_MISSING = "ALL_SYMBOL_EVIDENCE_MISSING"
REASON_PARTIAL_SYMBOL_EVIDENCE_MISSING = "PARTIAL_SYMBOL_EVIDENCE_MISSING"
REASON_COLLECTOR_PROCESS_MISSING = "COLLECTOR_PROCESS_MISSING"
REASON_NATIVE_WS_RED_PRESERVED = "NATIVE_WS_RED_PRESERVED"
REASON_NATIVE_WS_DEGRADED_PRESERVED = "NATIVE_WS_DEGRADED_PRESERVED"
REASON_CONTROL_EVIDENCE_MISSING = "CONTROL_EVIDENCE_MISSING"
REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL = "ALL_SYMBOL_SILENCE_BEYOND_CRITICAL"
REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING = "ALL_SYMBOL_SILENCE_BEYOND_WARNING"
REASON_CONTROLS_ADVANCING = "CONTROLS_ADVANCING"
REASON_CONTROLS_ALSO_STALE = "CONTROLS_ALSO_STALE"
REASON_SYMBOL_SILENCE = "SYMBOL_SILENCE"
REASON_MALFORMED_INPUT = "MALFORMED_INPUT"


def evaluate_liquidation_silence(
    *,
    now_ts: float,
    tracked_symbols: List[str],
    last_liquidation_ts_ms: Optional[Dict[str, Optional[int]]],
    mark_prices_age_sec: Optional[float],
    agg_trades_age_sec: Optional[float],
    native_ws_status: Optional[str] = None,
    collector_process_alive: Optional[bool] = None,
) -> Dict[str, Any]:
    """Pure decision function. All timestamps/ages are plain numbers so this
    is fully unit-testable with fixtures and safe to replay against
    historical data with an arbitrary now_ts (never the real wall clock).

    tracked_symbols: the canonical runtime symbol universe (discovered
        by the caller from logs/pids/collector_supervisor.json or
        equivalent -- never hardcoded here).
    last_liquidation_ts_ms: per-symbol last-liquidation epoch-ms, or None
        for a symbol whose evidence could not be read (malformed/missing
        query result) -- distinct from "key absent" which is treated the
        same way.
    """
    reasons: List[str] = []

    if not isinstance(tracked_symbols, list) or not tracked_symbols:
        return _result(STATUS_UNKNOWN, [REASON_NO_TRACKED_SYMBOLS], now_ts, {}, None, None, None, None)

    if not isinstance(last_liquidation_ts_ms, dict):
        return _result(STATUS_UNKNOWN, [REASON_MALFORMED_INPUT], now_ts, {}, None, None, None, None)

    if native_ws_status == "RED":
        return _result(
            STATUS_NATIVE_WS_UNHEALTHY,
            [REASON_NATIVE_WS_RED_PRESERVED],
            now_ts, {}, None, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            severity_override=SEVERITY_RED,
        )

    if collector_process_alive is False:
        return _result(
            STATUS_UNKNOWN, [REASON_COLLECTOR_PROCESS_MISSING], now_ts, {}, None,
            mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
        )

    per_symbol_age: Dict[str, Optional[float]] = {}
    missing_symbols: List[str] = []
    for sym in tracked_symbols:
        raw = last_liquidation_ts_ms.get(sym) if sym in last_liquidation_ts_ms else None
        if raw is None:
            missing_symbols.append(sym)
            per_symbol_age[sym] = None
        else:
            try:
                per_symbol_age[sym] = max(0.0, now_ts - float(raw) / 1000.0)
            except (TypeError, ValueError):
                missing_symbols.append(sym)
                per_symbol_age[sym] = None

    known_ages = {s: a for s, a in per_symbol_age.items() if a is not None}

    if not known_ages:
        return _result(
            STATUS_UNKNOWN, [REASON_ALL_SYMBOL_EVIDENCE_MISSING], now_ts, per_symbol_age, None,
            mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
        )
    if missing_symbols:
        reasons.append(REASON_PARTIAL_SYMBOL_EVIDENCE_MISSING + ":" + ",".join(sorted(missing_symbols)))

    all_symbol_age = min(known_ages.values())
    controls_evidence_present = mark_prices_age_sec is not None and agg_trades_age_sec is not None
    controls_advancing = bool(
        controls_evidence_present
        and mark_prices_age_sec < CONTROL_STREAM_FRESH_AGE_SEC
        and agg_trades_age_sec < CONTROL_STREAM_FRESH_AGE_SEC
    )

    # --- all-symbol silence beyond CRITICAL ---------------------------------
    if all_symbol_age >= ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC:
        if not controls_evidence_present:
            return _result(
                STATUS_UNKNOWN,
                reasons + [REASON_CONTROL_EVIDENCE_MISSING, REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            )
        if controls_advancing:
            return _result(
                STATUS_LIQUIDATION_TRANSPORT_OUTAGE,
                reasons + [REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL, REASON_CONTROLS_ADVANCING],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            )
        return _result(
            STATUS_CONTROL_STREAMS_STALE,
            reasons + [REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL, REASON_CONTROLS_ALSO_STALE],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
        )

    # --- all-symbol silence beyond WARNING -----------------------------------
    if all_symbol_age >= ALL_SYMBOL_SILENCE_WARNING_AGE_SEC:
        if not controls_evidence_present:
            return _result(
                STATUS_UNKNOWN,
                reasons + [REASON_CONTROL_EVIDENCE_MISSING, REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            )
        if not controls_advancing:
            return _result(
                STATUS_CONTROL_STREAMS_STALE,
                reasons + [REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING, REASON_CONTROLS_ALSO_STALE],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            )
        return _result(
            STATUS_ALL_SYMBOL_SILENCE_WARNING,
            reasons + [REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING, REASON_CONTROLS_ADVANCING],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
        )

    # --- per-symbol-only silence ----------------------------------------------
    silent_symbols = sorted(s for s, a in known_ages.items() if a >= SYMBOL_SILENCE_WARNING_AGE_SEC)
    if silent_symbols:
        return _result(
            STATUS_SYMBOL_SILENCE_WARNING,
            reasons + [REASON_SYMBOL_SILENCE + ":" + ",".join(silent_symbols)],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
        )

    if native_ws_status == "DEGRADED":
        return _result(
            STATUS_NATIVE_WS_UNHEALTHY,
            reasons + [REASON_NATIVE_WS_DEGRADED_PRESERVED],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            severity_override=SEVERITY_YELLOW,
        )

    return _result(
        STATUS_HEALTHY, reasons, now_ts, per_symbol_age, all_symbol_age,
        mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
    )


def _result(
    status: str,
    reasons: List[str],
    now_ts: float,
    per_symbol_age: Dict[str, Optional[float]],
    all_symbol_age: Optional[float],
    mark_prices_age_sec: Optional[float],
    agg_trades_age_sec: Optional[float],
    native_ws_status: Optional[str],
    severity_override: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "severity": severity_override or _STATUS_SEVERITY[status],
        "reasons": sorted(set(reasons)),
        "per_symbol_silence_age_sec": dict(per_symbol_age),
        "all_symbol_silence_age_sec": all_symbol_age,
        "mark_prices_age_sec": mark_prices_age_sec,
        "agg_trades_age_sec": agg_trades_age_sec,
        "native_ws_status": native_ws_status,
        "thresholds": {
            "symbol_silence_warning_age_sec": SYMBOL_SILENCE_WARNING_AGE_SEC,
            "all_symbol_silence_warning_age_sec": ALL_SYMBOL_SILENCE_WARNING_AGE_SEC,
            "all_symbol_silence_critical_age_sec": ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC,
            "control_stream_fresh_age_sec": CONTROL_STREAM_FRESH_AGE_SEC,
        },
        "policy_version": POLICY_VERSION,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "evaluated_at_ts": now_ts,
    }


# ---------------------------------------------------------------------------
# INTENDED future composition (NOT invoked by any production code path in
# this batch -- see module docstring). Defines how a future controlled-
# activation batch should fold this detector's severity into
# tools/heartbeat_watchdog.py's existing RED/YELLOW/GREEN top-level state,
# without ever downgrading a more severe verdict already in force.
# ---------------------------------------------------------------------------
_OVERALL_RANK = {SEVERITY_GREEN: 0, SEVERITY_UNKNOWN: 1, SEVERITY_YELLOW: 1, SEVERITY_RED: 2}


def compose_with_overall_severity(existing_overall_severity: str, detector_severity: str) -> str:
    """existing_overall_severity/detector_severity both in {GREEN, YELLOW,
    RED, UNKNOWN}. Returns the composed overall severity under the
    intended future mapping: detector YELLOW/UNKNOWN raises overall to at
    least YELLOW, detector RED forces overall RED, detector GREEN never
    downgrades an existing more-severe verdict. Pure function -- no
    heartbeat_watchdog.py call site exists for this yet."""
    a = _OVERALL_RANK.get(existing_overall_severity, 2)
    b = _OVERALL_RANK.get(detector_severity, 2)
    if a >= b:
        return existing_overall_severity
    return SEVERITY_YELLOW if b == 1 else SEVERITY_RED
