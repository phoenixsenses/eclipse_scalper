"""Deterministic liquidation-silence classification policy (v2, corrective).

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

--- v2 corrective changes (2026-07-11, independent-review corrective pass) --
1. COMPLETE-EVIDENCE GATING: an "all tracked symbols silent" claim
   (ALL_SYMBOL_SILENCE_WARNING / LIQUIDATION_TRANSPORT_OUTAGE) now requires
   usable evidence for EVERY tracked symbol. If any tracked symbol's
   evidence is missing while the known symbols are silent beyond
   warning/critical, the classification is PARTIAL_SYMBOL_EVIDENCE /
   UNKNOWN -- it must not over-claim an all-symbol transport outage from an
   incomplete symbol set (review MEDIUM 1: BTC/ETH critical + SOL missing
   previously mis-classified as LIQUIDATION_TRANSPORT_OUTAGE / RED).
2. DECISION-LOGIC FINGERPRINT: POLICY_FINGERPRINT is now the sha256 of a
   full deterministic POLICY_SPEC object covering every decision-critical
   semantic (precedence, boundary operators, aggregation method, complete-
   evidence requirement, native-WS/collector precedence, future-timestamp
   tolerance, schema version, ...) -- not just the four numeric thresholds
   plus a manually maintained version string (review MEDIUM 2). A material
   logic change now necessarily changes the fingerprint.
3. FUTURE-TIMESTAMP HANDLING: source timestamps in the future relative to
   now_ts within FUTURE_TIMESTAMP_TOLERANCE_SEC are clamped to age 0 (minor
   clock skew, tolerated); beyond tolerance they are recorded as an
   evidence anomaly and the reading is treated as unusable (not silently
   clamped to a healthy age-0) (review LOW).
4. SYMBOL NORMALIZATION: tracked symbols are de-duplicated / whitespace- and
   case-normalized / empties dropped before evaluation; an empty normalized
   universe fails visibly (review LOW).

Frozen numeric thresholds (3600/7200/9000/300) are UNCHANGED from v1; only
the surrounding evidence semantics were corrected.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Policy version / thresholds -- FROZEN. See
# reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md ("Frozen
# Detector Policy") for the calibration evidence behind every value below.
# The numeric thresholds are UNCHANGED across v1->v2; POLICY_FINGERPRINT is
# derived from the full POLICY_SPEC (see below) and changes for any semantic
# policy modification, not only a threshold edit.
# ---------------------------------------------------------------------------
POLICY_VERSION = "liquidation_silence_policy_v2_2026-07-11"

# Superseded v1 fingerprint (thresholds-only sha256), retained for governance
# provenance. The v2 POLICY_FINGERPRINT below MUST differ from this because
# the decision semantics changed in the corrective pass.
SUPERSEDED_V1_POLICY_VERSION = "liquidation_silence_policy_v1_2026-07-11"
SUPERSEDED_V1_POLICY_FINGERPRINT = (
    "9781e0ed8f7b4950e62bdb6b4e64773ef1f9f6e383749b92ac20641dec4ed9d8"
)

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
# chronological freshest-of-3 reconstruction over 307 hours of included
# healthy coverage (~12.8 elapsed days across 14 distinct calendar days) of
# combined post-2026-06-06 healthy data (2026-06-10/17, 2026-06-28/07-03,
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

# Clock-skew tolerance for source timestamps that land in the future
# relative to the evaluation instant (now_ts). Within this window a future
# timestamp is treated as age 0 (benign skew); beyond it the reading is
# recorded as an evidence anomaly and treated as unusable rather than
# silently clamped to a healthy age-0. Applied consistently to liquidation,
# control-stream, and component timestamps.
FUTURE_TIMESTAMP_TOLERANCE_SEC = 60.0

# Boundary operator semantics (recorded in POLICY_SPEC so a change to any of
# them changes POLICY_FINGERPRINT).
_OP_SILENCE = ">="  # age >= threshold triggers the silence tier
_OP_CONTROL_FRESH = "<"  # control age < fresh-budget counts as advancing

OUTPUT_SCHEMA_VERSION = "liquidation_silence_policy_output_v2"

# --- Classification enum ----------------------------------------------------
STATUS_HEALTHY = "HEALTHY"
STATUS_SYMBOL_SILENCE_WARNING = "SYMBOL_SILENCE_WARNING"
STATUS_ALL_SYMBOL_SILENCE_WARNING = "ALL_SYMBOL_SILENCE_WARNING"
STATUS_LIQUIDATION_TRANSPORT_OUTAGE = "LIQUIDATION_TRANSPORT_OUTAGE"
STATUS_CONTROL_STREAMS_STALE = "CONTROL_STREAMS_STALE"
STATUS_NATIVE_WS_UNHEALTHY = "NATIVE_WS_UNHEALTHY"
STATUS_PARTIAL_SYMBOL_EVIDENCE = "PARTIAL_SYMBOL_EVIDENCE"
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
    STATUS_PARTIAL_SYMBOL_EVIDENCE: SEVERITY_UNKNOWN,
    STATUS_UNKNOWN: SEVERITY_UNKNOWN,
}

# --- Reason codes ------------------------------------------------------------
REASON_NO_TRACKED_SYMBOLS = "NO_TRACKED_SYMBOLS"
REASON_EMPTY_NORMALIZED_UNIVERSE = "EMPTY_NORMALIZED_UNIVERSE"
REASON_SYMBOLS_NORMALIZED = "SYMBOLS_NORMALIZED"
REASON_ALL_SYMBOL_EVIDENCE_MISSING = "ALL_SYMBOL_EVIDENCE_MISSING"
REASON_PARTIAL_SYMBOL_EVIDENCE_MISSING = "PARTIAL_SYMBOL_EVIDENCE_MISSING"
REASON_KNOWN_SYMBOL_SILENCE_BEYOND_WARNING = "KNOWN_SYMBOL_SILENCE_BEYOND_WARNING"
REASON_KNOWN_SYMBOL_SILENCE_BEYOND_CRITICAL = "KNOWN_SYMBOL_SILENCE_BEYOND_CRITICAL"
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
REASON_FUTURE_LIQUIDATION_TS = "FUTURE_LIQUIDATION_TS_ANOMALY"
REASON_FUTURE_CONTROL_TS = "FUTURE_CONTROL_TS_ANOMALY"


def normalize_tracked_symbols(raw: Any) -> Dict[str, Any]:
    """Canonicalize a raw tracked-symbol universe before policy evaluation.

    - upper-cases and strips whitespace,
    - drops empty / non-string entries,
    - de-duplicates while preserving deterministic first-seen order.

    Returns {"symbols": [...], "dropped": [...], "raw_count": n}. An empty
    final universe is reported (the caller decides to fail visibly) -- there
    is no silent hardcoded fallback here (the detector's documented
    last-resort fallback lives in tools/liquidation_silence_detector.py and
    is explicitly reported via symbol_source there)."""
    symbols: List[str] = []
    dropped: List[str] = []
    seen = set()
    items = raw if isinstance(raw, list) else []
    for entry in items:
        if not isinstance(entry, str):
            dropped.append(repr(entry))
            continue
        norm = entry.strip().upper()
        if not norm:
            dropped.append(repr(entry))
            continue
        if norm in seen:
            # de-duplicated silently only in the sense of not re-adding; the
            # fact that de-dup happened is still surfaced via raw_count vs len.
            continue
        seen.add(norm)
        symbols.append(norm)
    return {"symbols": symbols, "dropped": dropped, "raw_count": len(items)}


def _age_from_ts_ms(raw: Any, now_ts: float) -> Tuple[Optional[float], Optional[str]]:
    """Compute a source age (seconds) from an epoch-ms timestamp with
    explicit future-timestamp handling.

    Returns (age_sec, anomaly). anomaly is:
      - None with a usable non-negative age (future within tolerance clamps
        to 0.0),
      - "MISSING" when raw is None,
      - "MALFORMED" when raw cannot be parsed as a number,
      - "FUTURE" when raw is in the future beyond
        FUTURE_TIMESTAMP_TOLERANCE_SEC (unusable; not clamped to a healthy
        age 0)."""
    if raw is None:
        return None, "MISSING"
    try:
        ts = float(raw) / 1000.0
    except (TypeError, ValueError):
        return None, "MALFORMED"
    age = now_ts - ts
    if age >= 0.0:
        return age, None
    if age >= -FUTURE_TIMESTAMP_TOLERANCE_SEC:
        return 0.0, None  # benign clock skew, tolerated
    return None, "FUTURE"


def _assess_controls(
    mark_prices_age_sec: Optional[float], agg_trades_age_sec: Optional[float]
) -> Tuple[bool, bool, List[str]]:
    """Decide whether the control streams constitute usable, advancing
    evidence. Handles future (negative-age) control timestamps: within
    tolerance a small future skew is treated as age 0; beyond tolerance the
    control reading is anomalous and cannot count as 'advancing'.

    Returns (evidence_present, advancing, anomaly_reasons)."""
    anomalies: List[str] = []
    present = mark_prices_age_sec is not None and agg_trades_age_sec is not None
    if not present:
        return False, False, anomalies

    def _eff(age: float, name: str) -> Optional[float]:
        if age >= 0.0:
            return age
        if age >= -FUTURE_TIMESTAMP_TOLERANCE_SEC:
            return 0.0
        anomalies.append(REASON_FUTURE_CONTROL_TS + ":" + name)
        return None

    eff_mark = _eff(float(mark_prices_age_sec), "mark_prices")
    eff_agg = _eff(float(agg_trades_age_sec), "agg_trades")
    if eff_mark is None or eff_agg is None:
        # An anomalous future control reading cannot be trusted as advancing.
        return True, False, anomalies
    advancing = eff_mark < CONTROL_STREAM_FRESH_AGE_SEC and eff_agg < CONTROL_STREAM_FRESH_AGE_SEC
    return True, advancing, anomalies


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
        equivalent -- never hardcoded here). Normalized/de-duplicated here.
    last_liquidation_ts_ms: per-symbol last-liquidation epoch-ms, or None
        for a symbol whose evidence could not be read (malformed/missing
        query result) -- distinct from "key absent" which is treated the
        same way.
    """
    reasons: List[str] = []

    # --- input validation ----------------------------------------------------
    if not isinstance(tracked_symbols, list):
        return _result(STATUS_UNKNOWN, [REASON_MALFORMED_INPUT], now_ts, {}, None,
                       None, None, None, tracked_symbol_count=0, known_symbol_count=0,
                       missing_symbols=[], complete_symbol_evidence=False)

    norm = normalize_tracked_symbols(tracked_symbols)
    symbols = norm["symbols"]
    if not symbols:
        return _result(
            STATUS_UNKNOWN, [REASON_NO_TRACKED_SYMBOLS, REASON_EMPTY_NORMALIZED_UNIVERSE],
            now_ts, {}, None, None, None, None,
            tracked_symbol_count=0, known_symbol_count=0, missing_symbols=[],
            complete_symbol_evidence=False,
        )
    if norm["raw_count"] != len(symbols) or norm["dropped"]:
        reasons.append(REASON_SYMBOLS_NORMALIZED)

    if not isinstance(last_liquidation_ts_ms, dict):
        return _result(
            STATUS_UNKNOWN, reasons + [REASON_MALFORMED_INPUT], now_ts, {}, None,
            None, None, None, tracked_symbol_count=len(symbols), known_symbol_count=0,
            missing_symbols=list(symbols), complete_symbol_evidence=False,
        )

    # --- upstream precedence: native WS RED (preserved, never re-derived) -----
    if native_ws_status == "RED":
        return _result(
            STATUS_NATIVE_WS_UNHEALTHY, reasons + [REASON_NATIVE_WS_RED_PRESERVED],
            now_ts, {}, None, mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            severity_override=SEVERITY_RED, tracked_symbol_count=len(symbols),
            known_symbol_count=0, missing_symbols=[], complete_symbol_evidence=False,
        )

    if collector_process_alive is False:
        return _result(
            STATUS_UNKNOWN, reasons + [REASON_COLLECTOR_PROCESS_MISSING], now_ts, {}, None,
            mark_prices_age_sec, agg_trades_age_sec, native_ws_status,
            tracked_symbol_count=len(symbols), known_symbol_count=0,
            missing_symbols=list(symbols), complete_symbol_evidence=False,
        )

    # --- per-symbol ages (with future-timestamp handling) --------------------
    per_symbol_age: Dict[str, Optional[float]] = {}
    missing_symbols: List[str] = []
    anomalies: List[str] = []
    for sym in symbols:
        raw = last_liquidation_ts_ms.get(sym)
        age, anomaly = _age_from_ts_ms(raw, now_ts)
        if anomaly is not None:
            missing_symbols.append(sym)
            per_symbol_age[sym] = None
            if anomaly == "FUTURE":
                anomalies.append(REASON_FUTURE_LIQUIDATION_TS + ":" + sym)
        else:
            per_symbol_age[sym] = age

    known_ages = {s: a for s, a in per_symbol_age.items() if a is not None}
    complete_symbol_evidence = len(missing_symbols) == 0
    counts = dict(
        tracked_symbol_count=len(symbols),
        known_symbol_count=len(known_ages),
        missing_symbols=sorted(missing_symbols),
        complete_symbol_evidence=complete_symbol_evidence,
    )

    if not known_ages:
        return _result(
            STATUS_UNKNOWN, reasons + anomalies + [REASON_ALL_SYMBOL_EVIDENCE_MISSING],
            now_ts, per_symbol_age, None, mark_prices_age_sec, agg_trades_age_sec,
            native_ws_status, **counts,
        )
    if missing_symbols:
        reasons.append(REASON_PARTIAL_SYMBOL_EVIDENCE_MISSING + ":" + ",".join(sorted(missing_symbols)))

    controls_present, controls_advancing, control_anomalies = _assess_controls(
        mark_prices_age_sec, agg_trades_age_sec
    )
    anomalies += control_anomalies

    # all_symbol_silence_age is only a meaningful "all tracked symbols have
    # been simultaneously silent this long" quantity when evidence is
    # complete. When incomplete we still report the freshest-known age (for
    # transparency) but never use it to assert an all-symbol claim.
    freshest_known_age = min(known_ages.values())
    all_symbol_age = freshest_known_age if complete_symbol_evidence else None

    # --- all-symbol silence beyond CRITICAL ---------------------------------
    if freshest_known_age >= ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC:
        if not complete_symbol_evidence:
            return _result(
                STATUS_PARTIAL_SYMBOL_EVIDENCE,
                reasons + anomalies + [REASON_KNOWN_SYMBOL_SILENCE_BEYOND_CRITICAL],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
                agg_trades_age_sec, native_ws_status, **counts,
            )
        if not controls_present:
            return _result(
                STATUS_UNKNOWN,
                reasons + anomalies + [REASON_CONTROL_EVIDENCE_MISSING, REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
                agg_trades_age_sec, native_ws_status, **counts,
            )
        if controls_advancing:
            return _result(
                STATUS_LIQUIDATION_TRANSPORT_OUTAGE,
                reasons + anomalies + [REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL, REASON_CONTROLS_ADVANCING],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
                agg_trades_age_sec, native_ws_status, **counts,
            )
        return _result(
            STATUS_CONTROL_STREAMS_STALE,
            reasons + anomalies + [REASON_ALL_SYMBOL_SILENCE_BEYOND_CRITICAL, REASON_CONTROLS_ALSO_STALE],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
            agg_trades_age_sec, native_ws_status, **counts,
        )

    # --- all-symbol silence beyond WARNING -----------------------------------
    if freshest_known_age >= ALL_SYMBOL_SILENCE_WARNING_AGE_SEC:
        if not complete_symbol_evidence:
            return _result(
                STATUS_PARTIAL_SYMBOL_EVIDENCE,
                reasons + anomalies + [REASON_KNOWN_SYMBOL_SILENCE_BEYOND_WARNING],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
                agg_trades_age_sec, native_ws_status, **counts,
            )
        if not controls_present:
            return _result(
                STATUS_UNKNOWN,
                reasons + anomalies + [REASON_CONTROL_EVIDENCE_MISSING, REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
                agg_trades_age_sec, native_ws_status, **counts,
            )
        if not controls_advancing:
            return _result(
                STATUS_CONTROL_STREAMS_STALE,
                reasons + anomalies + [REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING, REASON_CONTROLS_ALSO_STALE],
                now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
                agg_trades_age_sec, native_ws_status, **counts,
            )
        return _result(
            STATUS_ALL_SYMBOL_SILENCE_WARNING,
            reasons + anomalies + [REASON_ALL_SYMBOL_SILENCE_BEYOND_WARNING, REASON_CONTROLS_ADVANCING],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
            agg_trades_age_sec, native_ws_status, **counts,
        )

    # --- per-symbol-only silence ----------------------------------------------
    silent_symbols = sorted(s for s, a in known_ages.items() if a >= SYMBOL_SILENCE_WARNING_AGE_SEC)
    if silent_symbols:
        return _result(
            STATUS_SYMBOL_SILENCE_WARNING,
            reasons + anomalies + [REASON_SYMBOL_SILENCE + ":" + ",".join(silent_symbols)],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
            agg_trades_age_sec, native_ws_status, **counts,
        )

    if native_ws_status == "DEGRADED":
        return _result(
            STATUS_NATIVE_WS_UNHEALTHY,
            reasons + anomalies + [REASON_NATIVE_WS_DEGRADED_PRESERVED],
            now_ts, per_symbol_age, all_symbol_age, mark_prices_age_sec,
            agg_trades_age_sec, native_ws_status, severity_override=SEVERITY_YELLOW, **counts,
        )

    return _result(
        STATUS_HEALTHY, reasons + anomalies, now_ts, per_symbol_age, all_symbol_age,
        mark_prices_age_sec, agg_trades_age_sec, native_ws_status, **counts,
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
    *,
    tracked_symbol_count: int,
    known_symbol_count: int,
    missing_symbols: List[str],
    complete_symbol_evidence: bool,
    severity_override: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "severity": severity_override or _STATUS_SEVERITY[status],
        "reasons": sorted(set(reasons)),
        "per_symbol_silence_age_sec": dict(per_symbol_age),
        "all_symbol_silence_age_sec": all_symbol_age,
        "tracked_symbol_count": tracked_symbol_count,
        "known_symbol_count": known_symbol_count,
        "missing_symbols": list(missing_symbols),
        "complete_symbol_evidence": complete_symbol_evidence,
        "mark_prices_age_sec": mark_prices_age_sec,
        "agg_trades_age_sec": agg_trades_age_sec,
        "native_ws_status": native_ws_status,
        "thresholds": {
            "symbol_silence_warning_age_sec": SYMBOL_SILENCE_WARNING_AGE_SEC,
            "all_symbol_silence_warning_age_sec": ALL_SYMBOL_SILENCE_WARNING_AGE_SEC,
            "all_symbol_silence_critical_age_sec": ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC,
            "control_stream_fresh_age_sec": CONTROL_STREAM_FRESH_AGE_SEC,
            "future_timestamp_tolerance_sec": FUTURE_TIMESTAMP_TOLERANCE_SEC,
        },
        "policy_version": POLICY_VERSION,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "evaluated_at_ts": now_ts,
    }


# ---------------------------------------------------------------------------
# Decision-logic policy specification + fingerprint (v2).
#
# POLICY_SPEC is a fully deterministic description of every decision-critical
# semantic, not just the numeric thresholds. POLICY_FINGERPRINT = sha256 over
# its canonical (sort_keys) JSON. Any semantic change (a boundary operator,
# the classification precedence, the complete-evidence requirement, the
# native-WS precedence, the aggregation method, the future-timestamp
# tolerance, the output schema version, ...) changes the fingerprint -- a
# material logic change can no longer occur without moving it (review MEDIUM
# 2). sort_keys makes it insensitive to key ordering / whitespace.
# ---------------------------------------------------------------------------
def _policy_spec() -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "thresholds": {
            "symbol_silence_warning_age_sec": SYMBOL_SILENCE_WARNING_AGE_SEC,
            "all_symbol_silence_warning_age_sec": ALL_SYMBOL_SILENCE_WARNING_AGE_SEC,
            "all_symbol_silence_critical_age_sec": ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC,
            "control_stream_fresh_age_sec": CONTROL_STREAM_FRESH_AGE_SEC,
            "future_timestamp_tolerance_sec": FUTURE_TIMESTAMP_TOLERANCE_SEC,
        },
        "boundary_operators": {
            "silence_tier": _OP_SILENCE,
            "control_fresh": _OP_CONTROL_FRESH,
        },
        "age_clamping": {
            "future_within_tolerance": "clamp_to_zero",
            "future_beyond_tolerance": "anomaly_unusable",
            "negative_past": "not_applicable",
        },
        "complete_evidence_required_for_all_symbol_claim": True,
        "all_symbol_aggregation_method": "min_over_tracked_symbol_ages_requires_complete_evidence",
        "per_symbol_classification_rule": "known_symbol_age >= symbol_silence_warning -> SYMBOL_SILENCE_WARNING (never RED alone)",
        "decision_precedence": [
            "input_validation",
            "symbol_normalization",
            "native_ws_red_preserved",
            "collector_process_missing",
            "all_symbol_evidence_missing",
            "all_symbol_silence_critical",
            "all_symbol_silence_warning",
            "per_symbol_silence",
            "native_ws_degraded_preserved",
            "healthy",
        ],
        "native_ws_precedence": {
            "RED": "preserved_red_before_liquidation_analysis",
            "DEGRADED": "preserved_yellow_only_when_no_liquidation_specific_evidence",
        },
        "collector_precedence": "collector_process_alive_false -> unknown_before_liquidation_analysis",
        "stale_control_behavior": "all_symbol_silence + controls_not_advancing -> control_streams_stale/unknown (never transport_outage)",
        "missing_symbol_behavior": "any_tracked_symbol_missing -> no all-symbol claim; known-silence-beyond-tier -> partial_symbol_evidence/unknown",
        "missing_control_behavior": "material_silence + controls_absent -> unknown (never transport_outage)",
        "corrupt_evidence_behavior": "unparseable_symbol_ts -> treated_as_missing; future_beyond_tolerance -> anomaly_unusable",
        "unknown_severity_behavior": "unknown/control_stale/partial_symbol_evidence -> SEVERITY_UNKNOWN (fail-visible, not green)",
        "status_severity_map": dict(_STATUS_SEVERITY),
        "severity_composition_rank": "detector RED forces overall RED; YELLOW/UNKNOWN raise to >= YELLOW; GREEN never downgrades",
    }


def _fingerprint_of(spec: Dict[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


POLICY_SPEC = _policy_spec()
POLICY_FINGERPRINT = _fingerprint_of(POLICY_SPEC)


def _compute_policy_fingerprint() -> str:
    """Backwards-compatible accessor: recomputes the current policy
    fingerprint from POLICY_SPEC. Kept for existing callers/tests; the
    fingerprint is now derived from the full decision spec, not just the
    numeric thresholds."""
    return _fingerprint_of(_policy_spec())


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
