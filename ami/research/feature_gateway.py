"""BATCH-P6-001: mandatory feature gateway (FABLE REVIEW A F3 / REVIEW B F-B5).

Phase 6 research engines must not query ami_events/ami_chart tables (or any
ad-hoc script formula) directly. This module is the ONLY sanctioned entry
point:

  - fetch_events() / fetch_cycles(): fetch_events refuses to return a
    population that mixes REAL_LIQUIDATION with a PROXY_* source_quality
    (R-09/CONFLICT-008), via ami.identity.event_identity.assert_not_pooled.
    fetch_cycles returns ami_cycles rows (canonical-v1-only by construction).
  - fetch_level_features(): refuses to release ami_levels.touch_count /
    rejection_count / acceptance_count / last_touch_ts / strength_score
    while touch_stats_point_in_time=0 (F-B2 -- these are a single
    cumulative build-time aggregate, not a point-in-time-safe feature).
  - Every successful fetch records a researcher_exposure_ledger row
    (Phase 2 table, not a new ledger) so population assembly is auditable.
  - Only the five Phase 4 chart tables + ami_events/ami_cycles are
    queryable through this module; an unknown table name is rejected
    outright rather than silently proxied through to raw SQL.

This does not retroactively change any existing research script -- it is
the mandatory path for Phase 6+ work going forward (SCHEMA_DICTIONARY.md
"Phase 5 duplicate-engine denetimi" rule).
"""
from __future__ import annotations
import time
import uuid

from ami.identity.event_identity import SourceQuality, assert_not_pooled

KNOWN_FEATURE_TABLES = {
    "ami_events", "ami_cycles", "ami_candles", "ami_candle_morphology",
    "ami_swings", "ami_levels", "ami_pushes", "event_cycle_membership",
    "ami_candidate_universe",
    # BATCH-P7B-CANON: lifecycle/path tables -- all three require dedicated
    # fetch_* functions below (same precedent as ami_events/ami_levels), so
    # fetch_chart_feature() explicitly rejects them (see its own guard).
    "ami_signal_lifecycle", "ami_lifecycle_effective_transitions",
    "ami_lifecycle_path_observations",
}

# F-B2: blocked until ami_levels.touch_stats_point_in_time=1 rows exist.
_BLOCKED_LEVEL_COLUMNS = {
    "touch_count", "rejection_count", "acceptance_count", "last_touch_ts", "strength_score",
}
_SAFE_LEVEL_COLUMNS = {
    "level_id", "symbol", "level_type", "price", "origin_ts", "known_at_ts",
    "timeframe", "source_type", "level_definition_version",
}


class FeatureGatewayViolation(Exception):
    """Raised on pooled-population, blocked-column, or unknown-table/column access."""


class AmbiguousPathVersionError(FeatureGatewayViolation):
    """BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING (GOAL A).

    Raised by fetch_path_observations() when the rows matching a query span
    more than one path_definition_version and the caller neither pinned an
    exact version (equals={"path_definition_version": ...}) nor passed
    effective=True. ami_lifecycle_path_observations can legitimately hold
    more than one physical version of the same (signal_id, horizon_name) pair
    -- ami.lifecycle.path_candle_repair_correction documents path observations
    as "a re-derivable materialization, NOT append-only", and its targeted
    correction batch adds new rows under a new path_definition_version
    WITHOUT deleting the original ones. A caller that does not explicitly
    resolve this ambiguity must never silently receive a population that
    mixes physical versions -- that is exactly how a corrected (signal,
    horizon) pair gets double-counted, or how a frozen historical population
    silently drifts as new corrected rows are added later.

    Fix: for a genuine historical/versioned reproduction, pass
    equals={"path_definition_version": "<exact version>", ...}. For current,
    corrected-data research, use
    ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations()
    (which calls this function with effective=True internally, after
    reducing to exactly one row per (signal_id, horizon_name))."""


def _record_exposure(conn, research_context_id: str, exposure_category: str,
                      provenance: str = "batch-p6-001-feature-gateway") -> None:
    # uuid4, not a content/timestamp hash: exposure rows are an append-only
    # audit log (one row per fetch call), not an idempotent upsert keyed
    # entity -- multiple fetches in the same wave (or same millisecond) must
    # each get a distinct id rather than collide on a UNIQUE constraint.
    now = int(time.time() * 1000)
    exposure_id = "EXP-" + uuid.uuid4().hex[:24]
    conn.execute(
        "INSERT INTO researcher_exposure_ledger (exposure_id, claim_or_hypothesis_id, exposure_category, "
        "splits_seen_count, thresholds_seen_count, route_variants_seen_count, reports_seen_count, "
        "manual_override_log, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,0,0,0,0,NULL,?,?,?,?)",
        (exposure_id, research_context_id, exposure_category, 5, provenance, now, now),
    )
    conn.commit()


def fetch_events(conn, research_context_id: str, symbol: str | None = None,
                  source_quality: str | None = None) -> list[dict]:
    """Returns ami_events rows. Raises FeatureGatewayViolation if the
    resulting population mixes REAL_LIQUIDATION with any PROXY_* status."""
    cols = ["event_id", "symbol", "event_family", "anchor_ts_ms", "source_quality",
            "censor_status", "event_count", "notional", "event_end_ts_ms"]
    q = f"SELECT {', '.join(cols)} FROM ami_events"
    clauses, params = [], []
    if symbol is not None:
        clauses.append("symbol=?")
        params.append(symbol)
    if source_quality is not None:
        clauses.append("source_quality=?")
        params.append(source_quality)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(q, params).fetchall()
    qualities = [r[4] for r in rows]
    try:
        assert_not_pooled(qualities)
    except Exception as exc:  # PooledPopulationViolation
        raise FeatureGatewayViolation(str(exc)) from exc

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return [dict(zip(cols, r)) for r in rows]


def fetch_cycles(conn, research_context_id: str, symbol: str | None = None) -> list[dict]:
    """Returns ami_cycles rows (already canonical-v1-only by construction --
    only ami/identity/cycle_resolver.py writes this table)."""
    cols = ["cycle_id", "symbol", "start_ts_ms", "end_ts_ms", "cycle_definition_version",
            "event_count", "direction_conflict", "censored", "confidence"]
    q = f"SELECT {', '.join(cols)} FROM ami_cycles"
    params: list = []
    if symbol is not None:
        q += " WHERE symbol=?"
        params.append(symbol)
    rows = conn.execute(q, params).fetchall()

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return [dict(zip(cols, r)) for r in rows]


_LIFECYCLE_SIGNAL_COLUMNS = [
    "signal_id", "setup_id", "source_event_id", "independent_cycle_id", "symbol", "direction",
    "signal_birth_ts", "lifecycle_status", "lifecycle_reason_code", "evidence_layer", "is_proxy",
    "executability_status",
]


def fetch_lifecycle_signals(conn, research_context_id: str, symbol: str | None = None,
                             evidence_layer: str | None = None) -> list[dict]:
    """Returns ami_signal_lifecycle rows restricted to a curated,
    non-bookkeeping column set (excludes schema_version/provenance/
    created_at/updated_ms/identity_version/source_hash/code_commit). Raises
    FeatureGatewayViolation if the resulting population mixes
    evidence_layer='REAL' with 'PROXY' -- same pooling discipline as
    fetch_events()'s assert_not_pooled, but using ami_signal_lifecycle's own
    REAL/PROXY vocabulary (ami.lifecycle.canonical_schema.EvidenceLayer, a
    DIFFERENT enum from ami_events.source_quality's REAL_LIQUIDATION/PROXY_*
    -- never conflated, so ami.identity.event_identity.assert_not_pooled is
    not reused here)."""
    q = f"SELECT {', '.join(_LIFECYCLE_SIGNAL_COLUMNS)} FROM ami_signal_lifecycle"
    clauses, params = [], []
    if symbol is not None:
        clauses.append("symbol=?")
        params.append(symbol)
    if evidence_layer is not None:
        clauses.append("evidence_layer=?")
        params.append(evidence_layer)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(q, params).fetchall()
    out = [dict(zip(_LIFECYCLE_SIGNAL_COLUMNS, r)) for r in rows]
    layers = {r["evidence_layer"] for r in out}
    if "REAL" in layers and "PROXY" in layers:
        raise FeatureGatewayViolation(
            "population mixes ami_signal_lifecycle.evidence_layer='REAL' with 'PROXY' -- "
            "refusing to pool REAL and PROXY-derived signals in one research population"
        )

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return out


_EFFECTIVE_TRANSITION_COLUMNS = [
    "transition_id", "signal_id", "previous_status", "new_status", "transition_ts", "known_at_ts",
    "reason_code", "transition_version", "observation_mode", "correction_of",
]


def fetch_lifecycle_effective_transitions(conn, research_context_id: str,
                                           signal_id: str | None = None) -> list[dict]:
    """Queries ami_lifecycle_effective_transitions -- the corrections/
    supersessions-applied VIEW (ami.lifecycle.canonical_schema) -- NEVER the
    raw ami_lifecycle_transitions table directly. This is the enforcement
    point that keeps any research reading lifecycle transitions through this
    gateway from ever treating a superseded/reversed TERMINAL_CLOSE row as
    valid terminal evidence (the same discipline
    count_effective_closed_signals()/effective_lifecycle_status() already
    apply for status/interval queries)."""
    q = f"SELECT {', '.join(_EFFECTIVE_TRANSITION_COLUMNS)} FROM ami_lifecycle_effective_transitions"
    params: list = []
    if signal_id is not None:
        q += " WHERE signal_id=?"
        params.append(signal_id)
    rows = conn.execute(q, params).fetchall()

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return [dict(zip(_EFFECTIVE_TRANSITION_COLUMNS, r)) for r in rows]


_PATH_OBSERVATION_COLUMNS = [
    "observation_id", "signal_id", "horizon_name", "horizon_end_ts", "known_at_ts", "as_of_ts",
    "observation_status", "volatility_status", "reference_price", "reference_price_ts",
    "effective_path_start_ts", "endpoint_return_bps", "mfe_bps", "mae_bps", "time_to_mfe_ms",
    "time_to_mae_ms", "intrabar_order_status", "realized_vol_at_anchor",
    "endpoint_return_anchor_vol_units", "mfe_anchor_vol_units", "mae_anchor_vol_units",
    "horizon_outcome_class", "expected_candle_count", "observed_candle_count", "gap_count",
    "candle_definition_version", "path_definition_version",
]
_PATH_OBSERVATION_FILTERS = {
    "signal_id", "horizon_name", "observation_status", "volatility_status", "path_definition_version",
}


def fetch_path_observations(conn, research_context_id: str, equals: dict | None = None,
                             effective: bool = False) -> list[dict]:
    """Returns ami_lifecycle_path_observations rows restricted to a curated,
    non-bookkeeping column set (excludes observation_mode/provenance/
    schema_version/created_ms). `equals` is an allowlist-restricted
    {column: value} filter (same convention as fetch_chart_feature's
    equals=). No pooling guard is needed here (unlike fetch_events/
    fetch_lifecycle_signals) -- path observations are a derived
    measurement layer, already point-in-time-safe by construction
    (known_at_ts/as_of_ts columns self-describe each row).

    [BATCH: AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING,
    GOAL A] Repository-wide fail-closed path-version contract: if the rows
    matching this query span more than one path_definition_version, the
    caller must have EITHER pinned an exact version
    (equals={"path_definition_version": ...}) OR passed effective=True --
    otherwise this raises AmbiguousPathVersionError rather than silently
    returning a population that mixes physical versions. `effective=True` is
    reserved for the effective-path selector
    (ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations),
    which performs its own reduction to exactly one row per (signal_id,
    horizon_name) immediately after this call -- it is not a general escape
    hatch for ordinary consumers. This function never guesses "latest" by
    lexical version ordering; ambiguity is always resolved explicitly by the
    caller, never inferred."""
    equals = equals or {}
    unknown_filters = set(equals) - _PATH_OBSERVATION_FILTERS
    if unknown_filters:
        raise FeatureGatewayViolation(
            f"non-filterable column(s) for ami_lifecycle_path_observations: {unknown_filters}"
        )
    clauses, params = [], []
    for col, val in equals.items():
        clauses.append(f"{col}=?")
        params.append(val)
    q = f"SELECT {', '.join(_PATH_OBSERVATION_COLUMNS)} FROM ami_lifecycle_path_observations"
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(q, params).fetchall()
    result = [dict(zip(_PATH_OBSERVATION_COLUMNS, r)) for r in rows]

    if "path_definition_version" not in equals and not effective:
        distinct_versions = {r["path_definition_version"] for r in result}
        if len(distinct_versions) > 1:
            raise AmbiguousPathVersionError(
                f"fetch_path_observations(research_context_id={research_context_id!r}) matched rows "
                f"spanning {len(distinct_versions)} distinct path_definition_version values "
                f"({sorted(distinct_versions)}). A consumer must explicitly pin "
                "equals={'path_definition_version': '<exact version>'} for a historical/versioned read, "
                "or pass effective=True (reserved for the effective-path selector -- see "
                "ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations) to "
                "intentionally read across physical versions. Refusing to silently mix physical path rows."
            )

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return result


def fetch_level_features(conn, research_context_id: str, columns: list[str],
                          symbol: str | None = None) -> list[dict]:
    """Returns ami_levels rows restricted to `columns`. Raises
    FeatureGatewayViolation if any requested column is unknown or is a
    build-time-only aggregate (F-B2) not yet point-in-time-safe."""
    unknown = set(columns) - _SAFE_LEVEL_COLUMNS - _BLOCKED_LEVEL_COLUMNS
    if unknown:
        raise FeatureGatewayViolation(f"unknown ami_levels column(s): {unknown}")
    blocked = set(columns) & _BLOCKED_LEVEL_COLUMNS
    if blocked:
        raise FeatureGatewayViolation(
            f"ami_levels column(s) {blocked} are build-time cumulative aggregates, not point-in-time-safe "
            "(F-B2) -- blocked until touch_stats_point_in_time=1 rows exist"
        )

    q = f"SELECT {', '.join(columns)} FROM ami_levels"
    params: list = []
    if symbol is not None:
        q += " WHERE symbol=?"
        params.append(symbol)
    rows = conn.execute(q, params).fetchall()

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return [dict(zip(columns, r)) for r in rows]


_FILTERABLE_COLUMNS = {
    "ami_candles": {"symbol", "timeframe"},
    "ami_candle_morphology": set(),
    "ami_swings": {"symbol", "timeframe", "swing_type"},
    "ami_pushes": {"symbol", "timeframe", "direction"},
    "event_cycle_membership": {"cycle_definition_version", "is_canonical", "event_id"},
    "ami_candidate_universe": {"symbol", "timeframe", "is_event_aligned", "universe_definition_version"},
}


def fetch_chart_feature(conn, research_context_id: str, table: str, columns: list[str],
                        symbol: str | None = None, equals: dict | None = None) -> list[dict]:
    """Generic fetch for ami_candles/ami_candle_morphology/ami_swings/ami_pushes/
    event_cycle_membership. ami_levels and ami_events must go through their
    dedicated functions above (they carry extra safety rules). `equals` is an
    optional {column: value} filter restricted to a per-table allowlist
    (_FILTERABLE_COLUMNS) so column names are never interpolated from
    caller-controlled strings without validation."""
    if table in ("ami_levels", "ami_events", "ami_signal_lifecycle",
                 "ami_lifecycle_effective_transitions", "ami_lifecycle_path_observations"):
        raise FeatureGatewayViolation(f"{table} must be accessed via its dedicated fetch_* function")
    if table not in KNOWN_FEATURE_TABLES:
        raise FeatureGatewayViolation(f"unknown feature table {table!r} -- not in ami/chart canonical set")

    equals = equals or {}
    allowed = _FILTERABLE_COLUMNS.get(table, set())
    unknown_filters = set(equals) - allowed
    if unknown_filters:
        raise FeatureGatewayViolation(f"non-filterable column(s) for {table}: {unknown_filters}")

    clauses, params = [], []
    if symbol is not None:
        clauses.append("symbol=?")
        params.append(symbol)
    for col, val in equals.items():
        clauses.append(f"{col}=?")
        params.append(val)

    q = f"SELECT {', '.join(columns)} FROM {table}"
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(q, params).fetchall()

    _record_exposure(conn, research_context_id, "BLINDLY_PREREGISTERED")
    return [dict(zip(columns, r)) for r in rows]
