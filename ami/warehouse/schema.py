"""Canonical warehouse schema — BATCH-P1-001.

Tables per Reconstruction Protocol §7.1 (registry/lineage/namespace) and §16
(question registry). Every table carries schema_version + provenance +
created_at (master protocol §12 / §14 discipline: no silent mutation of
canonical history).

This module defines DDL only. It does not touch any existing store
(data/microstructure.db, data/ami/knowledge.sqlite, data/ami/research.sqlite,
reports/shadow/*.jsonl) and does not affect any running process.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "ami" / "canonical.sqlite"

# [PHASE 7A-P TEST ISOLATION SAFETY CLOSURE] immutable reference to the REAL
# canonical.sqlite path, independent of any test-time monkeypatch of
# DEFAULT_PATH above -- connect()'s fail-closed guard below always compares
# against THIS constant (never the mutable DEFAULT_PATH), so redirecting
# DEFAULT_PATH for test isolation can never accidentally disable the guard.
REAL_CANONICAL_PATH_IMMUTABLE = DEFAULT_PATH

# Set True ONLY by tests/conftest.py's session-scoped isolation fixture --
# never set anywhere in production code/CLAUDE.md-sanctioned scripts. While
# True, connect() refuses any WRITABLE connection to
# REAL_CANONICAL_PATH_IMMUTABLE (fail-closed) -- see connect() below.
_TEST_ISOLATION_ACTIVE = False

CANONICAL_SCHEMA_VERSION = 13

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_versions (
    component TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    applied_ms INTEGER NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS artifact_registry (
    artifact_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    content_hash TEXT,
    role TEXT NOT NULL,
    canonical_status TEXT NOT NULL,
    namespace TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_lineage (
    lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL,
    predecessor_id TEXT,
    relation TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifact_registry(artifact_id)
);

CREATE TABLE IF NOT EXISTS question_families (
    family_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    q_range_lo INTEGER,
    q_range_hi INTEGER,
    source_document TEXT NOT NULL,
    implementation_phase INTEGER,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS question_registry (
    question_id TEXT PRIMARY KEY,
    canonical_parent TEXT,
    family_id TEXT,
    question_text TEXT,
    text_status TEXT NOT NULL,
    current_status TEXT NOT NULL,
    evidence_layer TEXT,
    blocked_by TEXT,
    implementation_phase INTEGER,
    permission_ceiling TEXT NOT NULL,
    final_verdict TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    FOREIGN KEY (family_id) REFERENCES question_families(family_id)
);

CREATE TABLE IF NOT EXISTS contradiction_registry (
    contradiction_id TEXT PRIMARY KEY,
    left_ref TEXT NOT NULL,
    right_ref TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    resolution TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS operator_decision_queue (
    decision_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    current_evidence TEXT,
    recommended_options TEXT,
    risk_of_each_option TEXT,
    required_approval TEXT,
    status TEXT NOT NULL,
    operator_response TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    implemented_ms INTEGER
);

CREATE TABLE IF NOT EXISTS namespace_registry (
    namespace TEXT PRIMARY KEY,
    meaning TEXT NOT NULL,
    is_ami_artificial_market_intelligence INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
"""

# Phase 2 — Evidence, timestamp and contamination integrity (whitepaper §70 / §77 Phase A).
_SCHEMA_PHASE2 = """
CREATE TABLE IF NOT EXISTS evidence_contamination (
    contamination_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    hypothesis_birth_ts INTEGER,
    hypothesis_origin_split TEXT,
    splits_seen_before_freeze TEXT,
    contaminated_splits TEXT,
    eligible_validation_splits TEXT,
    fresh_forward_required INTEGER NOT NULL DEFAULT 0,
    evidence_status TEXT NOT NULL,
    maximum_evidence_ceiling TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS researcher_exposure_ledger (
    exposure_id TEXT PRIMARY KEY,
    claim_or_hypothesis_id TEXT NOT NULL,
    exposure_category TEXT NOT NULL,
    splits_seen_count INTEGER NOT NULL DEFAULT 0,
    thresholds_seen_count INTEGER NOT NULL DEFAULT 0,
    route_variants_seen_count INTEGER NOT NULL DEFAULT 0,
    reports_seen_count INTEGER NOT NULL DEFAULT 0,
    manual_override_log TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS mt_family_registry (
    family_id TEXT PRIMARY KEY,
    variants_tested INTEGER NOT NULL DEFAULT 0,
    effective_trials INTEGER NOT NULL DEFAULT 0,
    family_adjusted_significance REAL,
    threshold_stability TEXT,
    researcher_freedom_score REAL,
    minimum_economic_effect REAL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS causal_assumption_registry (
    assumption_id TEXT PRIMARY KEY,
    claim_or_hypothesis_id TEXT NOT NULL,
    assumed_arrow TEXT NOT NULL,
    causal_status TEXT NOT NULL,
    possible_confounders TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_events (
    event_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detected_ms INTEGER NOT NULL,
    period_start_ms INTEGER,
    period_end_ms INTEGER,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS market_structure_versions (
    version_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,
    effective_from_ms INTEGER NOT NULL,
    effective_to_ms INTEGER,
    fee_schedule TEXT,
    contract_spec TEXT,
    tick_size TEXT,
    event_definition_version TEXT,
    notes TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
"""

# Phase 3 — Event/cycle/path foundation (Protocol §7.4/§8, Observatory §5,
# whitepaper §65). ami_cycles is intentionally left EMPTY by this batch: a
# canonical cycle_definition_version and reset/censoring policy require
# operator approval (OD-003) and are NOT decided here. event_cycle_membership
# holds only non-canonical sensitivity candidates (is_canonical=0).
_SCHEMA_PHASE3 = """
CREATE TABLE IF NOT EXISTS ami_events (
    event_id TEXT PRIMARY KEY,
    event_family TEXT NOT NULL,
    symbol TEXT NOT NULL,
    venue TEXT,
    liquidation_side TEXT,
    anchor_ts_ms INTEGER NOT NULL,
    event_start_ts_ms INTEGER,
    event_end_ts_ms INTEGER,
    feature_available_ts_ms INTEGER,
    notional REAL,
    event_count INTEGER,
    duration_seconds REAL,
    max_single_share REAL,
    displacement_bps REAL,
    structural_location TEXT,
    source_quality TEXT NOT NULL,
    event_definition_version TEXT NOT NULL,
    censor_status TEXT,
    source_artifact_id TEXT,
    route_version TEXT,
    collector_version TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ami_cycles (
    cycle_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    start_ts_ms INTEGER,
    end_ts_ms INTEGER,
    cycle_definition_version TEXT NOT NULL,
    entry_state TEXT,
    peak_state TEXT,
    exit_state TEXT,
    event_count INTEGER,
    direction_conflict INTEGER NOT NULL DEFAULT 0,
    censored TEXT,
    confidence REAL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_cycle_membership (
    membership_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    candidate_cycle_key TEXT NOT NULL,
    cycle_definition_version TEXT NOT NULL,
    is_canonical INTEGER NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    FOREIGN KEY (event_id) REFERENCES ami_events(event_id),
    UNIQUE (event_id, cycle_definition_version)
);
"""

# Phase 4 — Chart-native object foundation (Chart-Native Extension §4/§6/§7).
# Closed-candle-only: no PARTIAL_CANDLE row is ever stored here (Obs §6.3);
# a still-forming bucket is skipped by the builder, never written as CLOSED.
_SCHEMA_PHASE4 = """
CREATE TABLE IF NOT EXISTS ami_candles (
    candle_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    venue TEXT,
    timeframe TEXT NOT NULL,
    open_ts_ms INTEGER NOT NULL,
    close_ts_ms INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    trade_count INTEGER,
    taker_buy_volume REAL,
    taker_sell_volume REAL,
    is_closed INTEGER NOT NULL DEFAULT 1,
    partial_status TEXT NOT NULL DEFAULT 'CLOSED',
    known_at_ts INTEGER NOT NULL,
    data_quality TEXT NOT NULL,
    source_hash TEXT,
    candle_definition_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    UNIQUE (symbol, timeframe, open_ts_ms, candle_definition_version)
);

CREATE TABLE IF NOT EXISTS ami_candle_morphology (
    candle_id TEXT PRIMARY KEY,
    range_abs REAL,
    body_ratio REAL,
    upper_wick_ratio REAL,
    lower_wick_ratio REAL,
    close_location_value REAL,
    open_location_value REAL,
    directional_body REAL,
    close_quality_label TEXT,
    morphology_definition_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    FOREIGN KEY (candle_id) REFERENCES ami_candles(candle_id)
);

CREATE TABLE IF NOT EXISTS ami_swings (
    swing_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    swing_type TEXT NOT NULL,
    pivot_ts INTEGER NOT NULL,
    pivot_price REAL NOT NULL,
    confirmation_ts INTEGER NOT NULL,
    confirmation_method TEXT NOT NULL,
    prominence_bps REAL,
    duration_bars INTEGER,
    left_strength INTEGER,
    right_strength INTEGER,
    known_at_ts INTEGER NOT NULL,
    swing_definition_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ami_levels (
    level_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    level_type TEXT NOT NULL,
    price REAL NOT NULL,
    origin_ts INTEGER NOT NULL,
    known_at_ts INTEGER NOT NULL,
    timeframe TEXT NOT NULL,
    touch_count INTEGER NOT NULL DEFAULT 0,
    rejection_count INTEGER NOT NULL DEFAULT 0,
    acceptance_count INTEGER NOT NULL DEFAULT 0,
    last_touch_ts INTEGER,
    strength_score REAL,
    touch_stats_point_in_time INTEGER NOT NULL DEFAULT 0,
    source_type TEXT NOT NULL,
    level_definition_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ami_pushes (
    push_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    start_swing_id TEXT NOT NULL,
    end_swing_id TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    displacement_bps REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    bars INTEGER,
    path_length_bps REAL,
    efficiency_ratio REAL,
    volume REAL,
    liquidation_notional REAL,
    average_speed REAL,
    peak_speed REAL,
    acceleration REAL,
    pullback_after_bps REAL,
    known_at_ts INTEGER NOT NULL,
    push_definition_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    FOREIGN KEY (start_swing_id) REFERENCES ami_swings(swing_id),
    FOREIGN KEY (end_swing_id) REFERENCES ami_swings(swing_id)
);
"""

# Phase 6 — scientific governance (Protocol §7.2/§7.7). Previously planned
# but not yet built; needed now to write research-wave verdicts to canonical
# SQL rather than leaving them only in Markdown reports.
_SCHEMA_PHASE6 = """
CREATE TABLE IF NOT EXISTS experiment_registry (
    experiment_id TEXT PRIMARY KEY,
    question_ids TEXT,
    hypothesis_id TEXT,
    preregistered_at INTEGER,
    frozen_population TEXT NOT NULL,
    frozen_features TEXT,
    frozen_target TEXT,
    frozen_thresholds TEXT,
    frozen_splits TEXT,
    frozen_economic_gate TEXT,
    frozen_statistical_gate TEXT,
    code_commit TEXT,
    dataset_hash TEXT NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    software_verdict TEXT NOT NULL,
    scientific_verdict TEXT NOT NULL,
    mutation_test_count INTEGER,
    mutation_test_passed INTEGER,
    supersedes_experiment_id TEXT,
    report_artifact_id TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiment_registry(experiment_id)
);
"""

# Phase 6 continued — all-timestamp candidate universe (Protocol §17.8).
# NOT conditioned on ami_events existing: every closed candle slot in the
# requested window is a candidate row, whether or not it aligns with a real
# event. is_event_aligned + aligned_event_id record the (possibly absent)
# link, they do not gate row existence.
_SCHEMA_PHASE6B = """
CREATE TABLE IF NOT EXISTS ami_candidate_universe (
    candidate_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    slot_ts_ms INTEGER NOT NULL,
    known_at_ts INTEGER NOT NULL,
    data_quality TEXT NOT NULL,
    is_event_aligned INTEGER NOT NULL DEFAULT 0,
    aligned_event_id TEXT,
    universe_definition_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    UNIQUE (symbol, timeframe, slot_ts_ms, universe_definition_version),
    FOREIGN KEY (aligned_event_id) REFERENCES ami_events(event_id)
);
"""


# BATCH-P7A (APPROVE PHASE 7A CANONICAL MIGRATION): verbatim copy of the SQL
# already validated (schema-identical, hash-verified) on a disposable copy
# in ami/lifecycle/canonical_schema.py's _SCHEMA -- not redesigned here.
# ami/lifecycle/canonical_schema.py keeps its own copy of this same DDL (its
# init_lifecycle_schema() is independently CREATE TABLE IF NOT EXISTS and
# therefore harmless/idempotent if called again) plus the identity/
# transition-validator/rebuild logic, which does not live here.
_SCHEMA_PHASE7A = """
CREATE TABLE IF NOT EXISTS ami_signal_lifecycle (
    signal_id TEXT PRIMARY KEY,
    setup_id TEXT NOT NULL,
    setup_version TEXT,
    source_event_id TEXT,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    timeframe TEXT,
    route_version TEXT,
    signal_birth_ts INTEGER NOT NULL,
    first_known_ts INTEGER,
    first_executable_ts INTEGER,
    last_confirmation_ts INTEGER,
    invalidation_ts INTEGER,
    terminal_ts INTEGER,
    lifecycle_status TEXT NOT NULL,
    lifecycle_reason_code TEXT,
    observation_mode TEXT NOT NULL,
    evidence_layer TEXT NOT NULL,
    is_proxy INTEGER NOT NULL,
    executability_status TEXT,
    identity_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    source_hash TEXT,
    code_commit TEXT,
    provenance TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    CHECK (direction IN ('LONG','SHORT','UNKNOWN')),
    CHECK (evidence_layer IN ('REAL','PROXY')),
    CHECK ((is_proxy=0 AND evidence_layer='REAL') OR (is_proxy=1 AND evidence_layer='PROXY')),
    CHECK (signal_birth_ts <= COALESCE(first_known_ts, signal_birth_ts)),
    CHECK (first_known_ts IS NULL OR first_executable_ts IS NULL OR first_known_ts <= first_executable_ts),
    CHECK (first_known_ts IS NULL OR last_confirmation_ts IS NULL OR last_confirmation_ts >= first_known_ts),
    CHECK (invalidation_ts IS NULL OR invalidation_ts >= signal_birth_ts),
    CHECK (terminal_ts IS NULL OR terminal_ts >= signal_birth_ts)
);
CREATE TABLE IF NOT EXISTS ami_lifecycle_transitions (
    transition_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    previous_status TEXT,
    new_status TEXT NOT NULL,
    transition_ts INTEGER NOT NULL,
    known_at_ts INTEGER NOT NULL,
    reason_code TEXT NOT NULL,
    transition_version INTEGER NOT NULL,
    observation_mode TEXT NOT NULL,
    correction_of TEXT,
    evidence_ref TEXT,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, previous_status, new_status, transition_ts, reason_code,
            transition_version, observation_mode),
    CHECK (known_at_ts >= transition_ts)
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_transitions_signal
    ON ami_lifecycle_transitions(signal_id, transition_ts);
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_source_event
    ON ami_signal_lifecycle(source_event_id);
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_cycle
    ON ami_signal_lifecycle(independent_cycle_id);
"""

# BATCH-P7A-P (APPROVE PHASE 7A-P CANONICAL PROVENANCE MIGRATION): verbatim
# copy of the effective-ledger view SQL already validated on a disposable
# copy in ami/lifecycle/canonical_schema.py's _EFFECTIVE_VIEW -- not
# redesigned here. Two-layer ledger contract: ami_lifecycle_transitions
# (above) is the immutable raw audit ledger; this view is the corrections/
# supersessions-applied EFFECTIVE lifecycle ledger downstream interval/
# duration research must query (see that module's docstring for the full
# pure-reversal exclusion rule).
_EFFECTIVE_LEDGER_VIEW = """
CREATE VIEW IF NOT EXISTS ami_lifecycle_effective_transitions AS
SELECT t.*
FROM ami_lifecycle_transitions t
WHERE t.transition_id NOT IN (
    SELECT correction_of FROM ami_lifecycle_transitions WHERE correction_of IS NOT NULL
)
AND NOT (
    t.correction_of IS NOT NULL
    AND EXISTS (
        SELECT 1 FROM ami_lifecycle_transitions orig
        WHERE orig.transition_id = t.correction_of
          AND orig.previous_status = t.new_status
          AND orig.new_status = t.previous_status
    )
);
"""

# BATCH-P7A-P (APPROVE PHASE 7A-P CANONICAL PROVENANCE MIGRATION): verbatim
# copy of the field-level provenance SQL already validated on a disposable
# copy in ami/lifecycle/canonical_field_provenance.py's _SCHEMA/_VIEW -- not
# redesigned here. Row-level (evidence_layer/is_proxy on ami_signal_lifecycle)
# vs field-level (this table, per (signal_id, field_name)) provenance are two
# distinct, never-conflated axes -- see that module's docstring.
_SCHEMA_PHASE7AP1_PROVENANCE = """
CREATE TABLE IF NOT EXISTS ami_lifecycle_field_provenance (
    provenance_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_classification TEXT NOT NULL,
    is_proxy INTEGER NOT NULL,
    derivation_method TEXT NOT NULL,
    source_reference TEXT NOT NULL,
    limitations TEXT,
    provenance_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    code_commit TEXT,
    source_hash TEXT,
    created_at INTEGER NOT NULL,
    UNIQUE (signal_id, field_name, provenance_version),
    CHECK (field_classification IN ('DETERMINISTIC_HISTORICAL_SAFE','HISTORICAL_PROXY',
                                    'FORWARD_ONLY','NOT_IMPLEMENTED','BLOCKED_BY_DATA')),
    CHECK ((field_classification='HISTORICAL_PROXY' AND is_proxy=1)
           OR (field_classification!='HISTORICAL_PROXY' AND is_proxy=0)),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id)
);
CREATE INDEX IF NOT EXISTS idx_field_provenance_signal
    ON ami_lifecycle_field_provenance(signal_id, field_name);
CREATE INDEX IF NOT EXISTS idx_field_provenance_classification
    ON ami_lifecycle_field_provenance(field_classification);
CREATE VIEW IF NOT EXISTS ami_lifecycle_direction_view AS
SELECT
    sl.signal_id,
    sl.direction,
    fp.field_classification AS direction_classification,
    fp.is_proxy AS direction_is_proxy,
    fp.derivation_method AS direction_derivation_method,
    fp.limitations AS direction_limitations
FROM ami_signal_lifecycle sl
LEFT JOIN ami_lifecycle_field_provenance fp
    ON fp.signal_id = sl.signal_id AND fp.field_name = 'direction'
    AND fp.provenance_version = 'field-provenance-v1';
"""

# BATCH-P7B-CANON (APPROVE PHASE 7B CANONICAL PATH METRICS MIGRATION): verbatim
# copy of the path-observation DDL already validated on a disposable copy in
# ami/lifecycle/path_schema.py's _SCHEMA (schema fingerprint
# 3a26ffa86ecec9d8b63eff9455e3cfbbd594cc59eb36896feeba4d3bf232f1e7,
# reproduced here byte-for-byte, printed directly from the interpolated
# f-string rather than retyped) -- not redesigned here. Additive-only
# (CREATE TABLE/INDEX IF NOT EXISTS); FK to ami_signal_lifecycle,
# two-independent-axes CHECK (observation_status=path-computability,
# volatility_status=vol-baseline-usability, coupled so volatility_status=
# NOT_APPLICABLE iff observation_status!=OK). See that module's docstring
# for the full candle-boundary/zero-extremum-timing/naming rationale.
_SCHEMA_PHASE7B = """
CREATE TABLE IF NOT EXISTS ami_lifecycle_path_observations (
    observation_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    horizon_name TEXT NOT NULL,
    horizon_end_ts INTEGER NOT NULL,
    known_at_ts INTEGER NOT NULL,
    as_of_ts INTEGER NOT NULL,
    observation_status TEXT NOT NULL,
    volatility_status TEXT NOT NULL,
    reference_price REAL,
    reference_price_ts INTEGER,
    effective_path_start_ts INTEGER,
    endpoint_return_bps REAL,
    mfe_bps REAL,
    mae_bps REAL,
    time_to_mfe_ms INTEGER,
    time_to_mae_ms INTEGER,
    intrabar_order_status TEXT,
    realized_vol_at_anchor REAL,
    endpoint_return_anchor_vol_units REAL,
    mfe_anchor_vol_units REAL,
    mae_anchor_vol_units REAL,
    horizon_outcome_class TEXT,
    expected_candle_count INTEGER NOT NULL,
    observed_candle_count INTEGER NOT NULL,
    gap_count INTEGER NOT NULL,
    candle_definition_version TEXT,
    path_definition_version TEXT NOT NULL,
    observation_mode TEXT NOT NULL,
    provenance TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, horizon_name, path_definition_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    CHECK (horizon_name IN ('scalp_30m', 'scalp_1h', 'swing_4h', 'swing_24h')),
    CHECK (observation_status IN ('OK', 'MISSING_REFERENCE_PRICE', 'MISSING_INTERNAL_GAP', 'EXCLUDED_NO_HORIZON_DATA', 'NOT_COMPUTABLE_DIRECTION')),
    CHECK (volatility_status IN ('OK', 'INVALID_VOLATILITY_BASELINE', 'NOT_APPLICABLE')),
    CHECK ((volatility_status='NOT_APPLICABLE' AND observation_status!='OK')
           OR (volatility_status!='NOT_APPLICABLE' AND observation_status='OK')),
    CHECK (intrabar_order_status IS NULL OR intrabar_order_status IN ('MFE_FIRST', 'MAE_FIRST', 'SAME_CANDLE_UNKNOWN')),
    CHECK (horizon_outcome_class IS NULL OR horizon_outcome_class IN ('CONTINUATION', 'REVERSAL', 'CHOP')),
    CHECK (mfe_bps IS NULL OR mfe_bps >= 0),
    CHECK (mae_bps IS NULL OR mae_bps <= 0),
    CHECK (known_at_ts >= horizon_end_ts),
    CHECK (expected_candle_count >= 0),
    CHECK (observed_candle_count >= 0),
    CHECK (gap_count >= 0)
);
CREATE INDEX IF NOT EXISTS idx_path_observations_signal
    ON ami_lifecycle_path_observations(signal_id);
CREATE INDEX IF NOT EXISTS idx_path_observations_status
    ON ami_lifecycle_path_observations(observation_status);
CREATE INDEX IF NOT EXISTS idx_path_observations_volatility_status
    ON ami_lifecycle_path_observations(volatility_status);
"""

# BATCH-AMI-BIRTH-TRUNCATED-GEOMETRY-CANONICAL-MIGRATION (operator approval:
# "APPROVE CONTROLLED REAL CANONICAL MIGRATION -- AMI BIRTH-TRUNCATED CASCADE
# GEOMETRY CANONICAL MIGRATION + IMMUTABLE BACKFILL"): verbatim copy of the
# geometry + field-provenance DDL already validated on a disposable copy in
# ami/geometry/birth_truncated_cascade_geometry.py's _SCHEMA, and the
# field-level quality-ledger DDL already validated on a disposable copy in
# ami/geometry/liquidation_source_quality_contract_v2.py's _SCHEMA -- not
# redesigned here. Additive-only (CREATE TABLE/INDEX/VIEW IF NOT EXISTS); FK
# to ami_signal_lifecycle/ami_events/ami_cycles.
#
# ONE quality-state mechanism only (pre-migration duplication review): this
# geometry table carries NO data_quality_status/coverage_assessment_version
# column and no row-level quality-assessment table of its own -- an earlier
# revision had both, which the operator's controlled-migration review
# flagged as a quality-state duplication risk against the field-level
# ami_birth_truncated_geometry_field_quality_v2 ledger below. That row-level
# pair was removed (never migrated here) rather than kept alongside it; see
# birth_truncated_cascade_geometry.py's module docstring for the full
# rationale. Current row/field quality status is exposed ONLY through
# ami_birth_truncated_geometry_field_quality_v2_effective (per-field) and
# ami_birth_truncated_geometry_row_quality_v2_effective (worst-case rollup).
_SCHEMA_PHASE_GEOMETRY = """
CREATE TABLE IF NOT EXISTS ami_birth_truncated_cascade_geometry (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    feature_available_ts_ms INTEGER NOT NULL,
    source_window_start_ts_ms INTEGER NOT NULL,
    source_window_end_ts_ms INTEGER NOT NULL,
    source_row_count INTEGER NOT NULL,
    source_row_manifest_sha256 TEXT NOT NULL,
    running_notional REAL NOT NULL,
    running_liq_count INTEGER NOT NULL,
    max_single_notional REAL NOT NULL,
    running_single_liq_dominance REAL NOT NULL,
    running_rate REAL NOT NULL,
    running_accel REAL NOT NULL,
    elapsed_since_first_sec REAL NOT NULL,
    inter_cluster_gap_sec REAL,
    known_at_classification TEXT NOT NULL,
    derivation_reference TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, feature_definition_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    FOREIGN KEY (source_event_id) REFERENCES ami_events(event_id),
    FOREIGN KEY (independent_cycle_id) REFERENCES ami_cycles(cycle_id),
    CHECK (source_window_start_ts_ms <= source_window_end_ts_ms),
    CHECK (source_window_end_ts_ms <= feature_available_ts_ms),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (source_row_count >= 1),
    CHECK (running_notional >= 0),
    CHECK (running_liq_count >= 1),
    CHECK (max_single_notional >= 0),
    CHECK (running_single_liq_dominance >= 0 AND running_single_liq_dominance <= 100),
    CHECK (running_rate >= 0),
    CHECK (elapsed_since_first_sec >= 0),
    CHECK (inter_cluster_gap_sec IS NULL OR inter_cluster_gap_sec >= 0)
);
CREATE INDEX IF NOT EXISTS idx_geometry_signal
    ON ami_birth_truncated_cascade_geometry(signal_id);
CREATE INDEX IF NOT EXISTS idx_geometry_cycle
    ON ami_birth_truncated_cascade_geometry(independent_cycle_id);

CREATE TABLE IF NOT EXISTS ami_birth_truncated_geometry_field_provenance (
    provenance_id TEXT PRIMARY KEY,
    feature_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_classification TEXT NOT NULL,
    derivation_method TEXT NOT NULL,
    source_reference TEXT NOT NULL,
    limitations TEXT,
    provenance_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    code_commit TEXT,
    source_hash TEXT,
    created_at INTEGER NOT NULL,
    UNIQUE (feature_id, field_name, provenance_version),
    CHECK (field_classification IN ('DETERMINISTIC_HISTORICAL_SAFE','HISTORICAL_PROXY',
                                    'FORWARD_ONLY','NOT_IMPLEMENTED','BLOCKED_BY_DATA')),
    FOREIGN KEY (feature_id) REFERENCES ami_birth_truncated_cascade_geometry(feature_id)
);
CREATE INDEX IF NOT EXISTS idx_geometry_field_provenance_feature
    ON ami_birth_truncated_geometry_field_provenance(feature_id, field_name);

CREATE TABLE IF NOT EXISTS ami_birth_truncated_geometry_field_quality_v2 (
    assessment_id TEXT PRIMARY KEY,
    feature_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    coverage_assessment_version TEXT NOT NULL,
    data_quality_status TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    provenance TEXT NOT NULL,
    assessed_at_ms INTEGER NOT NULL,
    UNIQUE (feature_id, field_name, coverage_assessment_version),
    FOREIGN KEY (feature_id) REFERENCES ami_birth_truncated_cascade_geometry(feature_id),
    CHECK (data_quality_status IN ('SOURCE_COMPLETE','SOURCE_GAPPED','SOURCE_COVERAGE_UNRESOLVED')),
    CHECK (window_start_ts_ms <= window_end_ts_ms)
);
CREATE INDEX IF NOT EXISTS idx_field_quality_v2_feature
    ON ami_birth_truncated_geometry_field_quality_v2(feature_id);
CREATE INDEX IF NOT EXISTS idx_field_quality_v2_status
    ON ami_birth_truncated_geometry_field_quality_v2(data_quality_status);

CREATE VIEW IF NOT EXISTS ami_birth_truncated_geometry_field_quality_v2_effective AS
SELECT q.feature_id, q.field_name, q.data_quality_status, q.coverage_assessment_version, q.assessed_at_ms
FROM ami_birth_truncated_geometry_field_quality_v2 q
WHERE q.assessed_at_ms = (
    SELECT MAX(q2.assessed_at_ms) FROM ami_birth_truncated_geometry_field_quality_v2 q2
    WHERE q2.feature_id = q.feature_id AND q2.field_name = q.field_name
);

CREATE VIEW IF NOT EXISTS ami_birth_truncated_geometry_row_quality_v2_effective AS
SELECT feature_id,
       CASE
           WHEN SUM(CASE WHEN data_quality_status='SOURCE_GAPPED' THEN 1 ELSE 0 END) > 0 THEN 'SOURCE_GAPPED'
           WHEN SUM(CASE WHEN data_quality_status='SOURCE_COVERAGE_UNRESOLVED' THEN 1 ELSE 0 END) > 0
               THEN 'SOURCE_COVERAGE_UNRESOLVED'
           ELSE 'SOURCE_COMPLETE'
       END AS data_quality_status,
       COUNT(*) AS fields_assessed_n
FROM ami_birth_truncated_geometry_field_quality_v2_effective
GROUP BY feature_id;
"""


# BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1 (operator approval, real
# migration step of "devam edelim" -> "Gerçek schema 11→12 migrasyonunu
# çalıştır"): verbatim copy of the DDL already validated in the disposable
# rehearsal (`ami/cvd/windowed_taker_flow.py::_SCHEMA`,
# `ami/cvd/cvd_source_quality_contract_v1.py::_SCHEMA`,
# `ami/cvd/aggtrades_repair_rehearsal.py::_SCHEMA`) -- not redesigned here.
# The ONLY permitted delta vs the rehearsal DDL (per
# S34_CVD_SCHEMA_11_TO_12_MIGRATION_PROPOSAL_2026-07-05.md §1.3/1.4): three
# FK lines added to the CVD signal-window feature/quality tables (the
# disposable DB has no parent tables to reference) + a rename of the staging
# table `ami_agg_trades_repaired_stage` -> canonical `ami_agg_trades_repaired`
# (identical columns) + its effective-view (superseded-row exclusion,
# geometry precedent) + the quality ledger's own effective-view
# (assessment_version-latest-wins, byte-identical to the accepted contract's
# `ami_cvd_window_quality_v1_effective` definition).
_SCHEMA_PHASE_CVD = """
CREATE TABLE IF NOT EXISTS ami_agg_trades_repaired (
    symbol TEXT NOT NULL,
    agg_trade_id INTEGER NOT NULL,
    ts_ms INTEGER NOT NULL,
    retrieved_at_ms INTEGER NOT NULL,
    price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    notional REAL NOT NULL,
    signed_quantity REAL NOT NULL,
    signed_notional REAL NOT NULL,
    is_buyer_maker INTEGER NOT NULL,
    taker_side TEXT NOT NULL,
    first_trade_id INTEGER,
    last_trade_id INTEGER,
    source_regime_id TEXT NOT NULL,
    retrieval_batch_id TEXT NOT NULL,
    retrieval_page_index INTEGER NOT NULL,
    source_provenance TEXT NOT NULL,
    source_quality_status TEXT NOT NULL,
    legacy_match_status TEXT NOT NULL,
    legacy_match_fingerprint TEXT,
    superseded_by_batch_id TEXT,
    data_version_id TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (symbol, agg_trade_id, retrieval_batch_id),
    CHECK (is_buyer_maker IN (0,1)),
    CHECK (taker_side IN ('BUY','SELL')),
    CHECK ((is_buyer_maker = 0) = (taker_side = 'BUY')),
    CHECK (legacy_match_status IN ('UNMATCHED','MATCHED_1TO1','AMBIGUOUS','CONFLICTING','NOT_ATTEMPTED')),
    CHECK (data_version_id = 'aggtrades-binance-fapi-repair-r1')
);
CREATE INDEX IF NOT EXISTS idx_repair_stage_ts
    ON ami_agg_trades_repaired(symbol, ts_ms, agg_trade_id);

CREATE VIEW IF NOT EXISTS ami_agg_trades_repaired_effective AS
  SELECT * FROM ami_agg_trades_repaired WHERE superseded_by_batch_id IS NULL;

CREATE TABLE IF NOT EXISTS ami_cvd_repair_batch_ledger (
    retrieval_batch_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    requested_start_ms INTEGER NOT NULL,
    requested_end_ms INTEGER NOT NULL,
    pagination_method TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    first_agg_trade_id INTEGER,
    last_agg_trade_id INTEGER,
    earliest_trade_ts_ms INTEGER,
    latest_trade_ts_ms INTEGER,
    page_overlap_rows INTEGER NOT NULL,
    missing_id_ranges TEXT NOT NULL,
    request_errors TEXT NOT NULL,
    truncation_flag INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    gap_manifest_sha256 TEXT NOT NULL,
    duplicate_manifest_sha256 TEXT NOT NULL,
    exact_reconstruction_verdict TEXT NOT NULL,
    data_version_id TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    CHECK (exact_reconstruction_verdict IN
           ('EXACT_RECONSTRUCTED','INCOMPLETE','FAILED','PROBE_ONLY'))
);

CREATE TABLE IF NOT EXISTS ami_cvd_windowed_flow (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    raw_interpretation_version TEXT NOT NULL,
    quality_contract_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    source_row_count INTEGER NOT NULL,
    legacy_row_count INTEGER NOT NULL,
    repair_row_count INTEGER NOT NULL,
    cvd_qty REAL,
    cvd_notional REAL,
    total_notional REAL,
    taker_buy_qty REAL,
    taker_sell_qty REAL,
    taker_buy_notional REAL,
    taker_sell_notional REAL,
    normalized_cvd REAL,
    source_row_manifest_sha256 TEXT NOT NULL,
    source_regime_ids TEXT NOT NULL,
    repair_method TEXT NOT NULL,
    repair_population_version TEXT,
    feature_available_ts_ms INTEGER NOT NULL,
    known_at_classification TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, feature_definition_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    FOREIGN KEY (source_event_id) REFERENCES ami_events(event_id),
    FOREIGN KEY (independent_cycle_id) REFERENCES ami_cycles(cycle_id),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600','BUCKET')),
    CHECK (evidence_layer = 'EXACT'),
    CHECK (window_start_ts_ms <= window_end_ts_ms),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (source_row_count = legacy_row_count + repair_row_count),
    CHECK (source_row_count >= 0),
    CHECK (repair_method IN ('NONE','AGGTRADES_REST','AGGTRADES_VISION_ARCHIVE')),
    CHECK (total_notional IS NULL OR total_notional >= 0),
    CHECK (normalized_cvd IS NULL OR (normalized_cvd >= -1.0 AND normalized_cvd <= 1.0))
);
CREATE INDEX IF NOT EXISTS idx_cvd_flow_signal ON ami_cvd_windowed_flow(signal_id);
CREATE INDEX IF NOT EXISTS idx_cvd_flow_cycle ON ami_cvd_windowed_flow(independent_cycle_id);

-- PROXY layer: PHYSICALLY separate table. Never pooled with the exact table;
-- a schema-level CHECK pins evidence_layer so a mixed population cannot even
-- be represented.
CREATE TABLE IF NOT EXISTS ami_cvd_windowed_flow_proxy (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    quality_contract_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    candle_timeframe TEXT NOT NULL,
    contained_candle_count INTEGER NOT NULL,
    last_contained_close_ts_ms INTEGER,
    proxy_cvd_qty REAL,
    proxy_taker_buy_qty REAL,
    proxy_taker_sell_qty REAL,
    candle_versions TEXT NOT NULL,
    source_row_manifest_sha256 TEXT NOT NULL,
    feature_available_ts_ms INTEGER NOT NULL,
    known_at_classification TEXT NOT NULL,
    descriptive_only INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, feature_definition_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    FOREIGN KEY (source_event_id) REFERENCES ami_events(event_id),
    FOREIGN KEY (independent_cycle_id) REFERENCES ami_cycles(cycle_id),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600','BUCKET')),
    CHECK (evidence_layer = 'PROXY'),
    CHECK (descriptive_only = 1),
    CHECK (window_start_ts_ms <= window_end_ts_ms),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (contained_candle_count >= 0),
    CHECK (last_contained_close_ts_ms IS NULL OR last_contained_close_ts_ms <= signal_birth_ts),
    CHECK ((contained_candle_count = 0) = (proxy_cvd_qty IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_cvd_proxy_signal ON ami_cvd_windowed_flow_proxy(signal_id);

-- explicit, non-silent BUCKET exclusion ledger
CREATE TABLE IF NOT EXISTS ami_cvd_bucket_exclusions (
    exclusion_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    reason TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, feature_definition_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    CHECK (reason = 'BUCKET_WINDOW_NOT_FROZEN_FOR_SIGNAL')
);

CREATE TABLE IF NOT EXISTS ami_cvd_window_quality_v1 (
    quality_id TEXT PRIMARY KEY,
    quality_contract_version TEXT NOT NULL,
    assessment_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    source_regime_ids TEXT NOT NULL,
    regime_spanning INTEGER NOT NULL,
    legacy_row_count INTEGER NOT NULL,
    repair_row_count INTEGER NOT NULL,
    total_row_count INTEGER NOT NULL,
    duplicate_count INTEGER NOT NULL,
    collision_count INTEGER NOT NULL,
    unresolved_match_count INTEGER NOT NULL,
    missing_minute_count INTEGER NOT NULL,
    repaired_minute_count INTEGER NOT NULL,
    cadence_proof TEXT NOT NULL,
    completeness_proof TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    feature_available_ts_ms INTEGER NOT NULL,
    source_provenance TEXT NOT NULL,
    data_version_id TEXT NOT NULL,
    feature_definition_version TEXT NOT NULL,
    assessed_at_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, quality_contract_version, assessment_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    CHECK (quality_status IN ('EXACT_RECONSTRUCTABLE','PROXY_ONLY','SOURCE_GAPPED',
                              'SOURCE_COVERAGE_UNRESOLVED','UNREPAIRABLE')),
    CHECK (evidence_layer IN ('EXACT','PROXY')),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600','BUCKET')),
    CHECK (window_start_ts_ms <= window_end_ts_ms),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (regime_spanning IN (0,1)),
    CHECK (total_row_count = legacy_row_count + repair_row_count)
);
CREATE INDEX IF NOT EXISTS idx_cvd_quality_signal
    ON ami_cvd_window_quality_v1(signal_id, window_id);

CREATE VIEW IF NOT EXISTS ami_cvd_window_quality_v1_effective AS
  SELECT q.* FROM ami_cvd_window_quality_v1 q
  WHERE q.assessed_at_ms = (
    SELECT MAX(q2.assessed_at_ms) FROM ami_cvd_window_quality_v1 q2
    WHERE q2.signal_id = q.signal_id AND q2.window_id = q.window_id
      AND q2.quality_contract_version = q.quality_contract_version);
"""


# BATCH-CASCADE-ABSORPTION-IMPACT-CANONICAL-MIGRATION-V1 (M-0035, operator
# approval): verbatim copy of the DDL already validated in the disposable
# rehearsal (`ami/absorption/cascade_absorption_impact_rehearsal.py::_SCHEMA`,
# accepted commit fc43e972) and frozen row-accounted in
# S34_CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1.json (commit
# 931cd3dd) -- not redesigned here. Per operator naming ruling, canonical
# production table names are `ami_absorption_impact_*` (the frozen contract's
# illustrative `ami_impact_*` and the rehearsal's disposable
# `absorption_impact_*` are both superseded by this naming only -- formula,
# units, window definitions, row identities, quality states, exclusion
# identity, FLOOR_USD_M and feature values are unchanged). The ONLY permitted
# delta vs the rehearsal DDL (same precedent as _SCHEMA_PHASE_CVD's three FK
# additions): FK lines added referencing the canonical identity tables (the
# disposable rehearsal DB had no parent tables to reference) + a `window_id`
# enum CHECK (rehearsal enforced this only in code, not in its disposable
# schema). No effective-view is added -- unlike CVD's quality table, this
# family's quality table has no `assessment_version` dimension in its UNIQUE
# constraint (one quality row per signal/window/quality_contract_version,
# ever), so there is no multi-assessment "latest wins" case to resolve.
_SCHEMA_PHASE_ABSORPTION_IMPACT = """
CREATE TABLE IF NOT EXISTS ami_absorption_impact_windowed_flow (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    trade_count INTEGER NOT NULL,
    native_rows_used INTEGER NOT NULL,
    repaired_rows_used INTEGER NOT NULL,
    signed_notional REAL NOT NULL,
    total_notional REAL NOT NULL,
    mark_price_start REAL,
    mark_price_end REAL,
    mark_return_bps REAL,
    floor_usd_m_applied INTEGER NOT NULL,
    floor_usd_m_value REAL NOT NULL,
    price_response_per_signed_notional REAL,
    evidence_layer TEXT NOT NULL,
    feature_available_ts_ms INTEGER NOT NULL,
    known_at_classification TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (symbol, signal_id, window_id, feature_definition_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    FOREIGN KEY (source_event_id) REFERENCES ami_events(event_id),
    FOREIGN KEY (independent_cycle_id) REFERENCES ami_cycles(cycle_id),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600')),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (evidence_layer = 'EXACT'),
    CHECK (direction IN ('LONG','SHORT')),
    CHECK (floor_usd_m_applied IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_absorption_impact_flow_symbol_window
    ON ami_absorption_impact_windowed_flow(symbol, window_id);
CREATE INDEX IF NOT EXISTS idx_absorption_impact_flow_signal
    ON ami_absorption_impact_windowed_flow(signal_id);

CREATE TABLE IF NOT EXISTS ami_absorption_impact_window_quality_v1 (
    quality_id TEXT PRIMARY KEY,
    quality_contract_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    confirmed_gap_overlap INTEGER NOT NULL,
    unresolved_gap_overlap INTEGER NOT NULL,
    before_collection_began INTEGER NOT NULL,
    repaired_rows_used INTEGER NOT NULL,
    native_rows_used INTEGER NOT NULL,
    assessed_at_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, quality_contract_version),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    CHECK (quality_status IN ('EXACT_RECONSTRUCTABLE','PROXY_ONLY','SOURCE_GAPPED',
                              'SOURCE_COVERAGE_UNRESOLVED','UNREPAIRABLE')),
    CHECK (evidence_layer = 'EXACT'),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600'))
);
CREATE INDEX IF NOT EXISTS idx_absorption_impact_quality_window
    ON ami_absorption_impact_window_quality_v1(window_id, quality_status);

-- immutable exclusion ledger; a signal/window pair excluded here must never
-- also appear in ami_absorption_impact_windowed_flow (proven by the
-- rehearsal's own test_signal_window_pair_never_in_both_windowed_flow_and_
-- exclusions, re-verified independently in the row-accounting freeze).
CREATE TABLE IF NOT EXISTS ami_absorption_impact_exclusions (
    exclusion_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    window_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, reason_code),
    FOREIGN KEY (signal_id) REFERENCES ami_signal_lifecycle(signal_id),
    CHECK (reason_code IN ('WINDOW_STARTS_BEFORE_COLLECTION_BEGAN','CONFIRMED_GAP_OVERLAP',
                           'UNRESOLVED_GAP_PROXIMITY')),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600'))
);
CREATE INDEX IF NOT EXISTS idx_absorption_impact_exclusions_window
    ON ami_absorption_impact_exclusions(window_id, reason_code);
"""


def connect(path: str | Path | None = None, read_only: bool = False) -> sqlite3.Connection:
    """Open the canonical warehouse. WAL + busy_timeout matches existing AMI store practice.

    FABLE-REVIEW-A F4: foreign_keys=ON so declared FKs (artifact_lineage,
    question_registry, event_cycle_membership) are actually enforced, not
    just documented. Seed order already respects this (artifact_ingest
    before registry_seed; ami_events before cooldown_sensitivity/cycle_resolver).

    [PHASE 7A-P TEST ISOLATION SAFETY CLOSURE] `path` defaults to the
    CURRENT value of the module-level DEFAULT_PATH, resolved at CALL time
    (not at function-definition time) -- this is what lets tests/conftest.py's
    session-scoped fixture redirect a bare/omitted-path connect() call to a
    disposable copy for the whole test session. Independent of that
    redirection, if `_TEST_ISOLATION_ACTIVE` is True (set only by that same
    fixture) and the resolved path is the REAL canonical.sqlite
    (REAL_CANONICAL_PATH_IMMUTABLE, never itself redirectable) with
    read_only=False, this raises immediately -- fail-closed backstop for any
    test that bypasses DEFAULT_PATH and hardcodes the real path directly.
    """
    if path is None:
        path = DEFAULT_PATH
    p = Path(path)
    if _TEST_ISOLATION_ACTIVE and not read_only and p.resolve() == Path(REAL_CANONICAL_PATH_IMMUTABLE).resolve():
        raise RuntimeError(
            "TEST_ISOLATION_SAFETY_BLOCKER: refusing to open a writable connection to the REAL "
            f"canonical.sqlite ({REAL_CANONICAL_PATH_IMMUTABLE}) during a test run. Tests must use "
            "a disposable copy -- see tests/conftest.py's session-scoped _isolate_real_canonical_db "
            "fixture, which redirects ami.warehouse.schema.DEFAULT_PATH automatically for any test "
            "that imports/passes it. If this fired from a test that hardcodes the real path as a "
            "literal string, fix the test to use DEFAULT_PATH (or mode=ro for genuine read-only checks)."
        )
    if read_only:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(p, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """CREATE TABLE IF NOT EXISTS does not add columns to an already-existing
    table (BATCH-P6-000: real canonical.sqlite already had ami_levels before
    touch_stats_point_in_time existed). This is the idempotent equivalent of
    'ALTER TABLE ... ADD COLUMN IF NOT EXISTS' for SQLite."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: safe to call on every process start (CREATE TABLE IF NOT EXISTS only)."""
    conn.executescript(_SCHEMA)
    conn.executescript(_SCHEMA_PHASE2)
    conn.executescript(_SCHEMA_PHASE3)
    conn.executescript(_SCHEMA_PHASE4)
    conn.executescript(_SCHEMA_PHASE6)
    conn.executescript(_SCHEMA_PHASE6B)
    conn.executescript(_SCHEMA_PHASE7A)
    conn.executescript(_EFFECTIVE_LEDGER_VIEW)
    conn.executescript(_SCHEMA_PHASE7AP1_PROVENANCE)
    conn.executescript(_SCHEMA_PHASE7B)
    conn.executescript(_SCHEMA_PHASE_GEOMETRY)
    conn.executescript(_SCHEMA_PHASE_CVD)
    conn.executescript(_SCHEMA_PHASE_ABSORPTION_IMPACT)
    _add_column_if_missing(conn, "ami_levels", "touch_stats_point_in_time",
                            "touch_stats_point_in_time INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        "INSERT INTO schema_versions (component, version, applied_ms, note) VALUES (?,?,strftime('%s','now')*1000,?) "
        "ON CONFLICT(component) DO UPDATE SET version=excluded.version, applied_ms=excluded.applied_ms, note=excluded.note",
        ("canonical_warehouse", CANONICAL_SCHEMA_VERSION, f"schema v{CANONICAL_SCHEMA_VERSION}"),  # FABLE-REVIEW-A F5
    )
    conn.commit()
