"""PHASE 7B-0 / 7B-0.1 / 7B-CANON: Path/MFE/MAE observation schema.

[PHASE 7B-CANON, APPLIED 2026-07-04] This schema is now MERGED into
ami/warehouse/schema.py as `_SCHEMA_PHASE7B` (byte-for-byte identical DDL,
schema fingerprint 3a26ffa86ecec9d8b63eff9455e3cfbbd594cc59eb36896feeba4d3bf232f1e7)
and applied to the real data/ami/canonical.sqlite (CANONICAL_SCHEMA_VERSION
9->10, see MIGRATION_LOG.md). `ami.warehouse.schema.init_schema()` is now the
canonical way this table gets created on any DB, including the real one.
This module's own `init_path_schema()`/`rollback_path_schema()` remain in
active use for disposable-copy testing/rehearsal (ami/lifecycle/
path_migration_rehearsal.py, tests) -- they are additive-only/idempotent and
safe to call against the (now-v10) real DB too, but production code reaches
this table via `ami.warehouse.schema.init_schema()` +
`ami.lifecycle.path_canonical_migration.run_canonical_migration()`, not via
this module directly.

Historical context (Phase 7B-0/7B-0.1, pre-migration): this module defined a
NEW, SEPARATE schema, not merged into ami/warehouse/schema.py's shared
init_schema() at the time, precisely so no existing script's routine
init_schema(DEFAULT_PATH) call could accidentally apply it to the real
data/ami/canonical.sqlite before the operator's explicit migration approval.

[PHASE 7B-0.1 semantic closure] PATH_SCHEMA_VERSION 1->2 / PATH_DEFINITION_VERSION
"path-v1"->"path-v2": two operator-mandated corrections landed together --
(a) `observation_status` was overloaded (it used to also carry
INVALID_VOLATILITY_BASELINE, conflating "is the price PATH computable" with
"is the anchor's realized-vol BASELINE usable for normalization" -- these are
now two independent columns, see below); (b) the MFE/MAE zero-extremum timing
rule changed (see compute_observation()'s docstring in path_metrics.py) --
this is a genuine formula/methodology change, not just a schema-shape change,
so PATH_DEFINITION_VERSION bumps too (immutable-versioned, same discipline as
candle/level/cycle_definition_version: a redefinition mints a new version, it
never silently mutates rows already written under the old one). No real data
was ever written under path-v1 to any persistent store (NO_CANONICAL_WRITE
was honored throughout 7B-0), so this bump has zero migration cost.

Reused, not reinvented:
  - horizon_name values           -> ami.research.w4_post_event_path_taxonomy.PATH_HORIZONS_MS
                                      keys, verbatim (scalp_30m/scalp_1h/swing_4h/swing_24h --
                                      no new horizon invented, per operator lock)
  - horizon_outcome_class values  -> ami.research.w4_post_event_path_taxonomy.classify_path's
                                      CONTINUATION/REVERSAL/CHOP taxonomy, verbatim (no parallel
                                      enum invented)
  - observation_mode              -> ami.lifecycle.canonical_schema.ObservationMode, verbatim
                                      (HISTORICAL_REPLAY only; FORWARD_OBSERVATION not written here)
  - identity style                -> deterministic sha256-of-natural-key, same pattern as
                                      generate_signal_id()/_transition_id() (no PID/UUID)
  - field-level provenance table  -> ami.lifecycle.canonical_field_provenance's EXISTING
                                      ami_lifecycle_field_provenance (signal_id, field_name)
                                      contract, reused verbatim (see path_field_provenance.py --
                                      no second/parallel provenance table)

STATUS TAXONOMY -- TWO INDEPENDENT AXES (operator lock, 7B-0.1):

  observation_status (path-computability ONLY, priority-ordered):
    OK                        -- reference price known, horizon matured, path window
                                 gap-free, direction known; endpoint/mfe/mae/timing/
                                 horizon_outcome_class all populated.
    MISSING_REFERENCE_PRICE    -- no fully-closed candle exists at/before signal_birth_ts.
    MISSING_INTERNAL_GAP        -- horizon matured but the path window (effective_path_start_ts
                                 .. horizon_end_ts) contains at least one non-AVAILABLE or
                                 entirely-missing 1m slot -- never computed over an incomplete path.
    EXCLUDED_NO_HORIZON_DATA    -- signal_birth_ts + horizon_ms is beyond the candle table's
                                 current maturity cutoff (the outcome has not happened yet).
    NOT_COMPUTABLE_DIRECTION    -- ami_signal_lifecycle.direction is not LONG/SHORT (e.g. a
                                 future UNKNOWN-prefix setup_id) -- MFE/MAE sign cannot be
                                 oriented; never defaulted to LONG.

  volatility_status (vol-normalization baseline ONLY, independent of the above):
    OK                          -- observation_status=='OK' AND the anchor's 60-candle
                                   realized_vol is a valid, positive number; the 3
                                   *_anchor_vol_units fields are populated.
    INVALID_VOLATILITY_BASELINE -- observation_status=='OK' but realized_vol_at_anchor is
                                   None or <=0 (insufficient/degenerate baseline); the 3
                                   *_anchor_vol_units fields are NULL, but
                                   endpoint_return_bps/mfe_bps/mae_bps/timing/
                                   horizon_outcome_class remain populated -- volatility
                                   invalidity never blanks out an otherwise-valid path.
    NOT_APPLICABLE               -- observation_status != 'OK' (the path itself was not
                                   computable, so there is nothing to normalize); vol
                                   normalization was never attempted.
  CHECK (volatility_status='NOT_APPLICABLE' iff observation_status!='OK') is enforced at
  the DB level below, not just in application code.

CANDLE BOUNDARY SEMANTICS (operator lock #1, 2026-07-04 approval, unchanged in 7B-0.1):
  reference_price      = close of the last 1m candle whose close_ts_ms <= signal_birth_ts
                          (fully closed strictly before-or-at signal birth).
  effective_path_start_ts = open_ts_ms of the first 1m candle whose open_ts_ms >= signal_birth_ts.
  The candle that STRADDLES signal_birth_ts (open_ts_ms < signal_birth_ts < close_ts_ms) is
  excluded from BOTH -- its close may postdate birth (disqualifying it as reference) and its
  open predates birth (disqualifying it from the path window). Its high/low never enter any
  MFE/MAE computation.

[PHASE 7B-0.1] ZERO-EXTREMUM TIMING (operator lock #2): the reference point (t=0,
signal_birth_ts itself) is treated as part of the observable path -- it trivially
contributes a 0-bps favorable AND 0-bps adverse excursion candidate. If no real candle in
the path window ever exceeds 0 favorably, MFE=0 and time_to_mfe_ms=0 (the reference point
IS the achieving point); symmetrically for MAE. This is never a silent fallback to "the
first candle's timestamp" -- see path_metrics.py's compute_observation() for the exact
tie-break rule (earliest timestamp among all points, real or the virtual t=0 point, that
achieve the extremum value).

[PHASE 7B-0.1] VOLATILITY NORMALIZATION NAMING (operator lock #5, option A selected): the
three vol-ratio fields are named endpoint_return_anchor_vol_units / mfe_anchor_vol_units /
mae_anchor_vol_units (not the bare *_vol_units used in 7B-0) precisely because the
denominator (realized_vol_at_anchor) is a FIXED 60-1m-candle realized-vol measured AT the
signal's own anchor point -- it is never rescaled by horizon_ms. A value of e.g. 2.0 in
mfe_anchor_vol_units for a scalp_30m row and 2.0 for a swing_24h row on the SAME signal do
NOT represent "the same number of horizon-scaled standard deviations" -- both use the
IDENTICAL anchor-vol denominator (realized_vol_at_anchor is identical across all 4 horizon
rows of one signal, since it does not depend on horizon_ms at all; see
test_realized_vol_at_anchor_identical_across_horizons_same_signal). The "anchor_vol" infix
makes this non-cross-horizon-comparable basis explicit in the field name itself, rather
than requiring a separate lookup column to discover it.

FIELD-LEVEL PROVENANCE: see ami/lifecycle/path_field_provenance.py -- this module's fields
are NOT self-certifying; every substantive (non-identity, non-bookkeeping) column here gets
an explicit row in the EXISTING ami_lifecycle_field_provenance table (field_name prefixed
"path_observations." to avoid any ambiguity with ami_signal_lifecycle's own 16 tracked
field names, since that shared table has no source-table column).
"""
from __future__ import annotations

PATH_SCHEMA_VERSION = 2
PATH_DEFINITION_VERSION = "path-v2"

_VALID_HORIZON_NAMES = ("scalp_30m", "scalp_1h", "swing_4h", "swing_24h")
_VALID_OBSERVATION_STATUSES = (
    "OK",
    "MISSING_REFERENCE_PRICE",
    "MISSING_INTERNAL_GAP",
    "EXCLUDED_NO_HORIZON_DATA",
    "NOT_COMPUTABLE_DIRECTION",
)
_VALID_VOLATILITY_STATUSES = ("OK", "INVALID_VOLATILITY_BASELINE", "NOT_APPLICABLE")
_VALID_INTRABAR_ORDER_STATUSES = ("MFE_FIRST", "MAE_FIRST", "SAME_CANDLE_UNKNOWN")
_VALID_HORIZON_OUTCOME_CLASSES = ("CONTINUATION", "REVERSAL", "CHOP")

_horizon_check = ", ".join(f"'{h}'" for h in _VALID_HORIZON_NAMES)
_status_check = ", ".join(f"'{s}'" for s in _VALID_OBSERVATION_STATUSES)
_vol_status_check = ", ".join(f"'{s}'" for s in _VALID_VOLATILITY_STATUSES)
_intrabar_check = ", ".join(f"'{s}'" for s in _VALID_INTRABAR_ORDER_STATUSES)
_outcome_check = ", ".join(f"'{s}'" for s in _VALID_HORIZON_OUTCOME_CLASSES)

_SCHEMA = f"""
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
    CHECK (horizon_name IN ({_horizon_check})),
    CHECK (observation_status IN ({_status_check})),
    CHECK (volatility_status IN ({_vol_status_check})),
    CHECK ((volatility_status='NOT_APPLICABLE' AND observation_status!='OK')
           OR (volatility_status!='NOT_APPLICABLE' AND observation_status='OK')),
    CHECK (intrabar_order_status IS NULL OR intrabar_order_status IN ({_intrabar_check})),
    CHECK (horizon_outcome_class IS NULL OR horizon_outcome_class IN ({_outcome_check})),
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


def init_path_schema(conn) -> None:
    """Additive only (CREATE TABLE/INDEX IF NOT EXISTS). Safe to call
    repeatedly. Caller controls which DB this connection points at -- NEVER
    call with ami.warehouse.schema.DEFAULT_PATH pointed at the real file in
    this batch."""
    conn.executescript(_SCHEMA)
    conn.commit()


def rollback_path_schema(conn) -> None:
    """Drops only the single table added by this module -- never touches
    ami_signal_lifecycle/ami_lifecycle_transitions/ami_lifecycle_field_provenance/
    ami_events/any pre-existing table."""
    conn.executescript("DROP TABLE IF EXISTS ami_lifecycle_path_observations;")
    conn.commit()


def schema_manifest(conn) -> dict:
    """Exact, code-derived (never hand-typed/guessed) structural description
    of ami_lifecycle_path_observations: ordered column list with type/
    nullability/PK, UNIQUE constraints, CHECK constraints (raw SQL text), FK
    list, and a schema fingerprint (reused from
    ami.lifecycle.migration_rehearsal.schema_fingerprint's hash-of-DDL
    approach, scoped to just this one table+its indexes)."""
    import hashlib

    cols_info = conn.execute("PRAGMA table_info(ami_lifecycle_path_observations)").fetchall()
    columns = [
        {"cid": c[0], "name": c[1], "type": c[2], "notnull": bool(c[3]),
         "default": c[4], "is_pk": bool(c[5])}
        for c in cols_info
    ]
    fk_info = conn.execute("PRAGMA foreign_key_list(ami_lifecycle_path_observations)").fetchall()
    foreign_keys = [{"table": f[2], "from": f[3], "to": f[4]} for f in fk_info]

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='ami_lifecycle_path_observations'"
    ).fetchone()
    create_sql = row[0] if row else None
    checks = []
    if create_sql:
        for line in create_sql.splitlines():
            stripped = line.strip().rstrip(",")
            if stripped.upper().startswith("CHECK"):
                checks.append(stripped)
    uniques = []
    if create_sql:
        for line in create_sql.splitlines():
            stripped = line.strip().rstrip(",")
            if stripped.upper().startswith("UNIQUE"):
                uniques.append(stripped)

    index_rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='ami_lifecycle_path_observations' "
        "AND sql IS NOT NULL ORDER BY name"
    ).fetchall()
    fingerprint_text = "\n".join([create_sql or ""] + [f"{n}|{s}" for n, s in index_rows])
    fingerprint = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()

    return {
        "table": "ami_lifecycle_path_observations",
        "column_count": len(columns),
        "columns": columns,
        "unique_constraints": uniques,
        "check_constraints": checks,
        "foreign_keys": foreign_keys,
        "indexes": [n for n, _ in index_rows],
        "schema_fingerprint": fingerprint,
        "create_sql": create_sql,
    }
