"""PHASE 7A.1: Canonical Lifecycle Schema Foundation.

DISPOSABLE_DB_ONLY / NO_LIVE_CANONICAL_DB_MIGRATION: this module defines a
NEW, SEPARATE schema (not merged into ami/warehouse/schema.py's shared
`_SCHEMA`/`init_schema()` in this batch) precisely so that no existing
script's routine `init_schema(DEFAULT_PATH)` call can accidentally apply it
to the real data/ami/canonical.sqlite. `init_lifecycle_schema()` is never
called against DEFAULT_PATH anywhere in this batch -- only against
tmp_path/disposable-copy connections, from tests and the migration
rehearsal harness. A future, SEPARATELY-approved "APPROVE PHASE 7A CANONICAL
MIGRATION" step would merge this into the shared warehouse schema and bump
CANONICAL_SCHEMA_VERSION 7->8 (proposed, not applied here).

Reconciled with existing infrastructure (reused, not reinvented):
  - source_event_id  -> ami_events.event_id (Protocol Sec7.4/Obs Sec5.1)
  - independent_cycle_id -> event_cycle_membership.candidate_cycle_key
    (cycle_definition_version="canonical-v1", is_canonical=1)
  - lifecycle_status -> ami.enums.TradeLifecycleState, verbatim (already
    live in ami/lifecycle/engine.py, not a new enum)
  - identity algorithm style -> ami.identity.event_identity.generate_event_id's
    deterministic-hash-of-natural-key pattern, reused verbatim (no PID/UUID)
  - evidence_layer/is_proxy -> ami.identity.event_identity.SourceQuality's
    REAL_LIQUIDATION vs PROXY_* distinction (a DIFFERENT axis from the
    per-FIELD reconstruction-method classification in
    ami/lifecycle/canonical_backfill.py's FIELD_CLASSIFICATION -- evidence_layer
    describes the underlying ANCHOR EVENT's source quality; field
    classification describes how a given LIFECYCLE FIELD's value was itself
    derived. The two are tracked separately, never conflated.)

ACTIVATION BOUNDARY (Phase 7A-0 finding, preserved verbatim, NOT changed
here): entry_ts_ms <= frozen_ms -> PRE_FREEZE/HISTORICAL;
entry_ts_ms > frozen_ms -> FORWARD_ELIGIBLE. This module does not implement
or touch forward_pipeline.py's activation logic at all (NO_NEW_FORWARD_BINDING,
NO_OBSERVER_START) -- ObservationMode.HISTORICAL_REPLAY is the only value
ever WRITTEN by this batch's backfill; FORWARD_OBSERVATION exists in the
enum for schema completeness but is not populated here.

CANONICAL QUERY CONTRACT -- "OPEN" is NOT "currently live/active" (Phase
7A-P1 semantic closure, round 3): every signal backfilled by this module is
lifecycle_status=OPEN (no genuinely validated terminal-detection mechanism
exists anywhere upstream -- see canonical_field_provenance.FIELD_PROVENANCE_SPECS
['terminal_ts']=NOT_IMPLEMENTED). OPEN here means "no evidence this signal's
terminal state has ever been validated", never "this trade/signal is
presently live". The taxonomy (LifecycleReasonCode/TradeLifecycleState) has
no UNKNOWN/CENSORED/UNRESOLVED counterpart, and none is invented here --
instead, the EXISTING terminal_ts=NOT_IMPLEMENTED field-provenance record is
the canonical gate: any terminal/hold-duration research MUST check that
classification (and query ami_lifecycle_effective_transitions, never the raw
ami_lifecycle_transitions table, for any interval/duration computation --
see effective_lifecycle_status()/count_effective_closed_signals() below)
before treating any signal as resolved, closed, or interval-bearing.
"""
from __future__ import annotations
import hashlib
import time
from enum import Enum

from ami.enums import TradeLifecycleState

LIFECYCLE_SCHEMA_VERSION = 3  # v1->v2 (BATCH-P7A-P1 semantic closure): setup_version relaxed to
                              # nullable (see migrate_setup_version_nullable). v2->v3 (BATCH-P7A-P1
                              # round 3): added ami_lifecycle_effective_transitions view (see below)
                              # -- proposed merge target = warehouse v9 (NOT applied to the real
                              # canonical.sqlite here)
IDENTITY_VERSION = "signal-identity-v1"
SETUP_VERSION_DEFAULT = "setup-v1"  # frozen placeholder: no per-route rule/parameter versioning
                                     # exists anywhere upstream yet (NOT_IMPLEMENTED, honestly labelled,
                                     # see canonical_backfill.FIELD_CLASSIFICATION); this constant is
                                     # itself immutable-versioned like candle/level/cycle_definition_version.

# [PHASE 7A-P1 semantic closure, identity contract] SETUP_VERSION_DEFAULT is
# formally an UNKNOWN_SETUP_VERSION identity token: a fixed placeholder used
# ONLY to keep generate_signal_id()'s hash-input stable across this
# historical backfill population -- it is NEVER an assertion that "setup-v1"
# is a real, observed per-route rule/parameter version (the canonical
# ami_signal_lifecycle.setup_version COLUMN is NULL/NOT_IMPLEMENTED; this
# token only ever appears inside the identity hash, never as stored data).
# SUPERSESSION CONTRACT (decided now, before it is ever needed): if a genuine
# per-route setup_version source is implemented in a future phase, existing
# signal_id values keyed on this token MUST NOT be silently recomputed or
# mutated. Either (a) a new IDENTITY_VERSION is minted so the same
# underlying event yields a NEW signal_id under the new version (the old id
# remains resolvable, never overwritten/reused), or (b) an explicit
# supersession link is recorded (correction-ledger-style, analogous to
# insert_transition's CORRECTION/correction_of pattern) from the old
# identity to the new one. Which of (a)/(b) applies is a decision for that
# future phase, NOT decided here -- this constant only locks in the
# "no silent identity mutation" constraint ahead of time.
UNKNOWN_SETUP_VERSION_TOKEN = SETUP_VERSION_DEFAULT


class ObservationMode(str, Enum):
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    FORWARD_OBSERVATION = "FORWARD_OBSERVATION"  # not written by this batch -- NO_OBSERVER_START


class EvidenceLayer(str, Enum):
    REAL = "REAL"
    PROXY = "PROXY"


class LifecycleReasonCode(str, Enum):
    """Deliberately minimal: identity/transition-ledger scope ONLY.
    NO_TIMING_PATH_LABEL_ENGINE / NO_CANCELLATION_LOGIC / NO_STOP_LOGIC /
    NO_HOLD_LOGIC / NO_REENTRY_LOGIC forbid adding path/progress/management
    reason codes in this batch -- those belong to a later, separately
    approved phase."""
    SIGNAL_BIRTH = "SIGNAL_BIRTH"
    TERMINAL_CLOSE = "TERMINAL_CLOSE"
    DATA_GAP = "DATA_GAP"
    CORRECTION = "CORRECTION"
    HISTORICAL_RECONSTRUCTION = "HISTORICAL_RECONSTRUCTION"


class ExecutabilityStatus(str, Enum):
    FORWARD_ONLY = "FORWARD_ONLY"
    BLOCKED_BY_DATA = "BLOCKED_BY_DATA"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class LifecycleIntegrityViolation(Exception):
    """Raised on invalid transition sequence, invalid status/reason code,
    append-only violation, or a timestamp-ordering constraint breach."""


_VALID_DIRECTIONS = {"LONG", "SHORT", "UNKNOWN"}
_TERMINAL_STATUSES = {TradeLifecycleState.CLOSED.value}

_SCHEMA = f"""
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

# [PHASE 7A-P1 semantic closure, round 3] Canonical two-layer ledger contract:
#   - RAW/immutable audit ledger  = ami_lifecycle_transitions (unchanged, append-only,
#     every row ever written, including now-superseded ones, is preserved forever).
#   - EFFECTIVE lifecycle ledger  = this view. It excludes:
#       (a) any row that has been superseded (referenced by another row's correction_of), and
#       (b) a correction row that is a PURE REVERSAL of what it corrects (i.e.
#           correction.previous_status == original.new_status AND
#           correction.new_status == original.previous_status) -- a pure reversal means the
#           pair, taken together, represents NO real market/lifecycle transition at all (the
#           original claim was invalid from the start; the correction exists only to undo it,
#           not to record a genuine second state change). A correction that changes to a
#           DIFFERENT new_status than the original's previous_status (a genuine, non-reversal
#           correction of a real transition) is NOT excluded -- only the superseded original is.
# Downstream research computing lifecycle DURATION/INTERVAL data (e.g. "how long was this
# signal OPEN before it CLOSED") MUST query this view, never the raw table directly -- reading
# the raw table for interval/duration purposes silently re-introduces exactly the invalid
# CLOSED interval this correction exists to remove. rebuild_current_state() (raw-table,
# latest-row-wins) and this view currently always agree for STATUS (both are read-safe for
# that purpose) -- this view is the additional, required surface for INTERVAL queries, which
# rebuild_current_state() was never designed to answer.
_EFFECTIVE_VIEW = """
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


def init_lifecycle_schema(conn) -> None:
    """Additive only (CREATE TABLE/INDEX/VIEW IF NOT EXISTS). Safe to call
    repeatedly (migration-rerun idempotency). Caller controls which DB this
    connection points at -- NEVER call with DEFAULT_PATH in this batch."""
    conn.executescript(_SCHEMA)
    conn.executescript(_EFFECTIVE_VIEW)
    conn.commit()


def rollback_lifecycle_schema(conn) -> None:
    """Drops only the objects added by this module (2 tables + the effective-
    ledger view) -- never touches ami_events/ami_cycles/experiment_*/any
    pre-existing table."""
    conn.executescript(
        "DROP VIEW IF EXISTS ami_lifecycle_effective_transitions;\n"
        "DROP TABLE IF EXISTS ami_lifecycle_transitions;\n"
        "DROP TABLE IF EXISTS ami_signal_lifecycle;\n"
    )
    conn.commit()


def migrate_setup_version_nullable(conn) -> bool:
    """PHASE 7A-P1 semantic closure (LIFECYCLE_SCHEMA_VERSION 1->2): relaxes
    ami_signal_lifecycle.setup_version from NOT NULL to nullable. SQLite has
    no ALTER COLUMN, so this rebuilds the table (standard SQLite 12-step
    procedure): create a new table with the relaxed constraint, copy all
    rows verbatim (signal_id and every other value preserved exactly --
    identity is NEVER recomputed here), drop the old table, rename.
    Idempotent: returns False (no-op) if the column is already nullable
    (e.g. a fresh init_lifecycle_schema() already created it nullable)."""
    info = {row[1]: row for row in conn.execute("PRAGMA table_info(ami_signal_lifecycle)").fetchall()}
    if info["setup_version"][3] == 0:  # (cid, name, type, notnull, dflt_value, pk) -- notnull=0 already
        return False

    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript("""
CREATE TABLE ami_signal_lifecycle__v9 (
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
""")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(ami_signal_lifecycle)").fetchall()]
        col_list = ", ".join(cols)
        conn.execute(f"INSERT INTO ami_signal_lifecycle__v9 ({col_list}) "
                     f"SELECT {col_list} FROM ami_signal_lifecycle")
        n_before = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        n_after = conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle__v9").fetchone()[0]
        if n_before != n_after:
            raise LifecycleIntegrityViolation(
                f"setup_version migration row-count mismatch: {n_before} -> {n_after}, aborting rebuild")
        conn.executescript("""
DROP TABLE ami_signal_lifecycle;
ALTER TABLE ami_signal_lifecycle__v9 RENAME TO ami_signal_lifecycle;
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_source_event ON ami_signal_lifecycle(source_event_id);
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_cycle ON ami_signal_lifecycle(independent_cycle_id);
""")
        conn.commit()
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if fk_errors:
            raise LifecycleIntegrityViolation(
                f"foreign_key_check failed after setup_version migration: {fk_errors}")
        return True
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def generate_signal_id(setup_id: str, setup_version: str, symbol: str, direction: str,
                        source_event_id: str | None = None, signal_birth_ts: int | None = None) -> str:
    """Deterministic, restart-stable hash of the natural key -- same pattern
    as ami.identity.event_identity.generate_event_id. No PID/UUID/row-ID;
    never derived from a mutable feature value. setup_version is an explicit
    part of the key (a setup redefinition mints a NEW identity, never
    silently reuses an old signal_id for different rules)."""
    if source_event_id is not None:
        key = f"{setup_id}|{setup_version}|{symbol}|{direction}|EVENT|{source_event_id}"
    else:
        if signal_birth_ts is None:
            raise ValueError("event-less signal identity requires signal_birth_ts (no fabricated anchor)")
        key = f"{setup_id}|{setup_version}|{symbol}|{direction}|NOEVENT|{signal_birth_ts}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"SIG-{digest}"


_SHORT_PREFIXES = ("SHORT_", "BUY_FADE")  # BUY_FADE is a SHORT-direction fade strategy
                                          # (failure_archive #8 "BUY-side short-squeeze fade",
                                          # #19 "BUYFADE SHORT->SHORT re-entry" -- grounded in
                                          # existing project records, not guessed fresh)
_LONG_PREFIXES = ("LONG_",)


def classify_direction_from_setup_id(setup_id: str) -> str:
    """HISTORICAL_PROXY method: prefix parse of the existing route-name
    string (ami_events.route_version tokens). Not a canonical route-registry
    lookup (none exists yet) -- returns 'UNKNOWN' rather than guessing for
    unrecognized prefixes."""
    upper = setup_id.upper()
    if upper.startswith(_LONG_PREFIXES):
        return "LONG"
    if any(upper.startswith(p) for p in _SHORT_PREFIXES):
        return "SHORT"
    return "UNKNOWN"


def _transition_id(signal_id: str, previous_status: str | None, new_status: str, transition_ts: int,
                    reason_code: str, transition_version: int, observation_mode: str) -> str:
    key = f"{signal_id}|{previous_status}|{new_status}|{transition_ts}|{reason_code}|{transition_version}|{observation_mode}"
    return "TRN-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _latest_transition(conn, signal_id: str) -> tuple[str, int] | None:
    """(new_status, transition_ts) of the most recent transition recorded
    for signal_id, ordered by transition_ts then transition_version --
    ties broken deterministically, never by insertion order alone."""
    row = conn.execute(
        "SELECT new_status, transition_ts FROM ami_lifecycle_transitions WHERE signal_id=? "
        "ORDER BY transition_ts DESC, transition_version DESC LIMIT 1",
        (signal_id,),
    ).fetchone()
    return (row[0], row[1]) if row else None


def validate_transition(conn, signal_id: str, previous_status: str | None, new_status: str) -> None:
    """Raises LifecycleIntegrityViolation on:
      - unknown new_status / previous_status (not in TradeLifecycleState)
      - previous_status == new_status (not a real transition)
      - broken chain (previous_status does not match the signal's current
        latest recorded new_status, for a non-genesis transition)
      - transition attempted FROM a terminal status (CLOSED)
      - a genesis transition (previous_status=None) attempted when the
        signal already has recorded transitions
    """
    valid_values = {s.value for s in TradeLifecycleState}
    if new_status not in valid_values:
        raise LifecycleIntegrityViolation(f"invalid new_status: {new_status!r}")
    if previous_status is not None and previous_status not in valid_values:
        raise LifecycleIntegrityViolation(f"invalid previous_status: {previous_status!r}")
    if previous_status == new_status:
        raise LifecycleIntegrityViolation(
            f"no-op transition rejected: previous_status == new_status == {new_status!r}")

    latest = _latest_transition(conn, signal_id)
    if previous_status is None:
        if latest is not None:
            raise LifecycleIntegrityViolation(
                f"genesis transition (previous_status=None) rejected: signal_id={signal_id!r} "
                f"already has a recorded transition (latest={latest[0]!r})")
        return
    if latest is None:
        raise LifecycleIntegrityViolation(
            f"non-genesis transition rejected: signal_id={signal_id!r} has no prior transition "
            f"(first transition for a signal must have previous_status=None)")
    latest_status, _latest_ts = latest
    if latest_status in _TERMINAL_STATUSES:
        raise LifecycleIntegrityViolation(
            f"transition from terminal status rejected: signal_id={signal_id!r} is already {latest_status!r}")
    if previous_status != latest_status:
        raise LifecycleIntegrityViolation(
            f"broken transition chain: signal_id={signal_id!r} previous_status={previous_status!r} "
            f"does not match latest recorded status {latest_status!r}")


def insert_transition(conn, signal_id: str, previous_status: str | None, new_status: str,
                       transition_ts: int, known_at_ts: int, reason_code: str,
                       observation_mode: str, transition_version: int = 1,
                       correction_of: str | None = None, evidence_ref: str | None = None,
                       provenance: str = "", schema_version: int = LIFECYCLE_SCHEMA_VERSION,
                       validate: bool = True) -> str:
    """The ONLY write path for ami_lifecycle_transitions. Append-only: never
    UPDATEs/DELETEs an existing row. Idempotent: re-inserting the identical
    (signal_id, previous_status, new_status, transition_ts, reason_code,
    transition_version, observation_mode) tuple is a silent no-op (checked
    before insert, backed by the table's UNIQUE constraint as a hard
    backstop). A correction of a previously-recorded transition MUST use a
    higher transition_version (or a later transition_ts with
    reason_code=CORRECTION) -- never overwrite; `validate=False` is the
    explicit escape hatch a correction row uses to bypass the normal
    chain/terminal checks (a correction is, by definition, revising a past
    claim, not extending the live chain).

    [PHASE 7A-P1 semantic closure, round 3] `validate=False` safety contract
    (fail-closed): permitted ONLY for reason_code=CORRECTION with a
    `correction_of` that references an EXISTING transition_id -- a normal
    transition writer (e.g. backfill_lifecycle's SIGNAL_BIRTH inserts) can
    never reach this bypass by construction (both conditions below are
    required together). A given `correction_of` target may be corrected at
    most ONCE (a second, DIFFERENT correction attempt against an
    already-corrected transition is rejected) -- re-submitting the identical
    correction tuple remains a silent idempotent no-op (caught by the
    existing-tuple check above, before these guards ever run)."""
    if reason_code not in {r.value for r in LifecycleReasonCode}:
        raise LifecycleIntegrityViolation(f"invalid reason_code: {reason_code!r}")
    if known_at_ts < transition_ts:
        raise LifecycleIntegrityViolation(
            f"known-at violation: known_at_ts={known_at_ts} < transition_ts={transition_ts}")

    # Idempotency check FIRST, before validation: re-submitting the exact
    # tuple that is already recorded must be a silent no-op even though the
    # chain/genesis/terminal checks in validate_transition() would now
    # reject it (the chain has already advanced past this point) -- only
    # genuinely NEW transitions are validated against the live chain state.
    tid = _transition_id(signal_id, previous_status, new_status, transition_ts, reason_code,
                          transition_version, observation_mode)
    existing = conn.execute(
        "SELECT 1 FROM ami_lifecycle_transitions WHERE transition_id=?", (tid,)
    ).fetchone()
    if existing:
        return tid  # idempotent no-op: identical tuple already recorded

    if not validate:
        if correction_of is None:
            raise LifecycleIntegrityViolation(
                "validate=False requires an explicit correction_of reference -- fail-closed: "
                "bypassing chain/terminal checks without a supersession target is forbidden")
        if reason_code != LifecycleReasonCode.CORRECTION.value:
            raise LifecycleIntegrityViolation(
                f"validate=False is only permitted for reason_code=CORRECTION, got {reason_code!r}")
        target_exists = conn.execute(
            "SELECT 1 FROM ami_lifecycle_transitions WHERE transition_id=?", (correction_of,)
        ).fetchone()
        if target_exists is None:
            raise LifecycleIntegrityViolation(
                f"correction_of={correction_of!r} does not reference an existing transition -- "
                "fail-closed: cannot supersede a non-existent transition")
        already_corrected = conn.execute(
            "SELECT transition_id FROM ami_lifecycle_transitions WHERE correction_of=?", (correction_of,)
        ).fetchone()
        if already_corrected is not None:
            raise LifecycleIntegrityViolation(
                f"transition {correction_of!r} has already been corrected by "
                f"{already_corrected[0]!r} -- a transition may only be corrected once")
    else:
        validate_transition(conn, signal_id, previous_status, new_status)

    now = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO ami_lifecycle_transitions (transition_id, signal_id, previous_status, new_status, "
        "transition_ts, known_at_ts, reason_code, transition_version, observation_mode, correction_of, "
        "evidence_ref, schema_version, provenance, created_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (tid, signal_id, previous_status, new_status, transition_ts, known_at_ts, reason_code,
         transition_version, observation_mode, correction_of, evidence_ref, schema_version, provenance, now),
    )
    return tid


def rebuild_current_state(conn, signal_id: str) -> dict | None:
    """Deterministic rebuild of the CURRENT lifecycle status purely from the
    append-only ledger (never a cache, never dashboard-maintained state).
    Folds all recorded transitions for signal_id in (transition_ts,
    transition_version) order and returns the final new_status + the
    transition_id it came from. Returns None if no transitions exist."""
    rows = conn.execute(
        "SELECT transition_id, new_status, transition_ts, transition_version FROM ami_lifecycle_transitions "
        "WHERE signal_id=? ORDER BY transition_ts ASC, transition_version ASC",
        (signal_id,),
    ).fetchall()
    if not rows:
        return None
    last = rows[-1]
    return {"current_status": last[1], "as_of_transition_id": last[0],
            "as_of_transition_ts": last[2], "n_transitions": len(rows)}


def effective_lifecycle_status(conn, signal_id: str) -> dict | None:
    """[PHASE 7A-P1 semantic closure, round 3] Companion to rebuild_current_state()
    that reads ami_lifecycle_effective_transitions (superseded rows AND their
    pure-reversal corrections excluded) instead of the raw table. Always
    agrees with rebuild_current_state() for CURRENT STATUS today (every
    correction recorded so far is a pure reversal, so the raw table's
    latest-row-wins logic and this effective-ledger fold produce the same
    answer) -- this function exists as the canonical, future-proof surface so
    a later NON-reversal correction (a genuine second real transition) cannot
    silently diverge status derived from the two ledgers. `n_effective_transitions`
    is the number DOWNSTREAM interval/duration research must use -- e.g. a
    signal whose only effective row is its genesis SIGNAL_BIRTH
    (n_effective_transitions=1) has NO closed interval to compute a duration
    from, regardless of what the raw (immutable, audit-only) ledger contains."""
    rows = conn.execute(
        "SELECT transition_id, new_status, transition_ts, transition_version "
        "FROM ami_lifecycle_effective_transitions WHERE signal_id=? "
        "ORDER BY transition_ts ASC, transition_version ASC", (signal_id,),
    ).fetchall()
    if not rows:
        return None
    last = rows[-1]
    return {"current_status": last[1], "as_of_transition_id": last[0],
            "as_of_transition_ts": last[2], "n_effective_transitions": len(rows)}


def count_effective_closed_signals(conn) -> int:
    """[PHASE 7A-P1 semantic closure, round 3] Number of DISTINCT signals that
    have at least one genuine (non-superseded, non-pure-reversal) CLOSED-type
    row in the effective ledger -- i.e. a REAL closed interval a duration
    query could legitimately compute. This is the canonical existence-check
    downstream terminal/hold-duration research must call BEFORE assuming any
    signal has a valid interval (see FIELD_PROVENANCE_SPECS['terminal_ts']:
    NOT_IMPLEMENTED -- OPEN under this backfill means 'terminal status not
    validated', never 'currently live/active')."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT signal_id) FROM ami_lifecycle_effective_transitions "
        "WHERE new_status=?", (TradeLifecycleState.CLOSED.value,),
    ).fetchone()
    return row[0]
