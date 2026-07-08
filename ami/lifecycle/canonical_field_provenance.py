"""PHASE 7A-P1: Canonical field-level provenance closure.

DISPOSABLE_DB_ONLY / NO_CANONICAL_DB_WRITE: this module is never called
against ami.warehouse.schema.DEFAULT_PATH in this batch -- only against
disposable-copy connections (see ami/lifecycle/provenance_rehearsal.py).

RECONCILIATION (done before designing anything new, per operator instruction):
searched every existing canonical table for a per-row-per-field provenance/
classification object. None exists:
  - data_quality_events      -- FEED-level gap/stale/schema events, not
                                per-signal-field derivation records.
  - causal_assumption_registry -- causal-DAG assumed_arrow/confounder
                                tracking, unrelated axis.
  - evidence_contamination   -- hypothesis freeze/contamination tracking.
  - mt_family_registry       -- multiple-testing family statistics.
  - market_structure_versions -- fee/contract/tick versioning.
  - ami_signal_lifecycle.provenance -- a single flat batch-tag string
    (e.g. "batch-p7a-canonical-migration"), identical across ALL rows of a
    backfill run -- row/batch-level, not per-field. This is the exact gap
    the operator identified.
No existing object is reused/extended; a NEW, minimal, general table is
added (not a second "truth layer" -- it never stores a competing copy of
signal_id/source_event_id/independent_cycle_id/timestamps; it only records,
per (signal_id, field_name), HOW that field's value in ami_signal_lifecycle
was derived).

SEMANTICS (operator-specified, load-bearing):
  - ami_signal_lifecycle.is_proxy/evidence_layer is a ROW-level axis
    (is the underlying ANCHOR EVENT real or proxy-detected) and is
    UNTOUCHED here -- it does NOT mean "every field on this row is a real
    observation." This module adds the ORTHOGONAL field-level axis.
  - Missing field-level provenance for a field that requires it must FAIL
    validation, never silently default to "real"/safe.
  - Route/setup-name-derived `direction` must never be exposed as
    REAL_OBSERVED_DIRECTION by any canonical query surface -- enforced here
    by a VIEW that always joins the value to its classification.
"""
from __future__ import annotations
import hashlib
import time

FIELD_PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_VERSION = "field-provenance-v1"

_VALID_CLASSIFICATIONS = {
    "DETERMINISTIC_HISTORICAL_SAFE", "HISTORICAL_PROXY", "FORWARD_ONLY",
    "NOT_IMPLEMENTED", "BLOCKED_BY_DATA",
}


class FieldProvenanceViolation(Exception):
    """Raised on missing/duplicate/malformed field-provenance records, or an
    attempt to expose a proxy-derived field without its classification."""


_SCHEMA = """
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
"""

# The canonical query/view/export contract: `direction` must never be read
# bare -- any consumer wanting to present it externally goes through this
# view, which always carries the field_classification/is_proxy/derivation
# alongside the value. Raw `SELECT direction FROM ami_signal_lifecycle`
# remains possible (SQL cannot forbid it) but is documented in
# SCHEMA_DICTIONARY.md as internal-bookkeeping-only, never external-facing.
_VIEW = f"""
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
    AND fp.provenance_version = '{PROVENANCE_VERSION}';
"""


def init_field_provenance_schema(conn) -> None:
    """Additive only. Safe to call repeatedly."""
    conn.executescript(_SCHEMA)
    conn.executescript(_VIEW)
    conn.commit()


def rollback_field_provenance_schema(conn) -> None:
    conn.executescript(
        "DROP VIEW IF EXISTS ami_lifecycle_direction_view;\n"
        "DROP TABLE IF EXISTS ami_lifecycle_field_provenance;\n"
    )
    conn.commit()


def _provenance_id(signal_id: str, field_name: str, provenance_version: str) -> str:
    key = f"{signal_id}|{field_name}|{provenance_version}"
    return "FPR-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


# Exact, code-grounded description per field -- reused verbatim from
# ami.lifecycle.canonical_backfill.FIELD_CLASSIFICATION (not re-derived,
# not guessed): each entry documents the ACTUAL implemented derivation, not
# an assumption about what it might be.
FIELD_PROVENANCE_SPECS: dict[str, dict] = {
    "source_event_id": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_copy",
        "source_reference": "ami_events.event_id",
        "limitations": None,
    },
    "independent_cycle_id": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_copy_via_join",
        "source_reference": "event_cycle_membership.candidate_cycle_key "
                             "(cycle_definition_version=canonical-v1, is_canonical=1)",
        "limitations": "NULL when the source event has no canonical-v1 cycle membership row",
    },
    "signal_birth_ts": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_copy",
        "source_reference": "ami_events.anchor_ts_ms",
        "limitations": None,
    },
    "terminal_ts": {
        # [PHASE 7A-P1 semantic closure, round 2] Was DETERMINISTIC_HISTORICAL_SAFE
        # (verbatim ami_events.event_end_ts_ms). Round 1 nulled this column
        # but left a live contradiction: the ledger still wrote a
        # TERMINAL_CLOSE transition (OPEN->CLOSED) off the SAME unvalidated
        # timestamp, so rebuild_current_state() still asserted CLOSED.
        # Round 2 corrects the ledger itself -- backfill_lifecycle() no
        # longer derives ANY status/transition claim from event_end_ts_ms
        # (every signal is OPEN/SIGNAL_BIRTH only), and
        # correct_unvalidated_terminal_close() retroactively reverses any
        # pre-existing TERMINAL_CLOSE-derived CLOSED status via an appended
        # CORRECTION transition (append-only -- the original row is never
        # edited/deleted).
        "field_classification": "NOT_IMPLEMENTED",
        "derivation_method": "not_computed",
        "source_reference": "N/A -- ami_events.event_end_ts_ms exists as a distinct source-event "
                             "concept (available via a direct join, source-event semantics) but is "
                             "intentionally NOT copied here and no longer drives lifecycle_status "
                             "(semantic mismatch: source-event window end != validated signal terminal)",
        "limitations": "No genuinely validated lifecycle-terminal detection mechanism exists yet "
                       "(future, separately-approved phase). The ami_lifecycle_transitions ledger no "
                       "longer contains any NEW TERMINAL_CLOSE row derived from event_end_ts_ms; any "
                       "such row inherited from a prior (now-superseded) backfill run is explicitly "
                       "reversed via a CORRECTION transition, never silently left in place.",
    },
    "symbol": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_copy",
        "source_reference": "ami_events.symbol",
        "limitations": None,
    },
    "setup_id": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "route_version_comma_split",
        "source_reference": "ami_events.route_version (ami.lifecycle.canonical_backfill._setup_tokens)",
        "limitations": "'UNKNOWN_SETUP' when route_version is NULL/empty",
    },
    "route_version": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_copy",
        "source_reference": "ami_events.route_version",
        "limitations": None,
    },
    "evidence_layer": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_mapping",
        "source_reference": "ami_events.source_quality (REAL_LIQUIDATION->REAL, PROXY_*->PROXY)",
        "limitations": None,
    },
    "is_proxy": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_mapping",
        "source_reference": "ami_events.source_quality (REAL_LIQUIDATION->0, PROXY_*->1)",
        "limitations": "Row-level anchor-event-source-quality axis -- distinct from, and not "
                       "a substitute for, this table's own per-field provenance for OTHER "
                       "columns (e.g. direction).",
    },
    "setup_version": {
        # [PHASE 7A-P1 semantic closure] Was frozen_placeholder_constant
        # (canonical column stored "setup-v1"). Corrected: no applied
        # per-route setup-versioning source exists, so the canonical column
        # is now NULL. SETUP_VERSION_DEFAULT still feeds signal_id's hash
        # (identity-generation input, unchanged -- identity NOT recomputed)
        # but is never exposed as canonical setup_version data.
        # identity_version and setup_version are distinct concepts, never
        # conflated.
        "field_classification": "NOT_IMPLEMENTED",
        "derivation_method": "not_computed",
        "source_reference": "N/A -- no applied per-route setup-versioning source exists; canonical "
                             "value is NULL (SETUP_VERSION_DEFAULT is an identity-hash input only, "
                             "tracked separately under identity_version, never stored here)",
        "limitations": "No per-route rule/parameter versioning exists anywhere upstream. "
                       "SETUP_VERSION_DEFAULT is formally an UNKNOWN_SETUP_VERSION identity token "
                       "(ami.lifecycle.canonical_schema.UNKNOWN_SETUP_VERSION_TOKEN) -- if a real "
                       "setup_version source is implemented later, existing signal_id values keyed "
                       "on this token must NOT be silently mutated (see that constant's docstring "
                       "for the pre-declared supersession contract: new IDENTITY_VERSION or an "
                       "explicit supersession link, decided at that future time).",
    },
    "direction": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "route_name_prefix_heuristic_v1",
        "source_reference": "ami_signal_lifecycle.setup_id via "
                             "ami.lifecycle.canonical_schema.classify_direction_from_setup_id "
                             "(prefix match: LONG_ -> LONG; SHORT_/BUY_FADE -> SHORT; else UNKNOWN)",
        "limitations": "No canonical route-direction registry exists; unrecognized prefixes "
                       "return UNKNOWN rather than being guessed. BUY_FADE is classified SHORT "
                       "per failure_archive #8 (\"BUY-side short-squeeze fade\") and #19 "
                       "(\"BUYFADE SHORT->SHORT re-entry\") precedent, not a verified route "
                       "contract. NEVER exposed as REAL_OBSERVED_DIRECTION by any canonical "
                       "view/query/export -- see ami_lifecycle_direction_view.",
    },
    "timeframe": {
        "field_classification": "NOT_IMPLEMENTED",
        "derivation_method": "not_computed",
        "source_reference": "N/A (no upstream source -- field never populated)",
        "limitations": "No canonical per-signal timeframe assignment exists yet.",
    },
    "first_known_ts": {
        "field_classification": "NOT_IMPLEMENTED",
        "derivation_method": "not_computed",
        "source_reference": "ami_events.feature_available_ts_ms (reserved column, always NULL upstream)",
        "limitations": "Never populated by any upstream pipeline.",
    },
    "first_executable_ts": {
        "field_classification": "FORWARD_ONLY",
        "derivation_method": "not_computed_by_design",
        "source_reference": "N/A (no historical-safe source exists for true executability timing)",
        "limitations": "True live-executability timestamp cannot be reconstructed "
                       "retroactively from candle/mark data; intentionally left NULL.",
    },
    "last_confirmation_ts": {
        "field_classification": "NOT_IMPLEMENTED",
        "derivation_method": "not_computed",
        "source_reference": "N/A (no upstream source -- concept not yet defined)",
        "limitations": "\"Confirmation\" is undefined without a path/progress engine (out of "
                       "scope for this phase -- NO_TIMING_PATH_ENGINE).",
    },
    "invalidation_ts": {
        "field_classification": "NOT_IMPLEMENTED",
        "derivation_method": "not_computed",
        "source_reference": "N/A (no upstream source -- concept not yet defined)",
        "limitations": "Same reason as last_confirmation_ts.",
    },
}


def backfill_field_provenance(conn_target, signal_ids: list[str],
                              provenance: str = "phase-7a-p1-field-provenance-closure") -> dict:
    """Idempotent: rerunning for the same signal_ids/fields upserts identical
    rows (unchanged row count and content). Every signal_id gets exactly one
    provenance row per field in FIELD_PROVENANCE_SPECS."""
    now = int(time.time() * 1000)
    n_written = 0
    for signal_id in signal_ids:
        for field_name, spec in FIELD_PROVENANCE_SPECS.items():
            pid = _provenance_id(signal_id, field_name, PROVENANCE_VERSION)
            is_proxy = 1 if spec["field_classification"] == "HISTORICAL_PROXY" else 0
            conn_target.execute(
                "INSERT INTO ami_lifecycle_field_provenance (provenance_id, signal_id, field_name, "
                "field_classification, is_proxy, derivation_method, source_reference, limitations, "
                "provenance_version, schema_version, code_commit, source_hash, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(signal_id, field_name, provenance_version) DO UPDATE SET "
                "field_classification=excluded.field_classification, is_proxy=excluded.is_proxy, "
                "derivation_method=excluded.derivation_method, source_reference=excluded.source_reference, "
                "limitations=excluded.limitations",
                (pid, signal_id, field_name, spec["field_classification"], is_proxy,
                 spec["derivation_method"], spec["source_reference"], spec["limitations"],
                 PROVENANCE_VERSION, FIELD_PROVENANCE_SCHEMA_VERSION,
                 "ami/lifecycle/canonical_field_provenance.py", None, now),
            )
            n_written += 1
    conn_target.commit()
    return {"signals_covered": len(signal_ids), "fields_per_signal": len(FIELD_PROVENANCE_SPECS),
            "provenance_rows_written": n_written}


def validate_field_provenance_complete(conn, signal_ids: list[str],
                                        required_fields: list[str] | None = None) -> None:
    """Raises FieldProvenanceViolation if ANY (signal_id, field) pair required
    by `required_fields` (default: all of FIELD_PROVENANCE_SPECS) has no
    provenance row -- missing provenance is a hard failure, never silently
    treated as DETERMINISTIC_HISTORICAL_SAFE/real."""
    fields = required_fields or list(FIELD_PROVENANCE_SPECS.keys())
    existing = {
        (r[0], r[1]) for r in conn.execute(
            "SELECT signal_id, field_name FROM ami_lifecycle_field_provenance "
            "WHERE provenance_version=?", (PROVENANCE_VERSION,)
        ).fetchall()
    }
    missing = [(sid, f) for sid in signal_ids for f in fields if (sid, f) not in existing]
    if missing:
        raise FieldProvenanceViolation(
            f"missing field provenance for {len(missing)} (signal_id, field) pairs "
            f"(e.g. {missing[:3]}) -- refusing to assume DETERMINISTIC_HISTORICAL_SAFE/real"
        )
