"""PHASE 7B-CANON: controlled canonical migration/backfill entry point for
the path-v2 package (ami_lifecycle_path_observations + its field-level
provenance).

This module composes the already-validated pieces (schema is now merged
into ami.warehouse.schema's shared init_schema() as _SCHEMA_PHASE7B;
ami.lifecycle.path_metrics.freeze_and_record; ami.lifecycle.
path_field_provenance.backfill_path_field_provenance) into ONE auditable,
idempotent entry point, mirroring the controlled-sequence discipline the
Phase 7A-P canonical migration established (SYSTEM_STATE Sec69) after the
SCHEMA_DRIFT_BLOCKER incident: schema first, then backfill, in one script,
never interleaved with an unrelated full-suite test run.

NOT_CALLED_AUTOMATICALLY: `run_canonical_migration()` takes an explicit
connection -- it is never invoked as an import side effect, and the module
has no module-level call to ami.warehouse.schema.connect(DEFAULT_PATH). The
operator-approved migration itself is run once, explicitly, from a
one-off script/shell invocation (recorded in MIGRATION_LOG.md), not from
this module's import.
"""
from __future__ import annotations

from ami.lifecycle.path_field_provenance import backfill_path_field_provenance
from ami.lifecycle.path_metrics import fetch_signals, freeze_and_record


def run_canonical_migration(conn, provenance: str = "phase-7b-canon-path-metrics-migration") -> dict:
    """Idempotent: schema creation is additive-only (CREATE TABLE/INDEX IF
    NOT EXISTS, already applied by ami.warehouse.schema.init_schema() before
    this is called); freeze_and_record()/backfill_path_field_provenance() are
    themselves upsert-idempotent. Rerunning this function against unchanged
    source data (ami_signal_lifecycle + ami_candles) produces byte-identical
    ami_lifecycle_path_observations content and an unchanged field-provenance
    row count.

    Caller is responsible for calling ami.warehouse.schema.init_schema(conn)
    first (this function does not call it itself, to keep schema-application
    and data-backfill as two explicit, separately-auditable steps, matching
    the Phase 7A-P sequence: migrate schema -> then backfill)."""
    path_result = freeze_and_record(conn, provenance=provenance)
    signal_ids = [s["signal_id"] for s in fetch_signals(conn)]
    provenance_result = backfill_path_field_provenance(conn, signal_ids, provenance=provenance)
    return {"path_metrics": path_result, "field_provenance": provenance_result}
