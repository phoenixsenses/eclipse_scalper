"""AMI BIRTH-TRUNCATED CASCADE GEOMETRY -- controlled canonical migration/
backfill entry point.

Composes the already-validated pieces (schema now merged into
ami.warehouse.schema's shared init_schema() as _SCHEMA_PHASE_GEOMETRY;
ami.geometry.birth_truncated_cascade_geometry.backfill; ami.geometry.
liquidation_source_quality_contract_v2.assess_geometry_rows/
backfill_field_quality) into ONE auditable, idempotent entry point, mirroring
the controlled-sequence discipline ami.lifecycle.path_canonical_migration
established (schema first, then backfill, in one script, never interleaved
with an unrelated full-suite test run).

NOT_CALLED_AUTOMATICALLY: `run_canonical_migration()` takes explicit
connections -- it is never invoked as an import side effect, and this module
has no module-level call to ami.warehouse.schema.connect(DEFAULT_PATH). The
operator-approved migration itself is run once, explicitly, from a one-off
script/shell invocation (recorded in MIGRATION_LOG.md), not from this
module's import.
"""
from __future__ import annotations

from ami.geometry import birth_truncated_cascade_geometry as geo
from ami.geometry import liquidation_source_quality_contract_v2 as v2
from ami.geometry.birth_truncated_geometry_rehearsal import (
    fetch_all_sell_liqs,
    fetch_events_by_id,
    fetch_long_signals,
)


def run_canonical_migration(conn, conn_liq,
                             provenance: str = "birth-truncated-geometry-canonical-migration") -> dict:
    """Idempotent: schema creation is additive-only (CREATE TABLE/INDEX/VIEW
    IF NOT EXISTS, already applied by ami.warehouse.schema.init_schema(conn)
    before this is called); geo.backfill()/v2.backfill_field_quality() are
    themselves upsert-idempotent (identical content) / fail-closed (content
    conflict under the same identity). Rerunning this function against
    unchanged source data (ami_signal_lifecycle + ami_events +
    data/microstructure.db:liquidations) produces byte-identical
    ami_birth_truncated_cascade_geometry + ami_birth_truncated_geometry_
    field_quality_v2 content.

    Caller is responsible for calling ami.warehouse.schema.init_schema(conn)
    first (this function does not call it itself, to keep schema-application
    and data-backfill as two explicit, separately-auditable steps, matching
    the Phase 7A-P/7B-CANON sequence: migrate schema -> then backfill).

    conn: writable connection to the canonical.sqlite (or its disposable copy).
    conn_liq: READ-ONLY connection to data/microstructure.db -- never written.
    """
    all_sell_liqs = fetch_all_sell_liqs(conn_liq)
    evidence = v2.fetch_quality_evidence(conn_liq, all_sell_liqs)

    signals = fetch_long_signals(conn)
    event_ids = {s["source_event_id"] for s in signals if s["source_event_id"]}
    events_by_id = fetch_events_by_id(conn, event_ids)

    geometry_result = geo.backfill(
        conn, all_sell_liqs, signals, events_by_id,
        reconstruct_anchors_fn=_reconstruct_anchors_fn(), provenance=provenance,
    )
    quality_rows = v2.assess_geometry_rows(
        conn, events_by_id, resolved_gaps=evidence["resolved_gaps"],
        sorted_all_market_liq_ts=evidence["sorted_all_market_liq_ts"],
        earliest_liq_ts_ms=evidence["earliest_liq_ts_ms"],
    )
    quality_result = v2.backfill_field_quality(conn, quality_rows, provenance=provenance)
    return {"geometry": geometry_result, "field_quality": quality_result}


def _reconstruct_anchors_fn():
    from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors
    return reconstruct_anchors
