"""AMI BIRTH-TRUNCATED CASCADE GEOMETRY -- disposable schema migration +
canonical backfill REHEARSAL, current accepted contract end-to-end:
ami.geometry.birth_truncated_cascade_geometry (immutable feature values +
field provenance) + ami.geometry.liquidation_source_quality_contract_v2
(the SOLE, field-level, append-only quality mechanism).

DISPOSABLE_DB_ONLY / NO_LIVE_CANONICAL_DB_MIGRATION / NO_AMI_EVENTS_MUTATION /
NO_OUTCOME_ANALYSIS: `run_disposable_rehearsal()` opens the real
canonical.sqlite ONLY read-only (`mode=ro`, to fingerprint schema + count
rows) and via `shutil.copy2` (read source, write only the new disposable
path) -- the real file is never opened for writing. `data/microstructure.db`
is likewise opened `mode=ro` only, and never written. No function in this
module reads MFE/MAE, returns, profit-or-loss, or any post-birth outcome
column.

Same discipline and precedent as ami/lifecycle/migration_rehearsal.py and
ami/lifecycle/short_noisy_v1_rehearsal.py.

HISTORY: an earlier revision of this module implemented its OWN row-level
gap-registry-cutoff quality classification (METHOD_B in the source-quality
reconciliation report) and passed it into
birth_truncated_cascade_geometry.backfill()'s (since-removed)
`quality_status_fn` parameter. METHOD_B was REJECTED by the operator
(absence of a gap-registry row is not evidence of completeness) in favor of
the field-level `liquidation-source-quality-contract-v2` implemented in
liquidation_source_quality_contract_v2.py. This module now composes the
geometry backfill (quality-free) with THAT module's classification/backfill
as two explicit, separately-auditable steps -- never re-implements
gap-registry-cutoff logic itself.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import sqlite3
from pathlib import Path

from ami.geometry import birth_truncated_cascade_geometry as geo
from ami.geometry import liquidation_source_quality_contract_v2 as v2
from ami.research.feature_gateway import fetch_lifecycle_signals
from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, TRAIN_FRACTION
from ami.research.w8_short_expanded_baseline import (
    _cycle_key,
    assert_zero_cycle_straddling,
    compute_global_cycle_split,
    split_rows_by_cycle_keys,
)

DIRECTION = "LONG"
GEOMETRY_SYMBOL = "ETHUSDT"
GEOMETRY_LIQ_SIDE = "SELL"
RESEARCH_CONTEXT_ID = "ami-birth-truncated-cascade-geometry-rehearsal"


def make_disposable_copy(source_path, disposable_path) -> None:
    Path(disposable_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, disposable_path)


def schema_fingerprint(conn) -> str:
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
    ).fetchall()
    text = "\n".join(f"{t}|{n}|{sql}" for t, n, sql in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _month_bucket(ts_ms: int) -> str:
    d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


# ---------------------------------------------------------------------------
# source-data readers (real-data, read-only; never mutate)
# ---------------------------------------------------------------------------

def fetch_long_signals(conn_target) -> list[dict]:
    signals = fetch_lifecycle_signals(conn_target, RESEARCH_CONTEXT_ID, symbol=GEOMETRY_SYMBOL)
    return [s for s in signals if s["direction"] == DIRECTION]


def fetch_events_by_id(conn_target, event_ids: set[str]) -> dict:
    if not event_ids:
        return {}
    placeholders = ",".join("?" for _ in event_ids)
    rows = conn_target.execute(
        f"SELECT event_id, anchor_ts_ms FROM ami_events WHERE event_id IN ({placeholders})",
        list(event_ids),
    ).fetchall()
    return {eid: {"anchor_ts_ms": ats} for eid, ats in rows}


def fetch_all_sell_liqs(conn_liq, symbol: str = GEOMETRY_SYMBOL, side: str = GEOMETRY_LIQ_SIDE) -> list[dict]:
    rows = conn_liq.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return [{"ts_ms": t, "notional": n} for t, n in rows]


# ---------------------------------------------------------------------------
# Goal E -- disposable backfill rehearsal + structural report
# ---------------------------------------------------------------------------

def _setup_composition(signals: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for s in signals:
        counts[s["setup_id"]] = counts.get(s["setup_id"], 0) + 1
    return dict(sorted(counts.items()))


def _monthly_distribution(signals: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for s in signals:
        m = _month_bucket(s["signal_birth_ts"])
        counts[m] = counts.get(m, 0) + 1
    return dict(sorted(counts.items()))


def compute_population_report(signals: list[dict]) -> dict:
    """Goal G structural/identity counts for a given population of LONG
    signal dicts (independent_cycle_id/source_event_id/signal_birth_ts
    present) -- no outcome value read anywhere."""
    split = compute_global_cycle_split(signals)
    return {
        "signal_n": len(signals),
        "source_event_n": len({s["source_event_id"] for s in signals if s["source_event_id"]}),
        "independent_cycle_n": split["total_cycle_n"],
        "train_cycle_n": split["train_cycle_n"],
        "test_cycle_n": split["test_cycle_n"],
        "cycle_straddling_violations": assert_zero_cycle_straddling(
            *split_rows_by_cycle_keys(signals, split["train_cycle_keys"], split["test_cycle_keys"])
        ),
        "monthly_distribution": _monthly_distribution(signals),
        "setup_composition": _setup_composition(signals),
        "min_bucket_n_verdict": (
            "OK" if split["train_cycle_n"] >= MIN_BUCKET_N and split["test_cycle_n"] >= MIN_BUCKET_N
            else "INSUFFICIENT_SAMPLE"
        ),
    }


def _run_backfill_pass(conn, all_sell_liqs, signals, events_by_id, evidence, provenance):
    """One geometry-backfill + field-quality-backfill pass (Goal E step,
    reused for the initial run, idempotent rerun, and post-rollback reapply)."""
    r = geo.backfill(
        conn, all_sell_liqs, signals, events_by_id,
        reconstruct_anchors_fn=_reconstruct_anchors_fn(), provenance=provenance,
    )
    quality_rows = v2.assess_geometry_rows(
        conn, events_by_id, resolved_gaps=evidence["resolved_gaps"],
        sorted_all_market_liq_ts=evidence["sorted_all_market_liq_ts"],
        earliest_liq_ts_ms=evidence["earliest_liq_ts_ms"],
    )
    qr = v2.backfill_field_quality(conn, quality_rows, provenance=provenance)
    return r, qr


def run_disposable_rehearsal(source_canonical_path, disposable_path, microstructure_path,
                              provenance: str = "birth-truncated-geometry-disposable-rehearsal") -> dict:
    report: dict = {
        "source_canonical_path": str(source_canonical_path), "disposable_path": str(disposable_path),
    }

    # --- read-only baseline against the REAL canonical.sqlite ---
    conn_ro = sqlite3.connect(f"file:{source_canonical_path}?mode=ro", uri=True)
    try:
        report["schema_fingerprint_before"] = schema_fingerprint(conn_ro)
        # [POST BATCH-AMI-BIRTH-TRUNCATED-GEOMETRY-CANONICAL-MIGRATION] once the
        # real migration is durably applied, the source itself already carries
        # ami_birth_truncated_cascade_geometry -- this flag lets callers/tests
        # distinguish "pre-migration state" (this rehearsal's rollback restores
        # the exact original fingerprint) from "post-migration steady state"
        # (rollback on the disposable copy necessarily DROPS objects the real
        # source already has, so fingerprint_before != fingerprint_after_rollback
        # by construction -- not a defect, same precedent as
        # ami.lifecycle.migration_rehearsal's test_schema_fingerprint_changes_only_by_addition).
        report["source_already_has_geometry_tables"] = conn_ro.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='ami_birth_truncated_cascade_geometry'"
        ).fetchone()[0] > 0
        report["pre_counts"] = {
            "ami_events": conn_ro.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0],
            "ami_signal_lifecycle": conn_ro.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0],
            "ami_lifecycle_path_observations": conn_ro.execute(
                "SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0],
            "experiment_registry": conn_ro.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0],
        }
    finally:
        conn_ro.close()

    # --- microstructure.db: read-only source data (never written) ---
    conn_liq = sqlite3.connect(f"file:{microstructure_path}?mode=ro", uri=True)
    try:
        all_sell_liqs = fetch_all_sell_liqs(conn_liq)
        evidence = v2.fetch_quality_evidence(conn_liq, all_sell_liqs)
    finally:
        conn_liq.close()
    report["sell_liquidation_row_n"] = len(all_sell_liqs)
    report["resolved_gap_row_n"] = len(evidence["resolved_gaps"])

    # --- disposable copy ---
    make_disposable_copy(source_canonical_path, disposable_path)
    conn = sqlite3.connect(disposable_path)
    conn.execute("PRAGMA foreign_keys=ON")

    signals = fetch_long_signals(conn)
    event_ids = {s["source_event_id"] for s in signals if s["source_event_id"]}
    events_by_id = fetch_events_by_id(conn, event_ids)

    # --- Goal G: coverage-based research gate, population identity BEFORE backfill ---
    report["all_reconstructable_candidate_population"] = compute_population_report(signals)

    # --- Goal E: schema + backfill (run 1) ---
    geo.init_schema(conn)
    v2.init_schema(conn)
    r1, qr1 = _run_backfill_pass(conn, all_sell_liqs, signals, events_by_id, evidence, provenance)
    report["backfill_run1"] = {k: v for k, v in r1.items() if k != "rejected"}
    report["backfill_run1_rejected"] = r1["rejected"]
    report["field_quality_backfill_run1"] = qr1
    hash_1 = geo.content_hash(conn)
    counts_1 = geo.row_counts(conn)
    quality_hash_1 = v2.content_hash(conn)
    quality_counts_1 = v2.row_counts(conn)

    row_quality_rows = conn.execute(
        "SELECT data_quality_status, COUNT(*) FROM ami_birth_truncated_geometry_row_quality_v2_effective "
        "GROUP BY data_quality_status"
    ).fetchall()
    report["data_quality_status_counts"] = dict(row_quality_rows)

    null_counts = {}
    for field in geo._FEATURE_FIELDS:
        null_counts[field] = conn.execute(
            f"SELECT COUNT(*) FROM ami_birth_truncated_cascade_geometry WHERE {field} IS NULL"
        ).fetchone()[0]
    report["per_feature_null_counts"] = null_counts

    accepted_signal_ids = {
        r[0] for r in conn.execute(
            "SELECT signal_id FROM ami_birth_truncated_cascade_geometry"
        ).fetchall()
    }
    accepted_signals = [s for s in signals if s["signal_id"] in accepted_signal_ids]
    report["accepted_population_report"] = compute_population_report(accepted_signals)

    complete_signal_ids = {
        r[0] for r in conn.execute(
            "SELECT g.signal_id FROM ami_birth_truncated_cascade_geometry g "
            "JOIN ami_birth_truncated_geometry_row_quality_v2_effective rq ON rq.feature_id = g.feature_id "
            "WHERE rq.data_quality_status='SOURCE_COMPLETE'"
        ).fetchall()
    }
    complete_signals = [s for s in signals if s["signal_id"] in complete_signal_ids]
    report["source_complete_only_population"] = (
        compute_population_report(complete_signals) if complete_signals else None
    )

    # --- Goal H: migration rehearsal safety ---
    # (H-1) idempotent rerun -- zero new rows, identical content
    r2, qr2 = _run_backfill_pass(conn, all_sell_liqs, signals, events_by_id, evidence, provenance + "-rerun")
    hash_2 = geo.content_hash(conn)
    counts_2 = geo.row_counts(conn)
    quality_hash_2 = v2.content_hash(conn)
    quality_counts_2 = v2.row_counts(conn)
    report["idempotent_rerun_row_counts_equal"] = (counts_1 == counts_2)
    report["idempotent_rerun_content_hash_equal"] = (hash_1 == hash_2)
    report["idempotent_rerun_accepted_n"] = r2["accepted_n"]
    report["idempotent_rerun_quality_counts_equal"] = (quality_counts_1 == quality_counts_2)
    report["idempotent_rerun_quality_hash_equal"] = (quality_hash_1 == quality_hash_2)

    # (H-2) conflicting content under the same identity fails closed (geometry)
    sample_signal_id = next(iter(accepted_signal_ids)) if accepted_signal_ids else None
    conflict_raised = False
    if sample_signal_id is not None:
        conn.execute(
            "UPDATE ami_birth_truncated_cascade_geometry SET running_notional = running_notional + 1 "
            "WHERE signal_id=?", (sample_signal_id,),
        )
        conn.commit()
        try:
            _run_backfill_pass(conn, all_sell_liqs, signals, events_by_id, evidence, provenance + "-conflict-probe")
        except geo.ImmutableGeometryConflict:
            conflict_raised = True
        # undo the synthetic mutation so downstream rollback/reapply checks
        # measure the module's own real, unmodified backfill output
        conn.execute(
            "UPDATE ami_birth_truncated_cascade_geometry SET running_notional = running_notional - 1 "
            "WHERE signal_id=?", (sample_signal_id,),
        )
        conn.commit()
    report["conflicting_content_fails_closed"] = conflict_raised

    # (H-3) old-reader compatibility: pre-existing tables byte-identical
    post_backfill_counts = {
        "ami_events": conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0],
        "ami_signal_lifecycle": conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0],
        "ami_lifecycle_path_observations": conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0],
        "experiment_registry": conn.execute("SELECT COUNT(*) FROM experiment_registry").fetchone()[0],
    }
    report["old_reader_counts_unchanged"] = (post_backfill_counts == report["pre_counts"])
    report["post_backfill_counts"] = post_backfill_counts

    # (H-4) rollback restores exact pre-migration schema/counts, reapply reproduces byte-identical state
    fingerprint_with_geometry = schema_fingerprint(conn)
    v2.rollback(conn)
    geo.rollback(conn)
    fingerprint_after_rollback = schema_fingerprint(conn)
    report["rollback_restores_pre_migration_fingerprint"] = (
        fingerprint_after_rollback == report["schema_fingerprint_before"]
    )
    counts_after_rollback = {
        "ami_events": conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0],
        "ami_signal_lifecycle": conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0],
    }
    report["rollback_preserved_existing_row_counts"] = (
        counts_after_rollback["ami_events"] == report["pre_counts"]["ami_events"]
        and counts_after_rollback["ami_signal_lifecycle"] == report["pre_counts"]["ami_signal_lifecycle"]
    )

    geo.init_schema(conn)
    v2.init_schema(conn)
    r3, qr3 = _run_backfill_pass(conn, all_sell_liqs, signals, events_by_id, evidence, provenance + "-reapply")
    hash_3 = geo.content_hash(conn)
    quality_hash_3 = v2.content_hash(conn)
    report["reapply_accepted_n"] = r3["accepted_n"]
    report["reapply_reproduces_byte_identical_content"] = (hash_3 == hash_1)
    report["reapply_reproduces_byte_identical_quality_content"] = (quality_hash_3 == quality_hash_1)
    report["schema_fingerprint_after_reapply_matches_first_migration"] = (
        schema_fingerprint(conn) == fingerprint_with_geometry
    )

    report["content_hash"] = hash_1
    report["quality_content_hash"] = quality_hash_1
    conn.close()
    return report


def _reconstruct_anchors_fn():
    from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors
    return reconstruct_anchors
