"""Disposable migration rehearsal harness for
ami/lifecycle/short_noisy_v1_rehearsal.py (SETUP_ID=SHORT_NOISY_BTC200K_CONFIRMED_V1).

DISPOSABLE_DB_ONLY / NO_REAL_CANONICAL_WRITE: mirrors
ami/lifecycle/path_migration_rehearsal.py's exact discipline -- the real
canonical.sqlite is opened ONLY `mode=ro` (fingerprint + row counts) or via
`shutil.copy2` (a read of the source, write only to the new disposable path).
data/microstructure.db (liquidations) is opened ONLY `mode=ro` -- never
copied, never written; it is pre-existing, immutable source data for THIS
batch (same posture CLAUDE.md already mandates for it).
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from ami.lifecycle.migration_rehearsal import schema_fingerprint
from ami.lifecycle.path_field_provenance import backfill_path_field_provenance
from ami.lifecycle.path_metrics import freeze_and_record
from ami.lifecycle.short_noisy_v1_rehearsal import (
    SETUP_ID,
    backfill_short_noisy_v1,
    backfill_short_noisy_v1_field_provenance,
    identify_candidates,
    rollback_short_noisy_v1,
)


def make_disposable_copy(source_path, disposable_path) -> None:
    Path(disposable_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, disposable_path)


def _signal_content_hash(conn, signal_ids: list[str]) -> str:
    if not signal_ids:
        return hashlib.sha256(b"").hexdigest()
    placeholders = ",".join("?" for _ in signal_ids)
    rows = conn.execute(
        f"SELECT signal_id, setup_id, setup_version, source_event_id, independent_cycle_id, symbol, "
        f"direction, timeframe, route_version, signal_birth_ts, lifecycle_status, lifecycle_reason_code, "
        f"evidence_layer, is_proxy, source_hash FROM ami_signal_lifecycle "
        f"WHERE signal_id IN ({placeholders}) ORDER BY signal_id",
        signal_ids,
    ).fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode("utf-8")).hexdigest()


def _overlap_matrix(conn, candidates: list[dict], signal_ids: list[str]) -> dict:
    membership = conn.execute(
        "SELECT event_id, candidate_cycle_key FROM event_cycle_membership "
        "WHERE cycle_definition_version='canonical-v1' AND is_canonical=1"
    ).fetchall()
    event_to_cycle = {r[0]: r[1] for r in membership}
    event_ids = [c["event_id"] for c in candidates]
    qualifying_cycles = sorted({event_to_cycle[eid] for eid in event_ids if eid in event_to_cycle})

    other_sig_rows = conn.execute(
        "SELECT source_event_id, setup_id, direction FROM ami_signal_lifecycle WHERE setup_id != ?",
        (SETUP_ID,),
    ).fetchall()
    signals_by_event: dict[str, list[tuple[str, str]]] = {}
    for source_event_id, setup_id, direction in other_sig_rows:
        signals_by_event.setdefault(source_event_id, []).append((setup_id, direction))

    cycle_to_events: dict[str, list[str]] = {}
    for eid, cyc in membership:
        cycle_to_events.setdefault(cyc, []).append(eid)

    cycles_already_short = 0
    cycles_newly_short = 0
    for cyc in qualifying_cycles:
        member_events = cycle_to_events.get(cyc, [])
        dirs = {d for me in member_events for (_s, d) in signals_by_event.get(me, [])}
        if "SHORT" in dirs:
            cycles_already_short += 1
        else:
            cycles_newly_short += 1

    route_overlap: dict[str, int] = {}
    for eid in event_ids:
        for setup_id, _direction in signals_by_event.get(eid, []):
            route_overlap[setup_id] = route_overlap.get(setup_id, 0) + 1

    return {
        "source_event_n": len(set(event_ids)),
        "distinct_qualifying_cycle_n": len(qualifying_cycles),
        "cycles_already_short_represented": cycles_already_short,
        "cycles_newly_short_represented": cycles_newly_short,
        "route_overlap_counts": route_overlap,
        "new_signal_n": len(signal_ids),
    }


def run_disposable_rehearsal(source_canonical_path, disposable_path, microstructure_db_path) -> dict:
    report: dict = {
        "source_canonical_path": str(source_canonical_path),
        "disposable_path": str(disposable_path),
        "microstructure_db_path": str(microstructure_db_path),
        "setup_id": SETUP_ID,
    }

    # 1. baseline: real DB fingerprint + counts, read-only. Baselines used for the
    # "unchanged"/"preserved" comparisons below are deliberately SCOPED to exclude SETUP_ID's
    # own rows from the start (WHERE setup_id != SETUP_ID / signal_id NOT IN <setup's signals>)
    # -- this makes every comparison correct regardless of whether SETUP_ID's 54 rows already
    # exist in source_canonical_path (e.g. a real, already-migrated canonical.sqlite) or not
    # (e.g. a pristine copy, or one where rollback_short_noisy_v1() was just run). An earlier
    # version of this harness compared against an UNFILTERED total-row-count baseline, which
    # only happened to work while SETUP_ID had never been applied anywhere -- fixed here.
    conn_ro = sqlite3.connect(f"file:{source_canonical_path}?mode=ro", uri=True)
    try:
        report["schema_fingerprint_before"] = schema_fingerprint(conn_ro)
        pre_signal_n = conn_ro.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0]
        pre_transition_n = conn_ro.execute("SELECT COUNT(*) FROM ami_lifecycle_transitions").fetchone()[0]
        pre_path_n = conn_ro.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0]
        pre_provenance_n = conn_ro.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
        pre_event_n = conn_ro.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
        pre_setup_signal_n = conn_ro.execute(
            "SELECT COUNT(*) FROM ami_signal_lifecycle WHERE setup_id=?", (SETUP_ID,)
        ).fetchone()[0]
        pre_other_setup_signal_n = conn_ro.execute(
            "SELECT COUNT(*) FROM ami_signal_lifecycle WHERE setup_id != ?", (SETUP_ID,)
        ).fetchone()[0]
        pre_other_setup_path_n = conn_ro.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE signal_id NOT IN "
            "(SELECT signal_id FROM ami_signal_lifecycle WHERE setup_id=?)", (SETUP_ID,),
        ).fetchone()[0]
        pre_other_setup_provenance_n = conn_ro.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE signal_id NOT IN "
            "(SELECT signal_id FROM ami_signal_lifecycle WHERE setup_id=?)", (SETUP_ID,),
        ).fetchone()[0]
    finally:
        conn_ro.close()
    # 0 (pristine/rolled-back source) or already-canonicalized (== the eventual candidate_n) are
    # both valid states; anything else (partial/duplicate write) is a real collision
    report["pre_existing_setup_id_signal_n"] = pre_setup_signal_n

    # 2. disposable copy (write side); liquidations opened read-only against the REAL,
    # never-copied microstructure.db
    make_disposable_copy(source_canonical_path, disposable_path)
    conn = sqlite3.connect(disposable_path)
    conn_liq = sqlite3.connect(f"file:{microstructure_db_path}?mode=ro", uri=True)

    try:
        # 3. deterministic-identity check: identify twice, independently, before any write
        candidates_a = identify_candidates(conn, conn_liq)
        candidates_b = identify_candidates(conn, conn_liq)
        report["identity_deterministic_across_reruns"] = (
            [c["event_id"] for c in candidates_a] == [c["event_id"] for c in candidates_b]
            and [(c["noisy_ts"], c["conf_ts"]) for c in candidates_a]
                == [(c["noisy_ts"], c["conf_ts"]) for c in candidates_b]
        )
        report["candidate_n"] = len(candidates_a)
        report["candidate_source_event_n"] = len({c["event_id"] for c in candidates_a})
        report["all_conf_ts_after_noisy_ts_plus_5m"] = all(
            c["conf_ts"] > c["noisy_ts"] + 5 * 60_000 for c in candidates_a
        )
        report["all_noisy_ts_after_anchor_plus_1m"] = all(
            c["noisy_ts"] > c["anchor_ts_ms"] + 60_000 for c in candidates_a
        )

        # 4. run 1: lifecycle backfill + field provenance + path provenance + path metrics
        r1 = backfill_short_noisy_v1(conn, conn, conn_liq)
        signal_ids = r1["signal_ids"]
        report["duplicate_conf_ts_did_not_merge_distinct_events"] = (
            len(set(signal_ids)) == len(signal_ids) == r1["candidate_n"]
        )
        report["all_signal_births_equal_conf_ts"] = all(
            s["signal_birth_ts"] == s["_conf_ts"] for s in r1["signals"]
        )
        fp1 = backfill_short_noisy_v1_field_provenance(conn, signal_ids)
        path_fp1 = backfill_path_field_provenance(conn, signal_ids)
        path1 = freeze_and_record(conn)
        content_hash_1 = _signal_content_hash(conn, signal_ids)

        report["lifecycle_run1"] = {k: v for k, v in r1.items() if k not in ("signals",)}
        report["field_provenance_run1"] = fp1
        report["path_field_provenance_run1"] = {k: v for k, v in path_fp1.items()}
        report["path_metrics_run1_summary"] = {k: v for k, v in path1.items() if k != "rows"}

        # 5. rerun (run 2) -- idempotency across every layer
        r2 = backfill_short_noisy_v1(conn, conn, conn_liq)
        fp2 = backfill_short_noisy_v1_field_provenance(conn, signal_ids)
        path_fp2 = backfill_path_field_provenance(conn, signal_ids)
        path2 = freeze_and_record(conn)
        content_hash_2 = _signal_content_hash(conn, signal_ids)

        report["idempotent_signal_upsert_count"] = (r1["signals_upserted"] == r2["signals_upserted"])
        report["idempotent_transitions_zero_new_on_rerun"] = (r2["transitions_attempted"] == r1["transitions_attempted"])
        report["idempotent_content_hash"] = (content_hash_1 == content_hash_2)
        report["idempotent_field_provenance_rows"] = (fp1["provenance_rows_written"] == fp2["provenance_rows_written"])
        report["idempotent_path_field_provenance_rows"] = (
            path_fp1["provenance_rows_written_this_call"] == path_fp2["provenance_rows_written_this_call"]
        )
        # scoped to path_definition_version='path-v2' -- freeze_and_record() (called above) always
        # writes exactly ONE row per (signal, horizon) under that version; counting ALL versions
        # unscoped would double-count any signal that ALSO happens to have a separate
        # ami.lifecycle.path_candle_repair_correction correction row (a different, independent
        # identity space -- see that module's docstring for why a corrected row is inserted
        # alongside, never in place of, the original "path-v2" row)
        row_count_new_path_obs_1 = conn.execute(
            f"SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE signal_id IN "
            f"({','.join('?' for _ in signal_ids)}) AND path_definition_version='path-v2'", signal_ids,
        ).fetchone()[0]
        report["new_path_observation_row_n"] = row_count_new_path_obs_1
        report["new_path_observation_row_n_expected_max"] = len(signal_ids) * 4

        # 6. old-reader compatibility -- pre-existing 270-signal population untouched
        post_pre_existing_signal_n = conn.execute(
            "SELECT COUNT(*) FROM ami_signal_lifecycle WHERE setup_id != ?", (SETUP_ID,)
        ).fetchone()[0]
        post_pre_existing_path_n = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE signal_id NOT IN "
            f"(SELECT signal_id FROM ami_signal_lifecycle WHERE setup_id=?)", (SETUP_ID,),
        ).fetchone()[0]
        post_pre_existing_provenance_n = conn.execute(
            "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE signal_id NOT IN "
            f"(SELECT signal_id FROM ami_signal_lifecycle WHERE setup_id=?)", (SETUP_ID,),
        ).fetchone()[0]
        post_event_n = conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0]
        report["old_reader_pre_existing_signal_n_unchanged"] = (post_pre_existing_signal_n == pre_other_setup_signal_n)
        report["old_reader_pre_existing_path_n_unchanged"] = (post_pre_existing_path_n == pre_other_setup_path_n)
        report["old_reader_pre_existing_provenance_n_unchanged"] = (
            post_pre_existing_provenance_n == pre_other_setup_provenance_n
        )
        report["old_reader_event_n_unchanged"] = (post_event_n == pre_event_n)

        # 7. overlap matrix (final, pre-rollback)
        report["overlap_matrix"] = _overlap_matrix(conn, candidates_a, signal_ids)

        # 8. final distributions
        report["final_observation_status_distribution"] = path1["status_counts"]
        report["final_volatility_status_distribution"] = path1["volatility_status_counts"]

        # 9. rollback rehearsal
        rb = rollback_short_noisy_v1(conn)
        report["rollback_result"] = rb
        report["rollback_signal_count_matches"] = (rb["signals_deleted"] == len(signal_ids))
        # after rollback, ALL of SETUP_ID's rows are gone -- total counts must equal the
        # OTHER-setups-only baselines (not the possibly-already-including-SETUP_ID totals)
        report["rollback_preserved_pre_existing_signal_n"] = (
            conn.execute("SELECT COUNT(*) FROM ami_signal_lifecycle").fetchone()[0] == pre_other_setup_signal_n
        )
        report["rollback_preserved_pre_existing_path_n"] = (
            conn.execute("SELECT COUNT(*) FROM ami_lifecycle_path_observations").fetchone()[0] == pre_other_setup_path_n
        )
        report["rollback_preserved_pre_existing_provenance_n"] = (
            conn.execute("SELECT COUNT(*) FROM ami_lifecycle_field_provenance").fetchone()[0]
            == pre_other_setup_provenance_n
        )
        report["rollback_preserved_event_n"] = (
            conn.execute("SELECT COUNT(*) FROM ami_events").fetchone()[0] == pre_event_n
        )

        # 10. reapply after rollback -- must reproduce byte-identical content
        r3 = backfill_short_noisy_v1(conn, conn, conn_liq)
        backfill_short_noisy_v1_field_provenance(conn, r3["signal_ids"])
        backfill_path_field_provenance(conn, r3["signal_ids"])
        path3 = freeze_and_record(conn)
        content_hash_3 = _signal_content_hash(conn, r3["signal_ids"])
        report["reapply_signal_ids_match"] = (set(r3["signal_ids"]) == set(signal_ids))
        report["reapply_content_hash_matches_pre_rollback"] = (content_hash_3 == content_hash_1)
        report["reapply_path_observation_row_n"] = conn.execute(
            f"SELECT COUNT(*) FROM ami_lifecycle_path_observations WHERE signal_id IN "
            f"({','.join('?' for _ in r3['signal_ids'])}) AND path_definition_version='path-v2'",
            r3["signal_ids"],
        ).fetchone()[0]

        # 11. schema fingerprint unchanged throughout (no DDL was ever applied by this batch)
        report["schema_fingerprint_after"] = schema_fingerprint(conn)
        report["schema_fingerprint_unchanged"] = (
            report["schema_fingerprint_after"] == report["schema_fingerprint_before"]
        )
    finally:
        conn_liq.close()
        conn.close()

    return report
