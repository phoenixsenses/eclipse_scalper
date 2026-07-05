"""AMI BIRTH-TRUNCATED CASCADE GEOMETRY -- immutable feature schema + backfill.

This module operates on a caller-supplied connection; it never opens
ami.warehouse.schema.DEFAULT_PATH itself. The real canonical.sqlite is only
ever written by an explicit, separately-audited migration entry point
(ami/geometry/birth_truncated_geometry_canonical_migration.py), the same
discipline as ami/lifecycle/path_canonical_migration.py composing
path_metrics.freeze_and_record()/path_field_provenance.backfill_path_field_provenance().
No function in this module is called anywhere in a live/shadow/execution path.

BACKGROUND (AMI BIRTH-TRUNCATED CASCADE GEOMETRY definition-audit batch):
`ami_events.notional` / `event_count` / `duration_seconds` / `event_end_ts_ms`
are derived from the S34 shadow ledger's per-ROUTE OPEN/CLOSE observations
(ami/identity/shadow_ledger_ingest.py), not from raw liquidation rows --
`event_end_ts_ms > signal_birth_ts` for 220/220 canonical LONG signals, and
`event_count` counts attached ROUTES, not liquidations. Those fields remain
UNCHANGED and UNUSED by this module; this module reconstructs cascade
geometry directly from data/microstructure.db:liquidations using the
project's own PRIOR, ALREADY-TESTED, already-known-at-safe (RUNNING_CLUSTER)
anchor-reconstruction algorithm:
tools.research_s34_knowable_anchor_continuation.reconstruct_anchors()
(imported, never reimplemented -- same constants as the live/shadow
executors: BUCKET_SEC=300, MIN_GAP_SEC=900, ACCEL_WIN_SEC=30,
THRESHOLD_USD=200_000, symbol=ETHUSDT, side=SELL).

Empirical finding from the definition-audit batch: signal_birth_ts equals
the reconstructed anchor_ts_ms for every current canonical LONG signal
(220/220), and truncating the liquidation input at signal_birth_ts before
calling reconstruct_anchors() never changes or drops the matched anchor
(0/220) -- the threshold-crossing snapshot depends, by construction, only on
rows with ts_ms <= the crossing timestamp. This is what "birth-truncated"
means here: the exact same RUNNING_CLUSTER definition, computed from an
input series defensively pre-filtered to ts_ms <= signal_birth_ts.

No new feature metric is introduced. No ami_events row is ever mutated.

QUALITY STATE -- ONE authoritative mechanism only (post source-quality-
contract-v2 batch): this table holds ONLY immutable feature values, their
source-window/manifest audit trail, and field-level provenance (Goal C).
It carries NO data_quality_status/coverage_assessment_version column and NO
row-level quality-assessment table of its own -- an earlier revision of this
module had both (the pre-contract-v2 Goal B row-level ledger), which the
operator's controlled-migration review flagged as a quality-state
duplication risk against the newer, authoritative, field-level
ami_birth_truncated_geometry_field_quality_v2 ledger
(ami/geometry/liquidation_source_quality_contract_v2.py). That earlier
row-level table/column pair was REMOVED here (never migrated to the real
canonical.sqlite) rather than kept alongside the field-level ledger --
consumers needing quality state must use
ami_birth_truncated_geometry_field_quality_v2_effective (per-field) or
ami_birth_truncated_geometry_row_quality_v2_effective (worst-case rollup),
never a column on this table.
"""
from __future__ import annotations

import hashlib
import json
import time

FEATURE_DEFINITION_VERSION = "s34-knowable-anchor-continuation-v1-birth-truncated"

SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
THRESHOLD_USD = 200_000.0
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WIN_SEC = 30
BUCKET_MS = BUCKET_SEC * 1000

# shared quality-status vocabulary (used by the SEPARATE, authoritative
# field-level ledger in liquidation_source_quality_contract_v2.py -- not a
# column on this table)
DATA_QUALITY_STATUSES = ("SOURCE_COMPLETE", "SOURCE_GAPPED", "SOURCE_COVERAGE_UNRESOLVED")

# ---------------------------------------------------------------------------
# Goal A -- schema DDL (additive-only; folded verbatim into
# ami/warehouse/schema.py's init_schema() by the canonical migration batch,
# the same way every prior _SCHEMA_PHASEn block was folded in after its own
# disposable rehearsal).
# ---------------------------------------------------------------------------
_SCHEMA = """
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

-- Goal C: field-level provenance, one row per (feature_id, field_name),
-- mirroring ami_lifecycle_field_provenance's existing classification
-- vocabulary exactly (no new enum invented).
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
"""

_FEATURE_FIELDS = (
    "running_notional", "running_liq_count", "max_single_notional",
    "running_single_liq_dominance", "running_rate", "running_accel",
    "elapsed_since_first_sec", "inter_cluster_gap_sec",
)


def init_schema(conn) -> None:
    """Idempotent: CREATE TABLE/INDEX IF NOT EXISTS only."""
    conn.executescript(_SCHEMA)
    conn.commit()


def _feature_id(signal_id: str, feature_definition_version: str) -> str:
    key = f"{signal_id}|{feature_definition_version}"
    return "FEAT-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


class ImmutableGeometryConflict(Exception):
    """Raised when a (signal_id, feature_definition_version) row already
    exists with DIFFERENT feature-value content than the one being written --
    geometry rows are immutable; a genuine redefinition must mint a new
    feature_definition_version, never overwrite an existing one."""


def _row_content_tuple(row: dict) -> tuple:
    return tuple(row.get(f) for f in _FEATURE_FIELDS) + (
        row["source_window_start_ts_ms"], row["source_window_end_ts_ms"],
        row["source_row_count"], row["source_row_manifest_sha256"],
    )


def reconstruct_signal_geometry(
    all_sell_liqs: list[dict], anchor_ts_ms: int, signal_birth_ts: int,
    *, reconstruct_anchors_fn,
) -> dict | None:
    """Re-derive the exact RUNNING_CLUSTER snapshot for one signal, using
    ONLY liquidation rows with ts_ms <= signal_birth_ts as input. Returns
    None if the anchor cannot be reproduced from the truncated input (would
    indicate a genuine known-at violation -- must never be silently
    swallowed by the caller)."""
    rows_truncated = [r for r in all_sell_liqs if r["ts_ms"] <= signal_birth_ts]
    if any(r["ts_ms"] > signal_birth_ts for r in rows_truncated):
        raise AssertionError("internal invariant violated: truncation filter let a future row through")
    ancs = reconstruct_anchors_fn(
        rows_truncated, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,), accel_window_sec=ACCEL_WIN_SEC,
    )
    match = next((a for a in ancs if int(a.anchor_ts_ms) == int(anchor_ts_ms)), None)
    if match is None:
        return None
    bucket_start = int(match.first_ts_ms)
    bucket_end = bucket_start + BUCKET_MS
    window_end = min(bucket_end, signal_birth_ts)
    window_rows = [r for r in rows_truncated if bucket_start <= r["ts_ms"] <= window_end]
    manifest_hash = hashlib.sha256(
        json.dumps([[r["ts_ms"], r["notional"]] for r in sorted(window_rows, key=lambda r: r["ts_ms"])]).encode()
    ).hexdigest()
    return {
        "running_notional": match.running_notional,
        "running_liq_count": match.running_liq_count,
        "max_single_notional": match.max_single_notional,
        "running_single_liq_dominance": match.running_single_liq_dominance,
        "running_rate": match.running_rate,
        "running_accel": match.running_accel,
        "elapsed_since_first_sec": match.elapsed_since_first_sec,
        "source_window_start_ts_ms": bucket_start,
        "source_window_end_ts_ms": window_end,
        "source_row_count": len(window_rows),
        "source_row_manifest_sha256": manifest_hash,
    }


def inter_cluster_gap_sec(sorted_anchor_ts: list[int], anchor_ts_ms: int) -> float | None:
    """None only when this is the first-ever accepted anchor in history --
    a genuinely undefined quantity, never coerced to 0.0."""
    idx = sorted_anchor_ts.index(anchor_ts_ms)
    if idx == 0:
        return None
    return (anchor_ts_ms - sorted_anchor_ts[idx - 1]) / 1000.0


_FIELD_PROVENANCE_DEFINITIONS = {
    "running_notional": "Cumulative SELL-liquidation notional from bucket first_ts_ms through min(bucket_end, signal_birth_ts) (reconstruct_anchors() forward accumulation, rows[:idx+1]).",
    "running_liq_count": "Count of SELL-liquidation rows in the same window (raw liquidation trade count -- NOT the ami_events.event_count route-count).",
    "max_single_notional": "Largest single SELL-liquidation notional observed in the same window.",
    "running_single_liq_dominance": "max_single_notional / running_notional * 100, at the same window end.",
    "running_rate": "running_notional / max(elapsed_since_first_sec, 1.0).",
    "running_accel": "Difference between two consecutive backward-looking ACCEL_WIN_SEC-second notional-rate windows ending at the snapshot ts; never NULL (empty-window sums evaluate to 0.0 in reconstruct_anchors()'s running_sum()).",
    "elapsed_since_first_sec": "snapshot_ts - bucket first_ts_ms, in seconds.",
    "inter_cluster_gap_sec": "Gap in seconds to the previous accepted 200K-threshold anchor; NULL only for the first anchor in the full reconstructed history (genuinely undefined, not zero).",
}


def backfill(
    conn, all_sell_liqs: list[dict], signals: list[dict], events_by_id: dict,
    *, reconstruct_anchors_fn, provenance: str,
) -> dict:
    """Backfill ami_birth_truncated_cascade_geometry (+ field-provenance) on
    the caller-supplied connection. Carries NO quality-status opinion --
    quality is a strictly separate, subsequent step
    (liquidation_source_quality_contract_v2.backfill_field_quality()) against
    the geometry rows this writes.

    signals: [{signal_id, source_event_id, independent_cycle_id, signal_birth_ts}, ...]
    events_by_id: {event_id: {anchor_ts_ms: int, ...}}
    """
    now = int(time.time() * 1000)
    all_anchor_ts = sorted({int(row["anchor_ts_ms"]) for row in events_by_id.values()})

    candidate_n = len(signals)
    accepted_n = 0
    rejected: dict[str, list[str]] = {}

    for s in signals:
        sid = s["signal_id"]
        eid = s["source_event_id"]
        birth = int(s["signal_birth_ts"])
        ev = events_by_id.get(eid)
        if ev is None:
            rejected.setdefault("NO_SOURCE_EVENT", []).append(sid)
            continue
        ats = int(ev["anchor_ts_ms"])
        if ats > birth:
            rejected.setdefault("ANCHOR_AFTER_BIRTH", []).append(sid)
            continue

        geo = reconstruct_signal_geometry(
            all_sell_liqs, ats, birth, reconstruct_anchors_fn=reconstruct_anchors_fn,
        )
        if geo is None:
            rejected.setdefault("ANCHOR_LOST_ON_TRUNCATION", []).append(sid)
            continue

        feature_id = _feature_id(sid, FEATURE_DEFINITION_VERSION)
        gap = inter_cluster_gap_sec(all_anchor_ts, ats)

        row = {
            "feature_id": feature_id, "signal_id": sid, "source_event_id": eid,
            "independent_cycle_id": s.get("independent_cycle_id"),
            "feature_available_ts_ms": birth,
            "running_notional": geo["running_notional"], "running_liq_count": geo["running_liq_count"],
            "max_single_notional": geo["max_single_notional"],
            "running_single_liq_dominance": geo["running_single_liq_dominance"],
            "running_rate": geo["running_rate"], "running_accel": geo["running_accel"],
            "elapsed_since_first_sec": geo["elapsed_since_first_sec"], "inter_cluster_gap_sec": gap,
            "source_window_start_ts_ms": geo["source_window_start_ts_ms"],
            "source_window_end_ts_ms": geo["source_window_end_ts_ms"],
            "source_row_count": geo["source_row_count"],
            "source_row_manifest_sha256": geo["source_row_manifest_sha256"],
        }

        # write-time assertions (cross-database / cross-table invariants that
        # cannot be expressed as an in-DB CHECK constraint against
        # microstructure.db or ami_signal_lifecycle directly)
        if row["feature_available_ts_ms"] != birth:
            raise AssertionError(f"{sid}: feature_available_ts_ms != canonical signal_birth_ts")
        if row["source_window_end_ts_ms"] > row["feature_available_ts_ms"]:
            raise AssertionError(f"{sid}: source_window_end_ts_ms > feature_available_ts_ms")
        if row["source_window_start_ts_ms"] > row["source_window_end_ts_ms"]:
            raise AssertionError(f"{sid}: source_window_start_ts_ms > source_window_end_ts_ms")

        existing = conn.execute(
            "SELECT feature_id, running_notional, running_liq_count, max_single_notional, "
            "running_single_liq_dominance, running_rate, running_accel, elapsed_since_first_sec, "
            "inter_cluster_gap_sec, source_window_start_ts_ms, source_window_end_ts_ms, "
            "source_row_count, source_row_manifest_sha256 "
            "FROM ami_birth_truncated_cascade_geometry WHERE signal_id=? AND feature_definition_version=?",
            (sid, FEATURE_DEFINITION_VERSION),
        ).fetchone()
        if existing is not None:
            existing_content = existing[1:]
            new_content = _row_content_tuple(row)
            if existing_content != new_content:
                raise ImmutableGeometryConflict(
                    f"signal_id={sid} feature_definition_version={FEATURE_DEFINITION_VERSION} already "
                    f"exists with DIFFERENT content -- geometry rows are immutable"
                )
            accepted_n += 1  # NOOP_IDENTICAL
            continue

        conn.execute(
            "INSERT INTO ami_birth_truncated_cascade_geometry "
            "(feature_id, feature_definition_version, signal_id, source_event_id, independent_cycle_id, "
            "feature_available_ts_ms, source_window_start_ts_ms, source_window_end_ts_ms, source_row_count, "
            "source_row_manifest_sha256, running_notional, running_liq_count, max_single_notional, "
            "running_single_liq_dominance, running_rate, running_accel, elapsed_since_first_sec, "
            "inter_cluster_gap_sec, known_at_classification, "
            "derivation_reference, schema_version, provenance, created_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (feature_id, FEATURE_DEFINITION_VERSION, sid, eid, row["independent_cycle_id"],
             birth, row["source_window_start_ts_ms"], row["source_window_end_ts_ms"],
             row["source_row_count"], row["source_row_manifest_sha256"],
             geo["running_notional"], geo["running_liq_count"], geo["max_single_notional"],
             geo["running_single_liq_dominance"], geo["running_rate"], geo["running_accel"],
             geo["elapsed_since_first_sec"], gap, "KNOWN_AT_SAFE",
             "tools.research_s34_knowable_anchor_continuation.reconstruct_anchors",
             1, provenance, now),
        )

        for field_name in _FEATURE_FIELDS:
            prov_id = "FPROV-" + hashlib.sha256(f"{feature_id}|{field_name}|{FEATURE_DEFINITION_VERSION}".encode()).hexdigest()[:24]
            conn.execute(
                "INSERT OR IGNORE INTO ami_birth_truncated_geometry_field_provenance "
                "(provenance_id, feature_id, field_name, field_classification, derivation_method, "
                "source_reference, limitations, provenance_version, schema_version, code_commit, "
                "source_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (prov_id, feature_id, field_name, "DETERMINISTIC_HISTORICAL_SAFE",
                 _FIELD_PROVENANCE_DEFINITIONS[field_name],
                 "tools.research_s34_knowable_anchor_continuation.reconstruct_anchors + "
                 "data/microstructure.db:liquidations",
                 None if field_name != "inter_cluster_gap_sec" else
                 "NULL exactly when this is the first anchor in reconstructed history.",
                 FEATURE_DEFINITION_VERSION, 1, None, row["source_row_manifest_sha256"], now),
            )
        accepted_n += 1

    conn.commit()
    return {
        "candidate_n": candidate_n, "accepted_n": accepted_n,
        "rejected_n": sum(len(v) for v in rejected.values()), "rejected": rejected,
    }


def content_hash(conn) -> str:
    rows = conn.execute(
        "SELECT feature_id, feature_definition_version, signal_id, source_event_id, independent_cycle_id, "
        "feature_available_ts_ms, source_window_start_ts_ms, source_window_end_ts_ms, source_row_count, "
        "source_row_manifest_sha256, running_notional, running_liq_count, max_single_notional, "
        "running_single_liq_dominance, running_rate, running_accel, elapsed_since_first_sec, "
        "inter_cluster_gap_sec, known_at_classification "
        "FROM ami_birth_truncated_cascade_geometry ORDER BY feature_id"
    ).fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()


def row_counts(conn) -> dict:
    return {
        "geometry": conn.execute("SELECT COUNT(*) FROM ami_birth_truncated_cascade_geometry").fetchone()[0],
        "field_provenance": conn.execute(
            "SELECT COUNT(*) FROM ami_birth_truncated_geometry_field_provenance").fetchone()[0],
    }


def rollback(conn) -> None:
    """Disposable-copy-only rollback: drops this module's tables/indexes,
    never touches ami_events/ami_signal_lifecycle/ami_cycles/path tables."""
    conn.executescript(
        "DROP TABLE IF EXISTS ami_birth_truncated_geometry_field_provenance;"
        "DROP TABLE IF EXISTS ami_birth_truncated_cascade_geometry;"
    )
    conn.commit()
