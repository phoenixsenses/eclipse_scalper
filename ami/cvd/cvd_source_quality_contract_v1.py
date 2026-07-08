"""cvd-source-quality-contract-v1 -- disposable-rehearsal implementation.

Classifies every (signal_id, window) source window of the CVD windowed
taker-flow feature into exactly one of the five frozen statuses:

    EXACT_RECONSTRUCTABLE   trade-level evidence for the ENTIRE window is
                            positively proven (native minute coverage AND
                            cadence proof AND no unresolved duplicate flag,
                            OR verified REST repair covering every missing
                            span with proven pagination continuity)
    PROXY_ONLY              only minute-level candle taker volume available
    SOURCE_GAPPED           confirmed trade-level loss inside the window and
                            no verified repair coverage for it (repair may
                            exist upstream but has NOT been proven for this
                            window -- fail closed)
    SOURCE_COVERAGE_UNRESOLVED  positive completeness proof is incomplete
                            (e.g. cadence proof unavailable, window outside
                            any proven coverage map, duplicate assessment
                            unresolved)
    UNREPAIRABLE            confirmed loss AND all repair sources confirmed
                            unavailable (reserved; never auto-assigned)

Fail-closed law: whenever proof is incomplete the classification degrades
(never upgrades): missing proof -> SOURCE_COVERAGE_UNRESOLVED, proven hole
without proven repair -> SOURCE_GAPPED. A window spanning source regimes is
NEVER automatically EXACT_RECONSTRUCTABLE: regime-spanning windows require
every member regime's own proof AND carry all regime ids.

The ledger is append-only and versioned (assessment_version); rewriting an
existing assessment with different content fails closed.
"""
from __future__ import annotations

import hashlib
import json
import time

QUALITY_CONTRACT_VERSION = "cvd-source-quality-contract-v1"

STATUSES = (
    "EXACT_RECONSTRUCTABLE",
    "PROXY_ONLY",
    "SOURCE_GAPPED",
    "SOURCE_COVERAGE_UNRESOLVED",
    "UNREPAIRABLE",
)

# Source-regime boundaries (ms, UTC) -- frozen from the accepted contract
# section 3. R3 start reuses the measured ALL_MARKET_TRANSITION_TS_MS
# constant frozen in liquidation-source-quality-contract-v2 (collector
# deploy boundary 2026-06-06 17:43:52.123Z).
REGIME_BOUNDARIES = (
    ("R0", None, 1775980020000),           # ... -> 2026-04-12 07:47:00
    ("R1", 1775980020000, 1777026446000),  # -> 2026-04-24 10:27:26
    ("R2", 1777026446000, 1780767832123),  # -> 2026-06-06 17:43:52.123
    ("R3", 1780767832123, None),           # -> present
)


class ImmutableCvdQualityConflict(Exception):
    """Same (feature target, assessment_version) with different content."""


_SCHEMA = """
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
"""


def init_schema(conn) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def regimes_for_window(window_start_ts_ms: int, window_end_ts_ms: int) -> list[str]:
    out = []
    for rid, lo, hi in REGIME_BOUNDARIES:
        lo_v = lo if lo is not None else -(1 << 62)
        hi_v = hi if hi is not None else (1 << 62)
        if window_start_ts_ms < hi_v and window_end_ts_ms >= lo_v:
            out.append(rid)
    return out


def classify_window(
    *,
    missing_minute_count: int,
    repaired_minute_count: int,
    coverage_map_available: bool,
    cadence_proof_pass,   # True / False / None (None = proof unavailable)
    duplicate_unresolved: bool,
    regime_ids: list[str],
    regime_proofs: dict,  # regime_id -> bool (positive completeness proof exists for this regime era)
    proxy_available: bool,
) -> str:
    """Pure classification function -- the single decision point, fail-closed.

    missing_minute_count: minutes in the window with ZERO local trade rows.
    repaired_minute_count: of those, minutes fully covered by VERIFIED repair
    rows (pagination-continuity-proven REST extraction).
    """
    if not coverage_map_available:
        return "SOURCE_COVERAGE_UNRESOLVED"
    if any(not regime_proofs.get(r, False) for r in regime_ids):
        # a window spanning (or inside) a regime era with no positive
        # completeness instrument never auto-passes
        return "SOURCE_COVERAGE_UNRESOLVED"
    if missing_minute_count > 0:
        if repaired_minute_count >= missing_minute_count:
            # every hole positively repaired with verified extraction
            if duplicate_unresolved:
                return "SOURCE_COVERAGE_UNRESOLVED"
            return "EXACT_RECONSTRUCTABLE"
        if proxy_available:
            return "PROXY_ONLY"
        return "SOURCE_GAPPED"
    # zero missing minutes: sub-minute completeness still needs cadence proof
    if cadence_proof_pass is None:
        return "SOURCE_COVERAGE_UNRESOLVED"
    if cadence_proof_pass is False:
        if proxy_available:
            return "PROXY_ONLY"
        return "SOURCE_GAPPED"
    if duplicate_unresolved:
        return "SOURCE_COVERAGE_UNRESOLVED"
    return "EXACT_RECONSTRUCTABLE"


_CONTENT_FIELDS = (
    "window_start_ts_ms", "window_end_ts_ms", "evidence_layer", "source_regime_ids",
    "regime_spanning", "legacy_row_count", "repair_row_count", "total_row_count",
    "duplicate_count", "collision_count", "unresolved_match_count",
    "missing_minute_count", "repaired_minute_count", "cadence_proof",
    "completeness_proof", "quality_status", "data_version_id",
)


def record_window_quality(conn, row: dict, *, assessment_version: str) -> str:
    """Append-only immutable write (experiment-ledger discipline)."""
    if row["quality_status"] not in STATUSES:
        raise ValueError(f"invalid quality_status {row['quality_status']!r}")
    qid = "CVDQ-" + hashlib.sha256(
        f"{row['signal_id']}|{row['window_id']}|{QUALITY_CONTRACT_VERSION}|{assessment_version}".encode()
    ).hexdigest()[:24]
    existing = conn.execute(
        "SELECT " + ", ".join(_CONTENT_FIELDS) +
        " FROM ami_cvd_window_quality_v1"
        " WHERE signal_id=? AND window_id=? AND quality_contract_version=? AND assessment_version=?",
        (row["signal_id"], row["window_id"], QUALITY_CONTRACT_VERSION, assessment_version)).fetchone()
    new_content = tuple(row.get(f) for f in _CONTENT_FIELDS)
    if existing is not None:
        if tuple(existing) != new_content:
            raise ImmutableCvdQualityConflict(
                f"{row['signal_id']}/{row['window_id']}/{assessment_version}: different content")
        return "NOOP_IDENTICAL"
    conn.execute(
        "INSERT INTO ami_cvd_window_quality_v1 (quality_id, quality_contract_version,"
        " assessment_version, signal_id, independent_cycle_id, symbol, signal_birth_ts,"
        " window_id, window_start_ts_ms, window_end_ts_ms, evidence_layer, source_regime_ids,"
        " regime_spanning, legacy_row_count, repair_row_count, total_row_count, duplicate_count,"
        " collision_count, unresolved_match_count, missing_minute_count, repaired_minute_count,"
        " cadence_proof, completeness_proof, quality_status, feature_available_ts_ms,"
        " source_provenance, data_version_id, feature_definition_version, assessed_at_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (qid, QUALITY_CONTRACT_VERSION, assessment_version, row["signal_id"],
         row.get("independent_cycle_id"), row["symbol"], row["signal_birth_ts"],
         row["window_id"], row["window_start_ts_ms"], row["window_end_ts_ms"],
         row["evidence_layer"], row["source_regime_ids"], row["regime_spanning"],
         row["legacy_row_count"], row["repair_row_count"], row["total_row_count"],
         row["duplicate_count"], row["collision_count"], row["unresolved_match_count"],
         row["missing_minute_count"], row["repaired_minute_count"], row["cadence_proof"],
         row["completeness_proof"], row["quality_status"], row["signal_birth_ts"],
         row["source_provenance"], row["data_version_id"], row["feature_definition_version"],
         int(time.time() * 1000)))
    return "INSERTED"


def content_hash(conn) -> str:
    rows = conn.execute(
        "SELECT quality_id, signal_id, window_id, quality_status, missing_minute_count,"
        " repaired_minute_count, cadence_proof, source_regime_ids, total_row_count"
        " FROM ami_cvd_window_quality_v1 ORDER BY quality_id").fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()
