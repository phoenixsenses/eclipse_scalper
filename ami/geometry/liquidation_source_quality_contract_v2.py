"""LIQUIDATION SOURCE-QUALITY CONTRACT V2 (`liquidation-source-quality-contract-v2`).

Operator-approved, authoritative, fail-closed source-quality contract for
`ami_birth_truncated_cascade_geometry` rows (see
`reports/research/s34/S34_LIQUIDATION_SOURCE_QUALITY_RECONCILIATION_2026-07-05.md`
for the reconciliation that led here). Supersedes METHOD_A/METHOD_B (both
reconciliation artifacts now, never a canonical contract) and rejects
cross-stream health as completeness evidence (May 2026 proves agg_trades/
mark_prices can stay healthy while the liquidations stream is completely
silent for 40+ days).

DISPOSABLE_DB_ONLY / NO_LIVE_CANONICAL_DB_MIGRATION / NO_OUTCOME_ANALYSIS:
every function here operates on caller-supplied connections/data; this
module never opens a real DB path itself, never reads MFE/MAE/profit-or-loss/
p-value, and never mutates `ami_birth_truncated_cascade_geometry`'s frozen feature
values or `ami_events`.

FIELD-LEVEL, not row-level (operator's explicit design choice): each of the
8 frozen RUNNING_CLUSTER fields has its OWN required source window (Goal D).
A row's overall status (if one is needed for a quick filter) is the WORST
status across its 8 field-level assessments -- never assigned independently.

CONTRACT SEMANTICS
------------------
A. SOURCE_COVERAGE_UNRESOLVED
   - the field's required window starts before the proven all-market
     `!forceOrder@arr` transition (measured, not assumed: 2026-06-06
     17:43:52.123 UTC -- the row immediately following the 40.14-day
     liquidations blackout; see ALL_MARKET_TRANSITION_TS_MS derivation
     below), where positive per-window cadence verification is structurally
     impossible (per-symbol subscriptions have natural multi-minute/hour
     silences); or
   - no raw liquidation message exists to bound the window's own edges.
   NO GAP RECORD IS EVER INTERPRETED AS COMPLETENESS.

B. SOURCE_COMPLETE
   - the field's ENTIRE required window is on/after the all-market
     transition, AND
   - the maximum raw all-market liquidation message cadence gap anywhere
     inside [window_start, window_end] (window edges included as boundary
     points -- a message-free stretch from window_start to the first real
     message, or from the last real message to window_end, counts) is
     <= CRITICAL_GAP_MS (300s, the original collector's own frozen
     stream-specific-reconnect threshold), AND
   - no resolved (start+end both known) registry gap intersects the window.

C. SOURCE_GAPPED
   - a resolved registry gap intersects the required window, OR
   - the all-market cadence check above finds a gap > CRITICAL_GAP_MS.
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import time

from ami.geometry import birth_truncated_cascade_geometry as geo

CONTRACT_VERSION = "liquidation-source-quality-contract-v2"

# Measured, not assumed: MIN(ts_ms) of the liquidations row that ends the
# 40.14-day blackout following the last per-symbol-era gap-registry row
# (2026-04-27 14:27:26.345 -> 2026-06-06 17:43:52.123); 171 distinct symbols
# observed in the following hour confirms the !forceOrder@arr all-market
# subscription mode (vs. 2-3 symbols throughout the entire Feb-Apr era).
ALL_MARKET_TRANSITION_TS_MS = 1780767832123

CRITICAL_GAP_MS = 300_000  # frozen collector constant (data.microstructure_collector's own
                            # historical stream-specific-reconnect threshold, git stash 07e1a1f9)

DATA_QUALITY_STATUSES = geo.DATA_QUALITY_STATUSES  # ("SOURCE_COMPLETE","SOURCE_GAPPED","SOURCE_COVERAGE_UNRESOLVED")

_ROW_WORST_CASE_ORDER = {"SOURCE_GAPPED": 2, "SOURCE_COVERAGE_UNRESOLVED": 1, "SOURCE_COMPLETE": 0}


def _iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat() if ms is not None else None


# ---------------------------------------------------------------------------
# Goal D -- feature-specific required source windows
# ---------------------------------------------------------------------------

def required_window(field_name: str, *, bucket_start_ts_ms: int, anchor_ts_ms: int,
                     prev_anchor_ts_ms: int | None, earliest_liq_ts_ms: int,
                     accel_win_sec: int = geo.ACCEL_WIN_SEC) -> tuple[int, int]:
    """Returns (window_start_ts_ms, window_end_ts_ms) -- the EXACT source
    interval whose completeness this field's already-computed value
    depends on. Never approximated with one row-wide window."""
    if field_name == "running_accel":
        # reconstruct_anchors()'s own frozen two-window definition: cur window
        # [t-accel_win_sec, t], prev window [t-2*accel_win_sec, t-accel_win_sec]
        return anchor_ts_ms - 2 * accel_win_sec * 1000, anchor_ts_ms
    if field_name == "inter_cluster_gap_sec":
        # requires enough positively-verified history to prove the PREVIOUS
        # accepted anchor was correctly identified under the frozen
        # MIN_GAP_SEC debounce -- never just "this anchor's own bucket".
        start = prev_anchor_ts_ms if prev_anchor_ts_ms is not None else earliest_liq_ts_ms
        return start, anchor_ts_ms
    # running_notional / running_liq_count / max_single_notional /
    # running_single_liq_dominance / running_rate / elapsed_since_first_sec:
    # all derived from the forward accumulation over [bucket_start, anchor_ts_ms].
    return bucket_start_ts_ms, anchor_ts_ms


# ---------------------------------------------------------------------------
# Goal A/C -- raw-cadence + resolved-gap evidence
# ---------------------------------------------------------------------------

def resolved_gap_overlaps(window_start_ts_ms: int, window_end_ts_ms: int,
                           resolved_gaps: list[tuple[int, int]]) -> bool:
    return any(window_start_ts_ms <= ge and gs <= window_end_ts_ms for gs, ge in resolved_gaps)


def max_cadence_gap_ms(window_start_ts_ms: int, window_end_ts_ms: int,
                        sorted_all_market_liq_ts: list[int]) -> int:
    """Window edges are themselves treated as mandatory boundary points -- a
    message-free stretch from window_start to the first real message (or
    from the last real message to window_end) counts as a gap. This is what
    lets a short, genuinely-quiet window (no liquidation happened, not "the
    collector died") still pass, while a long silent window cannot."""
    lo = bisect.bisect_left(sorted_all_market_liq_ts, window_start_ts_ms)
    hi = bisect.bisect_right(sorted_all_market_liq_ts, window_end_ts_ms)
    inside = sorted_all_market_liq_ts[lo:hi]
    seq = [window_start_ts_ms] + inside + [window_end_ts_ms]
    return max(b - a for a, b in zip(seq, seq[1:]))


def classify_field_window(window_start_ts_ms: int, window_end_ts_ms: int, *,
                           resolved_gaps: list[tuple[int, int]],
                           sorted_all_market_liq_ts: list[int]) -> tuple[str, str]:
    """Returns (status, reason). Contract-v2 GOAL A/B/C, applied to ONE
    already-determined feature-specific window."""
    if window_start_ts_ms < ALL_MARKET_TRANSITION_TS_MS:
        return ("SOURCE_COVERAGE_UNRESOLVED",
                f"window_start {_iso(window_start_ts_ms)} precedes the proven all-market "
                f"transition {_iso(ALL_MARKET_TRANSITION_TS_MS)} -- per-symbol-era cadence "
                f"cannot be positively verified")
    if resolved_gap_overlaps(window_start_ts_ms, window_end_ts_ms, resolved_gaps):
        return "SOURCE_GAPPED", "resolved registry gap intersects the required window"
    max_gap = max_cadence_gap_ms(window_start_ts_ms, window_end_ts_ms, sorted_all_market_liq_ts)
    if max_gap > CRITICAL_GAP_MS:
        return "SOURCE_GAPPED", f"all-market cadence gap {max_gap}ms exceeds CRITICAL_GAP_MS={CRITICAL_GAP_MS}ms"
    return "SOURCE_COMPLETE", f"all-market cadence verified, max_gap={max_gap}ms <= {CRITICAL_GAP_MS}ms"


# ---------------------------------------------------------------------------
# per-signal, all-8-fields classification
# ---------------------------------------------------------------------------

def classify_signal_fields(*, bucket_start_ts_ms: int, anchor_ts_ms: int,
                            prev_anchor_ts_ms: int | None, earliest_liq_ts_ms: int,
                            resolved_gaps: list[tuple[int, int]],
                            sorted_all_market_liq_ts: list[int]) -> dict[str, dict]:
    out = {}
    for field_name in geo._FEATURE_FIELDS:
        ws, we = required_window(
            field_name, bucket_start_ts_ms=bucket_start_ts_ms, anchor_ts_ms=anchor_ts_ms,
            prev_anchor_ts_ms=prev_anchor_ts_ms, earliest_liq_ts_ms=earliest_liq_ts_ms,
        )
        status, reason = classify_field_window(
            ws, we, resolved_gaps=resolved_gaps, sorted_all_market_liq_ts=sorted_all_market_liq_ts)
        out[field_name] = {"status": status, "reason": reason, "window_start_ts_ms": ws, "window_end_ts_ms": we}
    return out


def row_level_worst_case(field_statuses: dict[str, str]) -> str:
    """Goal D's row-level-rollup design: the WORST classification across all
    required feature windows (GAPPED worse than UNRESOLVED worse than
    COMPLETE) -- never an independent assessment, never used to silently
    upgrade a row to COMPLETE when any field is not."""
    return max(field_statuses.values(), key=lambda s: _ROW_WORST_CASE_ORDER[s])


# ---------------------------------------------------------------------------
# reusable read-only evidence gathering + per-geometry-row assessment
# (shared by the disposable rehearsal and the real canonical migration --
# ONE implementation, never duplicated per caller)
# ---------------------------------------------------------------------------

def fetch_quality_evidence(conn_liq, all_sell_liqs: list[dict]) -> dict:
    """Reads the liquidations-stream evidence classify_signal_fields() needs
    from a read-only microstructure.db connection: resolved registry gaps,
    the all-market-era raw cadence sequence, and the earliest available
    liquidation timestamp. `all_sell_liqs` is the SAME already-fetched
    SELL-only series ami.geometry.birth_truncated_cascade_geometry.
    reconstruct_signal_geometry() uses -- its own first row IS the earliest
    available liquidation evidence for this symbol/side, no extra query
    needed for that value."""
    resolved_gaps = conn_liq.execute(
        "SELECT start_ts_ms, end_ts_ms FROM gaps WHERE stream='liquidations' AND end_ts_ms IS NOT NULL"
    ).fetchall()
    sorted_all_market_liq_ts = [
        r[0] for r in conn_liq.execute(
            "SELECT ts_ms FROM liquidations WHERE ts_ms >= ? ORDER BY ts_ms",
            (ALL_MARKET_TRANSITION_TS_MS - CRITICAL_GAP_MS,),
        ).fetchall()
    ]
    earliest_liq_ts_ms = all_sell_liqs[0]["ts_ms"] if all_sell_liqs else 0
    return {
        "resolved_gaps": [tuple(g) for g in resolved_gaps],
        "sorted_all_market_liq_ts": sorted_all_market_liq_ts,
        "earliest_liq_ts_ms": earliest_liq_ts_ms,
    }


def assess_geometry_rows(conn, events_by_id: dict, *, resolved_gaps: list[tuple[int, int]],
                          sorted_all_market_liq_ts: list[int], earliest_liq_ts_ms: int) -> list[dict]:
    """Reads the ALREADY-BACKFILLED ami_birth_truncated_cascade_geometry rows
    (never mutates them) and computes contract-v2 field statuses for each,
    joining back to events_by_id for each row's anchor_ts_ms (the geometry
    table itself does not duplicate that column -- it stores
    feature_available_ts_ms=signal_birth_ts and source_window_start_ts_ms=
    bucket_start only). Returns [{feature_id, signal_id, field_statuses}, ...]
    ready for backfill_field_quality()."""
    rows = conn.execute(
        "SELECT feature_id, signal_id, source_event_id, source_window_start_ts_ms "
        "FROM ami_birth_truncated_cascade_geometry"
    ).fetchall()
    all_anchor_ts = sorted({int(r["anchor_ts_ms"]) for r in events_by_id.values()})
    out = []
    for feature_id, signal_id, source_event_id, bucket_start in rows:
        anchor_ts = int(events_by_id[source_event_id]["anchor_ts_ms"])
        pos = all_anchor_ts.index(anchor_ts)
        prev_anchor_ts_ms = all_anchor_ts[pos - 1] if pos > 0 else None
        field_statuses = classify_signal_fields(
            bucket_start_ts_ms=bucket_start, anchor_ts_ms=anchor_ts,
            prev_anchor_ts_ms=prev_anchor_ts_ms, earliest_liq_ts_ms=earliest_liq_ts_ms,
            resolved_gaps=resolved_gaps, sorted_all_market_liq_ts=sorted_all_market_liq_ts,
        )
        out.append({"feature_id": feature_id, "signal_id": signal_id, "field_statuses": field_statuses})
    return out


# ---------------------------------------------------------------------------
# schema -- append-only field-level quality assessment (Goal: immutability)
# ---------------------------------------------------------------------------

_SCHEMA = """
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

-- append-only effective view: latest assessment per (feature_id, field_name),
-- regardless of how many coverage_assessment_versions have since been appended
CREATE VIEW IF NOT EXISTS ami_birth_truncated_geometry_field_quality_v2_effective AS
SELECT q.feature_id, q.field_name, q.data_quality_status, q.coverage_assessment_version, q.assessed_at_ms
FROM ami_birth_truncated_geometry_field_quality_v2 q
WHERE q.assessed_at_ms = (
    SELECT MAX(q2.assessed_at_ms) FROM ami_birth_truncated_geometry_field_quality_v2 q2
    WHERE q2.feature_id = q.feature_id AND q2.field_name = q.field_name
);

-- row-level rollup (derived, NEVER an independent assessment): worst status
-- across all 8 fields' latest effective assessment
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


def init_schema(conn) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


class ImmutableFieldQualityConflict(Exception):
    """Raised when a (feature_id, field_name, coverage_assessment_version)
    row already exists with DIFFERENT content -- field quality assessments
    are append-only; a genuine reassessment must use a NEW
    coverage_assessment_version, never overwrite an existing one."""


def _assessment_id(feature_id: str, field_name: str, version: str) -> str:
    key = f"{feature_id}|{field_name}|{version}"
    return "FQV2-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def backfill_field_quality(conn, rows: list[dict], *, provenance: str,
                            coverage_assessment_version: str = CONTRACT_VERSION) -> dict:
    """rows: [{feature_id, field_statuses: {field_name: {status, reason,
    window_start_ts_ms, window_end_ts_ms}}}, ...]. Append-only: existing
    (feature_id, field_name, version) rows with IDENTICAL content are a
    silent no-op; DIFFERENT content raises ImmutableFieldQualityConflict.
    Never touches ami_birth_truncated_cascade_geometry's own frozen columns
    or source-row manifests."""
    now = int(time.time() * 1000)
    accepted_n = 0
    conflict_n = 0
    for row in rows:
        feature_id = row["feature_id"]
        for field_name, fs in row["field_statuses"].items():
            aid = _assessment_id(feature_id, field_name, coverage_assessment_version)
            existing = conn.execute(
                "SELECT data_quality_status, window_start_ts_ms, window_end_ts_ms, evidence "
                "FROM ami_birth_truncated_geometry_field_quality_v2 WHERE assessment_id=?",
                (aid,),
            ).fetchone()
            new_content = (fs["status"], fs["window_start_ts_ms"], fs["window_end_ts_ms"], fs["reason"])
            if existing is not None:
                if tuple(existing) != new_content:
                    conflict_n += 1
                    raise ImmutableFieldQualityConflict(
                        f"feature_id={feature_id} field_name={field_name} "
                        f"coverage_assessment_version={coverage_assessment_version} already exists "
                        f"with DIFFERENT content -- field quality assessments are append-only"
                    )
                accepted_n += 1
                continue
            conn.execute(
                "INSERT INTO ami_birth_truncated_geometry_field_quality_v2 "
                "(assessment_id, feature_id, field_name, coverage_assessment_version, data_quality_status, "
                "window_start_ts_ms, window_end_ts_ms, evidence, provenance, assessed_at_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (aid, feature_id, field_name, coverage_assessment_version, fs["status"],
                 fs["window_start_ts_ms"], fs["window_end_ts_ms"], fs["reason"], provenance, now),
            )
            accepted_n += 1
    conn.commit()
    return {"accepted_n": accepted_n, "conflict_n": conflict_n}


def row_counts(conn) -> dict:
    return {
        "field_quality_v2": conn.execute(
            "SELECT COUNT(*) FROM ami_birth_truncated_geometry_field_quality_v2").fetchone()[0],
    }


def content_hash(conn) -> str:
    import json
    rows = conn.execute(
        "SELECT feature_id, field_name, coverage_assessment_version, data_quality_status, "
        "window_start_ts_ms, window_end_ts_ms FROM ami_birth_truncated_geometry_field_quality_v2 "
        "ORDER BY feature_id, field_name, coverage_assessment_version"
    ).fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()


def rollback(conn) -> None:
    """Disposable-copy-only rollback: drops this module's own tables/views,
    never touches ami_birth_truncated_cascade_geometry or its Goal-A/B/C
    quality-assessment/field-provenance tables."""
    conn.executescript(
        "DROP VIEW IF EXISTS ami_birth_truncated_geometry_row_quality_v2_effective;"
        "DROP VIEW IF EXISTS ami_birth_truncated_geometry_field_quality_v2_effective;"
        "DROP TABLE IF EXISTS ami_birth_truncated_geometry_field_quality_v2;"
    )
    conn.commit()
