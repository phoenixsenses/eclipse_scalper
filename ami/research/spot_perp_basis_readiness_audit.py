"""BATCH-SPOT-PERP-BASIS-READINESS-AND-CONTRACT-V1 -- read-only, outcome-blind
source-coverage and known-at audit for `FAM_SPOT_PERP_BASIS_REVERSAL`
(canonical family name per the accepted selection artifact,
reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md).

This module performs NO schema creation, NO data write, NO outcome access.
Every function takes explicit read-only connections and returns plain
dicts/lists -- pure audit/measurement, matching the readiness-only nature
of this batch (is-identity discipline: the same "at-or-before" convention
already used by `ami.states.engine.StateEngine._px` for mark_prices, and
the same `FEED_LIMITS["spot_prices"]` = 10-minute healthy-age convention
already established elsewhere in this codebase for exactly this feed --
neither reinvented here).

No line of this module ever opens `ami_lifecycle_path_observations` or
selects any outcome column (`endpoint_return_bps`/`mfe_bps`/`mae_bps`).
"""
from __future__ import annotations

import bisect
import sqlite3

SYMBOL = "ETHUSDT"

# Reused verbatim from ami/states/engine.py's FEED_LIMITS -- the existing,
# already-established "max healthy age" convention for this exact feed, not
# a value invented for this readiness batch.
SPOT_PRICES_HEALTHY_AGE_MIN = 10.0
SPOT_PRICES_HEALTHY_AGE_MS = int(SPOT_PRICES_HEALTHY_AGE_MIN * 60 * 1000)

# mark_prices' own already-proven-clean known-at-safe convention (CVD/
# absorption precedent): effectively sub-second staleness across the whole
# population, reconfirmed empirically by this audit, not assumed.


def fetch_anchor_universe(canonical_conn: sqlite3.Connection) -> list[dict]:
    """signal_id/direction/independent_cycle_id/signal_birth_ts/
    source_event_id only -- no outcome column is ever selected here or
    anywhere else in this module."""
    rows = canonical_conn.execute(
        "SELECT signal_id, direction, independent_cycle_id, signal_birth_ts, source_event_id "
        "FROM ami_signal_lifecycle ORDER BY signal_birth_ts, signal_id"
    ).fetchall()
    return [
        {"signal_id": r[0], "direction": r[1], "independent_cycle_id": r[2],
         "signal_birth_ts": r[3], "source_event_id": r[4]}
        for r in rows
    ]


def fetch_sorted_timestamps(micro_conn: sqlite3.Connection, table: str, symbol: str = SYMBOL) -> list[int]:
    """Returns sorted ts_ms values for a price table (spot_prices or
    mark_prices), one symbol. Used only to build an in-memory sorted index
    for `nearest_at_or_before` -- never a full-table copy of any other
    column, and never of any other table."""
    if table not in ("spot_prices", "mark_prices"):
        raise ValueError(f"unexpected table for a price-timestamp index: {table!r}")
    rows = micro_conn.execute(
        f"SELECT ts_ms FROM {table} WHERE symbol=? ORDER BY ts_ms", (symbol,)
    ).fetchall()
    return [r[0] for r in rows]


def nearest_at_or_before(sorted_ts: list[int], target_ts_ms: int) -> int | None:
    """Deterministic 'at-or-before' lookup, identical convention to
    `ami.states.engine.StateEngine._px` and the absorption-impact family's
    `fetch_mark_price_at_or_before` -- structurally incapable of returning a
    future timestamp (bisect only ever looks left of `target_ts_ms`)."""
    i = bisect.bisect_right(sorted_ts, target_ts_ms)
    if i == 0:
        return None
    candidate = sorted_ts[i - 1]
    if candidate > target_ts_ms:
        raise AssertionError("KNOWN_AT_VIOLATION: nearest_at_or_before returned a future timestamp")
    return candidate


# ---------------------------------------------------------------------------
# Source coverage audit (Phase: SOURCE AUDIT)
# ---------------------------------------------------------------------------

def source_coverage_summary(micro_conn: sqlite3.Connection, table: str, symbol: str = SYMBOL) -> dict:
    row = micro_conn.execute(
        f"SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM {table} WHERE symbol=?", (symbol,)
    ).fetchone()
    return {"table": table, "symbol": symbol, "row_count": row[0],
            "first_ts_ms": row[1], "last_ts_ms": row[2]}


def inter_sample_gap_stats(sorted_ts: list[int]) -> dict:
    """Pure statistics over consecutive-sample gaps -- no anchor/outcome
    data involved. Used to characterize staleness risk empirically, since
    no gap ledger exists for spot_prices (confirmed: 0 rows in
    microstructure.db:gaps for stream='spot_prices')."""
    if len(sorted_ts) < 2:
        return {"n_samples": len(sorted_ts), "n_gaps": 0}
    diffs = sorted(sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1))
    n = len(diffs)
    largest = sorted(
        ((sorted_ts[i + 1] - sorted_ts[i], sorted_ts[i], sorted_ts[i + 1]) for i in range(len(sorted_ts) - 1)),
        reverse=True,
    )[:5]
    return {
        "n_samples": len(sorted_ts), "n_gaps": n,
        "min_gap_ms": diffs[0], "median_gap_ms": diffs[n // 2],
        "p95_gap_ms": diffs[int(n * 0.95)], "p99_gap_ms": diffs[int(n * 0.99)],
        "max_gap_ms": diffs[-1],
        "largest_gaps": [{"duration_ms": d, "start_ts_ms": s, "end_ts_ms": e} for d, s, e in largest],
        "gaps_over_5min": sum(1 for d in diffs if d > 5 * 60 * 1000),
        "gaps_over_30min": sum(1 for d in diffs if d > 30 * 60 * 1000),
        "gaps_over_60min": sum(1 for d in diffs if d > 60 * 60 * 1000),
    }


# ---------------------------------------------------------------------------
# Anchor accounting (Phase: ANCHOR ACCOUNTING) -- outcome-blind
# ---------------------------------------------------------------------------

QUALITY_ABSENT = "SOURCE_ABSENT_BEFORE_COLLECTION"
QUALITY_STALE = "SOURCE_STALE_BEYOND_HEALTHY_AGE"
QUALITY_FRESH = "EXACT_RECONSTRUCTABLE"


def classify_signal_spot_coverage(signal_birth_ts: int, spot_sorted_ts: list[int],
                                   healthy_age_ms: int = SPOT_PRICES_HEALTHY_AGE_MS) -> dict:
    near = nearest_at_or_before(spot_sorted_ts, signal_birth_ts)
    if near is None:
        return {"quality_status": QUALITY_ABSENT, "nearest_spot_ts_ms": None, "staleness_ms": None}
    staleness = signal_birth_ts - near
    if staleness < 0:
        raise AssertionError("KNOWN_AT_VIOLATION: negative staleness (future spot sample used)")
    status = QUALITY_FRESH if staleness <= healthy_age_ms else QUALITY_STALE
    return {"quality_status": status, "nearest_spot_ts_ms": near, "staleness_ms": staleness}


def anchor_accounting(canonical_conn: sqlite3.Connection, micro_conn: sqlite3.Connection,
                       healthy_age_ms: int = SPOT_PRICES_HEALTHY_AGE_MS) -> dict:
    """Full, outcome-blind accounting: for every anchor, classify spot
    coverage (absent/stale/fresh) and mark coverage (always expected fresh,
    reconfirmed not assumed). Never reads any outcome column, never applies
    an outcome-dependent eligibility rule -- exclusions here are 100%
    source-quality/coverage based, deterministic given only signal_birth_ts
    and the two price tables' own timestamps."""
    signals = fetch_anchor_universe(canonical_conn)
    spot_ts = fetch_sorted_timestamps(micro_conn, "spot_prices")
    mark_ts = fetch_sorted_timestamps(micro_conn, "mark_prices")

    rows = []
    for s in signals:
        spot_cls = classify_signal_spot_coverage(s["signal_birth_ts"], spot_ts, healthy_age_ms)
        mark_near = nearest_at_or_before(mark_ts, s["signal_birth_ts"])
        mark_staleness = (s["signal_birth_ts"] - mark_near) if mark_near is not None else None
        rows.append({
            **s,
            "spot_quality_status": spot_cls["quality_status"],
            "spot_staleness_ms": spot_cls["staleness_ms"],
            "mark_available": mark_near is not None,
            "mark_staleness_ms": mark_staleness,
        })

    total = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["spot_quality_status"]] = by_status.get(r["spot_quality_status"], 0) + 1

    by_status_by_direction: dict[str, dict[str, int]] = {}
    for r in rows:
        d = r["direction"]
        by_status_by_direction.setdefault(d, {})
        by_status_by_direction[d][r["spot_quality_status"]] = \
            by_status_by_direction[d].get(r["spot_quality_status"], 0) + 1

    fresh_rows = [r for r in rows if r["spot_quality_status"] == QUALITY_FRESH]
    fresh_cycles = {r["independent_cycle_id"] or f"NOCYCLE-{r['source_event_id']}" for r in fresh_rows}

    mark_absent = sum(1 for r in rows if not r["mark_available"])
    mark_stale_over_10s = sum(1 for r in rows if r["mark_staleness_ms"] and r["mark_staleness_ms"] > 10_000)

    return {
        "total_anchors": total,
        "healthy_age_ms_used": healthy_age_ms,
        "spot_quality_breakdown": by_status,
        "spot_quality_breakdown_by_direction": by_status_by_direction,
        "fresh_spot_rows": len(fresh_rows),
        "fresh_spot_independent_cycles": len(fresh_cycles),
        "mark_absent_rows": mark_absent,
        "mark_stale_over_10s_rows": mark_stale_over_10s,
        "reconciliation_ok": sum(by_status.values()) == total,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Known-at proof helpers (Phase: LOOKAHEAD AND KNOWN-AT PROOF)
# ---------------------------------------------------------------------------

def verify_no_lookahead(rows: list[dict]) -> dict:
    """Re-verifies, from the already-computed accounting rows, that no
    selected spot/mark sample postdates its signal's birth timestamp.
    Structural proof, not a trust assumption: `nearest_at_or_before` already
    raises on this, so a clean pass here is a redundant, defense-in-depth
    re-check over the materialized result set."""
    violations = []
    for r in rows:
        if r["spot_staleness_ms"] is not None and r["spot_staleness_ms"] < 0:
            violations.append({"signal_id": r["signal_id"], "feed": "spot_prices", "staleness_ms": r["spot_staleness_ms"]})
        if r["mark_staleness_ms"] is not None and r["mark_staleness_ms"] < 0:
            violations.append({"signal_id": r["signal_id"], "feed": "mark_prices", "staleness_ms": r["mark_staleness_ms"]})
    return {"known_at_violations": len(violations), "violations": violations}


def verify_duplicate_cycle_free(rows: list[dict]) -> dict:
    """Independent-cycle representative-rule duplicate check (same
    convention as CVD/absorption: earliest signal_birth_ts per
    independent_cycle_id among eligible signals) -- here scoped to the
    FRESH-spot-quality subset only, since that is the only outcome-blind
    eligibility gate this readiness batch defines."""
    fresh = [r for r in rows if r["spot_quality_status"] == QUALITY_FRESH]
    by_cycle: dict[str, dict] = {}
    for r in fresh:
        key = r["independent_cycle_id"] or f"NOCYCLE-{r['source_event_id']}"
        if key not in by_cycle or r["signal_birth_ts"] < by_cycle[key]["signal_birth_ts"]:
            by_cycle[key] = r
    return {
        "fresh_signal_count": len(fresh),
        "representative_cycle_count": len(by_cycle),
        "duplicates_collapsed": len(fresh) - len(by_cycle),
    }
