"""BATCH-BOOK-SPREAD-DYNAMICS-ROW-ACCOUNTING-FREEZE-V1.

Outcome-blind immutable row-accounting and lineage freeze for the accepted
child `H-BOOK-SPREAD-CHANGE-BPS-W300-V1` (`BOOK_SPREAD_CHANGE_BPS_W300_V1`).
Binds the operator-approved scientific definition (commit 6a449a64 /
FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1) to ONE exact, ordered,
reproducible anchor population and ONE exact set of selected source quotes,
before any canonical migration or outcome-linked work.

This module does NOT migrate, does NOT write canonical.sqlite, does NOT
access outcomes. It re-runs the accepted, deterministic rehearsal builder
(`ami.research.book_spread_dynamics_rehearsal`) into disposable space,
extracts five ordered manifests, and computes their component hashes plus a
single root hash. Independent replays must be byte-identical to each other
and to the accepted rehearsal evidence.

FROZEN ORDERING (this freeze's manifests): `signal_birth_ts ASC, anchor_id
ASC` -- deterministic, using only immutable anchor fields, never feature
values / quality / direction / outcomes. (The accepted rehearsal's own
content/row-manifest hashes use `anchor_id`-only ordering; both are
reproduced and reported.)

FROZEN SERIALIZATION: each record is the tuple of its fields in the fixed
order declared per manifest; each field rendered with `repr()` (full,
round-trippable float precision -- identical discipline to the accepted
rehearsal's content hash); fields joined by the unit separator U+001F;
records joined by the record separator U+001E; hashed with sha256.
"""
from __future__ import annotations

import hashlib
import sqlite3

from ami.research import book_spread_dynamics_rehearsal as REH

FREEZE_VERSION = "BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1"
FAMILY_ID = REH.FAMILY_ID
CHILD_WORKING_ID = REH.CHILD_WORKING_ID
FORMULA_VERSION = REH.FORMULA_VERSION
ORDERING_POLICY = "signal_birth_ts ASC, anchor_id ASC"
SERIALIZATION_POLICY = "per-field repr(); fields joined U+001F; records joined U+001E; sha256"

_FS = "\x1f"  # field separator
_RS = "\x1e"  # record separator

EXPECTED_TOTAL = 324
EXPECTED_EXACT = 196
EXPECTED_EXCLUDED = 128
EXPECTED_CYCLES = 97


def _hash_records(records: list[tuple]) -> str:
    payload = _RS.join(_FS.join(repr(v) for v in rec) for rec in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replay (independent rebuild via the accepted rehearsal builder)
# ---------------------------------------------------------------------------

def replay_into(disposable_conn: sqlite3.Connection, canonical_conn: sqlite3.Connection,
                micro_conn: sqlite3.Connection, input_manifest_id: str = "row-accounting-freeze-v1-replay") -> dict:
    """Builds a fresh disposable feature set via the accepted rehearsal
    builder (no copying of prior output). Returns the rehearsal counts."""
    return REH.run_rehearsal(disposable_conn, canonical_conn, micro_conn, input_manifest_id)


# ---------------------------------------------------------------------------
# Ordered manifests (Phase 3/4/5/6)
# ---------------------------------------------------------------------------

_ANCHOR_FIELDS = (
    "anchor_id", "canonical_signal_ts", "symbol", "venue", "market_segment", "quote_currency",
    "direction", "cycle_id", "source_quality_class", "exclusion_reason", "exclusion_endpoint",
    "is_cycle_representative", "specification_hash",
)

_EXACT_FIELDS = (
    "anchor_id", "cycle_id",
    "current_target_ts", "current_quote_id", "current_quote_ts", "current_quote_age_ms",
    "current_bid", "current_ask", "current_mid", "current_spread_bps",
    "historical_target_ts", "historical_quote_id", "historical_quote_ts", "historical_quote_age_ms",
    "historical_bid", "historical_ask", "historical_mid", "historical_spread_bps",
    "spread_change_bps_w300", "formula_version", "symbol", "venue", "market_segment", "quote_currency",
    "source_quality_class", "known_at_ts", "feature_available_ts", "is_cycle_representative",
    "specification_hash",
)

_EXCLUSION_FIELDS = (
    "anchor_id", "cycle_id", "source_quality_class", "exclusion_reason", "exclusion_endpoint",
    "current_quality_status", "historical_quality_status",
    "current_quote_age_ms", "historical_quote_age_ms", "specification_hash",
)


def ordered_anchor_manifest(conn: sqlite3.Connection) -> tuple[list[tuple], str]:
    rows = conn.execute(
        f"SELECT {', '.join(_ANCHOR_FIELDS)} FROM book_spread_change_w300 "
        "ORDER BY canonical_signal_ts ASC, anchor_id ASC").fetchall()
    return rows, _hash_records(rows)


def ordered_exact_feature_manifest(conn: sqlite3.Connection) -> tuple[list[tuple], str]:
    rows = conn.execute(
        f"SELECT {', '.join(_EXACT_FIELDS)} FROM book_spread_change_w300 "
        "WHERE source_quality_class='EXACT_RECONSTRUCTABLE' "
        "ORDER BY canonical_signal_ts ASC, anchor_id ASC").fetchall()
    return rows, _hash_records(rows)


def ordered_exclusion_manifest(conn: sqlite3.Connection) -> tuple[list[tuple], str]:
    rows = conn.execute(
        f"SELECT {', '.join(_EXCLUSION_FIELDS)} FROM book_spread_change_w300 "
        "WHERE source_quality_class != 'EXACT_RECONSTRUCTABLE' "
        "ORDER BY canonical_signal_ts ASC, anchor_id ASC").fetchall()
    return rows, _hash_records(rows)


def ordered_cycle_membership_manifest(conn: sqlite3.Connection) -> tuple[list[tuple], str]:
    """One record per (cycle_id, member anchor_id) among EXACT rows, ordered
    by cycle_id then signal_ts then anchor_id -- the full membership of each
    exact independent cycle."""
    rows = conn.execute(
        "SELECT cycle_id, anchor_id, canonical_signal_ts, is_cycle_representative "
        "FROM book_spread_change_w300 WHERE source_quality_class='EXACT_RECONSTRUCTABLE' "
        "ORDER BY cycle_id ASC, canonical_signal_ts ASC, anchor_id ASC").fetchall()
    return rows, _hash_records(rows)


def ordered_representative_manifest(conn: sqlite3.Connection) -> tuple[list[tuple], str]:
    rows = conn.execute(
        "SELECT cycle_id, anchor_id, canonical_signal_ts FROM book_spread_change_w300 "
        "WHERE is_cycle_representative=1 ORDER BY cycle_id ASC, anchor_id ASC").fetchall()
    return rows, _hash_records(rows)


# ---------------------------------------------------------------------------
# Accounting identities (Phase 13) and known-at revalidation (Phase 9)
# ---------------------------------------------------------------------------

def accounting_identities(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM book_spread_change_w300").fetchone()[0]
    by_class = dict(conn.execute(
        "SELECT source_quality_class, COUNT(*) FROM book_spread_change_w300 GROUP BY source_quality_class").fetchall())
    exact = by_class.get("EXACT_RECONSTRUCTABLE", 0)
    stale = by_class.get("STALE_SOURCE", 0)
    unavail = by_class.get("UNAVAILABLE_BEFORE_COLLECTION", 0)
    excluded = total - exact
    exact_cycles = conn.execute(
        "SELECT COUNT(DISTINCT cycle_id) FROM book_spread_change_w300 "
        "WHERE source_quality_class='EXACT_RECONSTRUCTABLE'").fetchone()[0]
    reps = conn.execute("SELECT COUNT(*) FROM book_spread_change_w300 WHERE is_cycle_representative=1").fetchone()[0]
    distinct_anchor = conn.execute("SELECT COUNT(DISTINCT anchor_id) FROM book_spread_change_w300").fetchone()[0]
    # every exact anchor has a feature value; no excluded anchor has one
    exact_with_value = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE source_quality_class='EXACT_RECONSTRUCTABLE' "
        "AND spread_change_bps_w300 IS NOT NULL").fetchone()[0]
    excluded_with_value = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE source_quality_class!='EXACT_RECONSTRUCTABLE' "
        "AND spread_change_bps_w300 IS NOT NULL").fetchone()[0]
    # representative is over distinct cycles
    reps_over_multi = conn.execute(
        "SELECT COUNT(*) FROM (SELECT cycle_id FROM book_spread_change_w300 WHERE is_cycle_representative=1 "
        "GROUP BY cycle_id HAVING COUNT(*)>1)").fetchone()[0]
    return {
        "total": total, "exact": exact, "stale": stale, "unavailable": unavail, "excluded": excluded,
        "invalid_crossed": by_class.get("INVALID_QUOTE_CROSSED", 0),
        "invalid_zero_neg": by_class.get("INVALID_QUOTE_ZERO_OR_NEG", 0),
        "invalid_locked": by_class.get("INVALID_QUOTE_LOCKED", 0),
        "repaired_exact": by_class.get("REPAIRED_EXACT", 0),
        "source_gapped": by_class.get("SOURCE_GAPPED", 0),
        "proxy_only": by_class.get("PROXY_ONLY", 0),
        "exact_independent_cycles": exact_cycles, "cycle_representatives": reps,
        "distinct_anchor_ids": distinct_anchor,
        "exact_with_feature_value": exact_with_value,
        "excluded_with_feature_value": excluded_with_value,
        "cycles_with_multiple_representatives": reps_over_multi,
        "identity_anchor": total == EXPECTED_TOTAL == (exact + stale + unavail) and distinct_anchor == EXPECTED_TOTAL,
        "identity_nonexact": excluded == EXPECTED_EXCLUDED == (stale + unavail),
        "identity_exact_cycles": exact == EXPECTED_EXACT and exact_cycles == EXPECTED_CYCLES,
        "identity_representatives": reps == EXPECTED_CYCLES and reps_over_multi == 0
            and exact_with_value == EXPECTED_EXACT and excluded_with_value == 0,
    }


def known_at_revalidation(conn: sqlite3.Connection) -> dict:
    """Re-prove no-lookahead at both endpoints for every row directly from the
    frozen lineage fields (independent of the builder's own in-flight check)."""
    cur_future = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE current_quote_ts IS NOT NULL "
        "AND current_quote_ts > current_target_ts").fetchone()[0]
    hist_future = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE historical_quote_ts IS NOT NULL "
        "AND historical_quote_ts > historical_target_ts").fetchone()[0]
    cur_stale = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE source_quality_class='EXACT_RECONSTRUCTABLE' "
        "AND current_quote_age_ms > ?", (REH.BOOK_TICKER_HEALTHY_AGE_MS,)).fetchone()[0]
    hist_stale = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE source_quality_class='EXACT_RECONSTRUCTABLE' "
        "AND historical_quote_age_ms > ?", (REH.BOOK_TICKER_HEALTHY_AGE_MS,)).fetchone()[0]
    identity_bad = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE symbol!='ETHUSDT' OR venue!='BINANCE_USDM_PERP' "
        "OR market_segment!='PERPETUAL_FUTURES' OR quote_currency!='USDT'").fetchone()[0]
    known_at_field = conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE known_at_ts != canonical_signal_ts "
        "OR feature_available_ts != canonical_signal_ts").fetchone()[0]
    return {
        "current_endpoint_future_quote_selections": cur_future,
        "historical_endpoint_future_quote_selections": hist_future,
        "current_endpoint_staleness_violations": cur_stale,
        "historical_endpoint_staleness_violations": hist_stale,
        "identity_violations": identity_bad,
        "known_at_field_violations": known_at_field,
        "all_zero": (cur_future == hist_future == cur_stale == hist_stale == identity_bad == known_at_field == 0),
    }


# ---------------------------------------------------------------------------
# Root hash (Phase 7)
# ---------------------------------------------------------------------------

def root_hash(component_hashes: dict[str, str]) -> str:
    """Deterministic root over the ordered (name, full-hash) pairs. Names are
    sorted; each pair rendered `name=hash`; joined by U+001E; sha256."""
    payload = _RS.join(f"{k}={component_hashes[k]}" for k in sorted(component_hashes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_all_manifests(conn: sqlite3.Connection) -> dict:
    a_rows, a_h = ordered_anchor_manifest(conn)
    e_rows, e_h = ordered_exact_feature_manifest(conn)
    x_rows, x_h = ordered_exclusion_manifest(conn)
    cm_rows, cm_h = ordered_cycle_membership_manifest(conn)
    r_rows, r_h = ordered_representative_manifest(conn)
    return {
        "ordered_anchor": {"rows": a_rows, "hash": a_h, "count": len(a_rows), "fields": _ANCHOR_FIELDS},
        "exact_feature": {"rows": e_rows, "hash": e_h, "count": len(e_rows), "fields": _EXACT_FIELDS},
        "exclusion": {"rows": x_rows, "hash": x_h, "count": len(x_rows), "fields": _EXCLUSION_FIELDS},
        "cycle_membership": {"rows": cm_rows, "hash": cm_h, "count": len(cm_rows),
                             "fields": ("cycle_id", "anchor_id", "canonical_signal_ts", "is_cycle_representative")},
        "representative": {"rows": r_rows, "hash": r_h, "count": len(r_rows),
                           "fields": ("cycle_id", "anchor_id", "canonical_signal_ts")},
    }
