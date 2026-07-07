"""BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1 (M-0036).

Controlled, outcome-blind canonical migration/backfill entry point for the
frozen W300 additive spread-change child (`BOOK_SPREAD_CHANGE_BPS_W300_V1`,
row-accounting root 33c4f4be..., freeze commit 54d00dca).

Composes the already-validated DDL (folded verbatim into
ami.warehouse.schema.init_schema() as _SCHEMA_PHASE_BOOK_SPREAD) with a
frozen-source-package backfill: every value written is copied verbatim from
the retained, hash-verified disposable rehearsal database
(.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite, accepted rehearsal
commit 6a449a64) -- NO recomputation, NO network call, NO outcome access.
The single disposable table `book_spread_change_w300` is split into three
canonical tables (feature 196 / quality 324 / exclusion 128); the only added
fields are the migration-provenance constants `row_accounting_root` and
`migration_id` plus two derived-but-deterministic fields
(`exact_eligibility_flag`, `exclusion_precedence_position`).

NOT_CALLED_AUTOMATICALLY: `run_canonical_migration()` takes explicit
connections; never an import side effect (CVD/absorption precedent).

Idempotency: rerunning against the SAME frozen source is a content-compare
NOOP for every row (never a duplicate insert, never an UPDATE/DELETE/REPLACE).
A same-key row with DIFFERENT content raises `ConflictNonIdentical`
(fail-closed).
"""
from __future__ import annotations

import hashlib
import sqlite3

MIGRATION_ID = "M-0036"
FORMULA_VERSION = "BOOK_SPREAD_CHANGE_BPS_W300_V1"
ROW_ACCOUNTING_ROOT = "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31"
SPECIFICATION_HASH = "ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212"

# frozen exclusion precedence (index = position); identical to the freeze module
_PRECEDENCE = (
    "UNAVAILABLE_BEFORE_COLLECTION", "INVALID_QUOTE_ZERO_OR_NEG", "INVALID_QUOTE_CROSSED",
    "INVALID_QUOTE_LOCKED", "STALE_SOURCE",
)


class ConflictNonIdentical(Exception):
    """A canonical row already exists under the same immutable key with
    DIFFERENT content -- hard stop, never overwritten."""


def _bsf(anchor_id: str) -> str:
    return "BSF-" + hashlib.sha256(f"ETHUSDT|{anchor_id}|{FORMULA_VERSION}".encode()).hexdigest()[:24]


def _bsq(anchor_id: str) -> str:
    return "BSQ-" + hashlib.sha256(f"ETHUSDT|{anchor_id}|{FORMULA_VERSION}|quality".encode()).hexdigest()[:24]


def _bsx(anchor_id: str) -> str:
    return "BSX-" + hashlib.sha256(f"ETHUSDT|{anchor_id}|{FORMULA_VERSION}|exclusion".encode()).hexdigest()[:24]


_FEATURE_COLS = (
    "feature_id", "formula_version", "anchor_id", "cycle_id", "signal_birth_ts", "symbol", "venue",
    "market_segment", "quote_currency", "direction", "current_target_ts", "current_quote_id",
    "current_quote_ts", "current_quote_age_ms", "current_bid", "current_ask", "current_mid",
    "current_spread_bps", "historical_target_ts", "historical_quote_id", "historical_quote_ts",
    "historical_quote_age_ms", "historical_bid", "historical_ask", "historical_mid",
    "historical_spread_bps", "spread_change_bps_w300", "source_quality_class", "known_at_ts",
    "feature_available_ts", "is_cycle_representative", "specification_hash", "row_accounting_root",
    "migration_id", "input_manifest_id", "created_ms",
)
_QUALITY_COLS = (
    "quality_id", "formula_version", "anchor_id", "signal_birth_ts", "cycle_id", "symbol",
    "source_quality_class", "exclusion_reason", "exclusion_endpoint", "exact_eligibility_flag",
    "is_cycle_representative", "current_quality_status", "historical_quality_status",
    "current_quote_age_ms", "historical_quote_age_ms", "specification_hash", "row_accounting_root",
    "migration_id", "input_manifest_id", "created_ms",
)
_EXCLUSION_COLS = (
    "exclusion_id", "formula_version", "anchor_id", "cycle_id", "source_quality_class",
    "exclusion_reason", "exclusion_endpoint", "exclusion_precedence_position", "current_quality_status",
    "historical_quality_status", "current_quote_age_ms", "historical_quote_age_ms", "specification_hash",
    "row_accounting_root", "migration_id", "input_manifest_id", "created_ms",
)

_PK = {
    "ami_book_spread_change_windowed_flow": "feature_id",
    "ami_book_spread_change_window_quality_v1": "quality_id",
    "ami_book_spread_change_exclusions": "exclusion_id",
}


def _rows_from_source(source_ro: sqlite3.Connection) -> dict:
    """Reads the frozen disposable rehearsal table and returns the three
    ordered target row lists (feature/quality/exclusion), values copied
    verbatim. Ordered by canonical_signal_ts, anchor_id (frozen ordering)."""
    src = source_ro.execute(
        "SELECT anchor_id, canonical_signal_ts, symbol, venue, market_segment, quote_currency, "
        "direction, cycle_id, is_cycle_representative, current_target_ts, current_quote_id, "
        "current_quote_ts, current_quote_age_ms, current_bid, current_ask, current_mid, "
        "current_spread_bps, current_quality_status, historical_target_ts, historical_quote_id, "
        "historical_quote_ts, historical_quote_age_ms, historical_bid, historical_ask, historical_mid, "
        "historical_spread_bps, historical_quality_status, spread_change_bps_w300, source_quality_class, "
        "exclusion_reason, exclusion_endpoint, known_at_ts, feature_available_ts, input_manifest_id, "
        "created_ms FROM book_spread_change_w300 ORDER BY canonical_signal_ts, anchor_id").fetchall()
    feature, quality, exclusion = [], [], []
    for r in src:
        (anchor_id, sig_ts, symbol, venue, seg, ccy, direction, cycle_id, is_rep, cur_tt, cur_qid,
         cur_qts, cur_age, cur_bid, cur_ask, cur_mid, cur_sp, cur_qs, hist_tt, hist_qid, hist_qts,
         hist_age, hist_bid, hist_ask, hist_mid, hist_sp, hist_qs, change, sqc, excl_reason, excl_ep,
         known_at, feat_avail, manifest_id, created_ms) = r
        eligible = 1 if sqc == "EXACT_RECONSTRUCTABLE" else 0
        quality.append((
            _bsq(anchor_id), FORMULA_VERSION, anchor_id, sig_ts, cycle_id, symbol, sqc, excl_reason,
            excl_ep, eligible, is_rep, cur_qs, hist_qs, cur_age, hist_age, SPECIFICATION_HASH,
            ROW_ACCOUNTING_ROOT, MIGRATION_ID, manifest_id, created_ms))
        if eligible:
            feature.append((
                _bsf(anchor_id), FORMULA_VERSION, anchor_id, cycle_id, sig_ts, symbol, venue, seg, ccy,
                direction, cur_tt, cur_qid, cur_qts, cur_age, cur_bid, cur_ask, cur_mid, cur_sp,
                hist_tt, hist_qid, hist_qts, hist_age, hist_bid, hist_ask, hist_mid, hist_sp, change,
                sqc, known_at, feat_avail, is_rep, SPECIFICATION_HASH, ROW_ACCOUNTING_ROOT, MIGRATION_ID,
                manifest_id, created_ms))
        else:
            exclusion.append((
                _bsx(anchor_id), FORMULA_VERSION, anchor_id, cycle_id, sqc, excl_reason, excl_ep,
                _PRECEDENCE.index(sqc), cur_qs, hist_qs, cur_age, hist_age, SPECIFICATION_HASH,
                ROW_ACCOUNTING_ROOT, MIGRATION_ID, manifest_id, created_ms))
    return {"feature": feature, "quality": quality, "exclusion": exclusion}


def _insert_plan(conn, table, cols, rows) -> dict:
    col_list = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    pk = _PK[table]
    pk_idx = cols.index(pk)
    inserted = noop = 0
    for row in rows:
        existing = conn.execute(
            f"SELECT {col_list} FROM {table} WHERE {pk}=?", (row[pk_idx],)).fetchone()
        if existing is not None:
            if tuple(existing) != tuple(row):
                raise ConflictNonIdentical(f"{table}: {pk}={row[pk_idx]} exists with different content")
            noop += 1
            continue
        conn.execute(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", row)
        inserted += 1
    return {"inserted": inserted, "noop_identical": noop, "source_rows": len(rows)}


def run_canonical_migration(conn: sqlite3.Connection, source_ro: sqlite3.Connection) -> dict:
    """Idempotent, content-identical split-copy from the frozen retained
    rehearsal database into the (already schema-migrated) canonical
    connection. Insert-only; no UPDATE/DELETE/REPLACE anywhere."""
    rows = _rows_from_source(source_ro)
    result = {
        "ami_book_spread_change_windowed_flow": _insert_plan(
            conn, "ami_book_spread_change_windowed_flow", _FEATURE_COLS, rows["feature"]),
        "ami_book_spread_change_window_quality_v1": _insert_plan(
            conn, "ami_book_spread_change_window_quality_v1", _QUALITY_COLS, rows["quality"]),
        "ami_book_spread_change_exclusions": _insert_plan(
            conn, "ami_book_spread_change_exclusions", _EXCLUSION_COLS, rows["exclusion"]),
    }
    conn.commit()
    return result


def canonical_counts(conn: sqlite3.Connection) -> dict:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _PK}


# ---------------------------------------------------------------------------
# Canonical replay hashes (Phase 12) -- rebuild the frozen manifests directly
# from the canonical destination tables, using the frozen ordering, to prove
# the migrated content reproduces the frozen manifest hashes.
# ---------------------------------------------------------------------------

_FS = "\x1f"
_RS = "\x1e"


def _hash_records(records) -> str:
    return hashlib.sha256(_RS.join(_FS.join(repr(v) for v in rec) for rec in records).encode("utf-8")).hexdigest()


def canonical_replay_hashes(conn) -> dict:
    """Rebuilds all five frozen manifests directly from the canonical
    destination tables, using the frozen ordering (signal_birth_ts ASC,
    anchor_id ASC) and serialization, and returns their hashes -- to be
    compared against the frozen component hashes. venue/market_segment/
    quote_currency are constants for this family (stored per-row on the
    feature table; supplied as literals for the quality-derived anchor
    manifest, where they are invariant); direction is joined from the
    canonical identity table for the anchor manifest."""
    anchor = conn.execute(
        "SELECT q.anchor_id, q.signal_birth_ts, q.symbol, 'BINANCE_USDM_PERP', 'PERPETUAL_FUTURES', 'USDT', "
        "s.direction, q.cycle_id, q.source_quality_class, q.exclusion_reason, q.exclusion_endpoint, "
        "q.is_cycle_representative, q.specification_hash "
        "FROM ami_book_spread_change_window_quality_v1 q "
        "JOIN ami_signal_lifecycle s ON s.signal_id=q.anchor_id "
        "ORDER BY q.signal_birth_ts, q.anchor_id").fetchall()
    exact = conn.execute(
        "SELECT anchor_id, cycle_id, current_target_ts, current_quote_id, current_quote_ts, "
        "current_quote_age_ms, current_bid, current_ask, current_mid, current_spread_bps, "
        "historical_target_ts, historical_quote_id, historical_quote_ts, historical_quote_age_ms, "
        "historical_bid, historical_ask, historical_mid, historical_spread_bps, spread_change_bps_w300, "
        "formula_version, symbol, venue, market_segment, quote_currency, source_quality_class, "
        "known_at_ts, feature_available_ts, is_cycle_representative, specification_hash "
        "FROM ami_book_spread_change_windowed_flow ORDER BY signal_birth_ts, anchor_id").fetchall()
    exclusion = conn.execute(
        "SELECT x.anchor_id, x.cycle_id, x.source_quality_class, x.exclusion_reason, x.exclusion_endpoint, "
        "x.current_quality_status, x.historical_quality_status, x.current_quote_age_ms, "
        "x.historical_quote_age_ms, x.specification_hash "
        "FROM ami_book_spread_change_exclusions x "
        "JOIN ami_book_spread_change_window_quality_v1 q ON q.anchor_id=x.anchor_id "
        "ORDER BY q.signal_birth_ts, x.anchor_id").fetchall()
    cycle_membership = conn.execute(
        "SELECT cycle_id, anchor_id, signal_birth_ts, is_cycle_representative "
        "FROM ami_book_spread_change_windowed_flow "
        "ORDER BY cycle_id, signal_birth_ts, anchor_id").fetchall()
    representative = conn.execute(
        "SELECT cycle_id, anchor_id, signal_birth_ts FROM ami_book_spread_change_windowed_flow "
        "WHERE is_cycle_representative=1 ORDER BY cycle_id, anchor_id").fetchall()
    return {
        "ordered_anchor_manifest": _hash_records(anchor),
        "exact_feature_manifest": _hash_records(exact),
        "exclusion_manifest": _hash_records(exclusion),
        "cycle_membership_manifest": _hash_records(cycle_membership),
        "representative_manifest": _hash_records(representative),
    }
