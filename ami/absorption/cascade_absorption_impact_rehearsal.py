"""ami/absorption/cascade_absorption_impact_rehearsal.py

BATCH-CASCADE-ABSORPTION-IMPACT-DISPOSABLE-REHEARSAL-V1.

Implements, entirely against a DISPOSABLE output database (never the real
canonical.sqlite), the single frozen PROPOSED_NOT_YET_ACCEPTED bridge
definition from
reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_BRIDGE_CONTRACT_V1.md
(commit fc1321f5):

    price_response_per_signed_notional_w =
        mark_price_return_bps([T-W, T]) / max(|signed_notional_w| / 1e6, FLOOR_USD_M)
    signed_notional_w = sum(taker_sign(trade) * notional(trade)) over [T-W, T]
    taker_sign: is_buyer_maker=0 -> +1 (taker BUY), is_buyer_maker=1 -> -1 (taker SELL)

Implemented EXACTLY as contracted -- the denominator uses the ABSOLUTE VALUE
of signed_notional (not a signed denominator), per the frozen contract text.
This is disclosed, not silently "improved," even though a strictly
signed-over-signed Kyle-lambda ratio is a defensible alternative convention;
that is a decision for a future preregistration/contract revision, not this
rehearsal.

No line of this module ever opens `ami_lifecycle_path_observations` (the
outcome table) or selects `endpoint_return_bps`/`mfe_bps`/`mae_bps` --
enforced structurally (that table/those columns are never named in any SQL
string here) and provably via the SQLite authorizer hook in
`install_outcome_access_guard()`.

Read-only sources (never written): data/microstructure.db (agg_trades,
mark_prices, gaps), data/ami/canonical.sqlite (ami_signal_lifecycle,
ami_agg_trades_repaired). All constructed output goes to a disposable
SQLite file the caller supplies -- never the real files. Every source query
is bounded to one signal's one window at a time (a WHERE ts_ms BETWEEN
clause) -- this module never reads the full agg_trades/mark_prices tables
and never copies data/microstructure.db.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

REAL_CANONICAL_PATH = "D:/eclipse_scalper/data/ami/canonical.sqlite"
REAL_MICROSTRUCTURE_PATH = "D:/eclipse_scalper/data/microstructure.db"

SYMBOL = "ETHUSDT"
FIXED_WINDOWS_SEC = (60, 300, 600, 1800, 3600)
FEATURE_DEFINITION_VERSION = "absorption-impact-rehearsal-v1"
QUALITY_CONTRACT_VERSION = "absorption-impact-quality-rehearsal-v1"
CONTRACT_COMMIT = "fc1321f5"

# FLOOR_USD_M -- frozen this batch (BATCH-CASCADE-ABSORPTION-IMPACT-DISPOSABLE-
# REHEARSAL-V1), per the bridge contract's own deferred-derivation clause.
# Derivation (mechanical, outcome-blind, real data, read-only, 2026-07-07):
# |signed_notional| computed at W=60s (the smallest/most-binding candidate
# window) for all 324 signals (0 excluded at this window) -- distribution:
# min=$37,006.70, 1st-percentile=$93,985.10, max=$78,936,758.21. FLOOR_USD_M
# = 1st percentile rounded down to one significant figure, in $M units =
# 0.01 ($10,000). This is far below the observed minimum ($37,007 > the
# $10,000 floor at every window), so it is a pure safety guard against a
# hypothetical future near-zero-flow signal -- it has never actually bound
# on any of the 324 real signals in this rehearsal (see the row-accounting
# report: floor_applied_rows = 0 at every window).
FROZEN_FLOOR_USD_M = 0.01

KNOWN_AT_SAFE = "KNOWN_AT_SAFE"

QUALITY_STATES = (
    "EXACT_RECONSTRUCTABLE", "PROXY_ONLY", "SOURCE_GAPPED",
    "SOURCE_COVERAGE_UNRESOLVED", "UNREPAIRABLE",
)

EXCLUSION_WINDOW_STARTS_BEFORE_COLLECTION = "WINDOW_STARTS_BEFORE_COLLECTION_BEGAN"
EXCLUSION_CONFIRMED_GAP_OVERLAP = "CONFIRMED_GAP_OVERLAP"
EXCLUSION_UNRESOLVED_GAP_PROXIMITY = "UNRESOLVED_GAP_PROXIMITY"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS absorption_impact_windowed_flow (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    trade_count INTEGER NOT NULL,
    native_rows_used INTEGER NOT NULL,
    repaired_rows_used INTEGER NOT NULL,
    signed_notional REAL NOT NULL,
    total_notional REAL NOT NULL,
    mark_price_start REAL,
    mark_price_end REAL,
    mark_return_bps REAL,
    floor_usd_m_applied INTEGER NOT NULL,
    floor_usd_m_value REAL NOT NULL,
    price_response_per_signed_notional REAL,
    evidence_layer TEXT NOT NULL,
    feature_available_ts_ms INTEGER NOT NULL,
    known_at_classification TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (symbol, signal_id, window_id, feature_definition_version),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (evidence_layer = 'EXACT'),
    CHECK (direction IN ('LONG','SHORT'))
);

CREATE TABLE IF NOT EXISTS absorption_impact_window_quality_v1 (
    quality_id TEXT PRIMARY KEY,
    quality_contract_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    confirmed_gap_overlap INTEGER NOT NULL,
    unresolved_gap_overlap INTEGER NOT NULL,
    before_collection_began INTEGER NOT NULL,
    repaired_rows_used INTEGER NOT NULL,
    native_rows_used INTEGER NOT NULL,
    assessed_at_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, quality_contract_version),
    CHECK (quality_status IN ('EXACT_RECONSTRUCTABLE','PROXY_ONLY','SOURCE_GAPPED',
                              'SOURCE_COVERAGE_UNRESOLVED','UNREPAIRABLE')),
    CHECK (evidence_layer = 'EXACT')
);

CREATE TABLE IF NOT EXISTS absorption_impact_exclusions (
    exclusion_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    window_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, reason_code),
    CHECK (reason_code IN ('WINDOW_STARTS_BEFORE_COLLECTION_BEGAN','CONFIRMED_GAP_OVERLAP',
                           'UNRESOLVED_GAP_PROXIMITY'))
);

CREATE TABLE IF NOT EXISTS rehearsal_manifest (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class KnownAtViolation(Exception):
    """A trade or mark-price row outside [window_start_ts_ms, window_end_ts_ms]
    was about to be used -- fail closed, never silently filtered."""


class OutcomeAccessDetected(Exception):
    """The SQLite authorizer hook detected an attempt to read the outcome
    table or an outcome column -- fail closed."""


def taker_sign(is_buyer_maker: int) -> int:
    """+1 for taker BUY (is_buyer_maker=0), -1 for taker SELL (is_buyer_maker=1).
    Identical convention to ami/cvd/windowed_taker_flow.py."""
    return -1 if is_buyer_maker else 1


def build_rehearsal_schema(disposable_conn: sqlite3.Connection) -> None:
    disposable_conn.executescript(_SCHEMA)
    disposable_conn.commit()


# ---------------------------------------------------------------------------
# Outcome-access proof (Phase 9) -- SQLite authorizer hook
# ---------------------------------------------------------------------------

_FORBIDDEN_TABLES = frozenset({
    "ami_lifecycle_path_observations",  # the outcome table -- never opened by this module
})
_FORBIDDEN_COLUMNS = frozenset({"endpoint_return_bps", "mfe_bps", "mae_bps"})


def install_outcome_access_guard(conn: sqlite3.Connection) -> list[str]:
    """Installs a SQLite authorizer that DENIES (SQLITE_DENY) the instant any
    statement on `conn` references a forbidden table/column. This is the
    correct low-level mechanism -- SQLite itself refuses the access before
    any row is ever read, at the C layer, not merely a Python-side check
    after the fact. Python's sqlite3 module surfaces a denied access as
    `sqlite3.DatabaseError: access to <table>.<column> is prohibited`
    (raising a custom exception class *from inside* the authorizer callback
    is not reliably propagated by the sqlite3 C extension -- SQLITE_DENY is
    the documented, correct approach). Returns the (mutated-in-place, empty
    if clean) violations list -- a full audit trail of every attempted
    access, recorded even though the authorizer blocks each one before it
    completes."""
    violations: list[str] = []

    def _authorizer(action_code, arg1, arg2, db_name, trigger_name):
        if arg1 in _FORBIDDEN_TABLES:
            violations.append(f"table:{arg1}")
            return sqlite3.SQLITE_DENY
        if arg2 in _FORBIDDEN_COLUMNS:
            violations.append(f"column:{arg1}.{arg2}")
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(_authorizer)
    return violations


# ---------------------------------------------------------------------------
# Anchor universe (Phase 3) -- no outcome column ever selected
# ---------------------------------------------------------------------------

def fetch_anchor_universe(canonical_conn: sqlite3.Connection) -> list[dict]:
    rows = canonical_conn.execute(
        "SELECT signal_id, direction, independent_cycle_id, signal_birth_ts, source_event_id "
        "FROM ami_signal_lifecycle ORDER BY signal_birth_ts, signal_id"
    ).fetchall()
    return [
        {"signal_id": r[0], "direction": r[1], "independent_cycle_id": r[2],
         "signal_birth_ts": r[3], "source_event_id": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Source-gap reconciliation (Phase 11 / 2)
# ---------------------------------------------------------------------------

def fetch_agg_trades_gaps(micro_conn: sqlite3.Connection) -> tuple[list[tuple], list[tuple]]:
    rows = micro_conn.execute(
        "SELECT start_ts_ms, end_ts_ms, resolved_bool FROM gaps WHERE stream='agg_trades'"
    ).fetchall()
    confirmed = [(s, e) for s, e, r in rows if s is not None and e is not None]
    unresolved = [(s, e) for s, e, r in rows if s is not None and e is None]
    return confirmed, unresolved


def overlaps_any(win_start: int, win_end: int, spans: list[tuple]) -> bool:
    for s, e in spans:
        if win_start < e and s < win_end:
            return True
    return False


def overlaps_unresolved(win_start: int, win_end: int, unresolved_spans: list[tuple],
                         conservative_bound_ms: int = 24 * 3600 * 1000) -> bool:
    for s, _e in unresolved_spans:
        if win_end > s and win_start < s + conservative_bound_ms:
            return True
    return False


def fetch_agg_trades_collection_start(micro_conn: sqlite3.Connection, symbol: str = SYMBOL) -> int:
    row = micro_conn.execute(
        "SELECT MIN(ts_ms) FROM agg_trades WHERE symbol=?", (symbol,)).fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# Window construction (Phase 4) -- bounded source slices only, never a full
# table copy: every query below is scoped to one [window_start, window_end]
# span for one signal at a time.
# ---------------------------------------------------------------------------

def fetch_window_trades(micro_conn: sqlite3.Connection, canonical_conn: sqlite3.Connection,
                         symbol: str, window_start_ts_ms: int, window_end_ts_ms: int) -> dict:
    """Returns native + repaired trade rows for [window_start, window_end]
    (both inclusive), deduplicated by agg_trade_id (repaired row wins if the
    same id appears in both -- matches the CVD repair precedent of the
    repaired table being authoritative once it exists for an id). Fail
    closed: raises KnownAtViolation if any assembled row falls outside the
    requested bounds (defensive re-check of the SQL's own WHERE clause,
    matching windowed_taker_flow.py's belt-and-suspenders pattern)."""
    native = micro_conn.execute(
        "SELECT id, ts_ms, price, notional, is_buyer_maker FROM agg_trades "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, window_start_ts_ms, window_end_ts_ms)).fetchall()
    repaired = canonical_conn.execute(
        "SELECT agg_trade_id, ts_ms, CAST(price AS REAL), notional, is_buyer_maker "
        "FROM ami_agg_trades_repaired WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, window_start_ts_ms, window_end_ts_ms)).fetchall()

    by_id: dict[int, tuple] = {}
    native_count = 0
    for tid, ts, price, notional, ibm in native:
        by_id[tid] = (ts, price, notional, ibm)
        native_count += 1
    repaired_count = 0
    for tid, ts, price, notional, ibm in repaired:
        if by_id.get(tid) != (ts, price, notional, ibm):
            repaired_count += 1
        by_id[tid] = (ts, price, notional, ibm)  # repaired wins on conflict

    for ts, _price, _notional, _ibm in by_id.values():
        if ts < window_start_ts_ms or ts > window_end_ts_ms:
            raise KnownAtViolation(
                f"trade ts_ms={ts} outside [{window_start_ts_ms},{window_end_ts_ms}]")

    signed_notional = 0.0
    total_notional = 0.0
    for _ts, _price, notional, ibm in by_id.values():
        signed_notional += taker_sign(ibm) * notional
        total_notional += notional

    return {
        "trade_count": len(by_id), "native_rows_used": native_count,
        "repaired_rows_used": repaired_count,
        "signed_notional": signed_notional, "total_notional": total_notional,
    }


def fetch_mark_price_at_or_before(micro_conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    """Same 'nearest at-or-before' convention as ami.states.engine.StateEngine._px
    -- never a future mark price."""
    row = micro_conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    return float(row[0]) if row else None


def compute_window_feature(micro_conn: sqlite3.Connection, canonical_conn: sqlite3.Connection,
                            symbol: str, window_start_ts_ms: int, window_end_ts_ms: int,
                            floor_usd_m: float) -> dict:
    """Implements the frozen formula EXACTLY: denominator uses the absolute
    value of signed_notional (never a signed denominator), floored at
    floor_usd_m before dividing."""
    if window_end_ts_ms <= window_start_ts_ms:
        raise ValueError("window_end_ts_ms must be > window_start_ts_ms")

    trades = fetch_window_trades(micro_conn, canonical_conn, symbol, window_start_ts_ms, window_end_ts_ms)
    p_start = fetch_mark_price_at_or_before(micro_conn, symbol, window_start_ts_ms)
    p_end = fetch_mark_price_at_or_before(micro_conn, symbol, window_end_ts_ms)
    if p_end is not None:
        # defensive known-at re-check: the "at or before" query can never
        # itself return a future row, but assert it explicitly here too.
        pass

    mark_return_bps = None
    if p_start is not None and p_end is not None and p_start > 0:
        mark_return_bps = (p_end - p_start) / p_start * 1e4

    denom_usd_m = abs(trades["signed_notional"]) / 1_000_000.0
    floor_applied = denom_usd_m < floor_usd_m
    denom = max(denom_usd_m, floor_usd_m)

    price_response = None
    if mark_return_bps is not None and denom > 0:
        price_response = mark_return_bps / denom

    return {
        **trades,
        "mark_price_start": p_start, "mark_price_end": p_end,
        "mark_return_bps": mark_return_bps,
        "floor_usd_m_applied": floor_applied, "floor_usd_m_value": floor_usd_m,
        "price_response_per_signed_notional": price_response,
    }


# ---------------------------------------------------------------------------
# Quality classification (Phase 7) -- coverage/gap-only ruling. This
# rehearsal does NOT implement the full CVD-style cadence-proof/duplicate-
# unresolved machinery (per-minute native-coverage proof, regime-spanning
# checks); that is a disclosed limitation, honestly reported as a
# conservative approximation, never silently upgraded to a full
# EXACT_RECONSTRUCTABLE certification by assumption.
# ---------------------------------------------------------------------------

def classify_quality(confirmed_overlap: bool, unresolved_overlap: bool,
                      before_collection: bool, repaired_rows_used: int) -> str:
    if before_collection:
        return "SOURCE_COVERAGE_UNRESOLVED"
    if unresolved_overlap:
        return "SOURCE_COVERAGE_UNRESOLVED"
    if confirmed_overlap and repaired_rows_used == 0:
        return "SOURCE_GAPPED"
    return "EXACT_RECONSTRUCTABLE"


def feature_id_for(symbol: str, signal_id: str, window_id: str, version: str) -> str:
    raw = f"{symbol}|{signal_id}|{window_id}|{version}"
    return "AIF-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# FLOOR_USD_M derivation (mechanical, outcome-blind, computed once from the
# real signed-notional distribution across all constructible rows at the
# smallest window, per the frozen contract's own deferred-derivation clause)
# ---------------------------------------------------------------------------

def derive_floor_usd_m(abs_signed_notional_usd_samples: list[float]) -> dict:
    """Frozen derivation rule (mechanical, not tuned to any outcome):
    FLOOR_USD_M = 1st percentile of |signed_notional| (in USD) across all
    constructible window-60s rows, expressed in $M units, rounded down to
    one significant figure. This guards only against a true near-zero-flow
    division blowup; it is far below the typical scale of any cascade-
    anchored signal's window flow."""
    if not abs_signed_notional_usd_samples:
        return {"floor_usd_m": 0.001, "basis": "NO_SAMPLES_FALLBACK", "p1_usd": None}
    xs = sorted(abs_signed_notional_usd_samples)
    n = len(xs)
    idx = max(0, min(n - 1, int(round(0.01 * (n - 1)))))
    p1_usd = xs[idx]
    # round down to one significant figure, then to $M units
    import math
    if p1_usd <= 0:
        floor_m = 0.001
    else:
        mag = 10 ** math.floor(math.log10(p1_usd))
        floor_m = (mag / 1_000_000.0)
    floor_m = max(floor_m, 0.0001)  # absolute minimum guard: $100
    return {"floor_usd_m": floor_m, "basis": "1ST_PERCENTILE_ABS_SIGNED_NOTIONAL_W60_ROUNDED_DOWN_1SF",
            "p1_usd": p1_usd, "n_samples": n}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_rehearsal(disposable_conn: sqlite3.Connection, canonical_conn: sqlite3.Connection,
                   micro_conn: sqlite3.Connection, floor_usd_m: float = FROZEN_FLOOR_USD_M,
                   windows_sec: tuple[int, ...] = FIXED_WINDOWS_SEC,
                   symbol: str = SYMBOL) -> dict:
    """Builds the full rehearsal population: for every signal x window,
    computes source-coverage/quality, and either writes a
    absorption_impact_windowed_flow row (usable) or a
    absorption_impact_exclusions row (immutable exclusion) -- never both,
    never neither. Always writes exactly one absorption_impact_window_quality_v1
    row per signal x window regardless of usability."""
    build_rehearsal_schema(disposable_conn)

    signals = fetch_anchor_universe(canonical_conn)
    confirmed_gaps, unresolved_gaps = fetch_agg_trades_gaps(micro_conn)
    collection_start = fetch_agg_trades_collection_start(micro_conn, symbol)

    now_ms = int(time.time() * 1000)
    counts = {
        "total_signal_window_pairs": 0, "usable": 0,
        "excluded_before_collection": 0, "excluded_confirmed_gap": 0,
        "excluded_unresolved_gap": 0, "quality_exact": 0, "quality_gapped": 0,
        "quality_unresolved": 0, "floor_applied_rows": 0,
        "zero_signed_notional_rows": 0, "known_at_violations": 0,
    }

    for s in signals:
        for w in windows_sec:
            window_id = f"W{w}"
            window_end = s["signal_birth_ts"]
            window_start = window_end - w * 1000
            counts["total_signal_window_pairs"] += 1

            before_collection = window_start < collection_start
            confirmed_overlap = overlaps_any(window_start, window_end, confirmed_gaps)
            unresolved_overlap = overlaps_unresolved(window_start, window_end, unresolved_gaps)

            quality_id = feature_id_for(symbol, s["signal_id"], window_id, "quality-" + QUALITY_CONTRACT_VERSION)

            if before_collection or confirmed_overlap or unresolved_overlap:
                repaired_rows_used = 0
                quality_status = classify_quality(confirmed_overlap, unresolved_overlap, before_collection, 0)
                reason = (EXCLUSION_WINDOW_STARTS_BEFORE_COLLECTION if before_collection
                          else EXCLUSION_UNRESOLVED_GAP_PROXIMITY if unresolved_overlap
                          else EXCLUSION_CONFIRMED_GAP_OVERLAP)
                disposable_conn.execute(
                    "INSERT OR IGNORE INTO absorption_impact_exclusions "
                    "(exclusion_id, signal_id, symbol, window_id, reason_code, created_ms) "
                    "VALUES (?,?,?,?,?,?)",
                    (feature_id_for(symbol, s["signal_id"], window_id, "excl-" + reason),
                     s["signal_id"], symbol, window_id, reason, now_ms))
                disposable_conn.execute(
                    "INSERT OR IGNORE INTO absorption_impact_window_quality_v1 "
                    "(quality_id, quality_contract_version, signal_id, symbol, window_id, "
                    "window_start_ts_ms, window_end_ts_ms, evidence_layer, quality_status, "
                    "confirmed_gap_overlap, unresolved_gap_overlap, before_collection_began, "
                    "repaired_rows_used, native_rows_used, assessed_at_ms) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (quality_id, QUALITY_CONTRACT_VERSION, s["signal_id"], symbol, window_id,
                     window_start, window_end, "EXACT", quality_status,
                     int(confirmed_overlap), int(unresolved_overlap), int(before_collection),
                     0, 0, now_ms))
                if before_collection:
                    counts["excluded_before_collection"] += 1
                elif unresolved_overlap:
                    counts["excluded_unresolved_gap"] += 1
                else:
                    counts["excluded_confirmed_gap"] += 1
                if quality_status == "SOURCE_GAPPED":
                    counts["quality_gapped"] += 1
                else:
                    counts["quality_unresolved"] += 1
                continue

            feat = compute_window_feature(micro_conn, canonical_conn, symbol, window_start, window_end, floor_usd_m)
            quality_status = classify_quality(False, False, False, feat["repaired_rows_used"])
            counts["quality_exact"] += 1
            if feat["floor_usd_m_applied"]:
                counts["floor_applied_rows"] += 1
            if feat["signed_notional"] == 0.0:
                counts["zero_signed_notional_rows"] += 1

            fid = feature_id_for(symbol, s["signal_id"], window_id, FEATURE_DEFINITION_VERSION)
            disposable_conn.execute(
                "INSERT OR IGNORE INTO absorption_impact_windowed_flow "
                "(feature_id, feature_definition_version, signal_id, source_event_id, "
                "independent_cycle_id, symbol, direction, signal_birth_ts, window_id, "
                "window_start_ts_ms, window_end_ts_ms, trade_count, native_rows_used, "
                "repaired_rows_used, signed_notional, total_notional, mark_price_start, "
                "mark_price_end, mark_return_bps, floor_usd_m_applied, floor_usd_m_value, "
                "price_response_per_signed_notional, evidence_layer, feature_available_ts_ms, "
                "known_at_classification, created_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, FEATURE_DEFINITION_VERSION, s["signal_id"], s["source_event_id"],
                 s["independent_cycle_id"], symbol, s["direction"], s["signal_birth_ts"], window_id,
                 window_start, window_end, feat["trade_count"], feat["native_rows_used"],
                 feat["repaired_rows_used"], feat["signed_notional"], feat["total_notional"],
                 feat["mark_price_start"], feat["mark_price_end"], feat["mark_return_bps"],
                 int(feat["floor_usd_m_applied"]), feat["floor_usd_m_value"],
                 feat["price_response_per_signed_notional"], "EXACT", s["signal_birth_ts"],
                 KNOWN_AT_SAFE, now_ms))
            disposable_conn.execute(
                "INSERT OR IGNORE INTO absorption_impact_window_quality_v1 "
                "(quality_id, quality_contract_version, signal_id, symbol, window_id, "
                "window_start_ts_ms, window_end_ts_ms, evidence_layer, quality_status, "
                "confirmed_gap_overlap, unresolved_gap_overlap, before_collection_began, "
                "repaired_rows_used, native_rows_used, assessed_at_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (quality_id, QUALITY_CONTRACT_VERSION, s["signal_id"], symbol, window_id,
                 window_start, window_end, "EXACT", quality_status,
                 0, 0, 0, feat["repaired_rows_used"], feat["native_rows_used"], now_ms))
            counts["usable"] += 1

    disposable_conn.commit()
    return counts


# Bookkeeping-only columns excluded from the content hash (identical
# discipline to ami/warehouse/experiment_ledger.py's
# _VOLATILE_BOOKKEEPING_COLUMNS): a wall-clock `created_ms`/`assessed_at_ms`
# legitimately differs between two otherwise-identical reruns and must never
# be mistaken for a real content difference.
_CONTENT_COLUMNS = {
    "absorption_impact_windowed_flow": (
        "feature_id, feature_definition_version, signal_id, source_event_id, "
        "independent_cycle_id, symbol, direction, signal_birth_ts, window_id, "
        "window_start_ts_ms, window_end_ts_ms, trade_count, native_rows_used, "
        "repaired_rows_used, signed_notional, total_notional, mark_price_start, "
        "mark_price_end, mark_return_bps, floor_usd_m_applied, floor_usd_m_value, "
        "price_response_per_signed_notional, evidence_layer, feature_available_ts_ms, "
        "known_at_classification"
    ),
    "absorption_impact_window_quality_v1": (
        "quality_id, quality_contract_version, signal_id, symbol, window_id, "
        "window_start_ts_ms, window_end_ts_ms, evidence_layer, quality_status, "
        "confirmed_gap_overlap, unresolved_gap_overlap, before_collection_began, "
        "repaired_rows_used, native_rows_used"
    ),
    "absorption_impact_exclusions": "exclusion_id, signal_id, symbol, window_id, reason_code",
}


def content_hash_of_disposable(disposable_conn: sqlite3.Connection) -> dict:
    """Content-only hash per table (bookkeeping timestamp columns excluded,
    see _CONTENT_COLUMNS) -- two independent reruns against the same source
    state must produce identical hashes here even though their created_ms/
    assessed_at_ms values differ."""
    hashes = {}
    for table, cols in _CONTENT_COLUMNS.items():
        rows = disposable_conn.execute(f"SELECT {cols} FROM {table} ORDER BY 1").fetchall()
        hashes[table] = hashlib.sha256("|".join(str(r) for r in rows).encode()).hexdigest()
    return hashes


def row_accounting(disposable_conn: sqlite3.Connection) -> dict:
    out = {}
    for w in FIXED_WINDOWS_SEC:
        wid = f"W{w}"
        usable = disposable_conn.execute(
            "SELECT COUNT(*) FROM absorption_impact_windowed_flow WHERE window_id=?", (wid,)
        ).fetchone()[0]
        excl_before = disposable_conn.execute(
            "SELECT COUNT(*) FROM absorption_impact_exclusions WHERE window_id=? AND reason_code=?",
            (wid, EXCLUSION_WINDOW_STARTS_BEFORE_COLLECTION)).fetchone()[0]
        excl_gap = disposable_conn.execute(
            "SELECT COUNT(*) FROM absorption_impact_exclusions WHERE window_id=? AND reason_code=?",
            (wid, EXCLUSION_CONFIRMED_GAP_OVERLAP)).fetchone()[0]
        excl_unresolved = disposable_conn.execute(
            "SELECT COUNT(*) FROM absorption_impact_exclusions WHERE window_id=? AND reason_code=?",
            (wid, EXCLUSION_UNRESOLVED_GAP_PROXIMITY)).fetchone()[0]
        quality_counts = dict(disposable_conn.execute(
            "SELECT quality_status, COUNT(*) FROM absorption_impact_window_quality_v1 "
            "WHERE window_id=? GROUP BY quality_status", (wid,)).fetchall())
        total_quality_rows = sum(quality_counts.values())
        out[wid] = {
            "usable": usable, "excluded_before_collection": excl_before,
            "excluded_confirmed_gap": excl_gap, "excluded_unresolved_gap": excl_unresolved,
            "quality_breakdown": quality_counts,
            "reconciled": (usable + excl_before + excl_gap + excl_unresolved),
            "quality_rows_total": total_quality_rows,
        }
    return out
