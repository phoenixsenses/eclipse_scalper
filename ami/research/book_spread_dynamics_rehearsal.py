"""BATCH-BOOK-SPREAD-DYNAMICS-DISPOSABLE-REHEARSAL-V1.

Builds, entirely against a DISPOSABLE output database (never the real
canonical.sqlite), the operator-approved primary child feature of
`FAM_BOOK_SPREAD_DYNAMICS`:

    BOOK_SPREAD_CHANGE_BPS_W300_V1

    mid_price(t)   = (best_ask(t) + best_bid(t)) / 2
    spread_bps(t)  = 10_000 * (best_ask(t) - best_bid(t)) / mid_price(t)
    spread_change_bps_w300 = spread_bps(t0) - spread_bps(t0 - 300s)

Sign: positive = expansion (widening) over W300; negative = compression;
zero = no change between the two governed reference points. Units: bps of
spread change. The additive difference (not a ratio/log/z-score) is the
operator ruling (FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1).

Reuses -- verbatim, not reinvented -- the accepted readiness quote-selection
and quality contract from `ami/research/spread_dynamics_readiness_audit.py`
(commit f115b9c1): the deterministic at-or-before selection with `id DESC`
tie-break, the 5-minute staleness tolerance (FEED_LIMITS["book_ticker"]),
the crossed/locked/zero/negative rules, and the immutable quality codes.

No line of this module ever opens `ami_lifecycle_path_observations` (the
outcome table), selects `endpoint_return_bps`/`mfe_bps`/`mae_bps`, or reads/
writes any experiment/nullifier/gate-receipt table -- enforced structurally
(none is ever named in any SQL string here) and provably via the SQLite
authorizer in `install_access_guard()`.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time

from ami.research.spread_dynamics_readiness_audit import (
    SYMBOL,
    BOOK_TICKER_HEALTHY_AGE_MS,
    QUALITY_EXACT, QUALITY_STALE, QUALITY_UNAVAILABLE,
    QUALITY_CROSSED, QUALITY_ZERO_NEG, QUALITY_LOCKED,
    select_quote_at_or_before, classify_quote,
    fetch_anchor_universe,
)

FORMULA_VERSION = "BOOK_SPREAD_CHANGE_BPS_W300_V1"
FAMILY_ID = "FAM_BOOK_SPREAD_DYNAMICS"
CHILD_WORKING_ID = "H-BOOK-SPREAD-CHANGE-BPS-W300-V1"
WINDOW_SEC = 300
WINDOW_MS = WINDOW_SEC * 1000
VENUE = "BINANCE_USDM_PERP"
MARKET_SEGMENT = "PERPETUAL_FUTURES"
QUOTE_CURRENCY = "USDT"
READINESS_COMMIT = "f115b9c1"

# Deterministic exclusion precedence, FROZEN before any count is inspected
# and independent of feature values/outcomes: when a row is not EXACT, its
# single row-level exclusion reason is the highest-precedence non-EXACT
# status among its two endpoints (index 0 = highest precedence).
_EXCLUSION_PRECEDENCE = (
    QUALITY_UNAVAILABLE,   # no quote exists at all -> most fundamental
    QUALITY_ZERO_NEG,      # corrupt price
    QUALITY_CROSSED,       # bid > ask
    QUALITY_LOCKED,        # bid == ask (zero-spread anomaly)
    QUALITY_STALE,         # a valid quote exists but is too old
)


def _precedence_rank(status: str) -> int:
    return _EXCLUSION_PRECEDENCE.index(status)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS book_spread_change_w300 (
    feature_id TEXT PRIMARY KEY,
    formula_version TEXT NOT NULL,
    anchor_id TEXT NOT NULL,
    canonical_signal_ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    venue TEXT NOT NULL,
    market_segment TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    direction TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    is_cycle_representative INTEGER NOT NULL,
    current_target_ts INTEGER NOT NULL,
    current_quote_id INTEGER,
    current_quote_ts INTEGER,
    current_quote_age_ms INTEGER,
    current_bid REAL,
    current_ask REAL,
    current_mid REAL,
    current_spread_bps REAL,
    current_quality_status TEXT NOT NULL,
    historical_target_ts INTEGER NOT NULL,
    historical_quote_id INTEGER,
    historical_quote_ts INTEGER,
    historical_quote_age_ms INTEGER,
    historical_bid REAL,
    historical_ask REAL,
    historical_mid REAL,
    historical_spread_bps REAL,
    historical_quality_status TEXT NOT NULL,
    spread_change_bps_w300 REAL,
    source_quality_class TEXT NOT NULL,
    exclusion_reason TEXT,
    exclusion_endpoint TEXT,
    known_at_ts INTEGER NOT NULL,
    feature_available_ts INTEGER NOT NULL,
    input_manifest_id TEXT NOT NULL,
    specification_hash TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (anchor_id, formula_version),
    CHECK (source_quality_class IN ('EXACT_RECONSTRUCTABLE','STALE_SOURCE',
        'UNAVAILABLE_BEFORE_COLLECTION','INVALID_QUOTE_CROSSED',
        'INVALID_QUOTE_ZERO_OR_NEG','INVALID_QUOTE_LOCKED')),
    CHECK (feature_available_ts = canonical_signal_ts),
    CHECK (known_at_ts = canonical_signal_ts),
    CHECK (symbol = 'ETHUSDT'),
    CHECK (venue = 'BINANCE_USDM_PERP'),
    CHECK (market_segment = 'PERPETUAL_FUTURES')
);

CREATE TABLE IF NOT EXISTS rehearsal_manifest (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class KnownAtViolation(Exception):
    """A selected quote postdates its target timestamp -- fail closed."""


# ---------------------------------------------------------------------------
# Access guard (Phase 6) -- SQLite authorizer denying outcome + governance
# ---------------------------------------------------------------------------

_FORBIDDEN_TABLES = frozenset({
    "ami_lifecycle_path_observations",  # the outcome table
    "experiment_registry", "experiment_results",
    "epistemic_test_nullifiers", "experiment_gate_receipts",
})
_FORBIDDEN_COLUMNS = frozenset({"endpoint_return_bps", "mfe_bps", "mae_bps"})
_WRITE_ACTIONS = frozenset({sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE})


def install_access_guard(conn: sqlite3.Connection) -> list[str]:
    """Installs a SQLite authorizer that DENIES (SQLITE_DENY) the instant any
    statement references an outcome/experiment/nullifier/gate-receipt table or
    an outcome column, OR attempts any write to those tables. Returns the
    (mutated-in-place, empty if clean) violations list."""
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


def build_rehearsal_schema(disposable_conn: sqlite3.Connection) -> None:
    disposable_conn.executescript(_SCHEMA)
    disposable_conn.commit()


def feature_id_for(anchor_id: str) -> str:
    raw = f"{SYMBOL}|{anchor_id}|{FORMULA_VERSION}"
    return "BSF-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def specification_hash() -> str:
    """Deterministic hash of the frozen spec constants -- pins the exact
    definition this rehearsal implements."""
    spec = "|".join([
        FAMILY_ID, CHILD_WORKING_ID, FORMULA_VERSION, str(WINDOW_MS), SYMBOL, VENUE,
        MARKET_SEGMENT, QUOTE_CURRENCY, str(BOOK_TICKER_HEALTHY_AGE_MS),
        "select=at_or_before;tiebreak=id_desc;diff=current_minus_historical",
        ",".join(_EXCLUSION_PRECEDENCE),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()


def _endpoint(micro_conn, target_ts: int) -> dict:
    q = select_quote_at_or_before(micro_conn, target_ts)
    if q is not None and q["quote_ts_ms"] > target_ts:
        raise KnownAtViolation(f"quote {q['quote_ts_ms']} > target {target_ts}")
    cls = classify_quote(q, target_ts, BOOK_TICKER_HEALTHY_AGE_MS)
    return {"quote": q, "cls": cls}


def _row_quality(cur_status: str, hist_status: str) -> tuple[str, str | None, str | None]:
    """Returns (source_quality_class, exclusion_reason, exclusion_endpoint)."""
    if cur_status == QUALITY_EXACT and hist_status == QUALITY_EXACT:
        return QUALITY_EXACT, None, None
    non_exact = []
    if cur_status != QUALITY_EXACT:
        non_exact.append(("current", cur_status))
    if hist_status != QUALITY_EXACT:
        non_exact.append(("historical", hist_status))
    # highest precedence (lowest rank) wins; tie -> current before historical
    non_exact.sort(key=lambda es: (_precedence_rank(es[1]), 0 if es[0] == "current" else 1))
    endpoint, status = non_exact[0]
    both = len(non_exact) == 2 and non_exact[0][1] == non_exact[1][1]
    return status, status, ("both" if both else endpoint)


def run_rehearsal(disposable_conn: sqlite3.Connection, canonical_conn: sqlite3.Connection,
                   micro_conn: sqlite3.Connection, input_manifest_id: str) -> dict:
    """Builds one accounting/feature row per canonical anchor into the
    disposable DB. Exactly one row per anchor (UNIQUE anchor_id); the row
    carries a feature value only when EXACT, else the frozen exclusion reason.
    Never joins to any outcome."""
    build_rehearsal_schema(disposable_conn)
    signals = fetch_anchor_universe(canonical_conn)
    spec_hash = specification_hash()
    now_ms = int(time.time() * 1000)

    counts = {"total_anchors": 0, "exact": 0, "stale": 0, "unavailable": 0,
              "invalid_crossed": 0, "invalid_zero_neg": 0, "invalid_locked": 0,
              "known_at_violations": 0}

    for s in signals:
        counts["total_anchors"] += 1
        t0 = s["signal_birth_ts"]
        th = t0 - WINDOW_MS

        cur = _endpoint(micro_conn, t0)
        hist = _endpoint(micro_conn, th)
        cur_cls, hist_cls = cur["cls"], hist["cls"]

        source_quality_class, exclusion_reason, exclusion_endpoint = _row_quality(
            cur_cls["quality_status"], hist_cls["quality_status"])

        spread_change = None
        if source_quality_class == QUALITY_EXACT:
            spread_change = cur_cls["spread_bps"] - hist_cls["spread_bps"]

        cq, hq = cur["quote"], hist["quote"]
        cycle_id = s["independent_cycle_id"] or f"NOCYCLE-{s['source_event_id']}"
        disposable_conn.execute(
            "INSERT OR IGNORE INTO book_spread_change_w300 ("
            "feature_id, formula_version, anchor_id, canonical_signal_ts, symbol, venue, "
            "market_segment, quote_currency, direction, cycle_id, is_cycle_representative, "
            "current_target_ts, current_quote_id, current_quote_ts, current_quote_age_ms, "
            "current_bid, current_ask, current_mid, current_spread_bps, current_quality_status, "
            "historical_target_ts, historical_quote_id, historical_quote_ts, historical_quote_age_ms, "
            "historical_bid, historical_ask, historical_mid, historical_spread_bps, historical_quality_status, "
            "spread_change_bps_w300, source_quality_class, exclusion_reason, exclusion_endpoint, "
            "known_at_ts, feature_available_ts, input_manifest_id, specification_hash, created_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (feature_id_for(s["signal_id"]), FORMULA_VERSION, s["signal_id"], t0, SYMBOL, VENUE,
             MARKET_SEGMENT, QUOTE_CURRENCY, s["direction"], cycle_id, 0,
             t0, (cq or {}).get("quote_id"), (cq or {}).get("quote_ts_ms"), cur_cls["staleness_ms"],
             (cq or {}).get("bid_price"), (cq or {}).get("ask_price"), cur_cls["mid_price"], cur_cls["spread_bps"],
             cur_cls["quality_status"],
             th, (hq or {}).get("quote_id"), (hq or {}).get("quote_ts_ms"), hist_cls["staleness_ms"],
             (hq or {}).get("bid_price"), (hq or {}).get("ask_price"), hist_cls["mid_price"], hist_cls["spread_bps"],
             hist_cls["quality_status"],
             spread_change, source_quality_class, exclusion_reason, exclusion_endpoint,
             t0, t0, input_manifest_id, spec_hash, now_ms))

        if source_quality_class == QUALITY_EXACT:
            counts["exact"] += 1
        elif source_quality_class == QUALITY_STALE:
            counts["stale"] += 1
        elif source_quality_class == QUALITY_UNAVAILABLE:
            counts["unavailable"] += 1
        elif source_quality_class == QUALITY_CROSSED:
            counts["invalid_crossed"] += 1
        elif source_quality_class == QUALITY_ZERO_NEG:
            counts["invalid_zero_neg"] += 1
        elif source_quality_class == QUALITY_LOCKED:
            counts["invalid_locked"] += 1

    # cycle-representative rule: earliest signal_birth_ts per cycle among EXACT rows
    _assign_cycle_representatives(disposable_conn)
    disposable_conn.commit()
    return counts


def _assign_cycle_representatives(disposable_conn: sqlite3.Connection) -> None:
    """One representative per independent cycle among EXACT_RECONSTRUCTABLE
    rows: the earliest canonical_signal_ts (deterministic; anchor_id
    tie-break). Sets is_cycle_representative=1 on exactly those rows."""
    rows = disposable_conn.execute(
        "SELECT anchor_id, cycle_id, canonical_signal_ts FROM book_spread_change_w300 "
        "WHERE source_quality_class='EXACT_RECONSTRUCTABLE' "
        "ORDER BY cycle_id, canonical_signal_ts, anchor_id").fetchall()
    rep_by_cycle: dict[str, str] = {}
    for anchor_id, cycle_id, ts in rows:
        if cycle_id not in rep_by_cycle:
            rep_by_cycle[cycle_id] = anchor_id
    for anchor_id in rep_by_cycle.values():
        disposable_conn.execute(
            "UPDATE book_spread_change_w300 SET is_cycle_representative=1 WHERE anchor_id=?",
            (anchor_id,))


# Bookkeeping-only column excluded from the content hash (same discipline as
# every prior rehearsal: a wall-clock created_ms legitimately differs between
# two otherwise-identical reruns).
_CONTENT_COLUMNS = (
    "feature_id, formula_version, anchor_id, canonical_signal_ts, symbol, venue, market_segment, "
    "quote_currency, direction, cycle_id, is_cycle_representative, current_target_ts, current_quote_id, "
    "current_quote_ts, current_quote_age_ms, current_bid, current_ask, current_mid, current_spread_bps, "
    "current_quality_status, historical_target_ts, historical_quote_id, historical_quote_ts, "
    "historical_quote_age_ms, historical_bid, historical_ask, historical_mid, historical_spread_bps, "
    "historical_quality_status, spread_change_bps_w300, source_quality_class, exclusion_reason, "
    "exclusion_endpoint, known_at_ts, feature_available_ts, input_manifest_id, specification_hash"
)


def content_rows(disposable_conn: sqlite3.Connection) -> list[tuple]:
    return disposable_conn.execute(
        f"SELECT {_CONTENT_COLUMNS} FROM book_spread_change_w300 ORDER BY anchor_id").fetchall()


def content_hash(disposable_conn: sqlite3.Connection) -> str:
    rows = content_rows(disposable_conn)
    return hashlib.sha256("|".join(repr(r) for r in rows).encode()).hexdigest()


def row_manifest_hash(disposable_conn: sqlite3.Connection) -> str:
    """Hash of the ordered (anchor_id, feature_id, source_quality_class,
    current_quote_id, historical_quote_id) manifest -- the identity/lineage
    fingerprint, separate from the full-content hash."""
    rows = disposable_conn.execute(
        "SELECT anchor_id, feature_id, source_quality_class, current_quote_id, historical_quote_id "
        "FROM book_spread_change_w300 ORDER BY anchor_id").fetchall()
    return hashlib.sha256("|".join(repr(r) for r in rows).encode()).hexdigest()


def accounting(disposable_conn: sqlite3.Connection) -> dict:
    total = disposable_conn.execute("SELECT COUNT(*) FROM book_spread_change_w300").fetchone()[0]
    by_class = dict(disposable_conn.execute(
        "SELECT source_quality_class, COUNT(*) FROM book_spread_change_w300 "
        "GROUP BY source_quality_class").fetchall())
    exact_rows = by_class.get(QUALITY_EXACT, 0)
    reps = disposable_conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE is_cycle_representative=1").fetchone()[0]
    exact_cycles = disposable_conn.execute(
        "SELECT COUNT(DISTINCT cycle_id) FROM book_spread_change_w300 "
        "WHERE source_quality_class='EXACT_RECONSTRUCTABLE'").fetchone()[0]
    distinct_anchor = disposable_conn.execute(
        "SELECT COUNT(DISTINCT anchor_id) FROM book_spread_change_w300").fetchone()[0]
    known_at_bad = disposable_conn.execute(
        "SELECT COUNT(*) FROM book_spread_change_w300 WHERE feature_available_ts != canonical_signal_ts "
        "OR known_at_ts != canonical_signal_ts").fetchone()[0]
    return {
        "total_rows": total, "distinct_anchor_ids": distinct_anchor,
        "by_source_quality_class": by_class, "exact_rows": exact_rows,
        "exact_independent_cycles": exact_cycles, "cycle_representatives": reps,
        "duplicate_cycle_representatives": reps - exact_cycles,
        "known_at_field_violations": known_at_bad,
        "reconciles_to_324": total == 324 and distinct_anchor == 324,
    }
