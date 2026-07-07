"""BATCH-SPREAD-EXPANSION-COMPRESSION-READINESS-AND-CONTRACT-V1 -- read-only,
outcome-blind source-coverage, quote-selection and known-at audit for
`FAM_BOOK_SPREAD_DYNAMICS` (canonical family name per the accepted selection
artifact, reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_
SELECTION_V1.md, candidate 3).

This module performs NO schema creation, NO data write, NO outcome access.
Every function takes explicit read-only connections and returns plain
dicts/lists -- pure audit/measurement, matching the readiness-only nature
of this batch.

Conventions reused verbatim, not reinvented:
  - `ami.states.engine.FEED_LIMITS["book_ticker"] = 5.0` minutes as the
    established "max healthy age" for this exact feed.
  - the same at-or-before quote-selection convention used throughout this
    codebase (StateEngine._px, absorption/impact fetch_mark_price_at_or_
    before), EXTENDED here with a mandatory deterministic tie-break: because
    ~75% of book_ticker rows share a ts_ms with another row (multiple
    sub-millisecond updates), `ORDER BY ts_ms DESC LIMIT 1` alone is
    NON-DETERMINISTIC (~6.5% of ms-collisions carry a different bid/ask).
    The frozen rule adds `id DESC` (the autoincrement PK = insertion order =
    latest update at that ms) so selection is reproducible.

No line of this module ever opens `ami_lifecycle_path_observations` or
selects any outcome column (`endpoint_return_bps`/`mfe_bps`/`mae_bps`).
"""
from __future__ import annotations

import sqlite3

SYMBOL = "ETHUSDT"

# Reused verbatim from ami/states/engine.py FEED_LIMITS["book_ticker"] = 5.0
BOOK_TICKER_HEALTHY_AGE_MIN = 5.0
BOOK_TICKER_HEALTHY_AGE_MS = int(BOOK_TICKER_HEALTHY_AGE_MIN * 60 * 1000)

# Immutable quality/exclusion codes (mirrors the absorption/basis vocabulary
# style; spread-specific invalid-quote codes added).
QUALITY_EXACT = "EXACT_RECONSTRUCTABLE"
QUALITY_STALE = "STALE_SOURCE"
QUALITY_UNAVAILABLE = "UNAVAILABLE_BEFORE_COLLECTION"
QUALITY_CROSSED = "INVALID_QUOTE_CROSSED"
QUALITY_ZERO_NEG = "INVALID_QUOTE_ZERO_OR_NEG"
QUALITY_LOCKED = "INVALID_QUOTE_LOCKED"  # bid == ask (zero spread); flagged, not silently kept


def fetch_anchor_universe(canonical_conn: sqlite3.Connection) -> list[dict]:
    """Identity columns only -- no outcome column is ever selected here or
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


def book_ticker_coverage(micro_conn: sqlite3.Connection, symbol: str = SYMBOL) -> dict:
    """Index-backed MIN/MAX only -- never a full-table COUNT/scan of the
    ~2-billion-row book_ticker table."""
    mn = micro_conn.execute(
        "SELECT MIN(ts_ms) FROM book_ticker WHERE symbol=?", (symbol,)).fetchone()[0]
    mx = micro_conn.execute(
        "SELECT MAX(ts_ms) FROM book_ticker WHERE symbol=?", (symbol,)).fetchone()[0]
    return {"symbol": symbol, "first_ts_ms": mn, "last_ts_ms": mx}


def select_quote_at_or_before(micro_conn: sqlite3.Connection, ts_ms: int,
                               symbol: str = SYMBOL) -> dict | None:
    """Deterministic no-lookahead quote selection: the latest valid quote at
    or before `ts_ms`, tie-broken by `id DESC` (insertion order) when
    multiple rows share the same ts_ms. Structurally incapable of returning
    a future quote (the WHERE clause bounds ts_ms <= target). Returns the raw
    quote dict or None if no quote exists at or before ts_ms."""
    row = micro_conn.execute(
        "SELECT ts_ms, id, bid_price, ask_price, bid_qty, ask_qty FROM book_ticker "
        "WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC, id DESC LIMIT 1",
        (symbol, ts_ms)).fetchone()
    if row is None:
        return None
    qts, qid, bid, ask, bq, aq = row
    if qts > ts_ms:
        raise AssertionError("KNOWN_AT_VIOLATION: selected a future book_ticker quote")
    return {"quote_ts_ms": qts, "quote_id": qid, "bid_price": bid, "ask_price": ask,
            "bid_qty": bq, "ask_qty": aq}


def spread_bps_and_mid(bid_price: float, ask_price: float) -> dict:
    """Frozen base formula: mid = (ask+bid)/2, spread_bps = 1e4*(ask-bid)/mid.
    IEEE-754 double, no rounding. Raises on invalid inputs -- callers must
    classify quality BEFORE computing (crossed/zero/negative never reach a
    division)."""
    if bid_price <= 0 or ask_price <= 0:
        raise ValueError("INVALID_QUOTE_ZERO_OR_NEG")
    if bid_price > ask_price:
        raise ValueError("INVALID_QUOTE_CROSSED")
    mid = (ask_price + bid_price) / 2.0
    spread_bps = 1e4 * (ask_price - bid_price) / mid
    return {"mid_price": mid, "spread_bps": spread_bps}


def classify_quote(quote: dict | None, target_ts_ms: int,
                   healthy_age_ms: int = BOOK_TICKER_HEALTHY_AGE_MS) -> dict:
    """Deterministic quality classification for a single at-or-before quote
    selection, in strict precedence order: unavailable -> zero/neg -> crossed
    -> locked -> stale -> exact. Every branch yields an immutable code; the
    caller never silently repairs, clips, or drops an anomaly."""
    if quote is None:
        return {"quality_status": QUALITY_UNAVAILABLE, "staleness_ms": None,
                "spread_bps": None, "mid_price": None}
    staleness = target_ts_ms - quote["quote_ts_ms"]
    if staleness < 0:
        raise AssertionError("KNOWN_AT_VIOLATION: negative staleness")
    bid, ask = quote["bid_price"], quote["ask_price"]
    if bid <= 0 or ask <= 0:
        return {"quality_status": QUALITY_ZERO_NEG, "staleness_ms": staleness,
                "spread_bps": None, "mid_price": None}
    if bid > ask:
        return {"quality_status": QUALITY_CROSSED, "staleness_ms": staleness,
                "spread_bps": None, "mid_price": None}
    if bid == ask:
        # locked/zero-spread book: flagged with its own code, never silently
        # treated as a normal 0-bps spread (a genuine zero-spread L1 on a
        # liquid perp is anomalous and worth an explicit, separate partition)
        return {"quality_status": QUALITY_LOCKED, "staleness_ms": staleness,
                "spread_bps": 0.0, "mid_price": (ask + bid) / 2.0}
    computed = spread_bps_and_mid(bid, ask)
    status = QUALITY_EXACT if staleness <= healthy_age_ms else QUALITY_STALE
    return {"quality_status": status, "staleness_ms": staleness, **computed}


def anchor_accounting(canonical_conn: sqlite3.Connection, micro_conn: sqlite3.Connection,
                       healthy_age_ms: int = BOOK_TICKER_HEALTHY_AGE_MS) -> dict:
    """Full, outcome-blind accounting for the spread LEVEL at signal birth
    (the minimal, zero-window candidate). For any windowed change feature the
    caller uses `windowed_pair_accounting` instead. Never reads any outcome
    column; exclusions are 100% source-quality/coverage based."""
    signals = fetch_anchor_universe(canonical_conn)
    rows = []
    for s in signals:
        q = select_quote_at_or_before(micro_conn, s["signal_birth_ts"])
        cls = classify_quote(q, s["signal_birth_ts"], healthy_age_ms)
        rows.append({**s, **cls,
                     "cycle_key": s["independent_cycle_id"] or f"NOCYCLE-{s['source_event_id']}"})

    total = len(rows)
    by_status: dict[str, int] = {}
    by_status_dir: dict[str, dict[str, int]] = {}
    for r in rows:
        by_status[r["quality_status"]] = by_status.get(r["quality_status"], 0) + 1
        by_status_dir.setdefault(r["direction"], {})
        by_status_dir[r["direction"]][r["quality_status"]] = \
            by_status_dir[r["direction"]].get(r["quality_status"], 0) + 1

    fresh = [r for r in rows if r["quality_status"] == QUALITY_EXACT]
    fresh_cycles = {r["cycle_key"] for r in fresh}
    return {
        "total_anchors": total,
        "healthy_age_ms_used": healthy_age_ms,
        "quality_breakdown": by_status,
        "quality_breakdown_by_direction": by_status_dir,
        "exact_rows": len(fresh),
        "exact_independent_cycles": len(fresh_cycles),
        "reconciliation_ok": sum(by_status.values()) == total,
        "rows": rows,
    }


def windowed_pair_accounting(canonical_conn: sqlite3.Connection, micro_conn: sqlite3.Connection,
                              window_sec: int,
                              healthy_age_ms: int = BOOK_TICKER_HEALTHY_AGE_MS) -> dict:
    """Outcome-blind coverage for a windowed change feature: an anchor is
    usable only if BOTH the birth quote (T) and the baseline quote (T-window)
    are EXACT_RECONSTRUCTABLE. Returns per-window usable rows and independent
    cycles -- the accounting a future spread-expansion/compression rehearsal
    would freeze once the operator rules on the feature form and window."""
    signals = fetch_anchor_universe(canonical_conn)
    usable = 0
    cycles = set()
    for s in signals:
        bts = s["signal_birth_ts"]
        qT = select_quote_at_or_before(micro_conn, bts)
        qB = select_quote_at_or_before(micro_conn, bts - window_sec * 1000)
        clsT = classify_quote(qT, bts, healthy_age_ms)
        clsB = classify_quote(qB, bts - window_sec * 1000, healthy_age_ms)
        if clsT["quality_status"] == QUALITY_EXACT and clsB["quality_status"] == QUALITY_EXACT:
            usable += 1
            cycles.add(s["independent_cycle_id"] or f"NOCYCLE-{s['source_event_id']}")
    return {"window_sec": window_sec, "both_endpoints_exact": usable,
            "independent_cycles": len(cycles)}


def verify_no_lookahead(rows: list[dict]) -> dict:
    """Defense-in-depth re-check over the materialized accounting rows:
    `select_quote_at_or_before` already raises on a future quote, so a clean
    pass here is redundant confirmation, not the sole guarantee."""
    violations = [r["signal_id"] for r in rows
                  if r.get("staleness_ms") is not None and r["staleness_ms"] < 0]
    return {"known_at_violations": len(violations), "violations": violations}


def verify_duplicate_cycle_free(rows: list[dict]) -> dict:
    fresh = [r for r in rows if r["quality_status"] == QUALITY_EXACT]
    by_cycle: dict[str, dict] = {}
    for r in fresh:
        key = r["cycle_key"]
        if key not in by_cycle or r["signal_birth_ts"] < by_cycle[key]["signal_birth_ts"]:
            by_cycle[key] = r
    return {"fresh_signal_count": len(fresh),
            "representative_cycle_count": len(by_cycle),
            "duplicates_collapsed": len(fresh) - len(by_cycle)}
