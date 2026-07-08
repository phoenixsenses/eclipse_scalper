"""Binance historical aggTrades probe + disposable repair staging.

Repair source (exact layer): GET /fapi/v1/aggTrades (USDT-M futures).
Pagination law (frozen for aggtrades-binance-fapi-repair-r1):

  1. First page: startTime/endTime bounded request (limit=1000).
  2. Every subsequent page: fromId = last returned agg_trade_id + 1
     (pure id-cursor -- gap-free and overlap-free BY CONSTRUCTION if and
     only if the returned ids are consecutive; consecutiveness is measured,
     never assumed).
  3. Stop when a page returns a trade with T > end_ms (trim it) or fewer
     rows than the limit at/beyond end_ms.
  4. Every page is manifested: request params, first/last agg_trade_id,
     row count, retry count, HTTP status. Page overlaps and missing id
     ranges are computed from the manifests -- a nonzero missing-id count
     inside [first_id, last_id] fails the extraction's continuity proof.

Determinism: an extraction is DETERMINISTIC only if a full rerun produces
byte-identical ordered content (ORDER BY symbol, ts_ms, agg_trade_id),
identical row counts and identical gap/duplicate manifests. Anything else
fails closed.

Staging is disposable-only: rows are written to a caller-supplied disposable
database, never to data/microstructure.db, never to canonical.sqlite.
Insertion is immutable: re-inserting the same (symbol, agg_trade_id) with
different content raises (supersession would be a new retrieval batch, never
an UPDATE).
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request

BASE_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
LIMIT = 1000
MAX_RETRIES = 5
RETRY_SLEEP_SEC = 2.0
REPAIR_POPULATION_VERSION = "aggtrades-binance-fapi-repair-r1"


class ImmutableRepairRowConflict(Exception):
    """Same (symbol, agg_trade_id) staged twice with different content."""


class PaginationContinuityError(Exception):
    """Missing agg_trade_id range inside a claimed-complete extraction."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ami_agg_trades_repaired_stage (
    symbol TEXT NOT NULL,
    agg_trade_id INTEGER NOT NULL,
    ts_ms INTEGER NOT NULL,
    retrieved_at_ms INTEGER NOT NULL,
    price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    notional REAL NOT NULL,
    signed_quantity REAL NOT NULL,
    signed_notional REAL NOT NULL,
    is_buyer_maker INTEGER NOT NULL,
    taker_side TEXT NOT NULL,
    first_trade_id INTEGER,
    last_trade_id INTEGER,
    source_regime_id TEXT NOT NULL,
    retrieval_batch_id TEXT NOT NULL,
    retrieval_page_index INTEGER NOT NULL,
    source_provenance TEXT NOT NULL,
    source_quality_status TEXT NOT NULL,
    legacy_match_status TEXT NOT NULL,
    legacy_match_fingerprint TEXT,
    superseded_by_batch_id TEXT,
    data_version_id TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (symbol, agg_trade_id, retrieval_batch_id),
    CHECK (is_buyer_maker IN (0,1)),
    CHECK (taker_side IN ('BUY','SELL')),
    CHECK ((is_buyer_maker = 0) = (taker_side = 'BUY')),
    CHECK (legacy_match_status IN ('UNMATCHED','MATCHED_1TO1','AMBIGUOUS','CONFLICTING','NOT_ATTEMPTED')),
    CHECK (data_version_id = 'aggtrades-binance-fapi-repair-r1')
);
CREATE INDEX IF NOT EXISTS idx_repair_stage_ts
    ON ami_agg_trades_repaired_stage(symbol, ts_ms, agg_trade_id);

CREATE TABLE IF NOT EXISTS ami_cvd_repair_batch_ledger (
    retrieval_batch_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    requested_start_ms INTEGER NOT NULL,
    requested_end_ms INTEGER NOT NULL,
    pagination_method TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    first_agg_trade_id INTEGER,
    last_agg_trade_id INTEGER,
    earliest_trade_ts_ms INTEGER,
    latest_trade_ts_ms INTEGER,
    page_overlap_rows INTEGER NOT NULL,
    missing_id_ranges TEXT NOT NULL,
    request_errors TEXT NOT NULL,
    truncation_flag INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    gap_manifest_sha256 TEXT NOT NULL,
    duplicate_manifest_sha256 TEXT NOT NULL,
    exact_reconstruction_verdict TEXT NOT NULL,
    data_version_id TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    CHECK (exact_reconstruction_verdict IN
           ('EXACT_RECONSTRUCTED','INCOMPLETE','FAILED','PROBE_ONLY'))
);
"""


def init_schema(conn) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _http_get(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "eclipse-cvd-rehearsal/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def fetch_window(symbol: str, start_ms: int, end_ms: int, *,
                 max_pages: int = 100000, http_get=_http_get, sleep=time.sleep) -> dict:
    """Fetch every aggTrade with start_ms <= T <= end_ms. Returns rows +
    per-page manifests + continuity analysis. Never raises on API refusal --
    records errors and returns a verdict-bearing result (fail-closed
    verdicts are computed, not assumed)."""
    rows = []       # raw dicts as returned (a,p,q,f,l,T,m)
    pages = []
    errors = []
    seen_ids = set()
    overlap_rows = 0
    truncated = False
    cursor_from_id = None
    page_idx = 0
    while page_idx < max_pages:
        if cursor_from_id is None:
            url = (f"{BASE_URL}?symbol={symbol}&startTime={start_ms}&endTime={end_ms}"
                   f"&limit={LIMIT}")
            method = "startTime/endTime"
        else:
            url = f"{BASE_URL}?symbol={symbol}&fromId={cursor_from_id}&limit={LIMIT}"
            method = "fromId"
        status = None
        body = None
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                status, body = http_get(url)
                break
            except Exception as e:  # noqa: BLE001 -- recorded, then retried
                last_err = f"{type(e).__name__}: {e}"
                sleep(RETRY_SLEEP_SEC * (attempt + 1))
        if body is None:
            errors.append({"page": page_idx, "url": url, "error": last_err})
            return _finalize(symbol, start_ms, end_ms, rows, pages, errors,
                             overlap_rows, truncated, failed=True)
        batch = json.loads(body)
        if isinstance(batch, dict):  # API error object
            errors.append({"page": page_idx, "url": url, "error": batch})
            return _finalize(symbol, start_ms, end_ms, rows, pages, errors,
                             overlap_rows, truncated, failed=True)
        kept = []
        page_done = False
        for tr in batch:
            if tr["T"] > end_ms:
                page_done = True
                continue
            if tr["T"] < start_ms:
                # only possible on the first page per API semantics; count as overlap-outside
                continue
            if tr["a"] in seen_ids:
                overlap_rows += 1
                continue
            seen_ids.add(tr["a"])
            kept.append(tr)
        rows.extend(kept)
        pages.append({
            "page_index": page_idx, "method": method,
            "returned_n": len(batch), "kept_n": len(kept),
            "first_a": batch[0]["a"] if batch else None,
            "last_a": batch[-1]["a"] if batch else None,
            "first_T": batch[0]["T"] if batch else None,
            "last_T": batch[-1]["T"] if batch else None,
            "http_status": status,
        })
        page_idx += 1
        if not batch:
            # empty FIRST bounded page = Binance says the range is empty
            # (complete answer); empty fromId page before reaching end_ms
            # means no data exists at/after the cursor -- treat as possible
            # truncation (fail closed)
            if method == "fromId":
                truncated = True
            break
        if page_done:
            break
        if len(batch) < LIMIT:
            if method == "startTime/endTime":
                # bounded-query semantics: fewer than limit rows on the
                # FIRST startTime/endTime page means the range was fully
                # served -- complete, NOT truncation (requires span < 1h,
                # which every extraction window in this batch satisfies)
                break
            # fromId page shorter than limit without crossing end_ms:
            # either the live edge or a retention boundary -- fail closed
            if batch[-1]["T"] < end_ms:
                truncated = True
            break
        cursor_from_id = batch[-1]["a"] + 1
    else:
        truncated = True
    return _finalize(symbol, start_ms, end_ms, rows, pages, errors,
                     overlap_rows, truncated, failed=False)


def _finalize(symbol, start_ms, end_ms, rows, pages, errors, overlap_rows,
              truncated, *, failed) -> dict:
    rows_sorted = sorted(rows, key=lambda r: (r["T"], r["a"]))
    ids = sorted(r["a"] for r in rows_sorted)
    missing_ranges = []
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1] + 1:
            missing_ranges.append([ids[i - 1] + 1, ids[i] - 1])
    content_sha = hashlib.sha256(json.dumps(
        [[r["a"], r["T"], r["p"], r["q"], bool(r["m"])] for r in rows_sorted]
    ).encode()).hexdigest()
    gap_sha = hashlib.sha256(json.dumps(missing_ranges).encode()).hexdigest()
    dup_sha = hashlib.sha256(json.dumps({"page_overlap_rows": overlap_rows}).encode()).hexdigest()
    return {
        "symbol": symbol,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "row_count": len(rows_sorted),
        "rows": rows_sorted,
        "pages": pages,
        "page_count": len(pages),
        "pagination_method": "startTime-then-fromId",
        "first_agg_trade_id": ids[0] if ids else None,
        "last_agg_trade_id": ids[-1] if ids else None,
        "earliest_trade_ts_ms": rows_sorted[0]["T"] if rows_sorted else None,
        "latest_trade_ts_ms": rows_sorted[-1]["T"] if rows_sorted else None,
        "page_overlap_rows": overlap_rows,
        "missing_id_ranges": missing_ranges,
        "request_errors": errors,
        "truncation_flag": bool(truncated),
        "content_sha256": content_sha,
        "gap_manifest_sha256": gap_sha,
        "duplicate_manifest_sha256": dup_sha,
        "failed": failed,
    }


def extraction_verdict(result: dict, *, probe_only: bool) -> str:
    """Fail-closed verdict: a failed or id-discontinuous or truncated
    extraction is NEVER 'EXACT_RECONSTRUCTED'."""
    if probe_only:
        return "PROBE_ONLY"
    if result["failed"] or result["request_errors"]:
        return "FAILED"
    if result["missing_id_ranges"] or result["truncation_flag"]:
        return "INCOMPLETE"
    if result["row_count"] == 0:
        return "INCOMPLETE"
    return "EXACT_RECONSTRUCTED"


def stage_rows(conn, result: dict, *, retrieval_batch_id: str, source_regime_id: str,
               source_quality_status: str = "EXACT_RECONSTRUCTABLE") -> int:
    """Immutable disposable staging of a fetch result."""
    now = int(time.time() * 1000)
    n = 0
    # map agg_trade_id -> page index for provenance (pages carry inclusive
    # [first_a, last_a] id ranges; first page that contains the id wins)
    page_ranges = [(p["first_a"], p["last_a"], p["page_index"])
                   for p in result["pages"] if p["first_a"] is not None]

    def _page_of(a: int) -> int:
        for lo, hi, idx in page_ranges:
            if lo <= a <= hi:
                return idx
        return -1
    for r in result["rows"]:
        price = str(r["p"])
        qty = str(r["q"])
        notional = float(r["p"]) * float(r["q"])
        m = 1 if r["m"] else 0
        sign = -1 if m else 1
        existing = conn.execute(
            "SELECT price, quantity, is_buyer_maker, ts_ms FROM ami_agg_trades_repaired_stage"
            " WHERE symbol=? AND agg_trade_id=? AND retrieval_batch_id=?",
            (result["symbol"], r["a"], retrieval_batch_id)).fetchone()
        if existing is not None:
            if tuple(existing) != (price, qty, m, r["T"]):
                raise ImmutableRepairRowConflict(
                    f"{result['symbol']}#{r['a']}@{retrieval_batch_id}: different content")
            continue
        conn.execute(
            "INSERT INTO ami_agg_trades_repaired_stage (symbol, agg_trade_id, ts_ms,"
            " retrieved_at_ms, price, quantity, notional, signed_quantity, signed_notional,"
            " is_buyer_maker, taker_side, first_trade_id, last_trade_id, source_regime_id,"
            " retrieval_batch_id, retrieval_page_index, source_provenance,"
            " source_quality_status, legacy_match_status, legacy_match_fingerprint,"
            " superseded_by_batch_id, data_version_id, created_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result["symbol"], r["a"], r["T"], now, price, qty, notional,
             sign * float(r["q"]), sign * notional, m, "SELL" if m else "BUY",
             r.get("f"), r.get("l"), source_regime_id, retrieval_batch_id,
             _page_of(r["a"]), "GET /fapi/v1/aggTrades",
             source_quality_status, "NOT_ATTEMPTED", None, None,
             REPAIR_POPULATION_VERSION, now))
        n += 1
    conn.commit()
    return n


def record_batch_ledger(conn, result: dict, *, retrieval_batch_id: str, verdict: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ami_cvd_repair_batch_ledger (retrieval_batch_id, symbol,"
        " requested_start_ms, requested_end_ms, pagination_method, page_count, row_count,"
        " first_agg_trade_id, last_agg_trade_id, earliest_trade_ts_ms, latest_trade_ts_ms,"
        " page_overlap_rows, missing_id_ranges, request_errors, truncation_flag, content_sha256,"
        " gap_manifest_sha256, duplicate_manifest_sha256, exact_reconstruction_verdict,"
        " data_version_id, created_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (retrieval_batch_id, result["symbol"], result["requested_start_ms"],
         result["requested_end_ms"], result["pagination_method"], result["page_count"],
         result["row_count"], result["first_agg_trade_id"], result["last_agg_trade_id"],
         result["earliest_trade_ts_ms"], result["latest_trade_ts_ms"],
         result["page_overlap_rows"], json.dumps(result["missing_id_ranges"]),
         json.dumps(result["request_errors"]), int(result["truncation_flag"]),
         result["content_sha256"], result["gap_manifest_sha256"],
         result["duplicate_manifest_sha256"], verdict, REPAIR_POPULATION_VERSION,
         int(time.time() * 1000)))
    conn.commit()


# ---------------------------------------------------------------------------
# Cross-source reconciliation (REST rows vs legacy local rows)
# ---------------------------------------------------------------------------

def legacy_fingerprint(ts_ms: int, price: float, quantity: float, is_buyer_maker: int) -> tuple:
    """Observable legacy fingerprint. Legacy rows carry NO agg_trade_id (the
    historical collector discarded Binance's a/f/l ids), so this composite is
    the ONLY joinable identity -- and it is NOT guaranteed unique."""
    return (int(ts_ms), float(price), float(quantity), int(bool(is_buyer_maker)))


def reconcile_rest_vs_legacy(rest_rows: list[dict], legacy_rows: list[tuple]) -> dict:
    """Deterministic multiset reconciliation over one bounded time span.

    rest_rows: raw REST dicts (a,p,q,T,m). legacy_rows: (ts_ms, price,
    quantity, is_buyer_maker) tuples from data/microstructure.db.

    Matching algorithm (frozen): exact fingerprint-multiset intersection.
    A REST row matches 1:1 iff its fingerprint occurs exactly once on both
    sides. Any fingerprint with multiplicity >1 on either side is a
    collision class, counted and treated as UNRESOLVED (never arbitrarily
    paired). No nearest-row or probabilistic matching exists here at all.
    """
    from collections import Counter
    rest_fp = Counter()
    for r in rest_rows:
        rest_fp[legacy_fingerprint(r["T"], float(r["p"]), float(r["q"]), 1 if r["m"] else 0)] += 1
    leg_fp = Counter(legacy_fingerprint(*t) for t in legacy_rows)

    one_to_one = 0
    unmatched_rest = 0
    unmatched_legacy = 0
    ambiguous_rest = 0
    ambiguous_legacy = 0
    one_to_many = 0     # rest multiplicity 1, legacy multiplicity > 1
    many_to_one = 0     # rest multiplicity > 1, legacy multiplicity 1
    many_to_many = 0
    conflicting = 0     # same (ts,p,q) both sides but differing maker flag only
    dup_multiplicity_hist = {}

    keys = set(rest_fp) | set(leg_fp)
    # side-flag conflict detection: same (ts, p, q) with opposite m present
    # on one side but not the other
    rest_tpq = Counter((k[0], k[1], k[2]) for k in rest_fp)
    leg_tpq = Counter((k[0], k[1], k[2]) for k in leg_fp)
    for k in keys:
        rn = rest_fp.get(k, 0)
        ln = leg_fp.get(k, 0)
        m = max(rn, ln)
        if m > 1:
            dup_multiplicity_hist[m] = dup_multiplicity_hist.get(m, 0) + 1
        if rn == 1 and ln == 1:
            one_to_one += 1
        elif rn >= 1 and ln == 0:
            tpq = (k[0], k[1], k[2])
            if leg_tpq.get(tpq, 0) > 0:
                conflicting += rn
            else:
                unmatched_rest += rn
        elif ln >= 1 and rn == 0:
            tpq = (k[0], k[1], k[2])
            if rest_tpq.get(tpq, 0) > 0:
                conflicting += ln
            else:
                unmatched_legacy += ln
        elif rn == 1 and ln > 1:
            one_to_many += 1
            ambiguous_legacy += ln
        elif rn > 1 and ln == 1:
            many_to_one += 1
            ambiguous_rest += rn
        else:
            many_to_many += 1
            ambiguous_rest += rn
            ambiguous_legacy += ln

    total_rest = sum(rest_fp.values())
    total_legacy = sum(leg_fp.values())
    deterministic = (unmatched_rest == 0 and one_to_many == 0 and many_to_one == 0
                     and many_to_many == 0 and conflicting == 0)
    return {
        "rest_rows": total_rest,
        "legacy_rows": total_legacy,
        "distinct_fingerprints_rest": len(rest_fp),
        "distinct_fingerprints_legacy": len(leg_fp),
        "exact_one_to_one": one_to_one,
        "unmatched_rest": unmatched_rest,
        "unmatched_legacy": unmatched_legacy,
        "one_to_many_collisions": one_to_many,
        "many_to_one_collisions": many_to_one,
        "many_to_many_collisions": many_to_many,
        "ambiguous_rest_rows": ambiguous_rest,
        "ambiguous_legacy_rows": ambiguous_legacy,
        "conflicting_side_flag_rows": conflicting,
        "duplicate_fingerprint_multiplicity_hist": dup_multiplicity_hist,
        "deterministic_supersession_feasible": deterministic,
    }
