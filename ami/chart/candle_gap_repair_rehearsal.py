"""BATCH: AMI HISTORICAL CANDLE GAP REMEDIATION -- source audit + disposable
rehearsal ONLY. NO_REAL_CANONICAL_WRITE: every function here operates on a
caller-supplied connection; this module never opens
ami.warehouse.schema.DEFAULT_PATH itself, matching the exact discipline of
every other disposable-rehearsal module in this project (path_migration_
rehearsal.py, short_noisy_v1_migration_rehearsal.py, etc.).

ROOT CAUSE (confirmed by direct audit, not assumed): every one of the 208
distinct missing-candle-run windows in `ami_candles` (ETHUSDT, 1m, full
2026-02-15..2026-07-03 history) has ZERO matching rows in
data/microstructure.db:agg_trades for the exact same window (epoch-exact
check, not a string/timezone-converted one -- an earlier ad-hoc spot-check
using `datetime.strptime(...).timestamp()` falsely suggested trades existed,
because that call implicitly used the local machine's UTC+3 offset instead
of UTC; the corrected, epoch-integer-only check confirmed zero rows). This
means the gaps are NOT a local candle-builder backfill bug (recoverable by
simply re-running ami.chart.candle_builder against agg_trades) -- they are a
genuine absence in the locally-collected trade archive, requiring the
project's own already-approved authoritative exchange source (Binance
USDT-M Futures, matching the existing `binanceusdm` ccxt integration in
ami/exchanges/binance.py and the project's liquidation-cascade/futures
architecture) to potentially recover.

SOURCE: Binance USDT-M Futures REST API, GET /fapi/v1/klines
(https://fapi.binance.com), interval=1m. NEVER spot (api.binance.com) --
this project trades/analyzes USDT-M perpetual futures exclusively, and
substituting spot candles for futures candles would silently change the
underlying instrument's price data (funding-driven basis divergence is real
and non-trivial for ETHUSDT), which is explicitly forbidden.

NO_INTERPOLATION / NO_SYNTHETIC_CANDLES: every candidate row in this module
is a REAL kline returned by the authoritative exchange for that EXACT
1-minute window -- nothing is computed, averaged, or fabricated. A window
Binance cannot return a row for is left as a genuine, explicitly reported
gap (IRRECOVERABLE_SOURCE_GAP), never silently filled.

CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR is a NEW, distinct version token
(never blended into "candle-agg_trades-v1") -- same discipline as every
other identity-bearing version constant in this project (setup_version,
path_definition_version, etc.): a different derivation method mints a new
version, it never silently reuses an old one. The UNIQUE(symbol, timeframe,
open_ts_ms, candle_definition_version) constraint means these repair rows
structurally cannot collide with or overwrite the original agg_trades-derived
rows, even if the real canonical DB were ever targeted by a future,
separately-approved batch.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
import time
from pathlib import Path

SYMBOL = "ETHUSDT"
TIMEFRAME = "1m"
CANDLE_MS = 60_000
CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR = "candle-binance-fapi-repair-v1"
SOURCE_IDENTIFIER = "binance-fapi-klines"
MARKET_TYPE = "USDT-M perpetual futures"


def make_disposable_copy(source_path, disposable_path) -> None:
    Path(disposable_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, disposable_path)


def _candle_id(symbol: str, timeframe: str, open_ts_ms: int, version: str) -> str:
    key = f"{symbol}|{timeframe}|{open_ts_ms}|{version}"
    return "CDL-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def raw_gap_manifest(conn, symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> dict:
    """Deterministic snapshot of every missing 1m open_ts_ms in [min,max] of
    the existing candle range, across ALL candle_definition_versions (a
    timestamp is 'present' if ANY version has a row there) -- this is the
    exact same full-history definition used by the read-only audit that
    preceded this batch."""
    row = conn.execute(
        f"SELECT MIN(open_ts_ms), MAX(open_ts_ms) FROM ami_candles WHERE symbol=? AND timeframe=?",
        (symbol, timeframe),
    ).fetchone()
    if row[0] is None:
        return {"gap_runs": [], "missing_n": 0, "manifest_hash": hashlib.sha256(b"").hexdigest()}
    lo, hi = row
    present = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT open_ts_ms FROM ami_candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ).fetchall()
    }
    gap_runs = []
    cur_start = None
    t = lo
    end = hi + CANDLE_MS
    while t < end:
        if t not in present:
            if cur_start is None:
                cur_start = t
        else:
            if cur_start is not None:
                gap_runs.append((cur_start, t))
                cur_start = None
        t += CANDLE_MS
    if cur_start is not None:
        gap_runs.append((cur_start, end))
    missing_n = sum((e - s) // CANDLE_MS for s, e in gap_runs)
    manifest_text = json.dumps(gap_runs, sort_keys=False)
    return {
        "gap_runs": gap_runs, "missing_n": missing_n,
        "manifest_hash": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    }


def validate_kline_row(raw_kline: list) -> tuple[bool, str | None]:
    """Rejects: non-1m-aligned open_time, invalid OHLC relationship,
    non-numeric fields. Never repairs/coerces -- a bad row is rejected
    outright, its reason recorded."""
    try:
        open_ts_ms = int(raw_kline[0])
        o, h, l, c, v = (float(raw_kline[1]), float(raw_kline[2]), float(raw_kline[3]),
                         float(raw_kline[4]), float(raw_kline[5]))
        close_ts_ms = int(raw_kline[6])
        n_trades = int(raw_kline[8])
        taker_buy_base = float(raw_kline[9])
    except (TypeError, ValueError, IndexError):
        return False, "NON_NUMERIC_OR_MALFORMED_FIELD"
    if open_ts_ms % CANDLE_MS != 0:
        return False, "NOT_1M_ALIGNED"
    if close_ts_ms != open_ts_ms + CANDLE_MS - 1:
        return False, "CLOSE_TIME_MISMATCH"
    if not (l <= o <= h and l <= c <= h and l > 0 and h > 0):
        return False, "INVALID_OHLC_RELATIONSHIP"
    return True, None


def build_candidate_rows(raw_klines_by_gap: dict, retrieval_ts_ms: int,
                          provenance: str = "batch-candle-gap-remediation-disposable-rehearsal") -> dict:
    """Converts raw Binance kline arrays into ami_candles row dicts. Rejects
    (never silently drops without accounting): duplicate open_ts_ms within
    the candidate batch, and any row failing validate_kline_row(). Returns
    accepted rows plus a reason-tagged rejection list."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_ts: set[int] = set()

    for gap_key, raw_klines in raw_klines_by_gap.items():
        for raw in raw_klines:
            ok, reason = validate_kline_row(raw)
            if not ok:
                rejected.append({"open_ts_ms": raw[0] if raw else None, "reason": reason})
                continue
            open_ts_ms = int(raw[0])
            if open_ts_ms in seen_ts:
                rejected.append({"open_ts_ms": open_ts_ms, "reason": "DUPLICATE_TIMESTAMP_IN_BATCH"})
                continue
            seen_ts.add(open_ts_ms)
            taker_buy = float(raw[9])
            volume = float(raw[5])
            source_hash = hashlib.sha256(
                f"{SOURCE_IDENTIFIER}|{SYMBOL}|{open_ts_ms}|{raw[1]}|{raw[2]}|{raw[3]}|{raw[4]}|{raw[5]}".encode()
            ).hexdigest()[:32]
            accepted.append({
                "candle_id": _candle_id(SYMBOL, TIMEFRAME, open_ts_ms, CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR),
                "symbol": SYMBOL, "venue": "BINANCE_USDM_FUTURES", "timeframe": TIMEFRAME,
                "open_ts_ms": open_ts_ms, "close_ts_ms": open_ts_ms + CANDLE_MS,
                "open": float(raw[1]), "high": float(raw[2]), "low": float(raw[3]), "close": float(raw[4]),
                "volume": volume, "trade_count": int(raw[8]),
                "taker_buy_volume": taker_buy, "taker_sell_volume": max(0.0, volume - taker_buy),
                "is_closed": 1, "partial_status": "CLOSED",
                "known_at_ts": open_ts_ms + CANDLE_MS,
                "data_quality": "AVAILABLE",
                "source_hash": source_hash,
                "candle_definition_version": CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR,
                "retrieval_ts_ms": retrieval_ts_ms,
                "source_identifier": SOURCE_IDENTIFIER,
                "provenance": provenance,
            })
    return {"accepted": accepted, "rejected": rejected}


def validate_no_conflict_with_existing(conn, candidate_rows: list[dict],
                                        symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> dict:
    """Proves every candidate open_ts_ms is genuinely a gap slot (no existing
    row for that timestamp under ANY candle_definition_version) before any
    write is attempted."""
    existing_ts = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT open_ts_ms FROM ami_candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ).fetchall()
    }
    conflicts = [r["open_ts_ms"] for r in candidate_rows if r["open_ts_ms"] in existing_ts]
    return {"conflict_n": len(conflicts), "conflicting_timestamps": conflicts[:20]}


def apply_repair_rows(conn, accepted_rows: list[dict], schema_version: int = 5) -> dict:
    """The ONLY write path for repair rows. INSERT ... ON CONFLICT(symbol,
    timeframe, open_ts_ms, candle_definition_version) DO UPDATE -- idempotent
    (same convention as ami.chart.candle_builder._upsert_candles), but since
    candle_definition_version is unique to this repair batch, this can never
    touch a pre-existing 'candle-agg_trades-v1' row."""
    now = int(time.time() * 1000)
    n_written = 0
    for r in accepted_rows:
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, venue, timeframe, open_ts_ms, close_ts_ms, "
            "open, high, low, close, volume, trade_count, taker_buy_volume, taker_sell_volume, "
            "is_closed, partial_status, known_at_ts, data_quality, source_hash, "
            "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, timeframe, open_ts_ms, candle_definition_version) DO UPDATE SET "
            "close=excluded.close, high=excluded.high, low=excluded.low, volume=excluded.volume, "
            "trade_count=excluded.trade_count, source_hash=excluded.source_hash, updated_ms=excluded.updated_ms",
            (r["candle_id"], r["symbol"], r["venue"], r["timeframe"], r["open_ts_ms"], r["close_ts_ms"],
             r["open"], r["high"], r["low"], r["close"], r["volume"], r["trade_count"],
             r["taker_buy_volume"], r["taker_sell_volume"], r["is_closed"], r["partial_status"],
             r["known_at_ts"], r["data_quality"], r["source_hash"], r["candle_definition_version"],
             schema_version, r["provenance"], now, now),
        )
        n_written += 1
    conn.commit()
    return {"rows_written": n_written}


def rederive_5m_with_source_traceability(conn, symbol: str = SYMBOL,
                                          provenance: str = "batch-candle-gap-remediation-5m-rederive") -> dict:
    """Re-derives 5m candles from the FULL combined 1m set (existing
    agg_trades-derived rows + repair rows together) via the EXISTING, UNMODIFIED
    ami.chart.candle_builder.derive_higher_timeframe() -- no separate
    authoritative 5m source is introduced, matching the operator's explicit
    instruction. The one addition: derive_higher_timeframe() itself hardcodes
    its module-level CANDLE_DEFINITION_VERSION constant onto every output row,
    which would silently mislabel a 5m bucket whose 1m children are partly or
    fully from the repair source. This function corrects ONLY that
    candle_definition_version label after calling the unmodified function --
    it does not change how OHLCV/volume/etc are computed, only which source
    version(s) each bucket is honestly attributed to (same multi-version
    comma-join convention ami.lifecycle.path_metrics.compute_observation()
    already uses for its own candle_definition_version field)."""
    from ami.chart.candle_builder import derive_higher_timeframe
    import hashlib

    cols = ["candle_id", "symbol", "venue", "timeframe", "open_ts_ms", "close_ts_ms", "open", "high",
            "low", "close", "volume", "trade_count", "taker_buy_volume", "taker_sell_volume",
            "is_closed", "partial_status", "known_at_ts", "data_quality", "source_hash",
            "candle_definition_version"]
    rows_1m = conn.execute(
        f"SELECT {', '.join(cols)} FROM ami_candles WHERE symbol=? AND timeframe='1m' ORDER BY open_ts_ms",
        (symbol,),
    ).fetchall()
    base_candles = [dict(zip(cols, r)) for r in rows_1m]

    candles_5m = derive_higher_timeframe(base_candles, symbol, "5m")

    bar_ms = 300_000
    by_5m_bucket: dict[int, list[str]] = {}
    for c in base_candles:
        bucket_open = c["open_ts_ms"] - (c["open_ts_ms"] % bar_ms)
        by_5m_bucket.setdefault(bucket_open, []).append(c["candle_definition_version"])

    n_corrected = 0
    for c5 in candles_5m:
        versions = sorted(set(by_5m_bucket.get(c5["open_ts_ms"], [])))
        correct_version = versions[0] if len(versions) == 1 else ",".join(versions)
        if c5["candle_definition_version"] != correct_version:
            c5["candle_definition_version"] = correct_version
            n_corrected += 1
            key = f"{symbol}|5m|{c5['open_ts_ms']}|{correct_version}"
            c5["candle_id"] = "CDL-" + hashlib.sha256(key.encode()).hexdigest()[:24]

    now = int(time.time() * 1000)
    n_written = 0
    for c in candles_5m:
        conn.execute(
            "INSERT INTO ami_candles (candle_id, symbol, venue, timeframe, open_ts_ms, close_ts_ms, "
            "open, high, low, close, volume, trade_count, taker_buy_volume, taker_sell_volume, "
            "is_closed, partial_status, known_at_ts, data_quality, source_hash, "
            "candle_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol, timeframe, open_ts_ms, candle_definition_version) DO UPDATE SET "
            "close=excluded.close, high=excluded.high, low=excluded.low, volume=excluded.volume, "
            "trade_count=excluded.trade_count, data_quality=excluded.data_quality, updated_ms=excluded.updated_ms",
            (c["candle_id"], c["symbol"], c["venue"], c["timeframe"], c["open_ts_ms"], c["close_ts_ms"],
             c["open"], c["high"], c["low"], c["close"], c["volume"], c["trade_count"],
             c["taker_buy_volume"], c["taker_sell_volume"], c["is_closed"], c["partial_status"],
             c["known_at_ts"], c["data_quality"], c["source_hash"], c["candle_definition_version"],
             7, provenance, now, now),
        )
        n_written += 1
    conn.commit()
    return {"rows_written": n_written, "version_corrected_n": n_corrected, "total_5m_buckets": len(candles_5m)}


def rollback_repair(conn) -> dict:
    """Deletes every row whose candle_definition_version references
    CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR -- this includes pure-repair 1m
    rows (exact match) AND any 5m row whose version is a comma-joined blend
    (e.g. "candle-agg_trades-v1,candle-binance-fapi-repair-v1", produced by
    rederive_5m_with_source_traceability() for a bucket with mixed-source 1m
    children) -- a LIKE match is required for those, an exact match alone
    would silently leave blended 5m repair rows behind. Every pure
    'candle-agg_trades-v1' row (no repair reference at all) is untouched."""
    cur = conn.execute(
        "DELETE FROM ami_candles WHERE candle_definition_version=? OR candle_definition_version LIKE ?",
        (CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR, f"%{CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR}%"),
    )
    conn.commit()
    return {"rows_deleted": cur.rowcount}


def run_disposable_rehearsal(source_canonical_path, disposable_path, raw_klines_by_gap: dict,
                              retrieval_ts_ms: int) -> dict:
    report: dict = {"source_canonical_path": str(source_canonical_path), "disposable_path": str(disposable_path)}

    make_disposable_copy(source_canonical_path, disposable_path)
    conn = sqlite3.connect(disposable_path)
    try:
        pre_manifest = raw_gap_manifest(conn)
        report["pre_repair_gap_manifest_hash"] = pre_manifest["manifest_hash"]
        report["pre_repair_missing_n"] = pre_manifest["missing_n"]

        built = build_candidate_rows(raw_klines_by_gap, retrieval_ts_ms)
        accepted, rejected = built["accepted"], built["rejected"]
        report["candidate_rows"] = len(accepted) + len(rejected)
        report["accepted_rows"] = len(accepted)
        report["rejected_rows_by_reason"] = {}
        for r in rejected:
            report["rejected_rows_by_reason"][r["reason"]] = report["rejected_rows_by_reason"].get(r["reason"], 0) + 1

        conflict_check = validate_no_conflict_with_existing(conn, accepted)
        report["conflict_check"] = conflict_check

        r1 = apply_repair_rows(conn, accepted)
        report["run1_rows_written"] = r1["rows_written"]

        post_manifest = raw_gap_manifest(conn)
        report["post_repair_gap_manifest_hash"] = post_manifest["manifest_hash"]
        report["post_repair_missing_n"] = post_manifest["missing_n"]
        report["remaining_unrepaired_gaps"] = post_manifest["gap_runs"]

        # idempotent rerun
        r2 = apply_repair_rows(conn, accepted)
        post_manifest_2 = raw_gap_manifest(conn)
        report["rerun_rows_written"] = r2["rows_written"]
        report["deterministic_rerun_manifest_matches"] = (
            post_manifest_2["manifest_hash"] == post_manifest["manifest_hash"]
        )
        # scoped to THIS call's own candidate timestamps -- a global COUNT(*) by version would
        # conflate this rehearsal's rows with any repair rows the source DB already carried
        # (e.g. a disposable copy of an already-repaired real canonical.sqlite)
        accepted_ts = [r["open_ts_ms"] for r in accepted]
        if accepted_ts:
            placeholders = ",".join("?" for _ in accepted_ts)
            count_after_run1 = conn.execute(
                f"SELECT COUNT(*) FROM ami_candles WHERE candle_definition_version=? "
                f"AND open_ts_ms IN ({placeholders})",
                [CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR] + accepted_ts,
            ).fetchone()[0]
        else:
            count_after_run1 = 0
        report["repair_row_count_after_rerun"] = count_after_run1
        report["rerun_did_not_duplicate_rows"] = (count_after_run1 == len(accepted))

        # rollback rehearsal
        pre_rollback_total = conn.execute("SELECT COUNT(*) FROM ami_candles").fetchone()[0]
        rb = rollback_repair(conn)
        report["rollback_result"] = rb
        post_rollback_manifest = raw_gap_manifest(conn)
        report["rollback_restores_pre_repair_manifest"] = (
            post_rollback_manifest["manifest_hash"] == pre_manifest["manifest_hash"]
        )
        post_rollback_total = conn.execute("SELECT COUNT(*) FROM ami_candles").fetchone()[0]
        report["rollback_row_delta"] = pre_rollback_total - post_rollback_total

        # reapply after rollback
        r3 = apply_repair_rows(conn, accepted)
        post_reapply_manifest = raw_gap_manifest(conn)
        report["reapply_matches_post_repair_manifest"] = (
            post_reapply_manifest["manifest_hash"] == post_manifest["manifest_hash"]
        )
        report["reapply_rows_written"] = r3["rows_written"]
    finally:
        conn.close()

    return report
