"""AMI CVD windowed taker-flow feature -- disposable-rehearsal implementation.

Implements the frozen contract
`S34_CVD_REPAIR_CONTRACT_AND_ANCHOR_DEFINITION_2026-07-05.md`:

- NO global CVD epoch, NO persistent cumulative state, NO restart-dependent
  reset state. The canonical quantity is a bounded, signal-relative trailing
  window: `[T - W, T]`, end-inclusive at `T`, start-inclusive at `T - W`,
  where `T = signal_birth_ts`.
- Frozen window family: 60s / 300s / 600s / 1800s / 3600s / BUCKET
  (= the signal's own frozen birth-truncated RUNNING_CLUSTER bucket window
  `[source_window_start_ts_ms, T]` from ami_birth_truncated_cascade_geometry).
- Sign convention (aggtrades-taker-side-v1): is_buyer_maker = 0 -> taker BUY
  (+1); is_buyer_maker = 1 -> taker SELL (-1). Verified in candle_builder.py,
  state_reconstruct.py and orderflow_chart.py.
- Exact evidence layer: trade-level rows only (live agg_trades and/or
  REST-repaired aggTrades). Proxy evidence layer: minute-quantized candle
  taker fields; NEVER equivalent to, pooled with, or substituted for the
  exact layer -- physically separate tables.

This module operates on caller-supplied connections; it never opens the real
canonical.sqlite itself and is never imported by any live/shadow/execution
path (geometry-module discipline).

No outcome fields exist anywhere in this module: no MFE, no MAE, no PnL,
no win rate, no future-path column.
"""
from __future__ import annotations

import hashlib
import json
import time

FEATURE_DEFINITION_VERSION = "s34-cvd-windowed-taker-flow-v1-birth-truncated"
RAW_INTERPRETATION_VERSION = "aggtrades-taker-side-v1"
REPAIR_POPULATION_VERSION = "aggtrades-binance-fapi-repair-r1"
QUALITY_CONTRACT_VERSION = "cvd-source-quality-contract-v1"

SYMBOL_SCOPE_V1 = "ETHUSDT"

# frozen window family -- 6 windows, no additions (contract section 1.1)
FIXED_WINDOWS_SEC = (60, 300, 600, 1800, 3600)
BUCKET_WINDOW_ID = "BUCKET"
WINDOW_IDS = tuple(f"W{w}" for w in FIXED_WINDOWS_SEC) + (BUCKET_WINDOW_ID,)

EVIDENCE_EXACT = "EXACT"
EVIDENCE_PROXY = "PROXY"

# BUCKET eligibility (documented canonical rule, NOT a silent drop):
# the BUCKET window is DEFINED only for signals that possess a frozen
# ami_birth_truncated_cascade_geometry row (feature_definition_version
# s34-knowable-anchor-continuation-v1-birth-truncated -- currently the 220
# canonical LONG signals). For any other signal the BUCKET window start was
# never frozen; deriving one post-hoc would be a NEW feature definition and
# would additionally require reading the liquidations source whose
# inferential use is currently blocked by source quality. Such signals are
# recorded in ami_cvd_bucket_exclusions (one explicit row each).
BUCKET_EXCLUSION_REASON = "BUCKET_WINDOW_NOT_FROZEN_FOR_SIGNAL"


class KnownAtViolation(AssertionError):
    """A source row with ts_ms > signal_birth_ts reached a window compute."""


class ImmutableCvdFeatureConflict(Exception):
    """Same (signal_id, window_id, feature_definition_version) written twice
    with different content -- feature rows are immutable."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ami_cvd_windowed_flow (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    raw_interpretation_version TEXT NOT NULL,
    quality_contract_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    source_row_count INTEGER NOT NULL,
    legacy_row_count INTEGER NOT NULL,
    repair_row_count INTEGER NOT NULL,
    cvd_qty REAL,
    cvd_notional REAL,
    total_notional REAL,
    taker_buy_qty REAL,
    taker_sell_qty REAL,
    taker_buy_notional REAL,
    taker_sell_notional REAL,
    normalized_cvd REAL,
    source_row_manifest_sha256 TEXT NOT NULL,
    source_regime_ids TEXT NOT NULL,
    repair_method TEXT NOT NULL,
    repair_population_version TEXT,
    feature_available_ts_ms INTEGER NOT NULL,
    known_at_classification TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, feature_definition_version),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600','BUCKET')),
    CHECK (evidence_layer = 'EXACT'),
    CHECK (window_start_ts_ms <= window_end_ts_ms),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (source_row_count = legacy_row_count + repair_row_count),
    CHECK (source_row_count >= 0),
    CHECK (repair_method IN ('NONE','AGGTRADES_REST','AGGTRADES_VISION_ARCHIVE')),
    CHECK (total_notional IS NULL OR total_notional >= 0),
    CHECK (normalized_cvd IS NULL OR (normalized_cvd >= -1.0 AND normalized_cvd <= 1.0))
);
CREATE INDEX IF NOT EXISTS idx_cvd_flow_signal ON ami_cvd_windowed_flow(signal_id);
CREATE INDEX IF NOT EXISTS idx_cvd_flow_cycle ON ami_cvd_windowed_flow(independent_cycle_id);

-- PROXY layer: PHYSICALLY separate table. Never pooled with the exact table;
-- a schema-level CHECK pins evidence_layer so a mixed population cannot even
-- be represented.
CREATE TABLE IF NOT EXISTS ami_cvd_windowed_flow_proxy (
    feature_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    quality_contract_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    independent_cycle_id TEXT,
    symbol TEXT NOT NULL,
    signal_birth_ts INTEGER NOT NULL,
    window_id TEXT NOT NULL,
    window_start_ts_ms INTEGER NOT NULL,
    window_end_ts_ms INTEGER NOT NULL,
    evidence_layer TEXT NOT NULL,
    candle_timeframe TEXT NOT NULL,
    contained_candle_count INTEGER NOT NULL,
    last_contained_close_ts_ms INTEGER,
    proxy_cvd_qty REAL,
    proxy_taker_buy_qty REAL,
    proxy_taker_sell_qty REAL,
    candle_versions TEXT NOT NULL,
    source_row_manifest_sha256 TEXT NOT NULL,
    feature_available_ts_ms INTEGER NOT NULL,
    known_at_classification TEXT NOT NULL,
    descriptive_only INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, window_id, feature_definition_version),
    CHECK (window_id IN ('W60','W300','W600','W1800','W3600','BUCKET')),
    CHECK (evidence_layer = 'PROXY'),
    CHECK (descriptive_only = 1),
    CHECK (window_start_ts_ms <= window_end_ts_ms),
    CHECK (window_end_ts_ms = signal_birth_ts),
    CHECK (feature_available_ts_ms = signal_birth_ts),
    CHECK (known_at_classification = 'KNOWN_AT_SAFE'),
    CHECK (contained_candle_count >= 0),
    CHECK (last_contained_close_ts_ms IS NULL OR last_contained_close_ts_ms <= signal_birth_ts),
    CHECK ((contained_candle_count = 0) = (proxy_cvd_qty IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_cvd_proxy_signal ON ami_cvd_windowed_flow_proxy(signal_id);

-- explicit, non-silent BUCKET exclusion ledger
CREATE TABLE IF NOT EXISTS ami_cvd_bucket_exclusions (
    exclusion_id TEXT PRIMARY KEY,
    feature_definition_version TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    reason TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE (signal_id, feature_definition_version),
    CHECK (reason = 'BUCKET_WINDOW_NOT_FROZEN_FOR_SIGNAL')
);
"""


def init_schema(conn) -> None:
    """Idempotent CREATE IF NOT EXISTS only. Disposable-DB use in this batch;
    the identical DDL is what the 11->12 migration proposal would fold into
    ami/warehouse/schema.py verbatim (geometry precedent)."""
    conn.executescript(_SCHEMA)
    conn.commit()


def taker_sign(is_buyer_maker) -> int:
    """aggtrades-taker-side-v1: is_buyer_maker == 0/False -> taker BUY (+1);
    is_buyer_maker == 1/True -> taker SELL (-1)."""
    return -1 if is_buyer_maker else 1


def window_bounds(signal_birth_ts: int, window_id: str, bucket_start_ts_ms=None):
    """Frozen boundary law: [T - W, T], BOTH ends inclusive."""
    if window_id == BUCKET_WINDOW_ID:
        if bucket_start_ts_ms is None:
            raise ValueError("BUCKET window requested without a frozen bucket start")
        if bucket_start_ts_ms > signal_birth_ts:
            raise KnownAtViolation("bucket_start_ts_ms > signal_birth_ts")
        return int(bucket_start_ts_ms), int(signal_birth_ts)
    if window_id not in WINDOW_IDS:
        raise ValueError(f"unknown window_id {window_id!r}")
    w_sec = int(window_id[1:])
    return int(signal_birth_ts) - w_sec * 1000, int(signal_birth_ts)


def compute_window_flow(rows, window_start_ts_ms: int, window_end_ts_ms: int) -> dict:
    """Compute the exact-layer quantities over trade rows.

    rows: iterable of dicts with keys ts_ms, price, quantity, notional,
    is_buyer_maker, source ('LEGACY'|'REST'), order_key (tuple used only for
    the deterministic manifest ordering).

    Fail-closed known-at rule: ANY row outside [window_start, window_end]
    raises -- rejection of post-T rows is an error, not a silent filter,
    because the caller is contractually required to pre-truncate (geometry
    precedent: filter first, compute second, assert nothing slipped through).
    """
    cvd_qty = 0.0
    cvd_notional = 0.0
    total_notional = 0.0
    buy_qty = sell_qty = 0.0
    buy_notional = sell_notional = 0.0
    legacy_n = rest_n = 0
    manifest_items = []
    for r in rows:
        ts = int(r["ts_ms"])
        if ts > window_end_ts_ms:
            raise KnownAtViolation(
                f"row ts_ms={ts} > window_end={window_end_ts_ms} (post-T row reached compute)")
        if ts < window_start_ts_ms:
            raise KnownAtViolation(
                f"row ts_ms={ts} < window_start={window_start_ts_ms} (pre-window row reached compute)")
        s = taker_sign(r["is_buyer_maker"])
        q = float(r["quantity"])
        n = float(r["notional"])
        cvd_qty += s * q
        cvd_notional += s * n
        total_notional += n
        if s > 0:
            buy_qty += q
            buy_notional += n
        else:
            sell_qty += q
            sell_notional += n
        if r.get("source") == "REST":
            rest_n += 1
        else:
            legacy_n += 1
        manifest_items.append((ts, r.get("order_key"), n, s))
    manifest_items.sort(key=lambda x: (x[0], str(x[1])))
    manifest = hashlib.sha256(
        json.dumps([[m[0], m[2], m[3]] for m in manifest_items]).encode()
    ).hexdigest()
    n_rows = legacy_n + rest_n
    return {
        "source_row_count": n_rows,
        "legacy_row_count": legacy_n,
        "repair_row_count": rest_n,
        "cvd_qty": cvd_qty if n_rows else None,
        "cvd_notional": cvd_notional if n_rows else None,
        "total_notional": total_notional if n_rows else None,
        "taker_buy_qty": buy_qty if n_rows else None,
        "taker_sell_qty": sell_qty if n_rows else None,
        "taker_buy_notional": buy_notional if n_rows else None,
        "taker_sell_notional": sell_notional if n_rows else None,
        # normalized CVD is mathematically defined only for a strictly
        # positive denominator -- never coerced to 0 on an empty window
        "normalized_cvd": (cvd_notional / total_notional) if total_notional > 0 else None,
        "source_row_manifest_sha256": manifest,
    }


def compute_proxy_window_flow(candles, window_start_ts_ms: int, window_end_ts_ms: int) -> dict:
    """PROXY layer (descriptive-only): 1m candles FULLY contained in
    [T-W, T]: open_ts_ms >= window_start AND close_ts_ms <= window_end.
    Candle availability respects candle-close time by construction (a candle
    whose close_ts_ms > T is NEVER included, so no end-of-minute leakage).
    Zero contained candles -> values are NULL (undefined), never fabricated 0.
    """
    used = []
    for c in candles:
        o, cl = int(c["open_ts_ms"]), int(c["close_ts_ms"])
        if c.get("taker_buy_volume") is None or c.get("taker_sell_volume") is None:
            # a candle without taker fields carries NO taker evidence --
            # it is not proxy-containable (fail-closed, never treated as 0)
            continue
        if o >= window_start_ts_ms and cl <= window_end_ts_ms:
            used.append(c)
        elif cl > window_end_ts_ms and o <= window_end_ts_ms:
            # partially-formed / straddling candle: excluded, and its
            # exclusion is exactly why P1 is never equivalent to Q2
            continue
    used.sort(key=lambda c: int(c["open_ts_ms"]))
    if not used:
        return {
            "contained_candle_count": 0, "last_contained_close_ts_ms": None,
            "proxy_cvd_qty": None, "proxy_taker_buy_qty": None,
            "proxy_taker_sell_qty": None, "candle_versions": json.dumps([]),
            "source_row_manifest_sha256": hashlib.sha256(b"[]").hexdigest(),
        }
    buy = sum(float(c["taker_buy_volume"]) for c in used)
    sell = sum(float(c["taker_sell_volume"]) for c in used)
    manifest = hashlib.sha256(json.dumps(
        [[int(c["open_ts_ms"]), float(c["taker_buy_volume"]), float(c["taker_sell_volume"])]
         for c in used]).encode()).hexdigest()
    return {
        "contained_candle_count": len(used),
        "last_contained_close_ts_ms": max(int(c["close_ts_ms"]) for c in used),
        "proxy_cvd_qty": buy - sell,
        "proxy_taker_buy_qty": buy,
        "proxy_taker_sell_qty": sell,
        "candle_versions": json.dumps(sorted({str(c.get("candle_definition_version")) for c in used})),
        "source_row_manifest_sha256": manifest,
    }


def _feature_id(signal_id: str, window_id: str, layer: str) -> str:
    key = f"{signal_id}|{window_id}|{FEATURE_DEFINITION_VERSION}|{layer}"
    return "CVDF-" + hashlib.sha256(key.encode()).hexdigest()[:24]


_EXACT_CONTENT_FIELDS = (
    "window_start_ts_ms", "window_end_ts_ms", "source_row_count", "legacy_row_count",
    "repair_row_count", "cvd_qty", "cvd_notional", "total_notional", "taker_buy_qty",
    "taker_sell_qty", "taker_buy_notional", "taker_sell_notional", "normalized_cvd",
    "source_row_manifest_sha256", "source_regime_ids", "repair_method",
)


def insert_exact_feature_row(conn, row: dict, *, provenance: str, schema_version: int = 1) -> str:
    """Immutable insert: new identity -> INSERT; identical rewrite -> NOOP;
    same identity + different content -> ImmutableCvdFeatureConflict."""
    if int(row["window_end_ts_ms"]) != int(row["signal_birth_ts"]):
        raise KnownAtViolation("window_end_ts_ms != signal_birth_ts")
    fid = _feature_id(row["signal_id"], row["window_id"], EVIDENCE_EXACT)
    existing = conn.execute(
        "SELECT " + ", ".join(_EXACT_CONTENT_FIELDS) +
        " FROM ami_cvd_windowed_flow WHERE signal_id=? AND window_id=? AND feature_definition_version=?",
        (row["signal_id"], row["window_id"], FEATURE_DEFINITION_VERSION)).fetchone()
    new_content = tuple(row.get(f) for f in _EXACT_CONTENT_FIELDS)
    if existing is not None:
        if tuple(existing) != new_content:
            raise ImmutableCvdFeatureConflict(
                f"{row['signal_id']}/{row['window_id']}: existing exact feature row has different content")
        return "NOOP_IDENTICAL"
    conn.execute(
        "INSERT INTO ami_cvd_windowed_flow (feature_id, feature_definition_version,"
        " raw_interpretation_version, quality_contract_version, signal_id, source_event_id,"
        " independent_cycle_id, symbol, signal_birth_ts, window_id, window_start_ts_ms,"
        " window_end_ts_ms, evidence_layer, source_row_count, legacy_row_count, repair_row_count,"
        " cvd_qty, cvd_notional, total_notional, taker_buy_qty, taker_sell_qty,"
        " taker_buy_notional, taker_sell_notional, normalized_cvd, source_row_manifest_sha256,"
        " source_regime_ids, repair_method, repair_population_version, feature_available_ts_ms,"
        " known_at_classification, schema_version, provenance, created_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (fid, FEATURE_DEFINITION_VERSION, RAW_INTERPRETATION_VERSION, QUALITY_CONTRACT_VERSION,
         row["signal_id"], row["source_event_id"], row.get("independent_cycle_id"), row["symbol"],
         row["signal_birth_ts"], row["window_id"], row["window_start_ts_ms"], row["window_end_ts_ms"],
         EVIDENCE_EXACT, row["source_row_count"], row["legacy_row_count"], row["repair_row_count"],
         row["cvd_qty"], row["cvd_notional"], row["total_notional"], row["taker_buy_qty"],
         row["taker_sell_qty"], row["taker_buy_notional"], row["taker_sell_notional"],
         row["normalized_cvd"], row["source_row_manifest_sha256"], row["source_regime_ids"],
         row["repair_method"],
         REPAIR_POPULATION_VERSION if row["repair_method"] != "NONE" else None,
         row["signal_birth_ts"], "KNOWN_AT_SAFE", schema_version, provenance,
         int(time.time() * 1000)))
    return "INSERTED"


def insert_proxy_feature_row(conn, row: dict, *, provenance: str, schema_version: int = 1) -> str:
    fid = _feature_id(row["signal_id"], row["window_id"], EVIDENCE_PROXY)
    existing = conn.execute(
        "SELECT contained_candle_count, proxy_cvd_qty, proxy_taker_buy_qty, proxy_taker_sell_qty,"
        " source_row_manifest_sha256 FROM ami_cvd_windowed_flow_proxy"
        " WHERE signal_id=? AND window_id=? AND feature_definition_version=?",
        (row["signal_id"], row["window_id"], FEATURE_DEFINITION_VERSION)).fetchone()
    new_content = (row["contained_candle_count"], row["proxy_cvd_qty"],
                   row["proxy_taker_buy_qty"], row["proxy_taker_sell_qty"],
                   row["source_row_manifest_sha256"])
    if existing is not None:
        if tuple(existing) != new_content:
            raise ImmutableCvdFeatureConflict(
                f"{row['signal_id']}/{row['window_id']}: existing proxy feature row has different content")
        return "NOOP_IDENTICAL"
    conn.execute(
        "INSERT INTO ami_cvd_windowed_flow_proxy (feature_id, feature_definition_version,"
        " quality_contract_version, signal_id, source_event_id, independent_cycle_id, symbol,"
        " signal_birth_ts, window_id, window_start_ts_ms, window_end_ts_ms, evidence_layer,"
        " candle_timeframe, contained_candle_count, last_contained_close_ts_ms, proxy_cvd_qty,"
        " proxy_taker_buy_qty, proxy_taker_sell_qty, candle_versions, source_row_manifest_sha256,"
        " feature_available_ts_ms, known_at_classification, descriptive_only, schema_version,"
        " provenance, created_ms)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (fid, FEATURE_DEFINITION_VERSION, QUALITY_CONTRACT_VERSION, row["signal_id"],
         row["source_event_id"], row.get("independent_cycle_id"), row["symbol"],
         row["signal_birth_ts"], row["window_id"], row["window_start_ts_ms"],
         row["window_end_ts_ms"], EVIDENCE_PROXY, "1m", row["contained_candle_count"],
         row["last_contained_close_ts_ms"], row["proxy_cvd_qty"], row["proxy_taker_buy_qty"],
         row["proxy_taker_sell_qty"], row["candle_versions"], row["source_row_manifest_sha256"],
         row["signal_birth_ts"], "KNOWN_AT_SAFE", 1, schema_version, provenance,
         int(time.time() * 1000)))
    return "INSERTED"


def record_bucket_exclusion(conn, signal_id: str, direction: str) -> None:
    ex_id = "CVDX-" + hashlib.sha256(
        f"{signal_id}|{FEATURE_DEFINITION_VERSION}".encode()).hexdigest()[:24]
    conn.execute(
        "INSERT OR IGNORE INTO ami_cvd_bucket_exclusions (exclusion_id,"
        " feature_definition_version, signal_id, direction, reason, schema_version, created_ms)"
        " VALUES (?,?,?,?,?,?,?)",
        (ex_id, FEATURE_DEFINITION_VERSION, signal_id, direction,
         BUCKET_EXCLUSION_REASON, 1, int(time.time() * 1000)))


def assert_not_pooled(rows) -> None:
    """Fail-closed pooling guard (contract section 8.3): a fetched population
    containing BOTH evidence layers raises immediately."""
    layers = {r["evidence_layer"] for r in rows}
    if len(layers) > 1:
        raise AssertionError(f"POOLED_EVIDENCE_LAYERS_FORBIDDEN: {sorted(layers)}")


def content_hash_exact(conn) -> str:
    rows = conn.execute(
        "SELECT feature_id, signal_id, window_id, window_start_ts_ms, window_end_ts_ms,"
        " source_row_count, legacy_row_count, repair_row_count, cvd_qty, cvd_notional,"
        " total_notional, taker_buy_qty, taker_sell_qty, taker_buy_notional, taker_sell_notional,"
        " normalized_cvd, source_row_manifest_sha256, source_regime_ids, repair_method"
        " FROM ami_cvd_windowed_flow ORDER BY feature_id").fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()


def content_hash_proxy(conn) -> str:
    rows = conn.execute(
        "SELECT feature_id, signal_id, window_id, contained_candle_count, proxy_cvd_qty,"
        " proxy_taker_buy_qty, proxy_taker_sell_qty, source_row_manifest_sha256"
        " FROM ami_cvd_windowed_flow_proxy ORDER BY feature_id").fetchall()
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()
