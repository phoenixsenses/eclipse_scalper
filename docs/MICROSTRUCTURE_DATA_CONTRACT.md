# Microstructure Data Contract

Date: 2026-03-09
Status: Active baseline contract
Scope: raw collector SQLite artifacts, canonical reader expectations, feature build inputs, and degradation rules

## Purpose

This document defines the canonical microstructure data contract used by:

- `data/microstructure_collector.py`
- `tools/check_data_ready.py`
- `src/microphys/io/sqlite_reader.py`
- `tools/build_micro_features.py`
- `data/features/micro_features.py`

The goal is not to describe an ideal future order-book system. The goal is to standardize the data surface that exists today and make degradation behavior explicit.

## Canonical Artifacts

Primary runtime artifacts:

- `data/microstructure.db`
- `data/event_diary.csv`
- `logs/collector_heartbeat.json`

Primary derived artifact root:

- `data/derived/micro_bars/`

## Symbol Contract

- Symbols must be normalized to canonical uppercase exchange form
- Current canonical examples:
  - `BTCUSDT`
  - `ETHUSDT`
- Producers may receive mixed-case or variant input, but persisted and derived records must use canonical symbol form
- Downstream readers should normalize via `canonical_symbol()` before filtering or grouping

## Timestamp Contract

- Raw SQLite timestamps are expected in Unix epoch milliseconds unless a reader explicitly detects text timestamps
- Canonical raw timestamp column name: `ts_ms`
- Accepted reader aliases exist for compatibility, but new producers should prefer `ts_ms`
- Derived feature outputs must expose:
  - `ts_ms`
  - `ts_utc`

## Required Raw Tables

The active collector baseline requires these tables in `data/microstructure.db`.

### `agg_trades`

Required columns:

- `ts_ms` INTEGER NOT NULL
- `symbol` TEXT NOT NULL
- `price` REAL NOT NULL
- `quantity` REAL NOT NULL
- `notional` REAL NOT NULL
- `is_buyer_maker` INTEGER NOT NULL

Semantics:

- One row per aggregated trade event
- `is_buyer_maker = 1` implies sell-side aggression
- `is_buyer_maker = 0` implies buy-side aggression

### `mark_prices`

Required columns:

- `ts_ms` INTEGER NOT NULL
- `symbol` TEXT NOT NULL
- `mark_price` REAL NOT NULL
- `funding_rate` REAL NULL
- `next_funding_time_ms` INTEGER NULL

Semantics:

- Mark price is the current baseline mid proxy
- This is not a true top-of-book feed

### `liquidations`

Required columns:

- `ts_ms` INTEGER NOT NULL
- `symbol` TEXT NOT NULL
- `side` TEXT NOT NULL
- `price` REAL NOT NULL
- `quantity` REAL NOT NULL
- `notional` REAL NOT NULL
- `trade_time_ms` INTEGER NOT NULL

Semantics:

- One row per liquidation event
- `side` indicates buy/sell liquidation side as emitted by the source stream

## Optional Raw Tables

There is currently no required true top-of-book table.

If a future producer adds one, the preferred fields are:

- `ts_ms`
- `symbol`
- `bid_px`
- `ask_px`
- `bid_qty`
- `ask_qty`

If present, the reader may prefer it over `mark_prices` for book-derived features.

## Reader Mapping Contract

`src/microphys/io/sqlite_reader.py` performs schema discovery with compatibility aliases, but the canonical kinds are:

- `trades`
- `book`
- `liquidations`

Current baseline mapping behavior:

- `agg_trades` -> `trades`
- `mark_prices` -> `book` fallback via `mid`/mark proxy
- `liquidations` -> `liquidations`

Important implication:

- the current system usually has a usable `mid`
- it usually does not have true `bid_px`, `ask_px`, `bid_qty`, `ask_qty`

## Derived Feature Contract

### Canonical per-record fields

The schema-agnostic feature layer in `data/features/micro_features.py` expects or emits:

- `ts_ms`
- `symbol`
- `mid`
- `spread`
- `imbalance`
- `trade_intensity`
- `micro_volatility`
- `ret_1`

### Canonical micro-bar fields

`tools/build_micro_features.py` emits the current micro-bar baseline:

- `ts_ms`
- `ts_utc`
- `symbol`
- `mid`
- `spread`
- `microprice`
- `buy_qty`
- `sell_qty`
- `trade_count`
- `qty_sum`
- `vwap`
- `ofi`
- `ofi_norm`
- `trade_intensity_qty_per_sec`
- `trade_intensity_trades_per_sec`
- `top_depth_imbalance`
- `rv_short`
- `liq_count`
- `liq_qty`
- `liq_sell_qty`
- `liq_buy_qty`
- `liq_imbalance`
- `liq_rate_per_sec`
- `bid_px`
- `ask_px`
- `bid_qty`
- `ask_qty`

### Bucket contract

- Buckets are deterministic
- Bucket resolution is controlled by `interval_ms`
- The current research baseline commonly uses sub-second to one-second style windows depending on the tool
- Each output row must map to exactly one deterministic bucket timestamp

## Degradation Rules

This contract explicitly allows degraded operation when no true top-of-book feed exists.

### Allowed degraded behavior

If no true book table exists:

- `mid` may fall back to `mark_price`
- `spread` may fall back to a proxy
- `microprice` may collapse toward `mid`
- `top_depth_imbalance` may be empty, weak, or synthetic

### Not allowed

The following must not happen silently:

- treating mark price as if it were proven bid/ask depth
- claiming true depth-derived confidence when only mark-price proxy exists
- mixing symbol variants in the same derived dataset
- emitting non-canonical timestamps without explicit conversion

## Freshness and Readiness Contract

Operational readiness is currently checked by `tools/check_data_ready.py`.

Baseline expectations:

- `data/microstructure.db` exists
- `data/event_diary.csv` exists
- at least one key symbol has fresh rows within the configured freshness window
- at least one key table with symbol and timestamp columns is fresh

This is a liveness/readiness contract, not a full research-fitness contract.

## Invariants

These invariants should hold for any compliant dataset:

- symbols are canonical uppercase
- timestamps are monotonically non-decreasing within a single sorted output stream
- `mid > 0` whenever emitted
- `spread >= 0` whenever emitted
- `trade_count >= 0`
- quantity and notional fields are non-negative
- derived rows are deterministic for the same raw window and configuration

## Producer Responsibilities

Collector-side producers must:

- write canonical symbols
- write millisecond timestamps
- preserve raw source event ordering as much as practical
- avoid partial schema drift without updating this contract

## Consumer Responsibilities

Reader/build consumers must:

- normalize symbols before querying
- handle missing true book depth explicitly
- avoid overstating confidence for degraded book-derived metrics
- preserve deterministic bucketization

## Known Gaps

These are not contract violations today, but they remain open follow-up work:

- no canonical true top-of-book feed in the active collector baseline
- no deterministic sample DB fixture yet
- no formal research fitness validator yet
- symbol canonicalization cleanup is not fully closed across all code paths

## Recommended Next Steps

1. Add a deterministic sample DB fixture matching this contract
2. Add a research fitness validator that checks this contract directly
3. Promote true top-of-book support to remove proxy/degraded behavior where possible
