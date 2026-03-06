# Research Microstructure Audit

## Scope

This audit covers the current research-side microstructure data path in the repository:

- writer layer in `data/`
- readiness/probe tools in `tools/`
- feature build path in `tools/build_micro_features.py`
- schema discovery in `src/microphys/io/sqlite_reader.py`
- current checked-in derived artifacts under `data/derived/`

Audit date: 2026-03-06

## Current Surface

The active research surface is not a top-level `features/` directory. The repo currently uses:

- `data/`
- `tools/`
- `src/microphys/`
- `strategies/`

Relevant paths:

- `data/microstructure_collector.py`
- `data/event_diary.py`
- `data/quality.py`
- `tools/check_data_ready.py`
- `tools/ingestion_check.py`
- `tools/data_layer_probe.py`
- `tools/db_introspect.py`
- `tools/build_micro_features.py`
- `src/microphys/io/sqlite_reader.py`
- `data/features/micro_features.py`

## Producer -> Artifact -> Consumer Map

### 1. Live collector

Producer:
- `data/microstructure_collector.py`

Primary artifact:
- `data/microstructure.db`

Tables written:
- `agg_trades`
- `mark_prices`
- `liquidations`

Additional runtime artifacts:
- `logs/collector_heartbeat.json`
- overall/component health via `tools.health_state`

Main downstream consumers:
- `tools/check_data_ready.py`
- `tools/ingestion_check.py`
- `tools/data_layer_probe.py`
- `tools/db_introspect.py`
- `tools/build_micro_features.py`
- many research/runtime tools reading `data/microstructure.db`

### 2. Event diary

Producer:
- `data/event_diary.py`

Input:
- `data/microstructure.db`

Artifact:
- `data/event_diary.csv`

Purpose:
- structured observation log, not a trading signal source

Main downstream consumers:
- `tools/check_data_ready.py`
- `tools/data_layer_probe.py`
- `tools/verify_data_layer.py`

### 3. Data readiness / liveness checks

Tools:
- `tools/check_data_ready.py`
- `tools/ingestion_check.py`
- `tools/data_layer_probe.py`
- `tools/db_introspect.py`

These do not produce features. They validate:

- db exists
- csv exists
- timestamps are fresh
- row counts progress
- collector heartbeat is fresh
- schema and likely core tables are discoverable

### 4. Feature build path

Producer:
- `tools/build_micro_features.py`

Input:
- `data/microstructure.db`

Reader:
- `src/microphys/io/sqlite_reader.py`

Derived artifact root:
- `data/derived/micro_bars/`

Per-run outputs:
- parquet partitions
- `manifest.json`

Observed checked-in derived branches also exist under:
- `data/derived/alpha_candidates/`
- `data/derived/alpha_eval/`
- `data/derived/execution_calibration/`
- `data/derived/impact/`
- `data/derived/physics/`
- `data/derived/physics_regime_metrics/`
- `data/derived/propagator/`
- `data/derived/regimes/`
- `data/derived/state/`

## Schema Contract

### Collector schema

`data/microstructure_collector.py` writes:

`agg_trades`
- `ts_ms`
- `symbol`
- `price`
- `quantity`
- `notional`
- `is_buyer_maker`

`mark_prices`
- `ts_ms`
- `symbol`
- `mark_price`
- `funding_rate`
- `next_funding_time_ms`

`liquidations`
- `ts_ms`
- `symbol`
- `side`
- `price`
- `quantity`
- `notional`
- `trade_time_ms`

### Feature reader contract

`src/microphys/io/sqlite_reader.py` discovers tables heuristically:

- `trades`
- `book`
- `liquidations`

Important detail:
- the current collector does not write a real top-of-book table
- `mark_prices` is therefore selected as the `book` source
- the reader maps `mark_price` into the `mid` field
- `bid_px`, `ask_px`, `bid_qty`, `ask_qty` are usually absent

### Practical consequence

`tools/build_micro_features.py` still works, but with degraded market microstructure richness:

- `mid` is usually available
- `spread` often falls back to a proxy
- `microprice` collapses toward `mid`
- `top_depth_imbalance` is usually weak or unavailable

This means the current pipeline is closer to:
- trade flow + mark-price features

than to:
- full order-book microstructure features

## Repository State Observed

Checked-in live files under `data/` do not currently include:

- `data/microstructure.db`
- `data/event_diary.csv`

Checked-in live files do include:

- `data/live/online_plan.json`

Checked-in derived artifacts exist under `data/derived/`.

This means research can inspect code and derived outputs immediately, but a fresh local ingestion loop still requires either:

1. running the collector and diary locally, or
2. creating a deterministic sample sqlite fixture for research tests

## Immediate Findings

### Finding 1: Planned ownership path differs from actual repo layout

The original collaboration split assumed a top-level `features/` directory.

Actual repo state:
- no top-level `features/`
- research feature code currently lives across:
  - `data/features/`
  - `tools/`
  - `src/microphys/`
  - `core/`

Impact:
- ownership should be described by functional paths, not by the original folder list alone

### Finding 2: Collector schema and feature builder are only partially aligned

The builder expects a discoverable `book` source.

Current writer only provides:
- aggregated trades
- mark price
- liquidations

Impact:
- spread, microprice, and depth imbalance are partly synthetic or empty
- any signal assuming real top-of-book depth should be treated as lower-confidence until a true book feed exists

### Finding 3: Readiness tools are stronger than the data contract itself

The repo has several liveness checks:
- `check_data_ready`
- `ingestion_check`
- `data_layer_probe`
- `db_introspect`

But there is no single canonical research doc that states:
- required tables
- required columns
- acceptable freshness
- which downstream features degrade when book data is missing

Impact:
- operational freshness is checked
- semantic research fitness is not yet documented centrally

### Finding 4: Derived outputs exist, but source lineage is not documented in one place

There are many `data/derived/*` outputs already checked in.

Impact:
- it is not yet obvious which generated dataset depends on:
  - raw collector tables
  - built micro bars
  - labels
  - later-stage evaluation scripts

## Recommended Research Start

### Phase 1: Ingestion contract hardening

First target:
- define the canonical research input contract for `data/microstructure.db`

Minimum contract:
- `agg_trades`, `mark_prices`, `liquidations` must exist
- timestamp column freshness thresholds must be explicit
- symbol normalization must be explicit
- research tools must declare whether they require:
  - mark-only data
  - trade+mark data
  - true top-of-book data

### Phase 2: Feature capability labeling

Mark each microstructure feature as one of:

- `mark_only`
- `trade_flow`
- `trade_plus_liq`
- `requires_book`

This prevents accidental misuse of synthetic proxies as real order-book features.

### Phase 3: Sample fixture path

Add a deterministic research fixture for:
- `data/microstructure.db`

Reason:
- local research iteration should not require live Binance collection every time
- tests for feature generation, no-lookahead checks, and validation become easier to reproduce

## First Concrete Tasks

1. Add a research-facing microstructure contract doc:
- required tables
- required columns
- freshness rules
- feature degradation rules

2. Add a contract validator focused on research fitness, not only liveness:
- table presence
- required columns
- symbol normalization
- data coverage by symbol
- whether true book fields exist

3. Label current features by dependency level:
- mark-only
- trade-only
- liq-assisted
- book-required

4. Add a deterministic sqlite fixture for microstructure research tests.

## Working Assumption For Person 1

Until a true order-book writer exists, the research lane should treat the live microstructure source as:

- strong for trade flow and liquidation studies
- acceptable for mark-price and funding context
- weak for book-depth and microprice research

That should drive feature prioritization for the next research tasks.
