# E-DER Prospective Microstructure Collector Contract V1

**Date:** 2026-08-19  
**Status:** `SPECIFICATION ONLY — IMPLEMENTATION NOT AUTHORIZED`  
**Purpose:** Forward Track B research-grade data contract

This contract defines what a future collector must preserve so prospective E-DER events are not limited to the historical proxy-only archive. It does not modify or authorize modification of any collector, database, schema, configuration, or live process.

## 1. Architecture principles

The architecture is `RAW FIRST`, `APPEND ONLY`, `VERSIONED`, `PROVENANCE PRESERVING`, `SEQUENCE AUDITABLE`, `MULTI-SYMBOL`, and `FORWARD CONFIRMATORY READY`.

Canonical flow:

`RAW IMMUTABLE DATA → VERIFIED NORMALIZED TABLE → DERIVED FEATURES`

Never retain only derived features. Every derived field must be reproducible from accepted raw records and versioned transformations.

## 2. Universe and synchronized coverage

The active E-DER research universe must be supplied to collectors through a versioned, timestamped universe artifact. It must reflect the candidate universe active at collection time, not a hard-coded historical 12-symbol list or BTC/ETH/SOL subset.

Every candidate event symbol requires synchronized core-stream coverage. BTC/common-market benchmark coverage remains mandatory. Universe changes create a new immutable version with effective-from time, source, reason, checksum, and collector acknowledgement.

## 3. Core stream A — forceOrder raw retention

Preserve every received forceOrder message before normalization, including:

- complete raw JSON and unfamiliar/future fields;
- top-level event time `E` and order trade time `T`;
- symbol, side, order type, time in force;
- order price `p`, original quantity `q`, average price `ap`, last filled quantity `l`, accumulated filled quantity `z`, status `X`, when present;
- local receive wall-clock time and, where feasible, local monotonic process time;
- stream/endpoint, source transport, connection/session ID;
- collector name/version, Git commit SHA, schema version, universe version;
- raw payload hash.

No field may be discarded merely because current research does not use it. Retention does not imply that forceOrder is a complete executed-liquidation tape; upstream throttling/censoring remains a semantic limitation.

## 4. Core stream B — aggTrade

Preserve at minimum:

- `E`, `T`, symbol, aggregate ID `a`, price, quantity, first/last trade IDs `f/l`, buyer-maker `m`;
- `nq` if present and every future/RPI-related field;
- raw JSON, wall/monotonic receive timestamps;
- `source=WS|REST`, endpoint, collector/schema/Git versions, connection/session ID, universe version and payload hash.

One aggTrade is not one taker order.

For REST fallback, preserve request interval, request/cursor IDs, response time, pagination and fallback reason. Never merge REST and WS rows without recoverable source identity. Preserve aggregate IDs so dedupe and continuity are deterministic and auditable.

## 5. Core stream C — sequence-valid incremental L2

Top-of-book bookTicker alone is insufficient for OFI, MLOFI, displayed-book transition, or replenishment research. The Level-A objective is raw incremental L2 per candidate symbol.

Preserve:

- complete raw diff-depth payload;
- exchange clock(s) available to the message;
- symbol and sequence fields `U/u/pu` or their venue equivalents;
- full bid and ask update arrays;
- wall/monotonic receive timestamps;
- source, endpoint, connection/session, collector version, Git SHA, schema version, universe version and payload hash.

### Reconstruction protocol

1. Open stream and buffer raw updates.
2. Obtain an authoritative snapshot with request provenance.
3. Apply buffered updates only under the exchange’s sequence contract.
4. Validate continuity for every update.
5. On a gap, invalidate the reconstructed book from the last proven state.
6. Log the gap/recovery and perform mandatory snapshot re-bootstrap.
7. Never silently forward-fill missing depth state.

Persist enough metadata to distinguish raw updates, reconstructed displayed book, and any sampled reconstructed book. Quantity removal is not called a cancellation unless execution versus cancellation cause is actually identifiable.

## 6. bookTicker cross-check

bookTicker may be retained independently as Research Bible Level C. Preserve raw bid/ask prices and quantities, available update IDs/times, all payload fields, receive clocks, source/session/version metadata and raw hash. It is a cross-check, never a replacement for sequence-valid L2.

## 7. Dated symbol metadata

At collector start and at a predefined periodic/checkpoint cadence, append a complete dated exchange metadata snapshot containing:

- raw exchangeInfo response and timestamp;
- tick size, step size, min/max price and quantity;
- price/quantity precision;
- contract status/type, onboard/listing date;
- all filters and multiplier-related fields;
- collector/schema/Git and universe versions;
- response hash and request provenance.

When metadata changes, append a new version. Never overwrite historical metadata in place or substitute today’s values for a prior event.

## 8. Optional second-tier derivatives context

Desirable but non-core context includes mark price, index price, funding, derivable basis, open interest, and BTC/common-market state. Each source follows the same raw/provenance/clock rules. Failure of optional context must not block, corrupt, or silently degrade core forceOrder/trade/L2 collection.

## 9. Clock contract

Every high-frequency source preserves both exchange-provided clock(s) and local receive wall-clock time. Preserve a monotonic process clock where feasible. Never replace an exchange clock with receive time, discard one after deriving another, or hide fallback clock selection.

The stored clocks and connection events must permit later estimation of feed delay, clock disagreement, ordering uncertainty, and reconnect effects. Host synchronization state and clock-monitor receipts should be versioned separately.

## 10. Record and partition provenance

Every record or immutable partition must trace to:

- collector name/version and Git SHA;
- schema version;
- stream/endpoint and WS/REST source;
- connection/session;
- collection start/end;
- exchange symbol;
- universe version;
- raw payload or partition checksum.

No future row should have unknown collector semantics because version metadata was discarded.

## 11. Source-aware gap and recovery system

Integrity accounting must include:

- heartbeat and last-event monitoring;
- sequence-gap detection where sequence semantics exist;
- disconnect/reconnect and subscription acknowledgement logs;
- snapshot/re-bootstrap and recovery status;
- REST fallback intervals and provenance;
- expected-versus-observed partition coverage where meaningful;
- explicit invalid intervals;
- duplicate counts and deterministic dedupe method;
- source-specific gap semantics.

Sparse event-driven streams such as forceOrder cannot treat absence of messages as automatic outage. Sequence streams, periodic streams and sparse streams require different gap rules. A gap registry must state what it can and cannot prove.

## 12. Storage and acceptance lifecycle

Raw storage is append-only, deterministically partitioned, checksummed and accompanied by manifests, rotation receipts, schema versions and validation receipts. Accepted raw partitions are immutable. Any repair requires a separate documented protocol that preserves the original, records before/after hashes, reason, authority and replacement lineage.

Normalized tables must retain the raw-record key/hash and transformation version. Derived features must retain normalized inputs, code version, parameter contract and quality state.

## 13. Future data levels

- **LEVEL A:** sequence-valid incremental L2 plus full trade/forceOrder timing and provenance. Supports OFI/MLOFI, displayed-book state transitions and better reaction-turnover analysis.
- **LEVEL B:** multi-level snapshots without event continuity. Supports weaker book-state analysis only.
- **LEVEL C:** bookTicker/top-of-book only. Supports L1 state only.

The target for active E-DER candidate symbols is Level A wherever feasible. Level A still does not automatically identify hidden liquidity, participant identity, market-maker behavior, or causality.

## 14. Research-grade acceptance tests

Before a new interval is called research-grade, validation must pass and emit immutable receipts for:

1. expected schema and unfamiliar-field retention;
2. raw-to-normalized row/field parity;
3. exchange and receive timestamp preservation;
4. payload-hash reproducibility;
5. forced reconnect behavior;
6. sequence-gap detection, invalidation and re-bootstrap;
7. REST fallback provenance and deterministic dedupe;
8. restart/session continuity;
9. partition completeness and checksum manifest;
10. duplicate handling;
11. candidate-universe coverage and change propagation;
12. metadata snapshot/version change handling;
13. optional-context failure isolation from core streams.

No profitability, E-DER return, win-rate, residual, or alpha statistic is an acceptance test.

## 15. Scientific boundary

Validated Track B data may support aggressive-flow surprise, quote-mid response, classical OFI, MLOFI, displayed LOB resilience, reaction turnover, add/remove displayed quantity dynamics, timing analysis and tick-regime conditioning under separately frozen definitions.

It does not by itself prove hidden liquidity, actor identity, absorption, seller exhaustion, market-maker conduct, or causality. Those claims need additional identification and falsification designs.

## 16. Implementation stage gate

Implementation remains `STOP`. A separate authorization must specify architecture, storage budget, retention, operational isolation, failure behavior, rollout, validation environment and rollback. No current collector, production configuration, market-data store, or live process may be changed under this document.
