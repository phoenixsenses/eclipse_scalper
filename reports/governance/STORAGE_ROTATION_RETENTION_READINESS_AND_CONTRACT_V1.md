# STORAGE_ROTATION_RETENTION_READINESS_AND_CONTRACT_V1

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-READINESS-AND-CONTRACT-V1
**Nature:** Operational-readiness, source-inventory, and storage-contract design only. No deletion, no archival, no VACUUM, no collector change, no schema change, no experiment/nullifier/gate-receipt, no outcome access.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## 0. Accepted checkpoint / portfolio context

Latest research-governance state (unaffected by this batch): `schema_version=14`, `experiment_registry=24`, `experiment_results=381`, `epistemic_test_nullifiers=2`, `experiment_gate_receipts=2`. Portfolio disposition (commit `c8c2156f`): `NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY` — `FAM_BOOK_SPREAD_DYNAMICS` LONG parked for sample growth (commit `93b7296d`), `FAM_SPOT_PERP_BASIS_REVERSAL` blocked by coverage (`1630f0a1`), `FAM_CASCADE_ABSORPTION_IMPACT` closed (`5e9e2e33`/`ba3ab906`). This batch does not reopen, touch, or depend on any of that scientific work — it exists because the raw storage estate is large and growing, independent of which research family is currently active.

The deeper alpha program (state transitions, exit/trade management, OI/funding intelligence, cross-instrument generalization, forward/live observation, and future independently-governed families) remains active and is not cancelled by this batch's scope.

---

## Phase 1 — Data-estate inventory

| Root | Purpose | Size | Live/concurrent writes | Journal mode | Backup status | May be copied | May be vacuumed | May ever be pruned |
|---|---|---|---|---|---|---|---|---|
| `data/microstructure.db` | primary raw collector estate (agg_trades/book_ticker/mark_prices/liquidations/open_interest/funding_rates/spot_prices/event_diary/gaps/etc.) | **758,774,398,976 bytes (~706.4 GiB)** at inspection end | **YES** — 4 active collectors write continuously (confirmed: WAL grew 5.27MB→5.80MB, file grew ~236MB, during this batch's ~24-minute inspection window) | WAL | none (too large for full backup; `MIGRATION_LOG.md`/`REPOSITORY_RUNTIME_AUDIT.md` note this is by design) | NO (this batch made zero copies; storage guardrail forbids it) | not in this batch; future maintenance gate only, requires downtime contract (Phase 11) | per-table, per Phase 3 classification — never wholesale |
| `data/ami/canonical.sqlite` | governed AMI research warehouse | 223,117,312 bytes (~212.8 MiB) | occasional, only during authorized migration/preregistration batches (none ran concurrently with this audit) | WAL, `auto_vacuum=0` (NONE) | 39 accepted backups under `data/ami/backups/` (2.8GB total) | via existing accepted transaction discipline only | not in this batch | **NEVER** (CANONICAL_IMMUTABLE) |
| `data/ami/knowledge.sqlite` | epistemic-gate/graveyard/nullifier/receipt store | 110,592 bytes (108 KiB) | same discipline as canonical | WAL, `auto_vacuum=0` | included in the same 39-backup set (paired snapshots, e.g. `knowledge_pre_G2_governed_execution_20260706.sqlite`) | via existing accepted transaction discipline only | not in this batch | **NEVER** (CANONICAL_IMMUTABLE) |
| `data/ami/backups/` | accepted pre-migration backup snapshots | 2.8 GB, 39 files | append-only, grows one pair per migration batch | n/a (flat files) | is itself the backup | already copies, not re-copied | n/a | **NEVER** without a dedicated, separately-authorized retention decision (out of scope here) |
| `.runtime_temp/` | disposable rehearsal/dry-run scratch | 3.7 MB | ephemeral, batch-scoped | n/a | none needed (disposable by design) | freely | n/a | routinely, per each batch's own cleanup discipline (already practiced this session) |
| `.pytest_temp/` | pytest scratch (project-root convention, distinct from the OS-temp scratchpad this session actually used) | 0 bytes (empty/absent) | ephemeral | n/a | none | freely | n/a | routinely |
| `runtime/` | live collector/executor checkpoints, state files, locks | 2.9 GB — but **2.8 GB of this is `runtime/chrome_user_data_copy`**, an unrelated browser-profile copy, **not part of the Eclipse Scalper data estate** (flagged, not touched, out of this batch's scope) | live state files, small | n/a | n/a | n/a | n/a | `chrome_user_data_copy` is a candidate for a *separate*, unrelated cleanup — not addressed here |
| `logs/` | collector/detector/paper/telemetry logs | 632 MB (`logs/archive` 358MB, `detector_runner.log` 71MB, `telemetry.jsonl` 38MB, etc.) | append-only, live | n/a | none | freely | n/a | age-based rotation is a reasonable future candidate, **not this batch's scope** (logs are not raw research source data) |
| `reports/` | this project's governance/research report corpus (including this batch's own artifacts) | 162 MB | append-only | n/a | version-controlled (git) | freely | n/a | never (accepted evidence) |
| `data/*.db` (ancillary) | `funding_history.db` (48MB, dead since 2026-05-12), `oi_history.db` (9MB, dead since 2026-05-14), `paper_trades.db` (70KB), `risk_state.db` (24KB), `s34_feature_factory.db` (1.5MB), `s34_intelligence.db` (28MB, live) | see Phase 2/3 | mixed (2 dead, 4 live/small) | mixed WAL | none dedicated | n/a | n/a | see Phase 3 per-table classification |
| Stray test-scratch DBs (`data/test_s34_gates_micro_*.db`, `test_s34_gates_trades_*`, `test_s34_micro_*`, `test_s34_old_micro_*`, `test_s34_report_micro_*`, `test_s34_report_trades_*`, `test_status_snapshot_*`, `test_tmp_logger.db`) | leftover pytest fixtures written directly under `data/` instead of `--basetemp` (pre-existing test-hygiene gap) | **416 files, ~13 MB total** | none (orphaned) | n/a | none | freely | n/a | **eligible for a future, separate, low-risk cleanup batch — not deleted here** |

Drive capacity (`D:`, hosting the entire project): **1.9 TB total, ~814 GB used, ~1.1 TB (58.9%) free** at inspection time.

No full database copy was created to produce this inventory (confirmed: `git status` shows no new large file; storage report at the end of this document lists every byte written).

---

## Phase 2 — Active database table inventory (`data/microstructure.db`)

All measurements below are bounded, index-backed queries (`MAX(id)` on the `INTEGER PRIMARY KEY AUTOINCREMENT` column, or `MIN(ts_ms)`/`MAX(ts_ms)` on an indexed column) — **no full-table `COUNT(*)` or unindexed scan was run.** Where a combined multi-table query timed out under live-write lock contention, each table was re-queried individually with a short timeout, which succeeded for all 8 re-attempted tables.

| Table | Purpose / producer | Row grain | PK | Timestamp fields | Symbol/venue | Approx. row count (upper bound) | Earliest ts (UTC) | Latest ts (UTC) | Span (days) | Append-only | Indexes | Downstream/research consumers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `agg_trades` | trade-tick collector | 1 row/trade | `id` AUTOINCREMENT | `ts_ms` (receipt) | `symbol` | ~393,648,692 | 2026-02-15T14:26:27.967Z | 2026-07-07T17:18:32.218Z | 142.1 | yes (repairs append corrections, don't rewrite) | `idx_trade_ts`, `idx_trade_symbol_ts` | `ami_agg_trades_repaired` (canonical, already materialized); CVD/absorption feature families |
| `book_ticker` | L1 best-bid/ask collector | 1 row/quote update | `id` AUTOINCREMENT | `ts_ms` (receipt) | `symbol` | ~4,791,706,520 (all symbols; ETHUSDT-only count from the accepted spread readiness audit was 2,077,780,064) | 2026-04-11T17:08:41.948Z | 2026-07-07T17:18:34.252Z | 87.0 | yes | `idx_bt_symbol_ts`, `idx_bt_ts` | `FAM_BOOK_SPREAD_DYNAMICS` (M-0036 canonical tables; also the still-open LONG sample-accrual path) |
| `mark_prices` | mark-price/funding collector | 1 row/tick | `id` AUTOINCREMENT | `ts_ms` (receipt) | `symbol` | ~21,099,827 | 2026-02-15T14:26:28Z | 2026-07-07T17:18:33.001Z | 142.1 | yes | `idx_mark_ts`, `idx_mark_symbol_ts` | funding-rate-adjacent research (funding_rates itself is dead) |
| `liquidations` | liquidation-event collector | 1 row/liquidation | `id` AUTOINCREMENT | `ts_ms` (receipt) | `symbol`, `side` | ~1,328,472 | 2026-02-15T14:30:18.195Z | 2026-07-06T10:06:39.307Z | 140.8 | yes | `idx_liq_ts`, `idx_liq_symbol_ts` | **direct anchor source** for `ami_events`/`ami_signal_lifecycle` — the root of the entire AMI research population |
| `open_interest` | OI poller (`data/oi_spot_poller.py`) | 1 row/(symbol,ts) | `(symbol, ts_ms)` | `ts_ms` | `symbol` | not independently counted (no AUTOINCREMENT id; composite PK) | 2026-03-28T12:00:00Z | 2026-07-07T17:18:01.694Z | 101.2 | yes | PK only | blocks `OPEN_INTEREST` family eligibility (last coverage recheck 15%, OD-012) |
| `funding_rates` | funding collector (dead) | 1 row/(symbol,ts) | `(symbol, ts_ms)` | `ts_ms` | `symbol` | 178 (prior accepted audit) | 2026-02-15T16:00:00Z | 2026-04-13T16:00:00Z | 57.0 (frozen — dead since) | yes, but frozen | PK only | `FUNDING_LEVEL_VELOCITY` family (source-blocked, OD-006) |
| `spot_prices` | spot-price poller | 1 row/(symbol,ts) | `(symbol, ts_ms)` | `ts_ms` | `symbol` | not independently counted | 2026-03-07T16:00:00Z | 2026-07-07T17:18:01.694Z | 122.1 | yes | PK, `idx_spot_symbol_ts` | `FAM_SPOT_PERP_BASIS_REVERSAL` (blocked by coverage, needs continued accrual) |
| `event_diary` | narrative event log | 1 row/event | `id` (no AUTOINCREMENT) | `ts_ms` | `symbol`, `event_type` | ~133,999 | 2026-03-14T00:25:06.058Z | 2026-06-05T15:53:03.004Z | 83.6 | yes | `idx_event_diary_ts_symbol_type` | audit/diagnostic only |
| `gaps` | source-gap ledger | 1 row/gap | `id` AUTOINCREMENT | `start_ts_ms`, `end_ts_ms` | `stream` | 812 (20 agg_trades, 741 liquidations, 51 mark_prices) | — | — | — | yes | `idx_gaps_stream_start` | every repair/quality-contract module (CVD, absorption) reads this |
| `detector_heartbeat` | live collector liveness | 1 row/heartbeat | not inspected further | — | — | small | — | — | — | yes | — | live monitoring only |
| `detector_log` | detector diagnostic log | — | — | — | — | not counted | — | — | — | yes | — | none (diagnostic only) |
| `detector_signals` | legacy detector output | — | — | — | — | not counted | — | — | — | yes | — | superseded by `ami_signal_lifecycle` |
| `basis_reversion_candidates` | ungoverned exploratory table | — | — | — | — | not counted | — | — | — | yes | — | none confirmed active |
| `liq_heatmap` | derived visualization aggregate | — | — | — | — | not counted | — | — | — | derived | — | dashboard only |
| `sol_s35_candidates` | ungoverned exploratory table | — | — | — | — | not counted | — | — | — | yes | — | none confirmed active |
| `vol_state` | live volatility-regime state | — | — | — | — | small | — | — | — | live-updated | — | active detectors |

`microstructure.db` PRAGMA: `page_size=4096`, `page_count=185,191,812` (≈758.5 GB, matches the file size), `freelist_count=0` (no reclaimable free pages — the file is fully packed; any future purge without a `VACUUM`/`incremental_vacuum` will free pages for *reuse by future inserts*, not shrink the file), `journal_mode=WAL`, `auto_vacuum=NONE`.

Some row counts (`open_interest`, `spot_prices`, and the small diagnostic tables) were **not** independently measured this batch beyond MIN/MAX(ts_ms), consistent with "do not perform a full-table COUNT... if it causes material I/O" — honestly marked `not independently counted` rather than fabricated.

---

## Phase 3 — Table retention classification

Full per-table classification (45 entries) is implemented as data in `ami/governance/storage_rotation_retention_readiness_v1.py::TABLE_REGISTRY`, verified by 27 focused tests. Summary by class:

| Class | Count | Policy | Examples |
|---|---|---|---|
| `CANONICAL_IMMUTABLE` | 15 | never deleted by rotation; never rewritten by archive restoration | `ami_signal_lifecycle`, `experiment_registry`, `failure_archive`, `epistemic_test_nullifiers` |
| `CONTINUITY_CRITICAL_ACTIVE` | 5 | never removed merely by age; governed by continuity semantics | `gaps`, `detector_heartbeat`, `vol_state`, `risk_state.db` |
| `RESEARCH_CRITICAL_COMPACT` | 8 | preserved indefinitely; no automatic purge based only on age | `liquidations`, `open_interest`, `funding_rates`, `spot_prices`, `event_diary`, `funding_history.db`, `oi_history.db`, `paper_trades.db` |
| `RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE` | 3 | target 30-day active horizon; older closed partitions archive-eligible; purge only after verification+activation | `agg_trades`, `book_ticker`, `mark_prices` |
| `DERIVED_REBUILDABLE` | 6 | may later be regenerated; deletion still requires an explicit future contract | `detector_log`, `detector_signals`, `basis_reversion_candidates`, `liq_heatmap`, `sol_s35_candidates`, `s34_feature_factory.db` |
| `TEMPORARY_DISPOSABLE` | 8 (patterns covering 416 files) | may be deleted under existing temp-cleanup rules, **not this batch** | the stray `data/test_s34_*.db` scratch files |

No table was left unclassified. `unclassified_tables()` is a fail-closed helper: any table observed in a future audit that isn't in `TABLE_REGISTRY` is reported, never silently treated as purge-eligible.

**Critical disclosure:** `book_ticker` and `mark_prices` — the two `RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE` tables most relevant to active research — are marked `CONDITIONAL` purge-eligibility, not unconditional, precisely because `FAM_BOOK_SPREAD_DYNAMICS` LONG is still accruing sample against `book_ticker`'s ongoing tail, and `mark_prices` is the last surviving funding-adjacent raw source. This directly implements the operator's research-dependency-guard requirement (§6/§9 below).

---

## Phase 4 — Storage growth and capacity audit

| Metric | Value |
|---|---|
| Drive total capacity | 1.9 TB |
| Drive free space (this batch) | ~1.1 TB (58.9%) |
| Drive used space | ~814 GB |
| `microstructure.db` size (this batch, inspection end) | 758,774,398,976 bytes (~706.4 GiB) |
| `microstructure.db` WAL size | 5,800,992 bytes (~5.8 MB), grew from 5,273,632 bytes during this batch's own ~24-minute inspection (live collectors, not this batch's writes) |
| `canonical.sqlite` size | 223,117,312 bytes (~212.8 MiB) |
| `knowledge.sqlite` size | 110,592 bytes (108 KiB) |
| Total accepted backups (`data/ami/backups/`) | 2.8 GB (39 files) |
| `.runtime_temp/` | 3.7 MB |
| `.pytest_temp/` | 0 (empty) |
| Existing Parquet/CSV/compressed archive | **none found** |
| Largest storage consumer (whole `data/`) | 734 GB, dominated by `microstructure.db` |
| Largest non-data consumer | `runtime/chrome_user_data_copy` (2.8 GB, unrelated to this project's data pipeline) |

### Growth-rate estimate (real historical data points, not fabricated)

Two accepted audit snapshots exist for `microstructure.db`:

| Date | Size | Source |
|---|---|---|
| 2026-07-03 20:23 | 684.7 GB | `REPOSITORY_RUNTIME_AUDIT.md` |
| 2026-07-07 20:xx (this batch) | 758.5 GB (706.4 GiB) | this batch's own `ls -la` |

**4-day growth: +73.8 GB → ≈18.5 GB/day.** Confidence: **MODERATE** — only two data points (no 7/14/30-day series exists in the repository), and the interval spans a period of unusually heavy collector/canonical activity (this entire multi-hour research-governance session); the true steady-state rate could differ. No 7-day, 14-day, or 30-day independently-measured growth estimate exists in the repository — **not fabricated here.**

**Disk-used discrepancy (disclosed, not resolved):** the prior audit recorded `D:` at 881 GB used / 982 GB free (2026-07-03); this batch measures 814 GB used / ~1.1 TB free (2026-07-07) — a **decrease** in total used space despite `microstructure.db` growing by ~74 GB, implying roughly ~141 GB of *other* data was freed elsewhere on the drive in the same window (cause not investigated in this batch — out of scope; flagged for operator awareness, not resolved).

### Projected capacity risk

At ≈18.5 GB/day (moderate confidence) against ~1.1 TB currently free: **≈60 days to `STORAGE_WARNING` (20% free / 200GB free, whichever binds first)**, assuming no other consumer changes and no archival begins. This is a rough projection from a 2-point series, presented as a planning input, not a committed forecast.

### Proposed operational thresholds (design only, Phase 4)

| State | %-free threshold | Absolute-free threshold | Permitted automated response | Prohibited automated response |
|---|---|---|---|---|
| `STORAGE_HEALTHY` | >20% | >200GB | none required | — |
| `STORAGE_WARNING` | ≤20% | ≤200GB | operator notification | any deletion, VACUUM, collector change |
| `STORAGE_CRITICAL` | ≤10% | ≤100GB | notification + block optional expensive batch jobs | any deletion, VACUUM, collector change |
| `STORAGE_EMERGENCY` | ≤5% | ≤50GB | notification + block optional jobs + pause nonessential derived-data builds (under a *separately authorized* runtime policy) | **still no automatic deletion of unverified data** |

The state function fails toward the more severe classification when the two inputs (percentage and absolute) disagree — proven by a dedicated test.

---

## Phase 5 — Archive schema contract (draft, not implemented)

For each `RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE` table (`agg_trades`, `book_ticker`, `mark_prices`):

| Element | Value |
|---|---|
| Archive schema version | `v1` (to be assigned at first real implementation) |
| Preserved columns | all source columns verbatim, including `id` (original AUTOINCREMENT PK) |
| Preserved SQL→Parquet types | `INTEGER`→`int64`, `REAL`→`double`, `TEXT`→`string` |
| Timestamp timezone | UTC, explicit (never naive/local) |
| Timestamp precision | millisecond, preserved as `int64` epoch-ms (not converted to a lossy Parquet `timestamp` logical type unless round-trip losslessness is independently proven) |
| Integer precision | 64-bit, no downcasting |
| Null handling | source `NOT NULL` columns remain non-nullable in the Parquet schema; nullable columns preserve nullability |
| Source-row ordering | ascending `id` within each partition file |
| Partition key | `(source_table, symbol, UTC_year, UTC_month)` |
| Compression | ZSTD |
| File naming | deterministic, e.g. `{table}/{symbol}/{yyyy}/{mm}/part.parquet` |
| Partial-file naming | `{...}/part.parquet.partial` — never queryable, never counted as published |
| Atomic publication | write `.partial` → verify → `os.replace` to final name (single filesystem rename, not a copy) |
| Schema fingerprint | sha256 of the canonicalized Parquet schema string |
| Duplicate-row rule | dedupe by source `id` (primary key uniqueness already guarantees no duplicates at the source; archive must preserve this, not introduce new duplicates) |
| Late-row rule | see Phase 8 |
| Repair-version rule | if a source repair changes a row after archival, the archive is never mutated in place — a new manifest version is published (Phase 6) |

This is a contract draft only — no Parquet writer exists yet (see Phase 14).

## Phase 6 — Archive manifest contract (draft)

Every future archive partition's manifest (small JSON, permitted per the operator's own ruling) must record: contract version, source DB identity, source table, UTC partition start/end, symbol/venue/market-segment, export cutoff, captured watermark (max source `id` at export time), row count, min/max source `id`, min/max exchange timestamp, min/max receipt timestamp, source schema hash, Parquet schema hash, Parquet file path/size/sha256, ordered scientific-content hash (row-tuple hash, same discipline as the AMI canonical-migration manifests already used this session), duplicate count, invalid-row count, late-row policy reference, repair status, source-gap status (cross-referenced against the `gaps` table), export timestamp, exporter version, verification status, restore-test status, and purge-authorization status (always `NOT_AUTHORIZED` until a future, separate activation gate sets it).

Manifest publication is atomic (same `.partial`→rename discipline as the Parquet file itself). A `.partial` archive or an unpublished manifest is never purge-eligible — enforced structurally by the two-phase contract below, not merely documented.

## Phase 7 — Two-phase archive/purge contract (draft, neither phase executed)

**Phase A (Archive, non-destructive):** capture watermark → select closed-partition rows → enforce timestamp/identity contract → export to `.partial` → validate schema/row-accounting/IDs/timestamps → compute hashes → read-parity check → restore-proof check → atomic Parquet publish → atomic manifest publish → mark `ARCHIVED_VERIFIED` → **source rows untouched.**

**Phase B (Purge, requires separate future authorization):** may run only when archive status is `ARCHIVED_VERIFIED`, restore/read proof is `PASS`, Parquet+manifest hashes match, row accounting matches, the late-arrival window is closed, the repair process for that partition is closed, the research-dependency guard (Phase 9) passes, the collector-continuity guard passes, the canonical-dependency guard passes, **and** explicit operator/policy authorization exists. Deletion itself must use bounded chunked deletes, explicit transaction boundaries, deterministic key ranges, checkpointing, restart safety, and post-delete reconciliation against the archive. No `UPDATE`/`REPLACE`, no full-table rewrite, no automatic `VACUUM`.

Neither phase is implemented or invoked by this batch (confirmed structurally — the readiness module contains no `execute()` call at all).

## Phase 8 — Concurrency and late-arrival contract

`agg_trades`/`book_ticker`/`mark_prices`/`liquidations` all use `id INTEGER PRIMARY KEY AUTOINCREMENT` — monotonically increasing, never reused. `ts_ms` is receipt time (collector-write time), confirmed conservatively ≥ exchange event time (per the accepted spread/absorption readiness audits' own "known-at-safe" analysis) — so exchange-timestamp-based ordering may occasionally lag `id` order by a small margin, but receipt-time ordering (the same convention already governing every canonical known-at contract in this project) is safe.

**Frozen future rule (matching the operator's preferred structure exactly):** export only fully-closed UTC months; capture the maximum immutable source `id` at export start as the watermark; include only rows with partition timestamp inside the closed month **and** `id` ≤ the captured watermark; any row that arrives later (with an `id` beyond the watermark) for that closed month becomes part of a repair-version, never a silent mutation of the published archive; a repaired archive publishes as a new manifest version, preserving all prior manifest history. `gaps`-table repairs (agg_trades: 20, liquidations: 741, mark_prices: 51, all already recorded) are the existing, proven mechanism for detecting exactly this kind of late/gap correction — the archive contract reuses it rather than inventing a parallel one.

WAL-mode read transactions can safely snapshot the source at a point in time (SQLite's own MVCC-like WAL read-consistency guarantee) — a bounded, deterministic-watermark read-only export does **not** require a full SQLite Online Backup API pass for tables of this size; a bounded id-range `SELECT` is sufficient and far cheaper.

## Phase 9 — Research dependency matrix

| Dependency | Required historical horizon | Canonical feature exists? | Raw replay still required? | Parquet direct-read acceptable? | Current blocker | Purge precondition |
|---|---|---|---|---|---|---|
| `FAM_BOOK_SPREAD_DYNAMICS` (LONG, parked) | `book_ticker`, full history to date, growing forward | yes, for the already-migrated 196/97 M-0036 population | **yes**, for every future anchor's pre-birth window as the sample grows toward the ≥67 target | yes, once implemented | parked for sample growth (58/67 eligible) | `book_ticker` partitions may not be purged while this child remains parked and could still need their pre-birth window |
| `FAM_SPOT_PERP_BASIS_REVERSAL` (blocked) | `spot_prices`, `mark_prices`, full history to date | no canonical bridging table yet | yes, once a readiness/bridging batch is authorized | yes, once implemented | coverage-blocked (54/324 aligned anchors) | neither `spot_prices` (RESEARCH_CRITICAL_COMPACT, never purged by age) nor `mark_prices` may be purged while this family could still be revisited |
| Open Interest family | `open_interest`, full history | no | yes, for any future coverage recheck | yes, once implemented | coverage-blocked, recheck pending | `open_interest` never purged by age (RESEARCH_CRITICAL_COMPACT) |
| Funding family | `funding_rates`, `mark_prices` | no | n/a (source dead) | yes | source-dead | both preserved indefinitely — they are now the only historical record |
| Depth/liquidity families (future) | `book_ticker` (depth is not currently collected — only L1) | no | n/a — L2 depth was never collected; this is a data-gap for a *future* family, not a `book_ticker` retention concern | n/a | source missing (no L2 collector) | none — `book_ticker` retention is independent of this future gap |
| Execution/fill studies (future) | `agg_trades`, real-fill ledgers (not yet built) | no | yes | yes, once implemented | infra not built | `agg_trades` partitions relevant to any future fill-study window must not be purged ahead of that infra existing |
| Forward/shadow validation (`E-HOUR17-FWD-001`, `E-CONVCOMP-FWD-001`) | ongoing, live | n/a — these are canonical-side accumulation experiments | no (they read canonical/lifecycle tables, not raw `microstructure.db` directly for their core logic) | n/a | actively accumulating | none direct on raw tables; canonical dependencies are `CANONICAL_IMMUTABLE` already |
| State-transition research (future, OD-016/017) | `liquidations`, `ami_events`/lifecycle | partial | yes for any raw-liquidation-anchored redefinition | yes | infra/data-population blocked (OD-016/017) | `liquidations` is `RESEARCH_CRITICAL_COMPACT`, already never-purged |
| Entry timing / exit-management / stop taxonomy / re-entry / multi-TF structure / regime-drift / cross-asset generalization (future, broader alpha program) | varies; mostly already-materialized canonical tables (`CANONICAL_IMMUTABLE`) plus potential future raw re-derivation | mostly yes | possible for any future re-derivation | yes | none currently active | no specific raw-table purge precondition beyond the general research-dependency guard below |

**General guard (binding on any future purge):** no `RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE` partition may be purged merely because it is older than the active horizon. The current `NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY` portfolio state does **not** by itself make any partition purge-safe — the broader research program (18+ families, several parked/blocked rather than closed, plus an unenumerated future roadmap) remains a live dependency until each specific partition clears Phase 7's full purge-precondition list.

## Phase 10 — Archive read and restore contract (draft)

**Mode 1 (direct Parquet research read, preferred):** predicate pushdown by time/symbol/venue, deterministic ascending-`id` ordering, mandatory schema-version + manifest verification before any read is trusted, no full-archive extraction, structural guarantee against future-data contamination (a research query for signals born before time T must never read an archive partition whose `export cutoff` postdates what was knowable at T — same known-at discipline as every canonical table).

**Mode 2 (minimal temporary slice restoration, fallback only):** restore only the required table(s)/time-range/symbol(s), write only to `.runtime_temp`/`.pytest_temp` (never OS temp, never a full historical database), verify row count + content hash against the source manifest, delete after use unless explicitly retained as evidence. Full-database restores solely to serve one test are explicitly disallowed by this contract.

Current test/tooling compatibility gap: existing research code (`ami/research/feature_gateway.py` and friends) reads `microstructure.db`/`canonical.sqlite` via plain `sqlite3`, with no Parquet reader anywhere in the codebase today. Building that reader is listed as a future component (Phase 14), not built here.

## Phase 11 — SQLite space-reclamation policy

`microstructure.db`: `auto_vacuum=NONE`, `freelist_count=0` (fully packed — no reclaimable free pages exist right now because nothing has been deleted yet). Deleting rows in the future will populate the freelist (pages become reusable by future inserts) but will **not** shrink the file on disk without a `VACUUM` (full rebuild, requires up to ~1x the database size in temporary free space — for a 706GB database, this could require **on the order of 700GB of temporary space**, a substantial and separately-risky operation) or `PRAGMA incremental_vacuum` (only available if `auto_vacuum=INCREMENTAL`, which this database is not currently set to — changing `auto_vacuum` itself requires a full `VACUUM` to take effect, per SQLite's own documented behavior).

**Frozen safety rule (binding on any future implementation):** no automatic `VACUUM` after routine archival purge, ever. Let SQLite reuse freed pages via its normal freelist mechanism first. Physical file shrinkage requires its own, separate, explicitly-authorized maintenance gate — one that itself proves sufficient free disk space for the temporary rebuild copy, a governed backup beforehand, and a downtime contract with the live collectors (a `VACUUM` on an actively-written database is far riskier than an offline one; this project's collectors write to `microstructure.db` essentially continuously). **Never run `VACUUM` when free disk space cannot safely hold the required temporary copy** (at current ~1.1TB free against a ~706GB database, there is headroom today, but this must be re-verified at execution time, not assumed from this document).

## Phase 12 — Health and alert contract (draft, not wired to automation)

States: `STORAGE_HEALTHY`, `STORAGE_WARNING`, `STORAGE_CRITICAL`, `STORAGE_EMERGENCY`, `ARCHIVE_LAGGING`, `ARCHIVE_VERIFICATION_FAILED`, `PURGE_BLOCKED_BY_RESEARCH_DEPENDENCY`, `PURGE_BLOCKED_BY_SOURCE_REPAIR`, `PURGE_BLOCKED_BY_CONTINUITY`, `STORAGE_STATE_UNKNOWN`. The coarse drive-capacity state is a pure function of two already-measured numbers (`storage_health_state(pct_free, abs_free_gb)`), fails toward the more severe classification on disagreement between the two inputs, and is proven deterministic by a dedicated test. At the real, measured state this batch found (58.9% free, ~1126GB free), the function returns `STORAGE_HEALTHY`. No live automation consumes this function yet.

## Phase 13 — Safety and failure modes

24 failure modes enumerated in `FAILURE_MODES` (Parquet write failure, partial file, manifest write failure, checksum/row-count/schema mismatch, duplicate/missing/late rows, source-gap or repair discovered after archive, collector restart during export, WAL growth during export, disk full during export or purge, interrupted chunk delete, archive file missing/corrupt, restore mismatch, incompatible schema, unsupported reader, active research dependency, unknown table classification, unknown timestamp semantics) — **every one resolves to `FAIL_CLOSED`, `deletion_permitted=False`**, proven by a dedicated test iterating the full table.

## Phase 14 — Implementation boundary draft (not built)

17 future components listed as metadata only (`FUTURE_COMPONENTS`): storage policy configuration, table registry (already exists as this batch's `TABLE_REGISTRY`, reusable), archive planner, read-only source snapshot/exporter, Parquet writer, manifest writer, verifier, restore tester, purge planner, chunked purger, storage-health reporter, archive catalog, CLI dry-run command, CLI apply command, scheduled-runner integration, Parquet research reader, minimal-slice restorer. None of these is implemented in this batch — the tuple holds plain strings, not callables (proven by a dedicated test).

---

## Phase 15 — Readiness verdict

**`STORAGE_ROTATION_RETENTION_READY_WITH_RESEARCH_DEPENDENCY_BLOCKERS`**

Justification: the active storage estate is understood (Phase 1-2), archive-eligible tables are unambiguous (3 of 45 classified entries: `agg_trades`, `book_ticker`, `mark_prices`), protected tables are unambiguous (15 `CANONICAL_IMMUTABLE` + 4 `CONTINUITY_CRITICAL_ACTIVE` + 7 `RESEARCH_CRITICAL_COMPACT`, none purge-eligible), timestamp/identity semantics are sufficient (receipt-time `ts_ms` + monotonic `id`, known-at-safe by the same convention already governing canonical known-at contracts), and no destructive action is needed to rehearse the contract in disposable space. **However**, future source-row *purge* (not archival rehearsal) is explicitly blocked for all three archive-eligible tables by unresolved research dependencies (`book_ticker` by the parked `FAM_BOOK_SPREAD_DYNAMICS` LONG sample-accrual path; `mark_prices`/`agg_trades` by the still-open basis/absorption/CVD research surface and the general "18-family portfolio, mostly parked-not-closed" state) — these blockers are explicit and do **not** prevent a non-destructive archive dry-run.

Parquet/ZSTD tooling availability was **not independently verified in this batch** (no `import pyarrow`/`fastparquet` check was run — doing so is appropriately scoped to the dry-run gate itself, which will need the library regardless; listing it here as a residual risk rather than silently assuming it, per Phase 15's own `BLOCKED_BY_ARCHIVE_TOOLING` disposition criteria — this batch does not claim tooling is confirmed available).

---

## Blockers and residual risks

1. **Tooling verification gap:** Parquet/ZSTD library availability in this environment was not checked this batch — the recommended dry-run gate must verify this as its first step and may itself discover a `BLOCKED_BY_ARCHIVE_TOOLING` condition.
2. **Two-point growth estimate:** only two `microstructure.db` size snapshots exist 4 days apart; the ≈18.5GB/day figure is directionally useful but not a high-confidence forecast.
3. **Disk-used discrepancy:** ~141GB of non-`microstructure.db` space was apparently freed elsewhere on `D:` between 2026-07-03 and this batch — cause not investigated, disclosed not resolved.
4. **`open_interest`/`spot_prices`/several diagnostic tables' exact row counts** were not independently measured (no AUTOINCREMENT `id` to cheaply bound; a full COUNT was avoided per the storage guardrail) — classified and protected regardless, since classification does not require an exact count.
5. **416 stray test-scratch `.db` files** under `data/` (a pre-existing test-hygiene gap, ~13MB total) are classified `TEMPORARY_DISPOSABLE` but **not deleted** in this batch — flagged for a future, separate, low-risk cleanup, out of scope here.
6. **`runtime/chrome_user_data_copy`** (2.8GB) is unrelated to this project's data estate — flagged, not touched, out of scope.
7. This document is a **contract draft**, not an implementation. No archive planner, writer, verifier, or purger exists yet.

---

## Storage report

| Item | Value |
|---|---|
| Peak temporary disk usage (this batch) | ~0 bytes beyond pytest's own bytecode cache — no fixture, no export, no copy was created |
| Temporary files created | none |
| Temporary files deleted | none |
| Temporary files retained | none |
| Final `.runtime_temp/` contents | unchanged (3.7 MB, pre-existing rehearsal artifacts from earlier batches this session) |
| Final `.pytest_temp/` contents | unchanged (empty) |
| Confirmation: no production archive created | **confirmed** — zero Parquet/manifest files exist anywhere in the repository after this batch |
| Confirmation: no live row deleted or changed | **confirmed** — see state-transition proof |
| Confirmation: no full database copy created | **confirmed** — every inspection query in this batch was a bounded `MIN`/`MAX`/`MAX(id)`/`PRAGMA` read against the live files via `mode=ro`; no `shutil.copy`/backup call was made |

---

## Next controlled gate (recommended, not begun)

**`BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1`** — must select one small, closed, non-current `book_ticker` (or `agg_trades`/`mark_prices`) UTC-month partition, export only to `.runtime_temp`, create no production archive, delete no source row, verify Parquet/ZSTD output and manifest, prove direct-read and minimal-slice-restore parity, test restart/idempotency and corruption detection, and prove zero live-database mutation throughout. **Not begun by this batch.**

## Verdict

**`STORAGE_ROTATION_RETENTION_READY_WITH_RESEARCH_DEPENDENCY_BLOCKERS`**

Stopping after this readiness accounting. No dry-run, archive, or purge action begins without new, separate operator instruction.
