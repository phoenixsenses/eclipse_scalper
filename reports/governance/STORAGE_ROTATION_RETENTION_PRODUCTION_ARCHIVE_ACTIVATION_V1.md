# STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_ACTIVATION_V1

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ARCHIVE-ACTIVATION-V1
**Nature:** Manual-only production archive activation. Publishes production partitions only under exact authorization receipts. No general activation, no scheduler, no purge, no VACUUM, no source mutation.
**Date:** 2026-07-08 · **Author:** Sonnet 5

---

## Headline result — PARTIAL

The full manual-only activation machinery (policy states, authorization-receipt model, single-writer catalog lock, concurrency-safe root-index rebuild, real staging-health reporting, six-file authorized publication, five new CLI commands) was **implemented and proven end-to-end on a live production partition** — `agg_trades`/`ETHUSDT`/`2026-02`/`v1` (23,957,222 rows) was published, re-verified (0 mismatches), direct-read-parity-matched, restore-parity-matched, and idempotent-rerun-confirmed (`NOOP_IDENTICAL_PRODUCTION_ARCHIVE`).

**The second live publication — `book_ticker`/`SOLUSDT`/`2026-04`/`v1` (114,404,095 rows) — did not complete.** Its export streamed ~1.95 GB of Parquet, then hung during finalization at ~3.0 GB resident memory (the RAM wall this project explicitly guards against — `CLAUDE.md`: "Paralel Python/PowerShell prosesi ÇALIŞTIRMA (RAM çöker)"). It never reached the final path and never entered the root catalog index. The abandoned gate-owned staging was cleaned; the source was never mutated.

Per the gate's own Phase 25 rule ("If exactly one new verified partition is published before an unexpected failure in the second publication ... do not delete the verified partition, use `PARTIAL_PUBLICATION_SOURCE_RETAINED`, report exact recovery gate, source remains fully retained, manual-only general activation must remain disabled until closure"), this batch closes:

**`STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_ACTIVATION_V1_PARTIAL_PUBLICATION_SOURCE_RETAINED`**

Final production estate: **2 verified partitions** (`mark_prices` rehearsal-legacy + `agg_trades` activation), not the target 3.

---

## Phase 1 — Pre-activation reconciliation

Before implementation: exactly one accepted partition existed (`mark_prices`/ETHUSDT/2026-05/v1, 5 files, parquet `6f919144…`), root index had 1 entry, `.staging` was empty, no unknown archive content, no conflicting entries. Confirmed immutable and rebuildable from the partition-local catalog entry.

## Phase 2 — Manual-only activation policy

`ami/storage/policy.py` gained **separate named states** (not one ambiguous boolean): `MANUAL_PRODUCTION_ARCHIVE_CREATION="ENABLED"`, `GENERAL_UNRESTRICTED_ACTIVATION="DISABLED"`, `PRODUCTION_SCHEDULER_STATE="DISABLED"`, `PRODUCTION_PURGE_STATE="DISABLED"`, `PRODUCTION_VACUUM_STATE="DISABLED"`, `SOURCE_RETENTION_REQUIREMENT="REQUIRED"`, read via `production_activation_states()`. `GENERAL_PRODUCTION_ACTIVATION_ENABLED` stays `False`. **Because this batch closes PARTIAL, the manual-only capability — though implemented and proven on `agg_trades` — is held un-blessed for general use until a closing recovery gate completes book_ticker.**

## Phase 3-4 — Authorization receipt model

`ami/storage/production_activation.py::build_authorization_receipt()` — an immutable, self-hashed (`receipt_sha256`), exact-partition-bound receipt with `action=CREATE_PRODUCTION_ARCHIVE_ONLY`, `exact_partition_plan_hash` (binds table/symbol/venue/segment/month/watermark/schema/root/version), `purge/scheduler/vacuum_authorization=PROHIBITED`, `source_retention_requirement=SOURCE_MUST_REMAIN_PRESENT`. `verify_authorization_receipt()` fails closed on self-hash mismatch, wrong action, plan-hash drift, wrong table/symbol/venue/segment, watermark drift, or expiry. Receipts are issued only by a **narrowly-scoped governance driver** (`issue_gate_authorized_receipt` / `gate_publish_partition`) — there is no CLI command that fabricates or self-approves a receipt (`production-archive-authorized` *consumes* a receipt path). New partitions retain `authorization_receipt.json` in the final directory; `mark_prices` is represented as legacy (no retroactive receipt — its partition-local files were never modified).

## Phase 5 — Catalog concurrency lock

`acquire_catalog_lock()` uses an atomic `O_CREAT|O_EXCL` exclusive create at `data/archives/raw_v1.catalog.lock`, writing job-identity/pid/host/plan-hash/owner-token-hash. Release is owner-token-verified (a non-owner is refused); a live lock causes fail-closed conflict; an unreadable lock is classified `CATALOG_LOCK_REPAIR_REQUIRED` and **never auto-deleted**; acquisition has a bounded timeout. The final rename + index update run under the lock.

## Phase 6 — Concurrency-safe root index

The root `catalog_index.json` is rebuilt from immutable partition-local catalog entries under the lock, `.partial`-first then atomically replaced, rejecting duplicates/conflicts/unverified/purge-authorized entries, self-hashed. It correctly holds **both** the rehearsal-era 5-file `mark_prices` entry and the activation-era 6-file `agg_trades` entry. Current state: **2 entries, self-hash `456892df5bb6f3e500f44132b727fee33dd0fb1cdb23ab7c2bc808558ced719c`.**

## Phase 7 — Real staging health

`ami/storage/health.py::scan_production_archive_health()` replaces the placeholder with a real read-only scan: active/abandoned/unrecognized staging directories, staging bytes, oldest staging timestamp, catalog-lock state, root-index presence/validity, verified/invalid partition counts — with explicit health states (`PRODUCTION_ARCHIVE_HEALTHY`, `…_STAGING_ACTIVE`, `…_ABANDONED_STAGING`, `…_CATALOG_LOCKED`, `…_CATALOG_LOCK_REPAIR_REQUIRED`, `…_INDEX_REPAIR_REQUIRED`, `…_VERIFICATION_FAILED`). Follows no reparse points, deletes nothing.

## Phase 8 — Candidate selection (deterministic, outcome-blind)

`select_candidate_partition()` — earliest eligible closed UTC month (≥ the table's earliest row, fully closed, outside the 30-day active horizon), then the symbol with the smallest positive bounded row count (lexicographic tie-break), recording every considered month and every symbol count.

| Table | Earliest eligible month | Symbol counts (bounded) | Selected |
|---|---|---|---|
| `agg_trades` | 2026-02 | BTCUSDT 26,376,020 · ETHUSDT 23,957,222 · SOLUSDT 0 | **ETHUSDT (23,957,222)** |
| `book_ticker` | 2026-04 | BTCUSDT 505,194,719 · ETHUSDT 540,644,508 · SOLUSDT 114,404,095 | **SOLUSDT (114,404,095)** |

## Phase 9-10 — Preflight and resource guards

Both plans were preflighted before the first publication. Resource guards (`check_resource_limits`, no hidden force): min-free ≥500 GB (actual ~1,120 GB), per-partition source ≤100 GB, parquet ≤30 GB, projected-free ≥400 GB — all satisfied for both plans. book_ticker's estimated parquet (~9 GB from 114M×80B) was well under the 30 GB cap; **the failure was not a cap violation but a runtime RAM wall during finalization.**

## Phases 11-14 — agg_trades: streaming publication (SUCCESS)

To stay within the RAM guardrail, the exporter was made **bounded-streaming** (`stream_export_to_parquet`: `fetchmany` batches → single `ParquetWriter`, incremental scientific-content hash — proven byte-identical to `write_table` for single-batch partitions, so `mark_prices` reproduction stays `6f919144…`). agg_trades six-file authorized publication:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `part-00000.parquet` | 317,348,832 | `7bbf05fd28ab30fd…` |
| `manifest.json` | 1,865 | `eb34359d9a650af6…` |
| `authorization_receipt.json` | 1,553 | `ec6f724386a0e5c2…` |
| `catalog_entry.json` | 2,127 | `d58d6d025699b77d…` |
| `_SUCCESS` | 26 | `f09de3a5280da3a3…` |
| **Total** | **317,354,403** | |

Row count 23,957,222; watermark 50,333,244; UTC `[2026-02-01, 2026-03-01)`; scientific-content hash `5b17f4e4be294b0a7a7ba75934095676ca8792d13351a4a4535652776ea2af25`; `production_status=PRODUCTION_VERIFIED`, `purge_authorization=PROHIBITED`. Post-publication re-verification (streaming, RAM-bounded): **0 mismatches**.

## Phase 15-16 — agg_trades: direct-read and restore parity (SUCCESS)

Direct-read parity: a fresh source re-read (streaming) produced hash `5b17f4e4…` — **exact match** to the published Parquet (23,957,222 rows, 0 mismatches). Minimal streaming restore into a disposable `.runtime_temp` SQLite slice (1.26 GB) reproduced the same hash exactly, then was deleted after the proof. Idempotent disposition: `NOOP_IDENTICAL_PRODUCTION_ARCHIVE`.

## book_ticker — attempted, failed on a RAM wall (Phases 11-14, incomplete)

- Candidate selection correctly chose `SOLUSDT`/2026-04 (114,404,095 rows, watermark 1,160,243,321).
- The streaming export ran the required `ORDER BY id` over 114M rows. Because the `(symbol, ts_ms)` index does not match the `id` sort order, SQLite performed a large sort (spilled to temp disk, ~12 min), then streamed ~1.95 GB of Parquet into staging.
- During finalization the process **hung at ~3.0 GB resident memory** and stopped progressing (the `.partial` was frozen for 13+ minutes, file handle still held). This is the machine's RAM wall for a 114M-row wide-row partition.
- The hung, gate-owned publish process (`.runtime_temp/publish_bt.py`, PID 14180, confirmed *not* a collector) was terminated; the abandoned staging directory (a never-published `.partial`, never renamed to final, never entered in the index) was removed.
- **No `book_ticker` production file exists; the root index remains at 2 entries; the source was never touched (all reads `mode=ro`, authorizer log empty).**

## Phase 23 — mark_prices legacy re-verification

`mark_prices`/ETHUSDT/2026-05/v1 re-verified: parquet still `6f91914400dcbe84…` (byte-identical), 5-file (no retroactive authorization receipt), root-index entry valid, no file modified. The activation-era receipt requirement applies only to new partitions.

## Phase 24 — Final production health state

| Field | Value |
|---|---|
| Verified production partitions | **2** (`mark_prices` legacy + `agg_trades` activation) — target was 3 |
| `book_ticker` | **not published** (failed, source retained) |
| Root catalog index | valid, 2 entries, self-hash `456892df…` |
| Catalog lock | absent (clean) |
| Active / abandoned / unrecognized staging | 0 / 0 / 0 |
| Manual production archive creation | ENABLED (implemented + proven on agg_trades) |
| General unrestricted activation | **DISABLED** |
| Scheduler | **DISABLED** |
| Purge | **DISABLED** |
| VACUUM | **DISABLED** |
| Source retention | REQUIRED |

## Phase 22 — CLI surface

Five new commands: `production-plan` (read-only), `production-archive-authorized` (consumes a receipt path, no arbitrary root/self-approval), `production-verify`, `production-catalog-rebuild` (lock-protected deterministic rebuild), `production-health`. No `activate-production`, `archive-all`, `archive-range`, `schedule`, `purge`, `delete`, `vacuum`, `compact`, `stop-collector`, or `restart-collector` command exists (confirmed by test).

---

## Verdict

**`STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_ACTIVATION_V1_PARTIAL_PUBLICATION_SOURCE_RETAINED`**

One new verified production partition (`agg_trades`) published and fully verified; the second (`book_ticker`) failed on an unexpected RAM wall before publication. The verified partitions are retained, the source is fully intact, and general/unrestricted activation, scheduler, purge, and VACUUM all remain disabled pending closure.

## Root cause and recovery gate

**Root cause:** the 114M-row `book_ticker` partition combines (a) an unavoidable large SQL sort (`ORDER BY id` vs. a `(symbol, ts_ms)` index) and (b) a ~2 GB single-file Parquet finalization, together exceeding this machine's practical RAM ceiling (~3 GB observed at hang). `mark_prices` (260K) and `agg_trades` (24M) were within reach; `book_ticker` (114M, 11 wide columns) was not, with the current single-file, single-pass streaming design.

**Recommended recovery gate:** `BATCH-STORAGE-PRODUCTION-ARCHIVE-BOOK-TICKER-RECOVERY-V1` — republish the same frozen `book_ticker`/SOLUSDT/2026-04 partition with a **memory-bounded export** (candidate mitigations, to be chosen there: multi-file partition `part-00000..part-NNNNN.parquet` with per-file row caps so no single-file finalization dominates RAM; a smaller streaming batch size threaded through the publisher; and/or an id-ordered scan that avoids the full sort). It must still prohibit source deletion, purge, scheduler, and VACUUM, keep the existing `mark_prices`/`agg_trades` partitions immutable, and — on success — bring the estate to 3 verified partitions and permit closing the manual-only-activation disposition.

## Blockers / residual risks

1. `book_ticker` production publication is unresolved (RAM-bound); the recovery gate above is required before the estate reaches 3 partitions.
2. The single-file Parquet + full-sort export design does not scale to ~100M+ row partitions on this machine — the recovery gate must address this before any larger partition is attempted.
3. Manual-only activation is proven but held un-blessed for general use until closure (per the PARTIAL rule).

## Storage report

| Item | Value |
|---|---|
| New verified production partition | `agg_trades` (317,354,403 bytes across 5 files) |
| Existing partition unchanged | `mark_prices` (2,479,924 bytes, `6f919144…`) |
| Root index | `catalog_index.json` 4,236 bytes, self-hash `456892df…` |
| book_ticker production bytes | 0 (never published) |
| Abandoned book_ticker staging | removed (was a ~1.95 GB `.partial`, never published) |
| Disposable restore/verify files | created under `.runtime_temp` and deleted after proof |
| Final `.staging` contents | empty |
| Final `.runtime_temp` activation contents | empty (restore slices deleted) |
| Drive free | ~1,118 GB |
| Full database copy | none |
| Source rows modified/deleted | 0 |
| Purge / scheduler / VACUUM capability added | none |

## Next controlled gate

**`BATCH-STORAGE-PRODUCTION-ARCHIVE-BOOK-TICKER-RECOVERY-V1`** (memory-bounded `book_ticker` republication). Not begun. After the estate reaches 3 verified partitions, the later independent gates remain: research-reader integration, scheduler, purge readiness/dependency release, bounded purge authorization, and finally `BATCH-STORAGE-SQLITE-PHYSICAL-RECLAMATION-MAINTENANCE-V1`.

Not begun by this batch.
