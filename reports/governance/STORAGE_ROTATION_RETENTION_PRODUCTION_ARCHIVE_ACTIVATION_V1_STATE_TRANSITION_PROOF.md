# STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_ACTIVATION_V1 — State-Transition Proof

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ARCHIVE-ACTIVATION-V1
**Date:** 2026-07-08 · **Author:** Sonnet 5
**Outcome:** `STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_ACTIVATION_V1_PARTIAL_PUBLICATION_SOURCE_RETAINED` — exactly **one** new production partition created (`agg_trades`); the second (`book_ticker`) failed on a RAM wall before publication. Source fully retained; all governance/canonical state unchanged.

---

## 1. Live-database + production mutation proof

| Metric | Value |
|---|---|
| Live source rows inserted / updated / deleted | 0 / 0 / 0 |
| Live source tables created / dropped | 0 / 0 |
| Live source indexes changed | 0 |
| Forced WAL checkpoints | 0 |
| VACUUM / incremental-vacuum executions | 0 / 0 |
| Collector stops / restarts / config changes | 0 / 0 / 0 |
| Scheduler jobs / startup hooks created | 0 / 0 |
| **New production partitions created** | **exactly 1** (`agg_trades`) |
| **New production Parquet files** | **exactly 1** |
| **New production manifests** | **exactly 1** |
| **New authorization receipts** | **exactly 1** |
| **New partition catalog entries** | **exactly 1** |
| **New success markers** | **exactly 1** |
| Root catalog index updates | deterministic, lock-protected (final state: 2 entries) |
| Partitions marked purge-eligible | 0 |
| Purge / delete / vacuum / unrestricted-activation commands exposed | 0 |
| Source rows purged | 0 |
| Canonical / knowledge rows changed | 0 / 0 |
| Schema changes / migrations | 0 / 0 |
| Outcome-value / TRAIN / TEST reads | 0 / 0 / 0 |
| Experiments / results / nullifiers / gate-receipts created or consumed | 0 |
| Route/bucket promotions | 0 |
| Runtime/risk/execution/paper/shadow/forward/live changes | 0 |
| Accepted backups / research evidence changed | 0 / 0 |
| Existing `mark_prices` production files changed | 0 |
| Cleanup outside gate-owned staging/temp | 0 |

**Process termination (disclosed):** one gate-owned python process was terminated — the hung `book_ticker` publisher (`.runtime_temp/publish_bt.py`, PID 14180), confirmed by its command line to be this batch's own failed job and **not** a collector. No collector, and no process other than this batch's own hung publisher, was stopped.

## 2. Canonical / knowledge immutability

| Field | Before | After |
|---|---|---|
| `canonical.sqlite` sha256 | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` | unchanged |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | unchanged |
| `schema_version` / `experiment_registry` / `experiment_results` / nullifiers / receipts | 14 / 24 / 381 / 2 / 2 | unchanged |

## 3. Source retention proof

All source access went through `ami.storage.source_access.open_read_only()` (`mode=ro` + `query_only` + authorizer). Every session's rejection log was empty (0 write attempts). The `agg_trades`/ETHUSDT/Feb-2026 population was independently re-read for direct-read parity and matched the published archive exactly (23,957,222 rows, hash `5b17f4e4…`). The `book_ticker`/SOLUSDT/Apr-2026 source population was read only (never mutated); its failed publication touched no source row. The existing `mark_prices` source population is likewise unchanged.

## 4. Existing partition immutability

`mark_prices`/ETHUSDT/2026-05/v1 re-verified: `part-00000.parquet` sha256 `6f91914400dcbe84…` (byte-identical to publication), still 5-file (no retroactive authorization receipt added), root-index entry valid, no partition-local file modified, no mtime changed by verification.

## 5. book_ticker failure — clean, non-destructive

The `book_ticker` publication hung during finalization (~3.0 GB RAM), never renamed staging to the final path, and never entered the root index. Its abandoned staging (a never-published `.partial`) was removed after the hung process was terminated. **No `book_ticker` production file exists; the root index holds exactly the 2 verified partitions (`mark_prices` + `agg_trades`); the source is intact.**

## 6. Root index / lock / staging final state

| Item | Value |
|---|---|
| Root index entries | 2 (`agg_trades`, `mark_prices`), self-hash `456892df5bb6f3e5…` |
| Catalog lock | absent (cleanly released after the standalone rebuild) |
| Active / abandoned / unrecognized staging | 0 / 0 / 0 |
| Final `.staging` contents | empty |

## 7. Focused tests

`tests/test_ami_storage_production_activation.py` — **45 passed**; `tests/test_ami_storage_production.py` — **55 passed**. Coverage: manual-only policy states, authorization-receipt build/verify + all rejection paths (wrong table/symbol/venue/segment/month/root/watermark/schema, altered, expired, missing), catalog-lock atomic acquisition + second-process conflict + owner-only release + unreadable-lock-repair-required + no-auto-delete, deterministic candidate selection (smallest-positive-symbol, lexicographic tie-break, earliest-month, zero-count exclusion), resource guards (low-free / oversized / no-force-override), six-file authorized publication + mismatched-receipt rejection + lock-released-after-completion + missing-receipt reverify, gate-driver idempotency (NOOP / never-v2 / no-rewrite), lock-protected deterministic index rebuild, streaming export/restore parity + output-cap rejection, CLI surface (5 new commands present, all forbidden commands absent, receipt-path required), and the real 2-partition estate (mark_prices immutable no-receipt, all-disabled health).

## 8. Regression

Paired across all 9 storage test files — **245 passed, 0 new attributable failures** (`policy_registry` 24, `job_state_cli` 24, `partition_planner` 21, `source_access` 14, `archive_verifier` 25, `catalog_reader_restorer` 22, `acceptance` 15, `production` 55, `production_activation` 45).

## 9. Verdict

**`STORAGE_ROTATION_RETENTION_PRODUCTION_ARCHIVE_ACTIVATION_V1_PARTIAL_PUBLICATION_SOURCE_RETAINED`**
**Recovery gate (not begun):** `BATCH-STORAGE-PRODUCTION-ARCHIVE-BOOK-TICKER-RECOVERY-V1`
**Execution stopped:** confirmed — exactly one new verified production partition (`agg_trades`) created; `book_ticker` failed on a RAM wall and was not published; the source is fully retained and byte-unchanged; general/unrestricted activation, scheduler, purge, and VACUUM all remain disabled.
