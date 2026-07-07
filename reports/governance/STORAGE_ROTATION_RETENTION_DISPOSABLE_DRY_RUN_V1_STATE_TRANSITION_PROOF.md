# STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1 — State-Transition Proof

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1
**Date:** 2026-07-07 · **Author:** Sonnet 5
**Outcome:** `STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_COMPLETE` — every mutable production/governance resource remains untouched; all activity was a bounded, read-only rehearsal confined to `.runtime_temp/storage_rotation_dry_run_v1/`.

---

## 1. Live-database mutation proof

| Metric | Value |
|---|---|
| Live rows inserted | 0 |
| Live rows updated | 0 |
| Live rows deleted | 0 |
| Live tables created | 0 |
| Live tables dropped | 0 |
| Live indexes changed | 0 |
| VACUUM executions | 0 |
| Incremental-vacuum executions | 0 |
| Forced WAL checkpoints | 0 |
| Collectors stopped | 0 |
| Collectors restarted | 0 |
| Collector configuration changes | 0 |
| Production archives created | 0 |
| Production manifests created | 0 |
| Production archive catalog changes | 0 |
| Partitions marked purge-eligible | 0 (manifest `purge_authorization` field hardcoded `PROHIBITED`) |
| Source rows purged | 0 |
| Canonical rows changed | 0 |
| Schema changes | 0 |
| Migration creation | 0 |
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| Outcome-value reads | 0 |
| Experiment creation | 0 |
| Experiment-result creation | 0 |
| Nullifier creation | 0 |
| Nullifier consumption | 0 |
| Gate-receipt creation | 0 |
| Gate-receipt update | 0 |
| Route/bucket promotion | 0 |
| Runtime/risk/execution delta | 0 |
| Paper/shadow/forward/live behavior delta | 0 |
| Accepted backups changed | 0 |
| Accepted research evidence changed | 0 |
| M-0036 rows changed | 0 |

**Direct proof, not inference:** the disposable driver connected to `microstructure.db` with a SQLite authorizer installed that denies every write-capable action (`INSERT`/`UPDATE`/`DELETE`/`CREATE`/`DROP`/`ALTER`/`REINDEX`/`ATTACH`/`TRANSACTION`). The authorizer's denial log recorded **zero entries** — meaning the exporter never even *attempted* a write, a stronger guarantee than "writes were blocked." The committed reusable module (`ami/governance/storage_rotation_retention_disposable_dry_run_v1.py`) additionally contains **zero** `.execute()`/`.executescript()`/`.executemany()` call sites and never imports `sqlite3`, `shutil`, `pyarrow`, or `os` — proven by two dedicated AST-walk tests, meaning the committed code is architecturally incapable of touching a database or the filesystem at all; the actual I/O lived only in the disposable, never-committed driver script.

## 2. Canonical / knowledge immutability

| Field | Value |
|---|---|
| `canonical.sqlite` sha256 | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` (unchanged, confirmed before and after this batch) |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged) |
| `schema_version` | 14 |
| `experiment_registry` | 24 |
| `experiment_results` | 381 |
| `epistemic_test_nullifiers` | 2 |
| `experiment_gate_receipts` | 2 |

## 3. `microstructure.db` — accurately reported (not falsely claimed static)

| Metric | Start | End |
|---|---|---|
| File size | 759,124,799,488 bytes | 759,124,799,488 bytes (unchanged — the ~10s rehearsal window did not coincide with a collector page-flush; this is observational, not a claim that the file can never change during a longer rehearsal) |
| WAL size | 8,157,632 bytes | 3,015,872 bytes (shrank — a live collector's own periodic checkpoint, not issued by this batch) |
| `mark_prices` table-wide `MAX(id)` | 13,265,132 (May-2026-partition-scoped watermark) | 21,101,480 (table-wide, all months, at rehearsal end) |

The table-wide `MAX(id)` jump reflects collector activity **outside** the selected May 2026 partition (i.e., June/July 2026 rows continuing to arrive) — it does **not** indicate any row was added to the selected partition itself. This is proven directly: `current_max_id_same_partition` (re-queried with the same `symbol`+date-range filter at the end of the rehearsal) equals the originally captured watermark exactly (13,265,132 = 13,265,132).

## 4. Focused tests

`tests/test_ami_governance_storage_rotation_retention_disposable_dry_run_v1.py` — **40/40 passed**. Covers: frozen partition-identity constants and closed-UTC-month/30-day-horizon/current-month-exclusion checks, rejected-candidate recording with reasons, selection-preference-order-not-silently-switched, the real dry-run's frozen results (row count, watermark, byte-identical run-A/run-B), size-cap compliance, canonical row-hash determinism/order-sensitivity/None-vs-zero distinction, schema-contract column preservation and nullability, `validate_partition_rows` across 6 scenarios (all-pass, wrong-symbol, out-of-range timestamp, above-watermark, duplicate, empty), manifest construction (disposable-not-production hardcoding, field count, watermark/hash recording), corruption detection (identical/mismatched/unreadable), verdict determination across 8 scenarios including fail-closed precedence (tooling-blocked first, live-write beats resource-limit), and structural guards proving the module has zero database/filesystem imports and zero execute-call sites.

## 5. Regression

Paired with the two prior storage-batch test files (readiness: 27 tests, discrepancy audit: 25 tests) — all still pass unaffected, since this batch's committed module shares no code path with either. Established baseline unaffected.

## 6. Storage report

| Item | Value |
|---|---|
| Peak disposable disk usage | 20,333,167 bytes (~19.4 MiB), entirely beneath `.runtime_temp/storage_rotation_dry_run_v1/` |
| Files created | 10 (2 Parquet builds, 1 manifest pair, 1 restored SQLite slice, 1 abandoned interruption partial, 1 interrupt-recovery Parquet, 2 corruption-test copies) |
| Files deleted | 3 immediately (abandoned partial + both corruption-test copies) + 7 more (all bulk binaries) after their hashes were recorded in `results.json` |
| Files retained | `driver.py` (19,810 bytes) + `results.json` (6,230 bytes) — small, hashed, bounded, no outcome data, only the selected raw partition's metadata |
| Final `.runtime_temp/storage_rotation_dry_run_v1/` contents | `driver.py`, `results.json` (28 KB total) |
| Final `.pytest_temp/` contents | unchanged (empty) |
| Full database copy created | **confirmed NOT created** |
| Production archive created | **confirmed NOT created** |
| Source row deleted or modified | **confirmed NOT occurred** |

## 7. Verdict

**`STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_COMPLETE`**
**`STORAGE_ROTATION_RETENTION_ARCHIVE_TOOLING_READY_FOR_BOUNDED_IMPLEMENTATION`**
**Next gate (not begun):** `BATCH-STORAGE-ROTATION-RETENTION-BOUNDED-IMPLEMENTATION-V1`
**Execution stopped:** confirmed — no production archival, source-row purge, scheduler integration, collector change, VACUUM, or outcome access occurred at any point in this batch.
