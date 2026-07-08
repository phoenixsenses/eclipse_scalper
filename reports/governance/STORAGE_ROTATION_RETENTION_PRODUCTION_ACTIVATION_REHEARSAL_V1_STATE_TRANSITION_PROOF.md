# STORAGE_ROTATION_RETENTION_PRODUCTION_ACTIVATION_REHEARSAL_V1 — State-Transition Proof

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ACTIVATION-REHEARSAL-V1
**Date:** 2026-07-08 · **Author:** Sonnet 5
**Outcome:** `STORAGE_ROTATION_RETENTION_PRODUCTION_ACTIVATION_REHEARSAL_V1_COMPLETE` — exactly one production archive partition created and verified; every other mutable resource (source rows, canonical/knowledge governance state, schema, collectors) remains untouched.

---

## 1. Live-database mutation proof

| Metric | Value |
|---|---|
| Source rows inserted by this batch | 0 |
| Source rows updated by this batch | 0 |
| Source rows deleted by this batch | 0 |
| Source tables created | 0 |
| Source tables dropped | 0 |
| Source indexes changed | 0 |
| Forced WAL checkpoints | 0 |
| VACUUM executions | 0 |
| Incremental-vacuum executions | 0 |
| Collector stops | 0 |
| Collector restarts | 0 |
| Collector configuration changes | 0 |
| Scheduler jobs created | 0 |
| Startup hooks created | 0 |
| **Production archive partitions created** | **exactly 1** |
| **Production Parquet files created** | **exactly 1** |
| **Production manifests created** | **exactly 1** |
| **Production partition catalog entries created** | **exactly 1** |
| **Production success markers created** | **exactly 1** |
| **Root catalog indexes created/atomically updated** | **exactly 1** |
| Partitions marked purge-eligible | 0 |
| Purge commands exposed | 0 (no `purge` subcommand exists anywhere in `ami/storage/cli.py`, confirmed by test) |
| Source rows purged | 0 |
| Canonical rows changed | 0 |
| Knowledge rows changed | 0 |
| Schema changes | 0 |
| Migrations created | 0 |
| Outcome-value reads | 0 |
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| Experiments created | 0 |
| Experiment results created | 0 |
| Nullifiers created | 0 |
| Nullifiers consumed | 0 |
| Gate receipts created | 0 |
| Gate receipts updated | 0 |
| Route/bucket promotions | 0 |
| Runtime/risk/execution changes | 0 |
| Paper/shadow/forward/live changes | 0 |
| Accepted backups changed | 0 |
| Accepted research evidence changed | 0 |
| Cleanup actions outside gate-owned staging/temp | 0 |

**Direct proof, not inference:** every real-database read this batch performed (initial live snapshot verification, the actual production publication run, the second idempotent-rerun CLI invocation, the post-publication re-verification, the direct-read/restore proofs, and the source-retention proof) went through `ami.storage.source_access.open_read_only()` — `mode=ro` + `query_only=ON` + a SQLite authorizer denying every write-capable action and writable PRAGMA. `assert_read_only_session_clean()` was asserted after every one of these sessions; the rejection log was empty every time (zero write attempts were ever issued, not merely blocked).

## 2. Canonical / knowledge immutability

| Field | Before | After |
|---|---|---|
| `canonical.sqlite` sha256 | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` (unchanged) |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged) |
| `schema_version` | 14 | 14 |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `epistemic_test_nullifiers` | 2 | 2 |
| `experiment_gate_receipts` | 2 | 2 |

Accepted migration artifacts (M-0036 rows, backups, all prior INCOMPLETE/COMPLETE research artifacts) unchanged — same sha256/count check used throughout this session.

## 3. `microstructure.db` — source table population proof

| Metric | Value |
|---|---|
| Selected table | `mark_prices` |
| Selected partition | `ETHUSDT`, UTC `[2026-05-01, 2026-06-01)` |
| Frozen watermark (this batch's fresh capture) | 13,265,132 |
| Row count before this batch's read | 260,657 |
| Row count after this batch's read (independently re-verified) | 260,657 |
| Scientific-content hash before | `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999` |
| Scientific-content hash after | `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999` |
| Batch-issued writes | **0** |

The frozen partition population is byte-stable across this entire batch. `microstructure.db`'s overall file size and WAL were not tracked as before/after deltas in this batch (live collectors continue writing outside the frozen partition, as in every prior batch) — the authoritative proof is the authorizer's empty rejection log across every session, not a file-size comparison.

## 4. Production archive inventory (permanent, outside Git)

| File | Size (bytes) | SHA-256 / self-hash | State |
|---|---|---|---|
| `part-00000.parquet` | 2,476,313 | `6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f` | VERIFIED |
| `manifest.json` | 1,898 | `61c9e226a61c04cfef1a776cc1bc2e897ec627fcbf8406e609fd2efddd496371` | VERIFIED |
| `catalog_entry.json` | 1,687 | `009ec5095ef9be50ee8b1a8185926a929152b07220cb89bd577df919f32888fb` | VERIFIED |
| `_SUCCESS` | 26 | (marker file, no separate hash tracked) | VERIFIED |
| `catalog_index.json` | 1,971 | `2f0bf51443dd044a6a624852d3cfe5ddaa94286cb9a3a5937b532e15d69fbd4e` (self-hash) | VERIFIED |
| **Total** | **2,482,981** | | |

All 5 files reside under `D:\eclipse_scalper\data\archives\raw_v1\` — confirmed excluded from Git via `.gitignore`'s new `data/archives/` rule (`git check-ignore -v` confirms; `git status` shows nothing under `data/archives/`).

## 5. Idempotency proof (concrete, not just asserted)

The CLI's `production-activation-rehearsal` command was invoked **twice** against the real database:

| Run | Result | Parquet mtime | Parquet SHA-256 |
|---|---|---|---|
| 1st | `PUBLISHED` | 2026-07-08 00:05 | `6f919144…` |
| 2nd | `NOOP_IDENTICAL_PRODUCTION_ARCHIVE` | 2026-07-08 00:05 (**unchanged**) | `6f919144…` (**unchanged**) |

No `version=v2` directory was ever created (`find data/archives -type d -name "version=*"` returns exactly one entry both before and after the second run).

## 6. Focused tests

`tests/test_ami_storage_production.py` — **55/55 passed**, including 5 tests that read the **real, live production archive** read-only (`test_real_production_archive_exists_and_verified`, `test_real_production_manifest_matches_accepted_hashes`, `test_real_production_reverification_zero_mismatches`, `test_real_root_index_deterministic_and_single_entry`, `test_real_idempotent_disposition_is_noop`) and 2 tests that read the **real source database** read-only (`test_source_retention_real_partition_row_count_unchanged`, `test_source_retention_zero_write_attempts`) — all others use disposable `tmp_path` fixtures exclusively. `tests/test_ami_storage_job_state_and_cli.py` gained 1 new test (24 total, was 23) confirming the new CLI command accepts no partition-identifying arguments.

## 7. Regression

Paired against all 8 other storage test files — **213/213 passed, 0 new attributable failures**:

| File | Tests |
|---|---|
| `test_ami_storage_policy_and_registry.py` | 24 |
| `test_ami_storage_partition_and_planner.py` | 21 |
| `test_ami_storage_source_access.py` | 14 |
| `test_ami_storage_archive_and_verifier.py` | 25 |
| `test_ami_storage_catalog_reader_restorer.py` | 22 |
| `test_ami_storage_acceptance.py` | 15 |
| `test_ami_governance_storage_rotation_retention_readiness_v1.py` | 27 |
| `test_ami_governance_storage_disk_usage_discrepancy_audit_v1.py` | 25 |
| `test_ami_governance_storage_rotation_retention_disposable_dry_run_v1.py` | 40 |

## 8. Storage report

| Item | Value |
|---|---|
| Production files retained | 5 files, 2,482,981 bytes total, all under `data/archives/raw_v1/` (gitignored) |
| Staging directories | 1 job's worth created and fully consumed (atomically renamed to final); `raw_v1.staging/` is empty at rest |
| Disposable restore proof file | 1 (`restored.sqlite`), deleted immediately after parity confirmation |
| Disposable corruption-test copies | multiple, all under pytest's `tmp_path`, auto-cleaned by the test runner |
| Final `data/archives/raw_v1.staging/` contents | empty |
| Final `.runtime_temp/storage_rotation_production_activation_rehearsal_v1/` contents | empty |
| Full database copy created | **confirmed NOT created** |
| Production archive created | **confirmed created — exactly 1** |
| Source row modified or deleted | **confirmed NOT occurred** |
| Purge available | **confirmed NOT available** |
| Scheduler available | **confirmed NOT available** |
| VACUUM available | **confirmed NOT available** |
| General production activation | **confirmed remains disabled** |

## 9. Verdict

**`STORAGE_ROTATION_RETENTION_PRODUCTION_ACTIVATION_REHEARSAL_V1_COMPLETE`**
**`STORAGE_ROTATION_RETENTION_SINGLE_PRODUCTION_PARTITION_VERIFIED_SOURCE_RETAINED`**
**Next gate (not begun):** `BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ARCHIVE-ACTIVATION-V1`
**Execution stopped:** confirmed — no general production activation, purge, scheduler integration, collector change, VACUUM, or outcome access occurred at any point in this batch. Exactly one production archive partition exists; the source table remains fully present and byte-unchanged.
