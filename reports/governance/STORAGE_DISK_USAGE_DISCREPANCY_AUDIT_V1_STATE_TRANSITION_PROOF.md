# STORAGE_DISK_USAGE_DISCREPANCY_AUDIT_V1 — State-Transition Proof

**Gate:** BATCH-STORAGE-DISK-USAGE-DISCREPANCY-AUDIT-V1
**Date:** 2026-07-07 · **Author:** Sonnet 5
**Outcome:** `STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED` — a **null state transition for every mutable resource this batch could affect**. All filesystem and database interaction was read-only metadata inspection.

---

## 1. Filesystem mutation proof

| Metric | Value |
|---|---|
| Files deleted | 0 |
| Directories deleted | 0 |
| Files moved | 0 |
| Files renamed | 0 |
| Files compressed | 0 |
| Sparse flags changed | 0 |
| Permissions changed | 0 |
| Ownership changed | 0 |
| Recycle Bin emptied | 0 |
| Shadow copies deleted | 0 |
| Restore points deleted | 0 |
| USN journal changed | 0 (confirmed not even enabled — nothing to change) |
| Databases copied | 0 |
| Database rows read for research analysis | 0 |
| Database rows inserted | 0 |
| Database rows updated | 0 |
| Database rows deleted | 0 |
| VACUUM executions | 0 |
| WAL checkpoints forced | 0 |
| Collectors stopped | 0 |
| Processes killed | 0 |
| Production archives created | 0 |
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| Experiment creation | 0 |
| Result creation | 0 |
| Nullifier creation/consumption | 0 |
| Gate-receipt creation/update | 0 |
| Schema change | 0 |
| Runtime/risk/execution change | 0 |
| Paper/shadow/forward/live change | 0 |

Every command this batch issued was one of: `Get-Volume`, `Get-PSDrive`, `Get-CimInstance` (read-only WMI/CIM queries), `fsutil volume diskfree`/`fsutil fsinfo ntfsinfo`/`vssadmin` (all denied, Error 5, no state changed by a denied command), `Get-CimInstance Win32_ShadowStorage` (query, no mutation), `fsutil usn queryjournal` (query, returned "not enabled", no mutation), `Get-ChildItem -Force` (read-only directory listing), `du -sh`/`du -sb` (read-only recursive size measurement), `stat -c%s` (read-only file-size query), and `sha256sum`-equivalent Python hashing of `canonical.sqlite`/`knowledge.sqlite` (read-only). No command in this batch's history has a write, delete, move, or rename semantic.

The readiness/audit module (`ami/governance/storage_disk_usage_discrepancy_audit_v1.py`) is pure arithmetic and classification logic — it never imports `os`, `shutil`, `sqlite3`, `subprocess`, or `pathlib`, and never calls `execute`/`remove`/`unlink`/`rmdir`/`rename`/`move`/`open` — proven structurally by two dedicated AST-walk tests, not merely claimed.

## 2. Canonical / knowledge immutability (byte-identical)

| Field | Before | After |
|---|---|---|
| `canonical.sqlite` sha256 | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` (unchanged) |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged) |
| `schema_version` | 14 | 14 |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `epistemic_test_nullifiers` | 2 | 2 |
| `experiment_gate_receipts` | 2 | 2 |

M-0036 canonical rows and every prior accepted family artifact remain unchanged (same sha256/count check used throughout this session).

## 3. `microstructure.db` — permitted concurrent growth, honestly reported

| Metric | Prior readiness batch end | This audit's end |
|---|---|---|
| File size | 758,774,398,976 bytes | 759,020,118,016 bytes |
| WAL size | 5,800,992 bytes | 6,109,992 bytes |
| Growth during this batch | — | 245,719,040 bytes (~234 MB, from live collectors, not this batch's own writes) |

Not falsely claimed unchanged, per this gate's own explicit allowance.

## 4. Focused tests

`tests/test_ami_governance_storage_disk_usage_discrepancy_audit_v1.py` — **25/25 passed**. Covers: GB/GiB unit conversion correctness (including the exact 2TB-drive figures measured this batch), the real-data reconciliation closing under the 2% threshold, negative-remaining-not-clamped safety (accounting-bug detection), 100%-unexplained edge case, unit-labeling reconciliation showing both interpretations with the decimal-GB delta proven smaller than the GiB-mislabeled delta (the batch's core finding, directly tested), chrome-copy classification from the real gathered evidence plus boundary cases (active-process-wins, recent-write-is-unknown-not-disposable), pytest-scratch classification across all four branches, verdict determination at all four dispositions including fail-closed precedence (measurement-inconsistency beats permission-block beats percentage), exact threshold boundary (98.0% passes, 97.999% does not), next-gate selection at and below the 1GB trigger using this batch's own real 13MB stray-file measurement, and two structural AST guards proving the module never imports a filesystem/database module or calls a mutating function.

## 5. Regression

Additive-only batch: one new pure-Python arithmetic/classification module (zero filesystem/database access, proven structurally) plus one new test file. No schema, no shared write path, no collector, no other family's code touched. Established baseline unaffected.

## 6. Storage report

| Item | Value |
|---|---|
| Peak temporary disk usage | 0 bytes |
| Temporary files created | 0 |
| Temporary files deleted | 0 |
| Temporary files retained | 0 |
| Final `.runtime_temp/` contents | unchanged |
| Final `.pytest_temp/` contents | unchanged (empty) |
| Data/backup/archive/database/browser-profile copy created | **confirmed NOT created** — every inspection was a metadata-only read (`du`, `stat`, `Get-Volume`/`Get-PSDrive`/`Get-CimInstance`, `fsutil` queries, sha256 of the two small governed databases only — never `microstructure.db` or any browser-profile file) |

## 7. Verdict

**`STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED`** (99.90% of used space reconciled, well past the 98% threshold)
**Next gate (not begun):** `BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1`
**Execution stopped:** confirmed — no deletion, move, rename, compression, permission change, VACUUM, WAL checkpoint, collector change, or outcome access occurred at any point in this batch.
