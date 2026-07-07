# STORAGE_ROTATION_RETENTION_READINESS_AND_CONTRACT_V1 — State-Transition Proof

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-READINESS-AND-CONTRACT-V1
**Date:** 2026-07-07 · **Author:** Sonnet 5
**Outcome:** `STORAGE_ROTATION_RETENTION_READY_WITH_RESEARCH_DEPENDENCY_BLOCKERS` — a **null state transition for every governed database**. `microstructure.db` grew concurrently under live collector writes (expected, disclosed, not this batch's own writes).

---

## 1. Live-database mutation proof

| Metric | Value |
|---|---|
| Live database rows deleted (by this batch) | 0 |
| Live database rows inserted (by this batch) | 0 |
| Live database rows updated (by this batch) | 0 |
| Live database tables created (by this batch) | 0 |
| Live database tables dropped (by this batch) | 0 |
| `VACUUM` executions | 0 |
| Incremental-vacuum executions | 0 |
| WAL checkpoints forced | 0 |
| Collectors stopped | 0 |
| Collectors restarted | 0 |
| Collector configuration changes | 0 |
| Archive production files created | 0 |
| Archive production files deleted | 0 |
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| Outcome-value reads | 0 |
| Experiment creation | 0 |
| Experiment-result creation | 0 |
| Nullifier creation | 0 |
| Nullifier consumption | 0 |
| Gate-receipt creation | 0 |
| Gate-receipt update | 0 |
| Schema change | 0 |
| Canonical migration | 0 |
| Route/bucket promotion | 0 |
| Runtime/risk/execution delta | 0 |
| Paper/shadow/forward/live behavior delta | 0 |

Every database interaction this batch performed was a bounded, read-only (`mode=ro` where applicable, `PRAGMA query_only=ON` on `microstructure.db`) `SELECT`, `MIN()`, `MAX()`, or `PRAGMA` statement. The readiness module itself (`ami/governance/storage_rotation_retention_readiness_v1.py`) contains **zero** `execute`/`executescript`/`executemany`/`connect` calls and **never imports `sqlite3`** — proven structurally by two AST-walk tests, not merely claimed.

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

M-0036 canonical rows, both book-spread INCOMPLETE preregistration artifacts, and every prior accepted family artifact remain unchanged (confirmed via the same sha256/count check used by every prior batch this session).

## 3. `microstructure.db` — permitted concurrent change, honestly reported (not falsely required to be unchanged)

Per this gate's own instruction, `microstructure.db`'s full-file hash is **not** required to remain unchanged, since 4+ live collectors write to it continuously and this batch ran alongside them.

| Metric | Inspection start | Inspection end |
|---|---|---|
| File size | 758,526,558,208 bytes (from the prior batch's own record, ~24 min earlier) | 758,774,398,976 bytes |
| WAL size | 5,273,632 bytes | 5,800,992 bytes |
| Collector activity status | live (4+ collectors) | live (unchanged) |

**Confirmation that the audit itself performed no writes:** every query issued against `microstructure.db` this batch was a `SELECT MAX(id)`, `SELECT MIN(ts_ms)`, `SELECT MAX(ts_ms)`, `SELECT COUNT(*) FROM gaps GROUP BY stream` (812 rows, cheap), or a `PRAGMA` read, all via a `PRAGMA query_only=ON` connection — a read-only pragma that causes SQLite itself to reject any write attempt at the connection level, an additional structural guarantee beyond just "we didn't write."

## 4. Focused tests

`tests/test_ami_governance_storage_rotation_retention_readiness_v1.py` — **27/27 passed**. Covers: exactly-one-class-per-table, `CANONICAL_IMMUTABLE`/`CONTINUITY_CRITICAL_ACTIVE`/`RESEARCH_CRITICAL_COMPACT` never purge-eligible, `RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE` conditional-not-unconditional, `book_ticker`'s specific research-dependency block, stray-test-file classification, unclassified-table fail-closed detection, 30-day (not 14-day) active-horizon default, Parquet/ZSTD-only archive format with forbidden-format guard, closed-UTC-month partitioning, the storage-health-state function's boundary behavior at all four thresholds plus its fail-toward-more-severe-on-disagreement property (dedicated test) and determinism, permitted-response-never-authorizes-deletion guard, all 24 failure modes fail-closed, and two AST-level structural guards proving the module never opens a database connection or calls a write-capable SQL method.

## 5. Regression

Additive-only batch: one new pure-Python design/policy module (zero database access, proven structurally) plus one new test file. No schema, no shared governance-write path, no collector, no other family's code touched. Established baseline unaffected — nothing in this batch's write-set overlaps with any test that pins governance counts, schema version, or canonical/knowledge content.

## 6. Storage report

| Item | Value |
|---|---|
| Peak temporary disk usage | ~0 bytes (no fixture, export, or copy created) |
| Temporary files created | 0 |
| Temporary files deleted | 0 |
| Temporary files retained | 0 |
| Final `.runtime_temp/` contents | unchanged (3.7 MB, pre-existing) |
| Final `.pytest_temp/` contents | unchanged (empty) |
| Production archive created | **confirmed NOT created** |
| Live row deleted or changed | **confirmed NOT occurred** |
| Full database copy created | **confirmed NOT created** (all inspection was bounded `mode=ro`/`PRAGMA query_only` reads) |

## 7. Verdict

**`STORAGE_ROTATION_RETENTION_READY_WITH_RESEARCH_DEPENDENCY_BLOCKERS`**
**Next gate (not begun):** `BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1`
**Execution stopped:** confirmed — no destructive action, no archive, no VACUUM, no collector change, no schema/canonical mutation, no outcome access occurred at any point in this batch.
