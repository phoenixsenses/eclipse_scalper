# S34_BOOK_SPREAD_DYNAMICS_CANONICAL_MIGRATION_V1

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1
**Migration ID:** M-0036
**Nature:** Controlled additive canonical migration only. No preregistration, no experiment ID, no nullifier action, no gate receipt, no TEST/outcome access, no scientific model, no route/bucket promotion.
**Depends on (source of truth, unedited):** readiness/definition commit `f115b9c1`, disposable rehearsal commit `6a449a64`, row-accounting freeze commit `54d00dca`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Family / child identity

| Field | Value |
|---|---|
| Family | `FAM_BOOK_SPREAD_DYNAMICS` |
| Child working ID | `H-BOOK-SPREAD-CHANGE-BPS-W300-V1` |
| Formula version | `BOOK_SPREAD_CHANGE_BPS_W300_V1` |
| Definition | `spread_change_bps = spread_bps(t0) − spread_bps(t0−300s)`, `mid=(ask+bid)/2`, `spread_bps=1e4·(ask−bid)/mid` |
| Source | Binance USD-M perp `book_ticker` L1 (ETHUSDT), at-or-before quote, 5-min staleness, `id DESC` tie-break |
| Specification hash | `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212` |
| Row-accounting root (frozen) | `33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31` |

The operator's freeze ruling (W300 horizon + additive spread-bps difference transform) is materialized verbatim. No definition, window, transform, unit, quality-precedence, or exclusion identity is changed by this migration.

---

## Migration-ID and schema-version ruling (from repository evidence)

- **Migration ID = M-0036.** `MIGRATION_LOG.md`'s prior entries end at M-0035 (canonical) / M-0034 (knowledge); M-0036 was free. Resolved, not assumed.
- **schema_version 13 → 14.** `ami/warehouse/schema.py::CANONICAL_SCHEMA_VERSION` + the `init_schema()` convention: each additive `_SCHEMA_PHASE_*` block increments the version by exactly 1. Confirmed by the unbroken precedent chain M-0030 (10→11), M-0031 (11→12), M-0035 (12→13). This batch adds one new `_SCHEMA_PHASE_BOOK_SPREAD` block (3 additive tables + their indexes), so the version increments 13→14 exactly as every prior additive phase did. The only authorized schema delta is those 3 new tables/indexes.

---

## Input package verification (pre-migration)

The retained frozen source (`.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite`) row-accounting root was independently recomputed **before** the migration and equals the frozen `33c4f4be…`. No feature value is recomputed; all rows are copied verbatim.

---

## Pre-migration checkpoint

| Field | Value |
|---|---|
| `data/ami/canonical.sqlite` absolute path | `D:\eclipse_scalper\data\ami\canonical.sqlite` |
| `schema_version` | 13 |
| Full-file sha256 | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` |
| `integrity_check` | ok |
| `foreign_key_check` | [] (clean) |
| Table count | 42 |
| `experiment_registry` / `experiment_results` | 24 / 381 |
| `epistemic_test_nullifiers` / `experiment_gate_receipts` | 2 / 2 |
| `researcher_exposure_ledger` | 1,180 |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` |
| Protected identity counts | `ami_events`=252, `ami_signal_lifecycle`=324, `ami_cycles`=167, `ami_birth_truncated_cascade_geometry`=220, `ami_cvd_windowed_flow`=1,840, `ami_absorption_impact_windowed_flow`=1,619 |

---

## Backup and disposable restore proof

- Backup: `data/ami/backups/canonical_pre_M0036_book_spread_dynamics_canonical_migration_20260707_151140.sqlite`, sha256 `3aefce83…` — **byte-exact** match to the live pre-migration file (manifest sidecar: `…_151140.manifest.json`). Method: `shutil.copy2` after `PRAGMA wal_checkpoint(FULL)`.
- Restored to a disposable path (`.runtime_temp/M0036_restore_verify/`, never overwriting the live file), verified: `schema_version=13`, **zero** `ami_book_spread_change_*` tables present, `integrity_check=ok`, `foreign_key_check=[]`, `experiment_registry=24`. Restored sha256 equals pre-migration. Disposable restore copy deleted after verification.

---

## Migration structures (schema 13→14)

Three tables added via `ami/warehouse/schema.py::_SCHEMA_PHASE_BOOK_SPREAD`, wired into `init_schema()`. All carry FK `anchor_id → ami_signal_lifecycle(signal_id)`, the frozen `row_accounting_root='33c4f4be…'` CHECK, and known-at CHECKs (`current_target_ts=signal_birth_ts`, `historical_target_ts=signal_birth_ts−300000`, `known_at_ts=signal_birth_ts`, `feature_available_ts=signal_birth_ts`, `current_quote_ts≤current_target_ts`, `historical_quote_ts≤historical_target_ts`). Insert-only / immutable.

| Table | Grain | Rows | Key constraints |
|---|---|---|---|
| `ami_book_spread_change_windowed_flow` | one row per EXACT anchor (feature) | 196 | PK `feature_id`; UNIQUE `(anchor_id, formula_version)`; `source_quality_class='EXACT_RECONSTRUCTABLE'`; `symbol='ETHUSDT'`, `venue='BINANCE_USDM_PERP'`, `market_segment='PERPETUAL_FUTURES'`, `quote_currency='USDT'` |
| `ami_book_spread_change_window_quality_v1` | one row per anchor (accounting) | 324 | PK `quality_id`; UNIQUE `(anchor_id, formula_version)`; `(exact_eligibility_flag=1) = (source_quality_class='EXACT_RECONSTRUCTABLE')` |
| `ami_book_spread_change_exclusions` | one row per non-exact anchor | 128 | PK `exclusion_id`; UNIQUE `(anchor_id, formula_version)`; `source_quality_class != 'EXACT_RECONSTRUCTABLE'`; `exclusion_precedence_position BETWEEN 0 AND 4` |

New module `ami/research/book_spread_dynamics_canonical_migration.py::run_canonical_migration()` copies all rows verbatim (0 recomputation, 0 network calls) from the retained frozen source `rehearsal_run1.sqlite`, deriving only the constant `row_accounting_root`/`migration_id` provenance and the mechanical `exact_eligibility_flag`/`exclusion_precedence_position`. Idempotency is content-compare (`NOOP_IDENTICAL`); a non-identical collision raises `ConflictNonIdentical` (fail-closed, matching M-0031/M-0035 precedent).

No outcome column, no alternative-window column, and no alternative-transform column exists in any of the three tables.

---

## Row accounting

| Metric | Value |
|---|---|
| Feature rows (EXACT) | 196 |
| Quality rows (all anchors) | 324 |
| Exclusion rows (non-exact) | 128 |
| Accounting identity | 324 = 196 EXACT + 22 STALE + 106 UNAVAILABLE ✓ |
| Exclusion identity | 128 = 22 STALE + 106 UNAVAILABLE ✓ |
| Quality breakdown | EXACT=196, STALE_SOURCE=22, UNAVAILABLE_BEFORE_COLLECTION=106 |
| Exact independent cycles | 97 |
| Cycle representatives | 97 (0 duplicate representatives) |
| EXACT rows in exclusion table | 0 |
| Excluded rows in feature table | 0 |
| Distinct `row_accounting_root` | `33c4f4be…` (single) |
| Distinct `migration_id` | `M-0036` (single) |

---

## Canonical replay — row-level frozen-manifest equality

The five frozen accounting manifests were **rebuilt directly from the destination canonical tables** (not from the retained manifest file) using the frozen ordering (`signal_birth_ts ASC, anchor_id ASC`) and serialization, and reproduce the frozen component hashes exactly:

| Manifest | Hash |
|---|---|
| ordered_anchor | `a77a8daf2a8d198d775436674a20a9bd5328dc071e2883938b7c331c17c534bb` |
| exact_feature | `b1eb902f5b3d1ea0f19b4b60d0ad999907a042b228adf506bbe09800a81e155b` |
| exclusion | `0694e43300710e1204c1b23643d9eacb9f10188c21aa0ceda572c28229cc8449` |
| cycle_membership | `e692ff1c8ce37b54a3349a501a38bd44f24865e75a51accc81c7e97399d29e18` |
| representative | `edadf5972cbbdddb0efa1db8234473ee089972f504d3bfbfafbae508238db246` |

All five match the frozen freeze → **full row-level population equality**, not merely matching aggregate counts. Frozen manifest sha256 (5-component composite): `0a65c45ffba906414c7a484e3f966e2405017eaea8990aded429dc35ed142c89`.

---

## Known-at and access proof

A SQLite authorizer (`SQLITE_DENY` on `ami_lifecycle_path_observations`, `epistemic_test_nullifiers`, `experiment_gate_receipts`; on columns `endpoint_return_bps`/`mfe_bps`/`mae_bps`; and on writes to `experiment_registry`/`experiment_results`) was installed around the data-copy + verification reads — **not** around the pre-existing, unrelated schema-DDL application, which legitimately owns earlier outcome-table `CREATE TABLE IF NOT EXISTS` definitions.

Result: **`outcome_or_governance_table_access=[]`, `outcome_column_access=[]`, `monitored_governance_write=[]`** — zero attempts, on both the live migration run and its idempotent rerun. Known-at is enforced structurally by table CHECK constraints (any violating row fails its INSERT); post-migration re-verification: 0 mismatches on `current_quote_ts`/`historical_quote_ts`/`known_at_ts`/`feature_available_ts`.

---

## Idempotent rerun

The migration was run a second time against the now-migrated live `canonical.sqlite`:

| Check | Result |
|---|---|
| Rows inserted | 0 (all three tables) |
| Rows `noop_identical` | 196 / 324 / 128 |
| Replay hashes | identical to run 1 (all 5 match frozen) |
| `schema_version` | unchanged (14) |
| Protected counts | unchanged |
| Full-file sha256 | **differs** — `schema_versions.applied_ms` bookkeeping timestamp is unconditionally upserted by `init_schema()` on every call, a pure wall-clock field with no data-content meaning (identical disclosure discipline to M-0035) |

**Result: `NOOP_IDENTICAL`** at the content level.

---

## Regression

Paired single-process sweep (`≤2 test files per pytest invocation`, sequential, `--basetemp` scratchpad, `-p no:cacheprovider`) across **83 files / 42 batches**. Result before the one fix below: **7 failing batches, 19 failing nodes.** After the fix: **18 failing nodes, all pre-existing waived; 0 net new deterministic failures introduced by M-0036.**

**Root-cause separation (proven, not asserted) — exactly one failure is attributable to this batch:**

1. **`test_ami_lifecycle_provenance_rehearsal.py::test_full_provenance_rehearsal_real_data`** — hardcoded `schema_version_before in (…,13)`, which passed at schema 13 and fails at 14. **Caused by this batch** (schema bump 13→14), the same precedent pattern as every prior schema bump. **Fixed**: tuple extended to `(…,13,14)` with matching comment, exactly as M-0035 extended it to 13. Verified green in isolation (provenance file: 2 passed + 1 failed → **3 passed**).

2. **The remaining 18 failures are pre-existing and provably NOT caused by M-0036:**
   - **Schema-hash "unchanged" pins (2):** `test_ami_epistemic_nullifier_enforcement_wiring.py::test_26_…` and `test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_22_23_…` both assert `version == 12`. They were **already failing at schema 13** (post-M-0035), so my 13→14 bump does not change their already-failing status.
   - **Governance-state drift pins (16):** the `..._preregistration_v1` invariant tests, the nullifier enforcement/legacy tests (`existing 22 experiments`, `retro audit 0 of 22`, `no new experiment`, `real nullifier and receipt state`, `gate receipt round-trip`), and the `..._001` governed-execution dress-rehearsal / verify-pre-execution tests. All pin `experiment_registry`/`experiment_results`/nullifier/receipt state that **M-0036 provably never touched** (protected-delta = ZERO; authorizer log empty on both runs; `knowledge.sqlite` byte-identical `710b3f68…` before and after). They went stale due to the **prior, separate** G2-CVD-PRIMARY-LONG governed execution (`60c3e26f`) and absorption preregistration work committed **before** this batch began (HEAD `fc43e972`). They are structurally incapable of having been changed by an outcome-blind additive schema/data migration.

Full failing-node list (18, post-fix):
```
test_ami_absorption_impact_preregistration_v1.py::test_no_experiment_created_and_canonical_invariants_hold
test_ami_cvd_primary_long_preregistration_v1.py::test_gate_receipt_mechanism_round_trips_on_disposable_copy
test_ami_cvd_primary_long_preregistration_v1.py::test_real_nullifier_and_receipt_state
test_ami_cvd_primary_long_preregistration_v1.py::test_no_experiment_created_and_canonical_invariants_hold
test_ami_epistemic_nullifier_enforcement_wiring.py::test_24_existing_22_historical_experiments_remain_unchanged
test_ami_epistemic_nullifier_enforcement_wiring.py::test_25_retro_audit_remains_0_of_22
test_ami_epistemic_nullifier_enforcement_wiring.py::test_26_canonical_schema_version_and_hash_unchanged
test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_17_18_existing_22_experiments_and_results_unchanged
test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_19_retro_audit_remains_0_of_22
test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_20_no_new_experiment_created_by_this_batch
test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_21_no_scientific_result_generated_by_this_batch_real_hash_unchanged
test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_22_23_canonical_schema_version_and_hash_unchanged
test_ami_research_cascade_absorption_impact_001.py::test_verify_pre_execution_reports_zero_errors_against_fresh_disposable_state
test_ami_research_cascade_absorption_impact_001.py::test_verify_pre_execution_detects_already_executed_state
test_ami_research_cascade_absorption_impact_001.py::test_governed_execution_dress_rehearsal_on_disposable_copies
test_ami_research_cvd_windowed_flow_001.py::test_verify_pre_execution_reports_zero_errors_against_real_db
test_ami_research_cvd_windowed_flow_001.py::test_execute_governed_run_blocks_on_identity_mismatch
test_ami_research_cvd_windowed_flow_001.py::test_governed_execution_dress_rehearsal_on_disposable_copies
```

The 13 focused migration tests (`test_ami_research_book_spread_dynamics_canonical_migration.py`) all pass.

**Operator decision:** accept `BOOK_SPREAD_DYNAMICS_CANONICAL_MIGRATION_V1_COMPLETE` with the 18 pre-existing, root-cause-proven-unrelated failures fully disclosed rather than silently reconciled or hidden under a waiver. Remediating those hardcoded G2-execution-era / earlier-schema checkpoints is explicitly out of scope for this migration and requires its own separate batch/operator decision.

---

## Final canonical validation

| Check | Value |
|---|---|
| Final `schema_version` | 14 |
| `ami_book_spread_change_windowed_flow` | 196 |
| `ami_book_spread_change_window_quality_v1` | 324 |
| `ami_book_spread_change_exclusions` | 128 |
| Accounting reconciliation | 324 = 196 + 22 + 106 ✓ |
| Known-at violations | 0 |
| Outcome reads / writes | 0 / 0 |
| `experiment_registry` / `experiment_results` | 24 / 381 (unchanged) |
| `epistemic_test_nullifiers` / `experiment_gate_receipts` | 2 / 2 (unchanged) |
| `knowledge.sqlite` sha256 | `710b3f68…` (unchanged, never opened) |
| Pre-existing protected tables | all content-identical (42→45 tables, +3 new only) |
| `integrity_check` | ok |
| `foreign_key_check` | [] |
| Final `canonical.sqlite` sha256 (post idempotent rerun) | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` |
| (first-migration sha256, pre-rerun) | `ddb9d72b8d1ff67c1092d824215a3806fe305d2a4d65b60707f14cb20b87adac` |

---

## Immutability after migration

Insert-only; no UPDATE/DELETE/REPLACE. Any future correction requires a new migration, a new version, an explicit relationship to V1, preservation of V1 evidence, and new operator authorization.

## Remaining risks

1. The 18 pre-existing failures (2 earlier-schema pins asserting `version==12`; 16 G2-execution-era governance pins) remain red in the full suite. They are proven unrelated to `FAM_BOOK_SPREAD_DYNAMICS` and require a separate remediation batch that is explicitly out of this migration's scope.
2. The migrated 196-row / 97-cycle usable population carries the permanent 22 STALE + 106 UNAVAILABLE exclusions; a future preregistration inherits exactly this frozen population.
3. The migration's frozen source is `rehearsal_run1.sqlite` specifically — an arbitrary but reproducible and disclosed choice; the retained run is content-identical to the frozen root already proven at freeze.

## Success verdicts

**`BOOK_SPREAD_DYNAMICS_CANONICAL_MIGRATION_V1_COMPLETE`**

**`BOOK_SPREAD_DYNAMICS_CANONICAL_DATA_READY_FOR_PREREGISTRATION`**

Stopping after canonical migration. No preregistration or hypothesis execution begins without new, separate operator instruction. Recommended next gate: **BATCH-BOOK-SPREAD-DYNAMICS-PREREGISTRATION-V1**.
