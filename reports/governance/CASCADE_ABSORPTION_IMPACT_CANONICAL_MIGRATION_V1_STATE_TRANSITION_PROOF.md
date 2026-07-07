# CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-CASCADE-ABSORPTION-IMPACT-CANONICAL-MIGRATION-V1
**Migration ID:** M-0035
**Purpose:** Migrate only the frozen absorption/impact data product from the accepted immutable rehearsal package into canonical SQL (schema 12→13).
**Prior checkpoint (unchanged, not reopened):** commit `931cd3dd` (`CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1_COMPLETE`), readiness verdict `ABSORPTION_IMPACT_ROW_ACCOUNTING_FROZEN_FOR_CANONICAL_MIGRATION`.
**Nature:** Canonical data migration only. No preregistration, no experiment ID, no nullifier action, no outcome/TEST access, no scientific model, no route/bucket promotion, no runtime/risk/execution modification.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Previous state root → new state root

| Field | Before | After |
|---|---|---|
| `schema_version` | 12 | **13** |
| `canonical.sqlite` sha256 | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` (post idempotent-rerun; the first-migration-only hash was `91cd794f84cfb366712914c9b5c7bf6800fc0934948d483bcd7c53c92f0cb6d0`, differing from the rerun's hash only in `schema_versions.applied_ms` bookkeeping) |
| Table count | 39 | 42 (+3: `ami_absorption_impact_windowed_flow`/`window_quality_v1`/`exclusions`) |
| `experiment_registry` / `experiment_results` | 23 / 350 | 23 / 350 (unchanged) |
| `knowledge.sqlite` | untouched | untouched (mtime unchanged throughout) |

---

## Naming ruling applied

Operator ruling: canonical production naming is `ami_absorption_impact_*`, superseding the contract's illustrative `ami_impact_*` and the rehearsal's disposable `absorption_impact_*`. Applied as a pure naming normalization — formula, units, windows, row identities, source manifests, quality states, exclusion identity, `FLOOR_USD_M=0.01`, and feature values are byte-identical to the frozen rehearsal (proven by content-hash parity, §Content verification below).

---

## Sequence executed

1. **Pre-flight collision check:** queried live `canonical.sqlite` for `schema_versions` rows and any table/view matching `ami_absorption_impact%`/`%absorption%`/`%impact%` — none found. Checked `MIGRATION_LOG.md`'s last entry (M-0034) — M-0035 was free.
2. **Input package verification:** recomputed sha256 of all 4 files from `fc1321f5` + 6 files from `fc43e972` (10 total), the code hash, the 4 retained rehearsal-evidence files, and 7 source/canonical table schema hashes (`agg_trades`, `mark_prices`, `gaps`, `ami_signal_lifecycle`, `ami_agg_trades_repaired`, `ami_events`, `ami_cycles`) — **23/23 matched the row-accounting freeze's recorded values exactly, zero drift.**
3. **Pre-migration checkpoint:** recorded absolute path, `schema_version=12`, full-file sha256, `integrity_check=ok`, `foreign_key_check=[]`, 39 tables/6 views, `experiment_registry=23`, `experiment_results=350`, `researcher_exposure_ledger=1176`, `knowledge.sqlite` nullifier=1/gate_receipt=1, protected identity counts (events=252/signals=324/cycles=167/geometry=220).
4. **Backup + disposable restore proof:** byte-exact backup (`data/ami/backups/canonical_pre_M0035_absorption_impact_canonical_migration_20260707_065549.sqlite`, sha256 identical to source), manifest sidecar recorded. Restored to a disposable path (never the live file), independently verified `schema_version=12`, zero new tables, `integrity_check=ok`, `foreign_key_check=[]`, all protected counts matching the checkpoint. Disposable restore copy deleted after verification.
5. **Schema implementation:** added `_SCHEMA_PHASE_ABSORPTION_IMPACT` to `ami/warehouse/schema.py` (3 tables, FK additions + `window_id` enum CHECK as the only permitted delta vs. the rehearsal DDL — same discipline as `_SCHEMA_PHASE_CVD`'s 3 FK-line precedent), bumped `CANONICAL_SCHEMA_VERSION` 12→13, wired into `init_schema()`. Verified on a fresh in-memory DB: 3/3 tables created, `schema_version=13`, `integrity_check=ok`, `foreign_key_check=[]`.
6. **Migration module:** new `ami/absorption/cascade_absorption_impact_canonical_migration.py::run_canonical_migration()` — verbatim, order-preserving copy from the frozen retained source, content-compare idempotent, `FrozenSourceRowConflict` on same-identity/different-content (same precedent as `ami/cvd/cvd_canonical_migration.py`).
7. **Focused tests first:** `tests/test_ami_absorption_impact_canonical_migration.py` (+7: not-called-automatically guard, count reproduction, idempotent rerun, content-hash parity vs. the frozen freeze's recorded values, conflict-raise, exclusion-identity-never-in-both-tables, protected-invariants-unchanged) — run against a disposable copy of the real `canonical.sqlite` — **7/7 passed**, including FK resolution against real parent-table data.
8. **Disposable production-shaped dry run:** ran the exact live-migration driver script against a disposable copy of the live `canonical.sqlite` first, with the SQLite authorizer scoped around the migration step — schema 12→13, 1,619/1,620/1 rows inserted, content hashes matched the frozen freeze exactly, `known_at_violations=0`, access log clean. Disposable copy deleted after verification.
9. **Real migration:** ran the identical driver script against the live `data/ami/canonical.sqlite`. Result: schema 12→13; 1,619 usable + 1,620 quality + 1 exclusion row inserted (0 conflicts); content hashes `f7c834cc…`/`5d1a205c…`/`5e3ae2e5…` — byte-identical to the frozen row-accounting freeze's values; `known_at_violations=0`; `foreign_key_check=[]`; `integrity_check=ok`; access log clean (zero outcome-table, outcome-column, or monitored-table access).
10. **Content verification:** full row-set comparison (declared content columns, bookkeeping excluded) between the migrated canonical tables and the retained rehearsal package — 0 missing, 0 extra, 0 value drift, exact per-window/quality/exclusion-identity match.
11. **Idempotent rerun:** ran the migration a second time against the now-migrated live file — 0 inserts, all rows `noop_identical`, content hashes unchanged, `schema_version` unchanged (13), protected counts unchanged, access log clean. Full-file hash changed only via the `schema_versions.applied_ms` bookkeeping upsert (disclosed, not data).
12. **Regression (collect-only + 2 full passes + fix + reconfirmation):** see dedicated section below.
13. **Final canonical validation:** re-verified every acceptance equation; recorded final hash.
14. **Documentation + `MIGRATION_LOG.md`:** this proof, the migration report, and one M-0035 row.

---

## Regression — full account, including the process violation

**Collect-only:** 987 tests, 76 files (unchanged before/after this batch's own test file addition, other than the +7 new tests).

**Two initial clean full paired passes** (properly sequential, single background invocation, no overlap) reproduced an **identical** failure set: batches 4 (3 failures), 7 (8 failures), 19 (2 failures), 21 (3 failures) = 16 failures. This determinism proof preceded any fix.

**Root-cause investigation** (detailed in the migration report) proved: 1 failure (batch 19, `test_full_provenance_rehearsal_real_data`) was caused by this batch's schema bump and was fixed (tuple extended to include 13); 1 failure (batch 19, `test_disposable_db_and_microstructure_db_untouched`) is a live-`microstructure.db`-collector timing artifact, proven by a clean isolated single-file rerun; the remaining 14 (batches 4, 7, 21) are proven pre-existing and unrelated to this family, caused by the separate, prior G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1 batch (commit `60c3e26f`).

**Process violation (disclosed):** a subsequent confirmation attempt was run incorrectly — a synchronous shell call that hit the tool's 2-minute timeout left its underlying process tree running detached, and a second, independent full-suite invocation was then started via `run_in_background` before the first had actually terminated. For a period, two full pytest sweeps were running concurrently — a violation of the repo's no-parallel-Python-processes guardrail. This was detected from anomalous symptoms (one batch pair took 308s vs. its normal ~140s; a spurious `FileNotFoundError: [WinError 3]` on a session-fixture file copy in `test_ami_geometry_liquidation_source_quality_contract_v2.py`; duplicate log lines for the same batch with two different timings). The contaminated logs (`M0035_final1.log`, `M0035_final2.log`) were discarded without being used as evidence. `ps aux` was checked clean of any eclipse_scalper-related process before proceeding.

**Clean confirmation pass** (single background invocation, verified no prior process running, post-fix): 987 collected, **973 passed, 14 failed** — exactly batches 4/7/21 (the proven-pre-existing set), batch 19 fully green (both sub-tests, including the previously-flaky microstructure.db check), and the transient `FileNotFoundError` did not recur — confirming it was contamination noise from the parallel-process violation, not a real, reproducible failure.

**Operator decision (recorded):** presented via `AskUserQuestion` — given the choice between (a) accept COMPLETE with full disclosure, (b) close INCOMPLETE per the letter of "both passes completely green," or (c) additionally fix the 13 pre-existing G2-related tests now — the operator selected **(a): accept as COMPLETE, disclose fully**, explicitly leaving the 13 G2-related test files untouched as out of this migration's scope.

---

## Known-at and access proof

SQLite authorizer (`SQLITE_DENY` on `ami_lifecycle_path_observations` table access, `endpoint_return_bps`/`mfe_bps`/`mae_bps` column access, and any INSERT/UPDATE/DELETE on `experiment_registry`/`experiment_results`/`epistemic_test_nullifiers`/`experiment_gate_receipts`) was installed **only around the migration's data-copy step and its immediate post-migration verification queries** — not around `init_schema()`, which legitimately must be able to define `ami_lifecycle_path_observations` (a pre-existing table from an earlier, unrelated migration phase) via its own `CREATE TABLE IF NOT EXISTS`. Scoping the authorizer any wider produced a false-positive `sqlite3.DatabaseError: not authorized` during ordinary, pre-existing schema DDL — caught during the disposable dry run, fixed before the real migration touched the live file.

Result (both the real migration and its idempotent rerun): `outcome_table_access=[]`, `outcome_column_access=[]`, `monitored_table_writes=[]`.

---

## Backup/restore proof (detail)

| Item | Value |
|---|---|
| Backup path | `data/ami/backups/canonical_pre_M0035_absorption_impact_canonical_migration_20260707_065549.sqlite` |
| Backup sha256 | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` (byte-exact to source) |
| Manifest | `…_065549.manifest.json` |
| Restore-proof record | `…_065549.restore_proof.json` |
| Restored `schema_version` | 12 |
| Restored `ami_absorption_impact_*` tables present | none |
| Restored protected counts | `experiment_registry`=23, `experiment_results`=350, `researcher_exposure_ledger`=1176, `ami_events`=252, `ami_signal_lifecycle`=324, `ami_cycles`=167, `ami_birth_truncated_cascade_geometry`=220 (all match pre-migration checkpoint) |
| `integrity_check` / `foreign_key_check` | ok / [] |
| Restored to live path | **never** |

---

## Table schemas (canonical, as applied)

See `ami/warehouse/schema.py::_SCHEMA_PHASE_ABSORPTION_IMPACT` for the exact DDL. Summary: `ami_absorption_impact_windowed_flow` (26 columns, PK `feature_id`), `ami_absorption_impact_window_quality_v1` (15 columns, PK `quality_id`), `ami_absorption_impact_exclusions` (6 columns, PK `exclusion_id`) — column lists, CHECK constraints, and FK targets identical to those specified in the row-accounting freeze's Phase 8 manifest (`S34_CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1.json`).

---

## Row accounting (final)

| Table | Row count | Content hash |
|---|---|---|
| `ami_absorption_impact_windowed_flow` | 1,619 | `f7c834cc8ebe90708e308629f1921a050d58520ad5560422b09406a7d1ca8942` |
| `ami_absorption_impact_window_quality_v1` | 1,620 | `5d1a205c7f79ca1b269307e34750c0d46dc104c8a799e9b4d01c862d307d7ba0` |
| `ami_absorption_impact_exclusions` | 1 | `5e3ae2e524fcdbd5d045698a5a14bd397ae2c21bf0ff9ae2f54f2502c35a3ff7` |

Universe reconciliation: 1,619 + 1 = 1,620 ✓. All three hashes byte-identical to the frozen row-accounting freeze's recorded values.

---

## Protected delta

| Table | Before | After |
|---|---|---|
| `ami_events` | 252 | 252 |
| `ami_signal_lifecycle` | 324 | 324 |
| `ami_cycles` | 167 | 167 |
| `ami_birth_truncated_cascade_geometry` | 220 | 220 |
| `ami_agg_trades_repaired` | 40,934 | 40,934 |
| `ami_cvd_windowed_flow` / `_proxy` | 1,840 / 1,840 | 1,840 / 1,840 |
| `experiment_registry` | 23 | 23 |
| `experiment_results` | 350 | 350 |
| `researcher_exposure_ledger` | 1,176 | 1,176 |
| `knowledge.sqlite` (nullifiers/receipts) | 1 / 1 | 1 / 1 (file never opened) |

**Protected delta = ZERO.** Only the 3 new absorption-impact tables were added; no existing table, row, runtime, risk, or execution file changed.

---

## Exact changed/added-file manifest (this commit)

| File | Status | Content |
|---|---|---|
| `ami/warehouse/schema.py` | Modified | `CANONICAL_SCHEMA_VERSION` 12→13, new `_SCHEMA_PHASE_ABSORPTION_IMPACT` block, wired into `init_schema()`. **Disclosure:** this file already carried a prior, uncommitted `_SCHEMA_PHASE_CVD` addition (schema 11→12, from the still-uncommitted M-0031 CVD migration) inherited from before this session — this commit necessarily bundles that pre-existing code alongside this batch's own v12→v13 addition, since both live in the same file; the v12→v13 delta is the only content this batch authored. |
| `ami/absorption/cascade_absorption_impact_canonical_migration.py` | New | migration implementation |
| `tests/test_ami_absorption_impact_canonical_migration.py` | New | 7 focused tests |
| `tests/test_ami_lifecycle_provenance_rehearsal.py` | Modified | `schema_version_before` tuple extended `(8,9,10,11,12)`→`(8,9,10,11,12,13)` — same established precedent as every prior schema-version bump. **Disclosure:** this file's diff (vs. the last commit) also carries a prior, uncommitted v12 extension from the same inherited M-0031 state; this batch's own delta is the v13 addition only. |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1.md` | New | migration report |
| `reports/governance/CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_STATE_TRANSITION_PROOF.md` | New | this document |
| `MIGRATION_LOG.md` | Modified | one new M-0035 row (prepended). **Disclosure:** this file already carried prior, uncommitted M-0028–M-0034 rows inherited from before this session; this batch's own delta is the M-0035 row only. |

Not included: preregistration artifacts, TEST results, `SYSTEM_STATE.md`/`IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md` (shared governance projections, explicitly out of this commit's scope per operator instruction), runtime/risk/execution changes, or unrelated repository cleanup.

---

## Storage guardrail

| Item | Value |
|---|---|
| Peak temporary disk usage this batch | ~223 MB momentary (one disposable copy of `canonical.sqlite` for the dry-run + one for the restore-proof, both under `.runtime_temp`, both deleted immediately after verification); pytest `--basetemp` scratch (~dozens of MB per batch, cleaned after each of the ~114 paired invocations across all regression attempts) |
| Files created this batch | disposable dry-run copy (deleted), disposable restore-proof copy (deleted), 3 small JSON evidence files retained under `.runtime_temp` (`pre_migration_checkpoint_M0035.json`, `M0035_live_migration_result.json`, `M0035_live_migration_rerun_result.json`, `M0035_final_canonical_validation.json` — ~4KB each), one contaminated-then-discarded pair of regression logs (deleted), one clean regression log (`M0035_clean_final.log`, retained under the OS scratchpad, not part of the repo) |
| Files retained | the above 4 small JSON evidence files under `D:\eclipse_scalper\.runtime_temp` (referenced by this proof); the pre-migration backup + manifest + restore-proof JSON under `data/ami/backups/` (permanent, per backup policy) |
| Files deleted | both disposable full-DB copies; the contaminated `M0035_final1.log`/`M0035_final2.log` and their associated `pytest_final1_batch*`/`pytest_final2_batch*` basetemp directories |
| Full `microstructure.db` copy made | never |
| Remaining under `.runtime_temp` | `absorption_impact_rehearsal_v1/` (unchanged, 4 files, 2.5MB, prior accepted evidence) + 4 new small JSON files (this batch, ~16KB total) |
| Remaining under `.pytest_temp` | none |

---

## Verdict

**`CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_COMPLETE`**

**`ABSORPTION_IMPACT_CANONICAL_DATA_READY_FOR_PREREGISTRATION`**

Accepted by explicit operator decision with the 13 pre-existing, root-cause-proven-unrelated test failures (batches 4/7/21) fully disclosed rather than silently reconciled. Stopping after canonical migration. No preregistration or hypothesis execution begins without new, separate operator instruction.
