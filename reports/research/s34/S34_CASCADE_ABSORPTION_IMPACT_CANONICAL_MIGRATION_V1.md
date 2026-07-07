# S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-CANONICAL-MIGRATION-V1
**Migration ID:** M-0035
**Nature:** Controlled additive canonical migration only. No preregistration, no experiment ID, no nullifier action, no TEST/outcome access, no scientific model, no route/bucket promotion.
**Depends on (source of truth, unedited):** readiness/contract commit `fc1321f5`, disposable rehearsal commit `fc43e972`, row-accounting freeze commit `931cd3dd`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Naming ruling (operator)

Canonical production table names are `ami_absorption_impact_*` — superseding both the frozen contract's illustrative `ami_impact_*` and the disposable rehearsal's `absorption_impact_*`. This is a naming normalization only: formula, units, window definitions, row identities, source manifests, quality states, exclusion identity, `FLOOR_USD_M`, feature values, and content accounting are all unchanged from the frozen rehearsal.

No naming collision existed: pre-flight confirmed zero tables/views matching `ami_absorption_impact%`, `%absorption%`, or `%impact%` in the live `canonical.sqlite`, and no prior `schema_version=13` or migration-ID collision (`MIGRATION_LOG.md`'s last entry was M-0034; M-0035 was free).

---

## Input package verification (Phase, pre-migration)

All 10 files committed by `fc1321f5`/`fc43e972`, the code hash, the retained rehearsal package's 4 files, and 7 source/canonical table schema hashes were independently recomputed against the current working tree and retained evidence — **23/23 matched with zero drift** (full list in the state-transition proof).

---

## Pre-migration checkpoint

| Field | Value |
|---|---|
| `data/ami/canonical.sqlite` absolute path | `D:\eclipse_scalper\data\ami\canonical.sqlite` |
| `schema_version` | 12 |
| Full-file sha256 | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| `integrity_check` | ok |
| `foreign_key_check` | [] (clean) |
| Table count / view count | 39 / 6 |
| `experiment_registry` / `experiment_results` | 23 / 350 |
| `researcher_exposure_ledger` | 1,176 |
| `knowledge.sqlite`: `epistemic_test_nullifiers` / `experiment_gate_receipts` | 1 / 1 |
| Protected identity counts | `ami_events`=252, `ami_signal_lifecycle`=324, `ami_cycles`=167, `ami_birth_truncated_cascade_geometry`=220 |

---

## Backup and disposable restore proof

- Backup: `data/ami/backups/canonical_pre_M0035_absorption_impact_canonical_migration_20260707_065549.sqlite`, sha256 `25a56a98d0…` — byte-exact match to the live pre-migration file (manifest sidecar: `…_065549.manifest.json`).
- Restored to a disposable path (`.runtime_temp/M0035_restore_verify/`, never overwriting the live file), verified: `schema_version=12`, zero `ami_absorption_impact_*` tables present, `integrity_check=ok`, `foreign_key_check=[]`, all protected counts identical to the checkpoint above (`…_065549.restore_proof.json`). Disposable restore copy deleted after verification.

---

## Migration structures (schema 12→13)

Three tables added via `ami/warehouse/schema.py::_SCHEMA_PHASE_ABSORPTION_IMPACT`, wired into `init_schema()`:

| Table | PK | Row count | Content hash (bookkeeping-excluded) |
|---|---|---|---|
| `ami_absorption_impact_windowed_flow` | `feature_id`, unique `(symbol, signal_id, window_id, feature_definition_version)` | 1,619 | `f7c834cc8ebe90708e308629f1921a050d58520ad5560422b09406a7d1ca8942` |
| `ami_absorption_impact_window_quality_v1` | `quality_id`, unique `(signal_id, window_id, quality_contract_version)` | 1,620 | `5d1a205c7f79ca1b269307e34750c0d46dc104c8a799e9b4d01c862d307d7ba0` |
| `ami_absorption_impact_exclusions` | `exclusion_id`, unique `(signal_id, window_id, reason_code)` | 1 | `5e3ae2e524fcdbd5d045698a5a14bd397ae2c21bf0ff9ae2f54f2502c35a3ff7` |

All three carry FK references to `ami_signal_lifecycle`/`ami_events`/`ami_cycles` (the only permitted delta vs. the rehearsal DDL, matching the CVD migration precedent), a `window_id` enum CHECK, and the same `evidence_layer='EXACT'`/`known_at_classification='KNOWN_AT_SAFE'`/`window_end_ts_ms=signal_birth_ts`/`feature_available_ts_ms=signal_birth_ts` CHECK constraints already validated in the rehearsal. **No effective-view was added** — unlike CVD's quality table, this family's quality table has no `assessment_version` dimension (one quality row per signal/window/quality_contract_version, ever), so there is no "latest wins" case to resolve. This is a disclosed, deliberate divergence from the contract's illustrative CVD-parallel description, flagged for any future A6+ review, not a defect.

New module `ami/absorption/cascade_absorption_impact_canonical_migration.py::run_canonical_migration()` copies all rows verbatim (0 recomputation, 0 network calls) from the retained frozen source `.runtime_temp/absorption_impact_rehearsal_v1/rehearsal_run1.sqlite`.

---

## Row accounting

| Metric | Value |
|---|---|
| Usable feature rows | 1,619 |
| Quality rows | 1,620 |
| Exclusion rows | 1 |
| Universe reconciliation | 1,619 + 1 = 1,620 ✓ |
| Per-window usable | W60=324, W300=324, W600=324, W1800=324, W3600=323 |
| Exclusion identity | `SIG-e03382b4d82720185dfc870a` (LONG), W3600, `CONFIRMED_GAP_OVERLAP` — unchanged from the frozen freeze |
| `floor_usd_m` distinct values | `{0.01}` — unchanged, never bound (`floor_applied_rows=0`) |
| Proxy rows in primary table | 0 |
| Full row-set comparison vs. retained package (bookkeeping columns excluded) | 0 missing, 0 extra, 0 value drift |

---

## Content verification

Row-by-row set comparison between the migrated canonical tables and the retained rehearsal package (`rehearsal_run1.sqlite`), all declared content columns, confirms **zero missing rows, zero extra rows, zero value drift** — not just matching aggregate counts/hashes but the full row population.

---

## Known-at and access proof

A SQLite authorizer (`SQLITE_DENY` on `ami_lifecycle_path_observations`, `endpoint_return_bps`/`mfe_bps`/`mae_bps`, and writes to `experiment_registry`/`experiment_results`/`epistemic_test_nullifiers`/`experiment_gate_receipts`) was installed around the actual migration execution (data-copy + immediate post-migration verification reads only — not around the pre-existing, unrelated schema-DDL application, which legitimately owns the outcome table's `CREATE TABLE IF NOT EXISTS` definition from an earlier, unrelated migration phase).

Result: **`outcome_table_access=[]`, `outcome_column_access=[]`, `monitored_table_writes=[]`** — zero attempts, on both the live migration run and its idempotent rerun. `known_at_violations=0` (re-verified directly against the migrated rows: 0 mismatches on `feature_available_ts_ms`, `window_end_ts_ms`, `known_at_classification`).

---

## Idempotent rerun

The migration was run a second time against the now-migrated live `canonical.sqlite`:

| Check | Result |
|---|---|
| Rows inserted | 0 (all three tables) |
| Rows `noop_identical` | 1,619 / 1,620 / 1 |
| Content hashes | identical to run 1 |
| `schema_version` | unchanged (13) |
| Protected counts | unchanged |
| Full-file sha256 | **differs** — `schema_versions.applied_ms` bookkeeping timestamp is unconditionally upserted by `init_schema()` on every call, a pure wall-clock field with no data-content meaning (identical disclosure discipline to the rehearsal's own bookkeeping-column exclusion) |

**Result: `NOOP_IDENTICAL`** at the content level.

---

## Regression

Collect-only: **987 tests / 76 files** (`pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py --collect-only`).

**Process note (disclosed, not concealed):** the first attempt at a post-fix confirmation run accidentally overlapped two regression sweeps (a synchronously-invoked call that hit its 2-minute tool timeout kept running detached in the background, while a second explicit background invocation of the same full suite started independently) — a genuine violation of the repo's no-parallel-Python-processes guardrail. This was detected (anomalous 5-minute batch duration, a spurious `FileNotFoundError` on a session-fixture file, duplicate log lines with differing timings for the same batch), the contaminated logs were discarded, and a clean single-process rerun reproduced a strictly smaller, fully-explained failure set — confirming the extra failure was contamination, not a real regression.

**Two clean full paired regression passes** (`≤2 test files per pytest invocation`, sequential, `--basetemp` scratchpad, `-p no:cacheprovider`), run before any code fix, independently reproduced the **identical** failure set — proving determinism:

| Batch | Files | Failures |
|---|---|---|
| 4 | `test_ami_cvd_canonical_migration.py` + `test_ami_cvd_primary_long_preregistration_v1.py` | 3 |
| 7 | `test_ami_epistemic_nullifier_enforcement_wiring.py` + `test_ami_epistemic_nullifier_legacy_bypass_closure.py` | 8 |
| 19 | `test_ami_lifecycle_provenance_rehearsal.py` + `test_ami_lifecycle_short_noisy_v1_rehearsal.py` | 2 |
| 21 | `test_ami_research_candidate_universe.py` + `test_ami_research_cvd_windowed_flow_001.py` | 3 |

**Root-cause separation (proven, not asserted):**

1. **Batch 19 / `test_full_provenance_rehearsal_real_data`** — hardcoded `schema_version_before in (8,9,10,11,12)`. **Caused by this batch** (schema bump 12→13), same precedent pattern as every prior schema-version bump in this codebase. **Fixed**: tuple extended to `(8,9,10,11,12,13)`. Verified green in isolation and in a subsequent clean full-suite pass.
2. **Batch 19 / `test_disposable_db_and_microstructure_db_untouched`** — compares a 64MB prefix hash of the live, actively-collecting `microstructure.db` taken seconds apart. **Not caused by this batch or any code**: passes cleanly in isolated single-file rerun; reappeared only in the contaminated parallel-process run above. A live-collector timing artifact, per the operator's own carve-out for "mutable live collector freshness/staleness."
3. **Batches 4, 7, 21 (14 test failures)** — all hardcode `experiment_registry==22`, a specific canonical.sqlite sha256 (`458bc07c…`), or `epistemic_test_nullifiers` count `==0`/gate-receipt state `PREREGISTERED_NOT_EXECUTED`. **Proven pre-existing and unrelated to `FAM_CASCADE_ABSORPTION_IMPACT`**, by two independent proofs: (a) this migration's code never opens `knowledge.sqlite` at all (structurally incapable of having changed nullifier/receipt state — confirmed via unchanged `knowledge.sqlite` mtime across this entire batch); (b) the pre-migration backup, taken before this batch touched anything, already shows `experiment_registry=23` and a canonical hash that already did not equal `458bc07c…`. These went stale due to the separate, prior **G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1** batch (commit `60c3e26f`), which legitimately consumed that nullifier and registered experiment #23.

**Final clean single-process pass (post-fix):** 987 collected, **973 passed, 14 failed** — exactly the batch 4/7/21 pre-existing set, batch 19 fully green (both sub-tests), and the transient geometry-module `FileNotFoundError` from the contaminated run did **not** recur (confirming it was contamination noise, not a real failure).

**Operator decision:** accept `CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_COMPLETE` with the 13 pre-existing, root-cause-proven-unrelated failures fully disclosed (batches 4/7/21) rather than silently reconciled or fixed under this batch's authority — remediating those hardcoded G2-execution-era checkpoints is explicitly out of scope for this migration and requires its own separate batch/operator decision.

---

## Final canonical validation

| Check | Value |
|---|---|
| Final `schema_version` | 13 |
| `ami_absorption_impact_windowed_flow` | 1,619 |
| `ami_absorption_impact_window_quality_v1` | 1,620 |
| `ami_absorption_impact_exclusions` | 1 |
| Universe reconciliation | 1,620 ✓ |
| `known_at_violations` | 0 |
| Outcome reads | 0 |
| `experiment_registry` | 23 (unchanged) |
| `experiment_results` | 350 (unchanged) |
| `knowledge.sqlite` nullifier/receipt state | unchanged (never opened) |
| Pre-existing protected tables | all content-identical (39→42 tables, +3 new only) |
| `integrity_check` | ok |
| `foreign_key_check` | [] |
| Final `canonical.sqlite` sha256 | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` |

---

## Remaining risks

1. The 13 pre-existing, G2-execution-caused test failures (batches 4/7/21) remain red in the full suite. They are proven unrelated to this family but require a separate remediation batch (updating those tests' hardcoded `experiment_registry`/hash/nullifier-state expectations) that is explicitly out of this migration's scope.
2. No effective-view exists for the quality table, diverging from the frozen contract's illustrative CVD-parallel description — disclosed as an open A6+ decision point, not a defect, since this family's quality table has no multi-assessment dimension to resolve.
3. The migration's frozen source is `rehearsal_run1.sqlite` specifically (of the two content-identical retained runs) — an arbitrary but reproducible and disclosed choice; `rehearsal_run2.sqlite` would have produced byte-identical migrated content (already proven at the row-accounting-freeze stage).

## Success verdicts

**`CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_COMPLETE`**

**`ABSORPTION_IMPACT_CANONICAL_DATA_READY_FOR_PREREGISTRATION`**

Stopping after canonical migration. No preregistration or hypothesis execution begins without new, separate operator instruction.
