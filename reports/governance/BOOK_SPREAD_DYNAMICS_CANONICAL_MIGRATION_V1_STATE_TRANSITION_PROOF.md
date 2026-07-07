# BOOK_SPREAD_DYNAMICS_CANONICAL_MIGRATION_V1 — State-Transition Proof

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1
**Migration ID:** M-0036 · **Type:** combined additive schema-and-data migration (3 new tables + frozen-source backfill)
**Date:** 2026-07-07 · **Author:** Sonnet 5
**Nature:** outcome-blind canonical migration only — no experiment, no nullifier action, no gate receipt, no preregistration, no TEST access, no scientific-definition change.

---

## 1. Accepted checkpoint (input state)

| Field | Value |
|---|---|
| Readiness / W300 definition commit | `f115b9c1` (`SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS` → operator resolution) |
| Disposable rehearsal commit | `6a449a64` (`BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_COMPLETE`) |
| Row-accounting freeze commit | `54d00dca` (`BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_COMPLETE`) |
| Operator ruling | `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1` — W300 horizon + additive `spread_change_bps` difference |
| Frozen row-accounting root | `33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31` |
| schema_version (before) | 13 |
| experiment_registry / experiment_results | 24 / 381 |
| epistemic_test_nullifiers / experiment_gate_receipts | 2 / 2 |
| Deterministic regression baseline at batch start | 18 pre-existing waived failures (see §11) |

---

## 2. Migration-ID and schema-version ruling

- **M-0036** resolved from `MIGRATION_LOG.md` (last canonical = M-0035, last knowledge = M-0034; M-0036 free). Not assumed.
- **schema_version 13 → 14** from `ami/warehouse/schema.py::CANONICAL_SCHEMA_VERSION` + `init_schema()` convention (each additive `_SCHEMA_PHASE_*` block = +1). Precedent chain: M-0030 (10→11), M-0031 (11→12), M-0035 (12→13). This batch adds one `_SCHEMA_PHASE_BOOK_SPREAD` block → +1 → 14. Neither "stays 13" nor "jumps arbitrarily" — the version tracks the additive-block count exactly.

---

## 3. Destination schema (only authorized delta)

Three insert-only tables added to `init_schema()` via `_SCHEMA_PHASE_BOOK_SPREAD`, each FK `anchor_id → ami_signal_lifecycle(signal_id)`, each carrying the frozen `row_accounting_root='33c4f4be…'` CHECK and known-at CHECKs:

| Table | Grain | Rows | PK / UNIQUE |
|---|---|---|---|
| `ami_book_spread_change_windowed_flow` | EXACT anchor (feature) | 196 | `feature_id` / `(anchor_id, formula_version)` |
| `ami_book_spread_change_window_quality_v1` | anchor (accounting) | 324 | `quality_id` / `(anchor_id, formula_version)` |
| `ami_book_spread_change_exclusions` | non-exact anchor | 128 | `exclusion_id` / `(anchor_id, formula_version)` |

Write-set boundary: only the 3 new tables + their indexes + the `schema_versions` canonical row. **Forbidden writes all zero:** outcomes, experiment_registry, experiment_results, nullifiers, gate_receipts, route/bucket/paper/shadow/forward/live/risk/execution, other feature families, `knowledge.sqlite`. No outcome / alternative-window / alternative-transform column exists in any table.

---

## 4. Backup + restore proof

| Item | Value |
|---|---|
| Backup path | `data/ami/backups/canonical_pre_M0036_book_spread_dynamics_canonical_migration_20260707_151140.sqlite` |
| Backup sha256 | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` |
| Byte-exact vs live pre-migration | YES |
| Restore target (disposable) | `.runtime_temp/M0036_restore_verify/canonical_restored.sqlite` (deleted after proof; live file never overwritten) |
| Restored schema_version | 13 |
| Restored book_spread tables present | none |
| Restored integrity / fk | ok / [] |
| Restored sha256 == pre-migration | YES |

---

## 5. Source verification (no recomputation)

Retained frozen source `.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite`; row-accounting root recomputed **before** migration = `33c4f4be…` = frozen. All feature values copied verbatim; zero recomputation, zero network calls.

---

## 6. Inserted-row + canonical-replay accounting

- First-run inserted: 196 / 324 / 128; conflicts 0.
- Identities: `324 = 196 EXACT + 22 STALE + 106 UNAVAILABLE`; `128 = 22 + 106`; 97 cycles = 97 representatives (0 dup); 0 EXACT-in-exclusions; 0 excluded-in-feature; single root `33c4f4be…`; single migration_id `M-0036`.
- **Canonical replay rebuilt from destination tables reproduces all 5 frozen manifest hashes** (ordered_anchor `a77a8daf…`, exact_feature `b1eb902f…`, exclusion `0694e433…`, cycle_membership `e692ff1c…`, representative `edadf597…`). This IS the row-level frozen-manifest equality proof (mismatch count 0).

---

## 7. First-run and idempotent NOOP

| Run | Inserted | NOOP-identical | schema_version | replay==frozen |
|---|---|---|---|---|
| 1 (apply) | 196 / 324 / 128 | — | 14 | YES |
| 2 (rerun) | 0 / 0 / 0 | 196 / 324 / 128 | 14 | YES |

Rerun verdict: `NOOP_IDENTICAL` at content level. Full-file sha256 differs across runs **only** because `init_schema()` upserts the `schema_versions.applied_ms` wall-clock bookkeeping field on every call; the 3 migrated tables are byte-identical (proven by identical replay hashes + NOOP + unchanged counts). Same disclosed behavior as M-0035.

---

## 8. Migration checksum

| Artifact | sha256 |
|---|---|
| `ami/research/book_spread_dynamics_canonical_migration.py` | `d5ac1dca70c47f4818d7ad1de21e2fa695452b50d87586dc260d1e160cd80a7c` |
| `ami/warehouse/schema.py` | `5868da8c3de1870939d3272e1fd426b153a9838acc07cee2326d06aba0db717b` |
| `tests/test_ami_research_book_spread_dynamics_canonical_migration.py` | `7f8c2339642fabb25cec5658894eb623285f700a5b02bf9ed9b86807fd1cb460` |
| frozen 5-component manifest | `0a65c45ffba906414c7a484e3f966e2405017eaea8990aded429dc35ed142c89` |
| row_accounting_root | `33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31` |
| specification_hash | `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212` |

---

## 9. Pre / post canonical hashes

| State | canonical.sqlite sha256 | schema_version | tables |
|---|---|---|---|
| Pre-migration | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` | 13 | 42 |
| Post first-run | `ddb9d72b8d1ff67c1092d824215a3806fe305d2a4d65b60707f14cb20b87adac` | 14 | 45 |
| Post idempotent rerun (current) | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` | 14 | 45 |
| `knowledge.sqlite` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged) | — | — |

---

## 10. No-outcome-access proof

SQLite authorizer around the data step + verification reads (not around unrelated pre-existing schema DDL):

| Channel | Result (run 1 / rerun) |
|---|---|
| Outcome/governance table access (`ami_lifecycle_path_observations`, `epistemic_test_nullifiers`, `experiment_gate_receipts`) | [] / [] |
| Outcome column access (`endpoint_return_bps`, `mfe_bps`, `mae_bps`) | [] / [] |
| Monitored governance writes (`experiment_registry`, `experiment_results`) | [] / [] |
| Outcome reads / writes | 0 / 0 |
| Experiment / result / nullifier / gate-receipt creation or mutation | 0 |
| Route/bucket promotion | 0 |

Secondary AST control: neither the migration module nor the schema block carries any outcome/experiment/nullifier column name in any SQL literal. Known-at enforced by table CHECK constraints (violating rows fail INSERT); post-migration re-verification 0 mismatches.

---

## 11. Regression comparison vs. accepted baseline

Paired single-process sweep, 83 files / 42 batches. **Before fix: 19 failing nodes. After fix: 18 failing nodes, 0 net new deterministic failures from M-0036.**

**Attributable to this batch (1, fixed):**
- `test_ami_lifecycle_provenance_rehearsal.py::test_full_provenance_rehearsal_real_data` — tuple `schema_version_before in (…,13)` passed at 13, fails at 14. Fixed by extending to `(…,13,14)` (M-0035 precedent). Verified: provenance file 2p+1f → **3 passed**.

**Pre-existing, proven NOT caused by M-0036 (18):**
- 2 schema-hash pins assert `version == 12` (`test_26_…`, `test_22_23_…`) — already failing since M-0035 bumped to 13; unaffected by 13→14.
- 16 governance-state pins (preregistration invariants, nullifier/receipt/experiment-count, governed-execution dress rehearsals, verify-pre-execution) — depend on governance rows that M-0036 **provably never touched** (protected-delta ZERO, authorizer empty, `knowledge.sqlite` byte-identical). Stale from the prior G2-CVD governed execution (`60c3e26f`) and absorption preregistration committed before batch start (HEAD `fc43e972`).

The 13 focused migration tests all pass. Full 18-node list in the migration report §Regression.

---

## 12. Protected-delta (ZERO)

| Table | Before | After |
|---|---|---|
| ami_events | 252 | 252 |
| ami_signal_lifecycle | 324 | 324 |
| ami_cycles | 167 | 167 |
| ami_birth_truncated_cascade_geometry | 220 | 220 |
| ami_agg_trades_repaired | 40,934 | 40,934 |
| ami_cvd_windowed_flow | 1,840 | 1,840 |
| ami_absorption_impact_windowed_flow | 1,619 | 1,619 |
| experiment_registry | 24 | 24 |
| experiment_results | 381 | 381 |
| researcher_exposure_ledger | 1,180 | 1,180 |

Only the 3 new book-spread tables were added.

---

## 13. Storage report

- No copy of `data/microstructure.db` made. Backup placed under `data/ami/backups/` (not OS temp). Disposable restore/dry-run under `.runtime_temp/` (deleted after proof). Retained rehearsal/freeze evidence never deleted or mutated.

---

## 14. Verdict and next gate

**Verdict:** `BOOK_SPREAD_DYNAMICS_CANONICAL_MIGRATION_V1_COMPLETE`
**Disposition:** `BOOK_SPREAD_DYNAMICS_CANONICAL_DATA_READY_FOR_PREREGISTRATION` (authorizes no automatic next step)
**Recommended next gate:** `BATCH-BOOK-SPREAD-DYNAMICS-PREREGISTRATION-V1` — must separately resolve outcome ID / split / nullifier / gate receipt / model, outcome-blind until its own TEST authorization. **Do not begin automatically.**
