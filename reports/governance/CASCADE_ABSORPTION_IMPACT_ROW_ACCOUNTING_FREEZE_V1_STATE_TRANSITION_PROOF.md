# CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-CASCADE-ABSORPTION-IMPACT-ROW-ACCOUNTING-FREEZE-V1
**Purpose:** Freeze the outcome-blind absorption/impact reconstruction universe, formula, source commitments, quality partitions, exact row accounting, and expected canonical migration contents — row-accounting and migration-input freeze only.
**Prior checkpoint (unchanged, not reopened):** commit `fc43e972` (`CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1_COMPLETE`), success verdict `ABSORPTION_IMPACT_REHEARSAL_READY_FOR_ROW_ACCOUNTING_FREEZE`, `experiment_registry`=23, `experiment_results`=350, `schema_version`=12, `epistemic_test_nullifiers`=1.
**Nature:** No canonical migration, no schema_version change, no preregistration, no experiment ID, no nullifier action, no TEST/outcome access, no predictive model, no profitability inspection, no formula/window/floor alteration, no runtime/risk/execution/shadow/paper/forward/live modification.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Source-of-truth resolution:** recomputed sha256 of all 4 files committed by `fc1321f5` and all 6 files committed by `fc43e972`, directly from the current working tree (no `git show` extraction needed — these commits are already the tip of history for these files and the tree is clean for these paths). All 10 recomputed hashes matched the values already recorded inside the rehearsal's own `manifest.json`/state-transition proof — **zero drift, no accepted artifact rewritten**.
2. **Phase 1 (formula freeze):** transcribed the frozen formula, all parameters, and `FLOOR_USD_M=0.01` unedited from the accepted rehearsal report and code (`ami/absorption/cascade_absorption_impact_rehearsal.py`, module docstring + `FROZEN_FLOOR_USD_M` derivation comment). No value altered.
3. **Phase 2 (source manifest freeze):** computed sha256 of the 7 relevant table schemas (`agg_trades`, `mark_prices`, `gaps` in `microstructure.db`; `ami_signal_lifecycle`, `ami_agg_trades_repaired`, `ami_events`, `ami_cycles` in `canonical.sqlite`) via `sqlite_master.sql`, read-only. Computed sha256 of the `classify_quality()` function source region as the "quality-contract hash" (no separate quality-contract file exists for this family, unlike CVD's `cvd_source_quality_contract_v1.py`). Recomputed sha256 of all 4 retained rehearsal-evidence files (`manifest.json`, `rehearsal_result.json`, `rehearsal_run1.sqlite`, `rehearsal_run2.sqlite`) — matched the original manifest's own recorded hashes for the two `.sqlite` files exactly. Promoted the retained package to immutable rehearsal evidence (all promotion preconditions met, see freeze MD §Phase 2).
4. **Phase 3 (primary-key freeze):** ran read-only SQL against both retained disposable databases (`rehearsal_run1.sqlite`, `rehearsal_run2.sqlite`) to independently count distinct `feature_id`/`quality_id`/`exclusion_id`, check for duplicate `(signal_id, window_id)` pairs within `windowed_flow`, and check for any `(signal_id, window_id)` pair present in both `windowed_flow` and `exclusions`. Result: 1,620 expected primary keys confirmed, 0 duplicates, 0 pairs in both tables, both runs identical.
5. **Phase 4 (per-window accounting freeze):** queried per-window usable/excluded counts and per-window content hashes (3 tables × 5 windows = 15 hashes) from both retained databases; all identical between run1 and run2. Traced the single `W3600` source gap to its exact signal identity (`SIG-e03382b4d82720185dfc870a`, LONG) and its exact missing interval (`microstructure.db:gaps` row `id=766`, `2026-04-21T14:10:35.146Z`–`14:17:05.985Z`, confirmed/closed) via a read-only join against `ami_signal_lifecycle` (identity columns only — `signal_id`, `direction`, `independent_cycle_id`, `signal_birth_ts`, `source_event_id`, no outcome column) and `microstructure.db:gaps`.
6. **Phase 5 (content-hash freeze):** defined and applied the three-way hash-type distinction (raw file byte hash / deterministic scientific-content hash / semantic hash excluding bookkeeping), explicitly listing and justifying the two excluded bookkeeping columns (`created_ms`, `assessed_at_ms`). Computed a package-level raw-byte hash and a package-level semantic-content hash by hashing sorted `name:hash` pairs — new derived values, not present in the original rehearsal artifacts, computed this batch for the freeze's own completeness requirement.
7. **Phase 6 (exact/proxy freeze):** re-asserted the existing rule (native EXACT primary, book-depth `LOW_FIDELITY_PROXY_ONLY`, no proxy layer constructed) — no new computation required, no proxy rows exist to check.
8. **Phase 7 (known-at freeze):** ran 5 independent read-only column-mismatch checks (`feature_available_ts_ms`, `window_end_ts_ms`, `known_at_classification`, `evidence_layer`, window-duration) against both retained databases — all 0 mismatches, both runs. Did **not** reread any outcome table (none exists in either retained disposable database — verified by listing `sqlite_master` table names in both files).
9. **Phase 8 (expected canonical migration manifest):** wrote out full column/constraint/index/policy specifications for the three tables, reusing the rehearsal's exact validated schema (`_SCHEMA` in `cascade_absorption_impact_rehearsal.py`) rather than inventing a new one. Explicitly flagged the naming discrepancy between the contract's illustrative `ami_impact_*` names and the rehearsal's actual validated `absorption_impact_*` names as an open A5 decision — not resolved unilaterally.
10. **Phase 9 (migration acceptance equations):** transcribed the required equations from the frozen numbers above; no new judgment introduced.
11. **Phase 10 (independent freeze verification):** the checks in steps 4-8 above collectively constitute this phase — performed entirely against the two retained disposable databases, no rebuild against `microstructure.db`, no new disposable database created. Result: `ROW_ACCOUNTING_IDENTICAL_TO_ACCEPTED_REHEARSAL`, zero drift.
12. **Live database validation:** re-queried `experiment_registry`, `experiment_results`, `schema_version`, `researcher_exposure_ledger`, `epistemic_test_nullifiers`, `experiment_gate_receipts` against the real `mode=ro` connections — all unchanged from the values recorded at the close of the `fc43e972` rehearsal.
13. **Cleanup:** deleted the one working diagnostic file created this batch (`independent_verification_run1_run2.json`, 6,713 bytes, under `.runtime_temp`) after incorporating its results into the freeze MD/JSON. The four originally-retained rehearsal-evidence files are unchanged and untouched.

## Why this batch could freeze row accounting without touching the real database's write path or rereading outcomes

Every fact in Phases 1-9 was either (a) transcribed unedited from the two already-accepted artifacts (`fc1321f5`, `fc43e972`), or (b) independently recomputed by read-only SQL against the two **retained disposable databases**, which contain only `absorption_impact_windowed_flow`, `absorption_impact_window_quality_v1`, `absorption_impact_exclusions`, and `rehearsal_manifest` — structurally no outcome table exists in either file to be read, let alone reread. The only queries against the real `canonical.sqlite`/`microstructure.db` this batch were: (i) two identity-column-only lookups (`ami_signal_lifecycle` for the gapped signal's identity, `microstructure.db:gaps` for the exact missing interval — both are pre-existing, already-governed data-quality/identity tables, not outcome tables) and (ii) six live-database-state count checks (`experiment_registry`, `experiment_results`, `schema_version`, `researcher_exposure_ledger`, `epistemic_test_nullifiers`, `experiment_gate_receipts`), all via `mode=ro` connections, structurally incapable of writing.

## Real database state — unchanged (proof)

| Check | Before this batch (= after `fc43e972`) | After this batch |
|---|---|---|
| `data/ami/canonical.sqlite` sha256 | `25a56a98d0…` | `25a56a98d0…` (unchanged) |
| `experiment_registry` | 23 | 23 |
| `experiment_results` | 350 | 350 |
| `schema_version` | 12 | 12 |
| `researcher_exposure_ledger` | 1,176 | 1,176 |
| `data/ami/knowledge.sqlite`: `epistemic_test_nullifiers` | 1 | 1 |
| `data/ami/knowledge.sqlite`: `experiment_gate_receipts` | 1 | 1 |
| Outcome-table (`ami_lifecycle_path_observations`) reads | 0 (proven at rehearsal time by a live SQLite authorizer) | **0** — not reread this batch; structurally absent from both retained disposable databases used for all Phase 3/4/7/10 verification |

Every connection to the real files during this batch was `mode=ro`, structurally incapable of writing, independent of application-level discipline.

## Exact changed/added-file manifest (this commit)

| File | Status | Content |
|---|---|---|
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1.md` | New | row-accounting freeze report (10 phases) |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1.json` | New | machine-readable companion |
| `reports/governance/CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1_STATE_TRANSITION_PROOF.md` | New | this document |

Not included: any canonical migration, DDL, preregistration artifact, TEST result, shared unrelated governance-projection change (`SYSTEM_STATE.md`/`IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md`/`MIGRATION_LOG.md`), runtime modification, or repository-wide cleanup. No immutable-manifest file separate from the JSON companion was created — the JSON companion itself carries the full source/hash/manifest content required by Phase 2, avoiding an unnecessary fourth artifact.

## Storage guardrail accounting

| Item | Value |
|---|---|
| Peak temporary disk usage this batch | 6,713 bytes (`independent_verification_run1_run2.json` under `.runtime_temp\absorption_impact_rehearsal_v1\`, a read-only-query scratch file) |
| Full database copies created | 0 |
| Large temp DB copies under `C:\Users\...\AppData\Local\Temp` | 0 |
| Files created this batch | 1 (the diagnostic JSON above) + the 3 committed artifacts |
| Files retained at completion | the 3 committed artifacts; the 4 pre-existing rehearsal-evidence files (`manifest.json`, `rehearsal_result.json`, `rehearsal_run1.sqlite`, `rehearsal_run2.sqlite`, 2.5 MB total, unchanged) — now formally promoted to immutable rehearsal evidence by this freeze |
| Files deleted at completion | `independent_verification_run1_run2.json` (its results are fully captured in the two freeze artifacts; retaining it separately would duplicate information already canonicalized) |
| Never touched | `data\microstructure.db`, `data\ami\canonical.sqlite` (write path), `data\ami\knowledge.sqlite` (write path), accepted `data\ami\backups\*`, any prior immutable evidence artifact, any active runtime checkpoint/ledger |

## Required validations (proven)

- TEST/outcome reads: **0**
- New experiment count: **0**
- New experiment result count: **0**
- New nullifier count: **0**
- Consumed-nullifier delta: **0**
- Live `canonical.sqlite`: unchanged (hash-verified)
- Live `knowledge.sqlite`: unchanged (count-verified: `epistemic_test_nullifiers`=1, `experiment_gate_receipts`=1)
- `schema_version`: remains **12**
- `experiment_registry`: remains **23**
- `experiment_results`: remains **350**
- Feature/window/floor change: **none**
- Runtime/risk/execution delta: **0**
- Route or bucket promotion: **0**
- Canonical migration: **did not occur**

No code changes were required for this batch (pure verification/documentation), so no regression suite was rerun.

---

## Verdict

**`CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1_COMPLETE`**

Readiness verdict: **`ABSORPTION_IMPACT_ROW_ACCOUNTING_FROZEN_FOR_CANONICAL_MIGRATION`**

Independent verification result: **`ROW_ACCOUNTING_IDENTICAL_TO_ACCEPTED_REHEARSAL`** — zero drift found across all counts, primary keys, quality partitions, exclusion reasons, and content hashes.

Stopping after the freeze. No canonical migration begins without new, separate operator instruction.
