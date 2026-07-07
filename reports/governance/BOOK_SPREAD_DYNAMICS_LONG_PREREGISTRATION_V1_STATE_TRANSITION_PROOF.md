# BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1 — State-Transition Proof

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-LONG-PREREGISTRATION-V1
**Date:** 2026-07-07 · **Author:** Sonnet 5
**Outcome:** `BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1_INCOMPLETE` — null state transition. Every governance count, canonical row, and knowledge row is identical before and after.

---

## 1. Accepted checkpoint

Prior mixed-direction preregistration `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`, commit `a4722117`, verdict `BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE` — **immutable, unread as prior TEST exposure** for this new child (proven in §3). Operator scope ruling `reports/governance/FAM_BOOK_SPREAD_DYNAMICS_LONG_CHILD_SCOPE_V1.md` freezes direction/sign/effect-floor/model before any outcome access in this child's history.

## 2. New child identity

| Field | Value |
|---|---|
| `canonical_family_id` | `FAMv1:85cfe11ceeadbbe8` |
| Distinct from parent (`FAMv1:2d102e7b70820470`) | yes |
| Distinct from mixed-direction child family_id | yes |
| Resolved from | `resolve_canonical_family_id("FAM_BOOK_SPREAD_DYNAMICS", "H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1")` |

## 3. Prior-exposure proof (mixed-direction attempt is NOT exposure)

| Check | Result |
|---|---|
| `experiment_gate_receipts` for `FAMv1:85cfe11ceeadbbe8` (new LONG family) | 0 rows |
| `epistemic_test_nullifiers` for `FAMv1:85cfe11ceeadbbe8` | 0 rows |
| `experiment_gate_receipts` for `FAMv1:2d102e7b70820470` (mixed-direction family, from the PRIOR batch) | 0 rows — confirming the prior INCOMPLETE closure itself created nothing to be exposed to |
| Graveyard hits | 0/31 |

## 4. LONG population (Phase 4, complete)

Read directly from `ami_book_spread_change_windowed_flow.direction` (immutable, birth-known column):

| | Count |
|---|---|
| Structural LONG representatives | 70 |
| Outcome-compatible (effective `observation_status='OK'`) | 58 |
| Excluded | 12 (11 `MISSING_INTERNAL_GAP`, 1 `EXCLUDED_NO_HORIZON_DATA`) |
| Duplicate cycle IDs | 0 |
| SHORT rows present | 0 (structurally excluded by the `direction='LONG'` filter itself) |

## 5. Split resolution and sample-sufficiency block (Phase 5)

Reused the existing `TRAIN_FRACTION=0.7` cycle-grouped chronological convention verbatim (same as `w4_post_event_path_taxonomy.py`, `w8_short_expanded_baseline.py`, CVD, and Absorption). Applied to the 58 eligible LONG representatives:

| Metric | Value | Frozen minimum | Pass? |
|---|---|---|---|
| TRAIN n | 40 | 30 | ✓ |
| **TEST n** | **18** | **20** | **✗** |
| Total n | 58 | 50 | ✓ |
| Residual df (TEST−rank) | 16 | 15 | ✓ |
| Overlap | 0 | 0 | ✓ |

**TEST n=18 is the sole failing condition** — an outcome-blind, mechanical fact of applying the reused split to the reused eligibility rule; no researcher discretion altered it. Per the gate's Phase 5 instruction, this stops the batch INCOMPLETE.

## 6. Nullifier (audit-only, never persisted)

A nullifier value was computed once for documentation completeness (`derive_test_nullifier(family_id, split_version, test_cycle_ids)` = `33dcdeb2…`), confirmed unconsumed by any prior experiment across the entire real `epistemic_test_nullifiers` table — but **no row was inserted**. `epistemic_test_nullifiers` remains at 2 rows, unchanged.

## 7. Governance state — before and after (identical)

| Table | Before | After |
|---|---|---|
| `canonical.sqlite` schema_version | 14 | 14 |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `epistemic_test_nullifiers` | 2 | 2 |
| `experiment_gate_receipts` | 2 | 2 |

Confirmed via direct query against the real (read-only) databases. No `INSERT`/`UPDATE`/`DELETE` was issued against either real file — `resolve_family_and_child_identity`, `resolve_long_population`, `resolve_outcome_metadata`, and `resolve_split` are pure read/compute functions.

## 8. Resolved conflict with the gate prompt's literal Phase-11 expectation

The operator prompt's Phase 11 states a *successful* preregistration should move `experiment_registry` 24→25, `nullifiers` 2→3, `gate_receipts` 2→3. Repository evidence (both the CVD and Absorption preregistration artifacts, verbatim: *"No experiment_registry row exists yet"*) establishes `experiment_registry` is written only at governed-execution time, never at preregistration; and `epistemic_test_nullifiers` rows exist only as consumption records (`consume_test_evidence()` is the sole writer, and insertion IS consumption — there is no "registered but unconsumed nullifier row" mechanism in the codebase). Per the gate's own precedence rule ("repository state wins for exact ... enforcement conventions"), this is disclosed rather than silently reconciled. It is moot for this specific batch's outcome, since INCOMPLETE created neither a registry row nor a nullifier row regardless of which convention applied — but the resolution is recorded here so a future COMPLETE attempt for this or any other child is not blocked by the same ambiguity.

## 9. No-outcome-access proof

| Channel | Count |
|---|---|
| `endpoint_return_bps` reads | 0 |
| `mfe_bps` reads | 0 |
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| Nullifier consumption attempts | 0 |
| Experiment/result/receipt writes | 0 |

## 10. Focused tests

`tests/test_ami_research_book_spread_dynamics_long_preregistration_v1.py` — **24/24 passed**. Covers prior-V1-immutability, new-child/family-ID distinctness, graveyard/exposure-clean (including the mixed-direction-attempt-is-not-exposure proof), outcome-reused-verbatim with AST-scoped guard, LONG population (structural 70 / eligible 58) with SHORT-absence proof, row-accounting-root CHECK enforcement, split reuses the frozen 70/30 convention with deterministic hashes, the TEST=18<20 failure isolated to exactly one rule, full-chain zero-mutation proof, and real-DB governance-state confirmation.

## 11. Regression

Additive-only batch (one new module, one new test file, two governance docs). No schema, no shared governance-write path, no other family's code touched. Established 18-pre-existing-failure baseline unaffected.

## 12. Storage report

No `microstructure.db` copy. No canonical backup (nothing mutated). Disposable copies existed only under pytest's own `--basetemp`, removed by the runner. No OS-temp file retained.

## 13. Verdict and next step

**Verdict:** `BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1_INCOMPLETE`
**Authorization state:** `INCOMPLETE_NO_AUTHORIZATION_ISSUED`
**Next step:** operator decision — wait for LONG population growth (eligible n≥67 needed to clear TEST≥20 under a 70/30 cut, i.e. 9 more eligible LONG cycles) or explicitly authorize a different, still outcome-blind split ratio as its own separate scoping decision. No default recommended.
**Execution stopped:** confirmed — no TRAIN/TEST access, no model fit, no nullifier consumption, no route/bucket work, no runtime/risk/execution touched at any point.
