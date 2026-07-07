# BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1 — State-Transition Proof

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-PREREGISTRATION-V1
**Date:** 2026-07-07 · **Author:** Sonnet 5
**Outcome:** `BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE` — this batch is a **null state transition**: every governance count, canonical row, and knowledge row is identical before and after.

---

## 1. Accepted checkpoint

Readiness `f115b9c1`, disposable rehearsal `6a449a64`, row-accounting freeze `54d00dca`, canonical migration `5267a15a` (M-0036, schema 13→14). All read-only inputs to this batch; none re-verified-and-mutated here.

## 2. Family/child identity resolution (Phase 1, complete)

| Field | Value |
|---|---|
| `canonical_family_id` | `FAMv1:2d102e7b70820470` |
| Resolved from | `resolve_canonical_family_id("FAM_BOOK_SPREAD_DYNAMICS", "H-BOOK-SPREAD-CHANGE-BPS-W300-V1-DIRECTION-NEUTRAL")` |
| Graveyard hits | 0 (of 31 curated fingerprints) |
| Prior `epistemic_test_nullifiers` rows for this family_id | 0 |
| Prior `experiment_gate_receipts` rows for this family_id | 0 |
| Prior `experiment_registry` rows referencing this family | 0 |

## 3. Outcome resolution (Phase 2, complete)

Reused `endpoint_return_bps@swing_24h` verbatim — identical identity to the CVD and Absorption preregistrations. 0 outcome values read (proven by an AST-scoped test scanning only `.execute()`-family call arguments for `endpoint_return_bps`/`mfe_bps` tokens — 0 hits — plus manual confirmation that `resolve_outcome_metadata`'s only query selects `observation_status`, grouped, never a value column).

## 4. Direction/sign resolution (Phase 3, blocked — proof of the block)

Population direction composition, read directly from the M-0036 canonical table (`ami_book_spread_change_windowed_flow.direction`, immutable, birth-known):

| | 196 EXACT rows | 97 representatives |
|---|---|---|
| LONG | 120 | 70 |
| SHORT | 76 | 27 |

Three resolution paths were evaluated and all three collide with an explicit prohibition in the gate:

1. **Population restriction by direction** — would introduce a new, previously-unfrozen researcher degree of freedom (no earlier gate in this family's chain — readiness, rehearsal, freeze, migration — scoped by direction).
2. **Outcome direction-flip** — would derive a new, family-specific outcome (explicitly forbidden).
3. **Interaction term / per-direction subgroup fit** — explicitly forbidden (no interactions; no subgroup rescue). Cross-checked against the repository's own established handling of direction-mixed populations (`ami/research/w8_hold_baseline.py`, which stratifies by direction into separate cells rather than pooling) — confirming that pooling with a single additive sign is not how this repository resolves direction-mixed designs, and stratifying is exactly what this gate's single-primary-model constraint forbids for a first preregistration.

No path remains. Per the gate's own Phase 3 instruction, this closes `INCOMPLETE` rather than asserting an arbitrary sign.

## 5. Governance state — before and after (identical)

| Table | Before | After |
|---|---|---|
| `canonical.sqlite` schema_version | 14 | 14 |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `epistemic_test_nullifiers` (knowledge.sqlite) | 2 | 2 |
| `experiment_gate_receipts` (knowledge.sqlite) | 2 | 2 |
| `canonical.sqlite` sha256 | `0604b0da…` | `0604b0da…` (unchanged) |
| `knowledge.sqlite` sha256 | `710b3f68…` | `710b3f68…` (unchanged) |

**Confirmed via direct query against the real (read-only) databases, not inferred.** No `INSERT`/`UPDATE`/`DELETE` was issued against either real file by this batch — `resolve_family_and_child_identity`, `resolve_population`, `resolve_outcome_metadata`, and `resolve_direction_and_sign` are all pure read/compute functions; none accepts a write-capable path, and the focused test suite independently proves (via before/after count comparison inside disposable copies, plus direct real-DB assertions) that no table changed.

## 6. No-outcome-access proof

| Channel | Count |
|---|---|
| `endpoint_return_bps` reads | 0 |
| `mfe_bps` reads | 0 |
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| Nullifier consumption attempts | 0 |
| Experiment/result/receipt writes | 0 |

## 7. Focused tests

`tests/test_ami_research_book_spread_dynamics_preregistration_v1.py` — **17/17 passed**. Covers identity/constant determinism, graveyard-clean, prior-exposure-clean, read-only enforcement, population accounting (196/97), direction-mixed proof, row-accounting-root drift rejection (schema CHECK + module guard), outcome-reused-verbatim with AST-scoped no-value-access guard, the Phase-3 INCOMPLETE resolution (both blocked and trivial branches), full-chain zero-mutation proof, and real-DB governance-count/no-orphaned-receipt confirmation.

## 8. Regression

This batch adds one new read-only module (`ami/research/book_spread_dynamics_preregistration_v1.py`) and one new test file. It does not modify `ami/warehouse/schema.py`, any canonical/knowledge DDL, or any other family's code — no schema-version-dependent or governance-count-dependent test anywhere in the suite can be affected by this batch's changes. No additional full-suite regression sweep was required beyond the M-0036 batch's already-established 18-pre-existing-failure baseline (unchanged; this batch touches none of those files).

## 9. Storage report

No `microstructure.db` copy. No canonical backup (nothing was mutated — no destructive or migratory operation occurred, so no backup was required or created). Disposable canonical/knowledge copies existed only under pytest's own `--basetemp` scratchpad directory during the test run and were removed by the test runner afterward. No file was created or retained under OS temp.

## 10. Verdict and next step

**Verdict:** `BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE`
**Authorization state:** `INCOMPLETE_NO_AUTHORIZATION_ISSUED`
**Next step:** not a controlled research gate — an operator scoping decision on whether to authorize a direction-restricted re-attempt (mirroring the CVD/Absorption LONG-only precedent). No default recommended; the fork is surfaced, not resolved, by this batch.
**Execution stopped:** confirmed — no TRAIN/TEST access, no model fit, no nullifier consumption, no route/bucket work, no runtime/risk/execution change occurred at any point in this batch.
