# S34_BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-LONG-PREREGISTRATION-V1
**Status:** `BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1_INCOMPLETE`. No TEST outcome was read (nor TRAIN, nor any outcome value). No `experiment_registry` row, no TEST nullifier consumption, no gate receipt was created.
**Date:** 2026-07-07 · **Author:** Sonnet 5

This document records a clean, outcome-blind sample-sufficiency stop: the frozen 70/30 chronological split, applied to the LONG-only population under the operator's own scoping ruling, yields a TEST set of 18 cycles — below the frozen minimum of 20. Per the gate's own instruction, this closes INCOMPLETE without lowering the rule.

---

## 0. Operator scope ruling (accepted, recorded separately)

`reports/governance/FAM_BOOK_SPREAD_DYNAMICS_LONG_CHILD_SCOPE_V1.md` freezes: direction=LONG, expected sign=NEGATIVE, effect-size floor β≤−1.0, model=OLS+intercept+HC3 no controls. This document does not re-derive or re-litigate any of those — it resolves the remaining outcome-blind phases (identity, outcome reuse, population, split) against the real database.

## Phase 1-2 — Family/child identity, graveyard, exposure (COMPLETE)

| Field | Value |
|---|---|
| Parent family | `FAM_BOOK_SPREAD_DYNAMICS` (`FAMv1:2d102e7b70820470`) |
| Prior incomplete mixed-direction child | `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`, commit `a4722117` — **immutable, unaffected** |
| New child ID | `H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1` |
| Resolved `canonical_family_id` | `FAMv1:85cfe11ceeadbbe8` — distinct from both the parent and the mixed-direction child |
| Graveyard | 0/31 hits, CLEAN |
| Prior TEST exposure for this family_id | 0 nullifier rows, 0 gate receipts — genuinely unconsumed |
| Mixed-direction attempt counted as exposure? | **No** — it created no gate receipt and no nullifier for anyone to be exposed to (verified: `experiment_gate_receipts`/`epistemic_test_nullifiers` for the *mixed-direction* family_id `FAMv1:2d102e7b70820470` are also empty) |

## Phase 3 — Existing outcome (COMPLETE, reused verbatim)

`endpoint_return_bps@swing_24h` (`ami_lifecycle_path_observations`), continuous, **not** direction-flipped — identical, unmodified identity to the mixed-direction attempt, CVD, and Absorption. No direction adjustment, no derivation. Zero outcome values read.

## Phase 4 — LONG population (COMPLETE, outcome-blind)

| Metric | Value |
|---|---|
| Structural LONG representatives (`direction='LONG'`, `is_cycle_representative=1`, formula/root matching) | **70** |
| Outcome-compatible (`observation_status='OK'`, effective/corrected selection) | **58** |
| Excluded | **12** (11 `MISSING_INTERNAL_GAP`, 1 `EXCLUDED_NO_HORIZON_DATA`) |
| Duplicate cycle IDs among eligible | 0 |
| Missing representatives | 0 |
| Structural population hash (70, ordered) | `6f7119cec64de592953a743ab8cb0b1c0dcd321797e9faf3e3a355c7aa481d6c` |
| Eligible population hash (58, ordered) | `a22d4746e5b1b149edeb8e4d3ea79c8e359a6f69934435edc9afd5ec86002d79` |

Every SHORT row is structurally absent from this population by construction (`direction='LONG'` filter at the SQL level) — confirmed by a dedicated test (`test_short_never_appears_in_long_population`).

## Phase 5 — Split resolution and sample sufficiency: **BLOCKED, closes the batch**

Reused the existing canonical chronological split convention verbatim — cycle-grouped, chronological, `TRAIN_FRACTION=0.7` cut by cycle count (the same convention `w4_post_event_path_taxonomy.py`/`w8_short_expanded_baseline.py`/the CVD and Absorption preregistrations all use). No new split was created; no boundary was moved.

| Field | Value |
|---|---|
| `split_version` | `SPLITv1:9a01a5190e25526f` |
| Total eligible | 58 |
| TRAIN n | **40** |
| TEST n | **18** |
| TRAIN/TEST overlap | 0 |
| Straddling | none (TRAIN's latest representative precedes TEST's earliest) |
| Ordered TRAIN cycle-set hash | `4909253243fc2f0c58f04ff48dfe336a48a51d4e714bade87de0dff3523205e4` |
| Ordered TEST cycle-set hash | `a14a9239837a8f74e10ac13016542c864b21bcf9b27c0aaef53cdb75fa7aa03b` |
| Residual df (TEST n − design rank 2) | 16 |

### Sufficiency check against the frozen minimums

| Rule | Frozen minimum | Actual | Pass? |
|---|---|---|---|
| TRAIN ≥ 30 | 30 | 40 | ✓ |
| **TEST ≥ 20** | **20** | **18** | **✗ FAILS** |
| Total ≥ 50 | 50 | 58 | ✓ |
| Residual df ≥ 15 | 15 | 16 | ✓ |
| TRAIN/TEST overlap = 0 | 0 | 0 | ✓ |

**TEST n=18 is the sole blocking condition** — every other rule (including residual degrees of freedom, which would independently have passed at 16≥15) is satisfied. This is not a degrees-of-freedom problem or a design-rank problem; it is a plain shortfall of 2 cycles against the frozen minimum, produced mechanically by applying the reused 70/30 split to the reused eligibility rule. No researcher choice altered this outcome.

Per the gate's own explicit instruction — *"If the exact LONG outcome-compatible TEST set is below 20 ... close INCOMPLETE. Do not lower the sample rule."* — this preregistration attempt stops here.

## Phases 6+: not attempted

Predictor scaling, model fitting readiness beyond what is already fixed by the operator ruling, nullifier derivation-for-registration, experiment registration, and gate-receipt issuance were **not** performed. The nullifier value that *would* have been derived from `(family_id, split_version, test_cycle_ids)` was computed once for audit/documentation purposes only (`33dcdeb2110f78c13cbe476a7c17850bdb34d0010d09e0e3f2def2467c9ee0a1`) and confirmed unconsumed by any other family or experiment — but it is **not persisted anywhere** (no row was inserted into `epistemic_test_nullifiers`, no gate receipt references it). An INCOMPLETE closure creates nothing.

## Governance state (before this attempt = after this attempt)

| Field | Before | After |
|---|---|---|
| `schema_version` | 14 | 14 |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `epistemic_test_nullifiers` | 2 | 2 |
| `experiment_gate_receipts` | 2 | 2 |

**Zero governance-table writes.**

### Resolved conflict with this gate's own literal Phase-11 expected-delta text (repository evidence wins, per this gate's own precedence rule)

The prompt's Phase 11 states an expected post-state of `experiment_registry: 25` / `nullifiers: 3` / `gate_receipts: 3` for a **successful** preregistration. Two independent, unambiguous repository precedents (`S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md`: *"No experiment_registry row exists yet"*; `S34_CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1.md`: same statement, same wording) establish that `experiment_registry` is written only at **governed execution** time (`record_experiment_registry`, called exclusively from `execute_governed_run`/`register_experiment_with_gates`), never at preregistration. Separately, `epistemic_test_nullifiers` rows are created **only** by `consume_test_evidence()`, whose own docstring and every call site treat insertion as consumption itself — there is no code path that inserts an "unconsumed" nullifier row, and this gate explicitly forbids consuming one here. Both of these facts would apply regardless of whether this batch had reached a COMPLETE verdict; they are noted here for completeness since this batch closed INCOMPLETE and created neither anyway (both counts are unchanged: 24 and 2 respectively).

## Focused tests

`tests/test_ami_research_book_spread_dynamics_long_preregistration_v1.py` — **24/24 passed**. Covers: prior-V1-immutability proof, new child/family-ID distinctness (from both parent and the mixed-direction child), graveyard-clean, exposure-clean (including proof the mixed-direction INCOMPLETE attempt created no exposure), outcome reused verbatim with an AST-scoped no-value-access guard, LONG structural population (70) and eligible population (58/12 excluded) with a dedicated SHORT-absence proof, row-accounting-root drift rejected by the schema's own CHECK, split reuses the frozen 70/30 convention with deterministic hashes, the TEST=18<20 sufficiency failure (isolated to exactly one failing rule), zero-mutation proof across the full identity→population→outcome→split chain, and confirmation of real-DB governance-count/no-orphaned-receipt state.

## Regression

Additive-only batch (one new module, one new test file, two new governance docs); touches no schema, no other family's code, no shared governance-write path. No additional full-suite sweep required beyond the established 18-pre-existing-failure baseline (unaffected).

## Storage report

No temporary database exceeded a few hundred KB (pytest `--basetemp` disposable copies only, deleted by the runner). No `microstructure.db` copy. No canonical/knowledge write occurred against the real files.

## Status

**`BOOK_SPREAD_DYNAMICS_LONG_PREREGISTRATION_V1_INCOMPLETE`**

## Recommended next step

Not a controlled research gate — an operator decision on how to close the 2-cycle TEST shortfall: (a) wait for additional data collection to grow the LONG population past the point where a 70/30 cut clears TEST≥20 (would need eligible n≥67, i.e. 9 more eligible LONG cycles beyond the current 58), or (b) explicitly authorize a different, still-outcome-blind resolution (e.g., a non-70/30 split ratio, chosen before any outcome access) as its own separately-versioned scoping decision. This batch recommends neither by default.
