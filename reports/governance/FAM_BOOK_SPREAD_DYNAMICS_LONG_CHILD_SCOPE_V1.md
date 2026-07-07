# FAM_BOOK_SPREAD_DYNAMICS — LONG Child Scope V1

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-LONG-PREREGISTRATION-V1, Phase 1
**Date:** 2026-07-07 · **Author:** Sonnet 5 (recording an explicit operator ruling)

---

## Parent family and prior incomplete child

- **Parent family:** `FAM_BOOK_SPREAD_DYNAMICS` (`canonical_family_id = FAMv1:2d102e7b70820470`).
- **Prior mixed-direction child:** `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`, closed `BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE` (commit `a4722117`). **This document does not amend, reopen, or supersede that closure. It remains immutable and INCOMPLETE.**

## New LONG child identity

- **New child ID:** `H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1`.
- **`hypothesis_id` frozen for this attempt:** `H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1`.
- **`question_ids` frozen for this attempt:** `FAM_BOOK_SPREAD_DYNAMICS` (unchanged — same parent family).
- **Resolved `canonical_family_id`:** `FAMv1:85cfe11ceeadbbe8` — distinct from the mixed-direction child's `FAMv1:2d102e7b70820470` (different `hypothesis_id` text → different hash, by construction of `resolve_canonical_family_id`).

This is a **new child**, not an amendment: it has its own `canonical_family_id`, its own graveyard check, its own exposure check, and its own nullifier space. The mixed-direction child's INCOMPLETE closure is unaffected and unread as prior TEST exposure for this new child (confirmed below — the mixed-direction attempt created no gate receipt and no nullifier consumption for anyone to be exposed to).

## Outcome-blind LONG rationale (operator ruling, verbatim basis)

1. **Directional coherence.** The reused canonical outcome (`endpoint_return_bps@swing_24h`) carries an absolute price-return sign, not a signal-relative one. A single expected coefficient sign is interpretable only once one immutable signal direction is frozen for the population.
2. **No derived outcome.** Restricting the base population to the existing immutable `direction='LONG'` field allows `endpoint_return_bps@swing_24h` to be reused verbatim — no direction-flip, no direction-adjustment, no new outcome is created.
3. **No interaction or subgroup rescue.** The LONG child uses one primary population and one primary model. No direction interaction, no per-direction model comparison, no post-TEST subgroup rescue.
4. **Outcome-blind sample-capacity rationale.** Frozen representative population: LONG = 70 independent-cycle representatives, SHORT = 27. Under the existing 70/30 chronological split convention and the repository's TEST≥20 minimum-sample rule, the SHORT population (27 total, ≈8 TEST at a 70/30 cut) cannot reasonably support a governed TEST set. LONG is the only direction with structural potential to clear the minimum — though, per this document's own instruction, the *exact* outcome-compatible count must still be resolved structurally during preregistration, and if it falls short, the batch must close INCOMPLETE rather than lowering the rule (see the paired preregistration artifact for that resolution).
5. **Degree-of-freedom containment.** This ruling chooses one direction once, before outcome access. SHORT is not tested, compared, or used as a diagnostic anywhere in this or the paired preregistration artifact.

## SHORT non-evaluation statement

**SHORT has not been scientifically rejected.** **SHORT has not received TEST exposure.** No outcome value (TRAIN or TEST) was read to choose LONG over SHORT — the choice was made purely from immutable, outcome-blind identity fields (`direction`) and structural cycle-representative counts. SHORT remains available only for a future, independently authorized child, after either sufficient sample capacity accumulates or a separately governed scientific contract (e.g., a different split convention, a different minimum-sample rule change authorized in its own right) is established. This document authorizes no SHORT work of any kind.

## Expected sign

**Frozen: NEGATIVE.**

Mechanistic interpretation (operator ruling, outcome-blind): positive `spread_change_bps_w300` means the executable L1 spread widened into signal birth. Within a LONG-only signal population, widening represents short-horizon liquidity withdrawal, execution fragility, and impaired immediate price support. Greater widening is therefore expected to associate with a *lower* subsequent `endpoint_return_bps@swing_24h`; compression/normalization is expected to be relatively more supportive of a positive LONG path. This sign was not selected from TRAIN results, TEST results, correlations, coefficients, PnL, win rate, MFE, MAE, or subgroup performance — it is frozen here, before any outcome access in this child's own history.

## Effect-size relevance rule

**Frozen: β ≤ −1.0** — at least 1.0 basis point lower `endpoint_return_bps@swing_24h` for each +1.0 spread-basis-point of W300 spread expansion. Raw predictor units (spread basis points); no standardization for the primary interpretation. Fixed before outcome access; may not be changed using TRAIN or TEST behavior.

## Model and no-control ruling

**Frozen: OLS with intercept, HC3 heteroskedasticity-robust standard errors, `spread_change_bps_w300` as the sole predictor, no controls.** Rationale: this first LONG child tests the incremental univariate association of the frozen spread-dynamics feature within one coherent direction; no control set was frozen earlier for this family; adding controls now would introduce unnecessary researcher degrees of freedom; adjusted models require independently preregistered child hypotheses.

## Scope limitation

This ruling and its paired preregistration artifact govern **only** `H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1`. It does not authorize: any SHORT work, any alternative window, any alternative transform, any control addition, any model-family change, any threshold/bin/quantile construction, any subgroup or regime model, or any change to the frozen mixed-direction child's own (immutable) INCOMPLETE status.

## Amendment policy

Immutable once committed. Any change to direction scope, child ID, outcome ID, expected sign, effect threshold, sample rule, population, split, predictor, model, or controls requires a new, separately versioned child and a new gate cycle before any TEST access. This document may not be silently patched.
