# S34_BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-PREREGISTRATION-V1
**Status:** `BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE`. No TEST outcome was read (nor TRAIN, nor any outcome value). No `experiment_registry` row, no TEST nullifier, no gate receipt was created.
**Date:** 2026-07-07 · **Author:** Sonnet 5

This document is binding as a closure record. It does not freeze a hypothesis for execution — Phase 3 (direction / expected-sign resolution) could not be completed without violating an explicit prohibition, so the batch stops here per its own protocol ("if the outcome definition and mechanism do not imply one defensible sign without result inspection, stop INCOMPLETE").

---

## 0. Research question (unchanged from readiness)

Does pre-birth L1 spread expansion (relative to its own pre-window baseline) contain continuous incremental predictive information for `endpoint_return_bps@swing_24h`, controlling for the same frozen control set? (`S34_SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1.md`, commit `f115b9c1`)

---

## Phase 1 — Family and child identity (COMPLETE)

| Field | Value |
|---|---|
| Family name | `FAM_BOOK_SPREAD_DYNAMICS` |
| Child ID | `H-BOOK-SPREAD-CHANGE-BPS-W300-V1` |
| `question_ids` (frozen for this attempt) | `FAM_BOOK_SPREAD_DYNAMICS` |
| `hypothesis_id` (frozen for this attempt) | `H-BOOK-SPREAD-CHANGE-BPS-W300-V1-DIRECTION-NEUTRAL` |
| Resolved `canonical_family_id` | `FAMv1:2d102e7b70820470` |
| Formula version | `BOOK_SPREAD_CHANGE_BPS_W300_V1` |
| Specification hash | `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212` |
| Row-accounting root | `33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31` |
| Migration | M-0036, schema_version=14 |

**Graveyard check** (real `knowledge.sqlite`, 31 curated fingerprints, `match_graveyard()` against `question_ids | hypothesis_id | full spec text`): **0 hits**. Not a graveyard retest of any known-failed family (spread/bid/ask/quote/widen/thin/liquidity keywords all clean, per the readiness doc's own check and independently reproduced this session).

**Prior exposure check** (real `knowledge.sqlite`, this exact `canonical_family_id`): 0 rows in `epistemic_test_nullifiers`, 0 rows in `experiment_gate_receipts`. **Genuinely unconsumed.**

**Registry check** (real `canonical.sqlite`): 0 existing `experiment_registry` rows referencing `BOOK_SPREAD`/`SPREAD` in `question_ids`/`hypothesis_id`.

## Phase 2 — Existing canonical outcome (COMPLETE, reused verbatim)

| Field | Value |
|---|---|
| `outcome_id` | `endpoint_return_bps@swing_24h` |
| Table | `ami_lifecycle_path_observations` |
| Dependent-variable type | continuous |
| Direction semantics | **NOT direction-flipped** (absolute price-return sign) — identical, unmodified convention to the CVD and Absorption preregistrations |
| Reused verbatim / newly derived | **verbatim / not derived** — no new outcome is proposed |
| Structural compatibility with this population | confirmed: 77/97 representatives have `observation_status='OK'` (effective/corrected selection); 8 `EXCLUDED_NO_HORIZON_DATA`, 35 (of which some overlap corrected) `MISSING_INTERNAL_GAP` before correction — only `observation_status` and `path_definition_version` were read (existence/coverage metadata), **zero outcome values accessed** |

No alternative outcome ID was proposed or considered — the repository-wide convention (used identically by both prior mechanism families) resolves this unambiguously.

## Phase 3 — Direction and expected sign: **BLOCKED, closes the batch**

### Population direction composition (outcome-blind, real DB)

| | 196 EXACT rows | 97 representative cycles |
|---|---|---|
| LONG | 120 | 70 |
| SHORT | 76 | 27 |

The M-0036 canonical population is **direction-mixed**, confirmed both at the full EXACT-row level and at the cycle-representative level used for modeling. This was expected: unlike `FAM_CASCADE_ABSORPTION_IMPACT` and the CVD family — whose hypothesis IDs, base universes, and readiness/rehearsal/freeze artifacts scoped `direction='LONG'` from inception — `FAM_BOOK_SPREAD_DYNAMICS`'s hypothesis text, readiness contract, rehearsal, and row-accounting freeze **never mention a direction restriction**. The freeze's own representative-selection rule is stated explicitly as using "no outcome/feature/direction/subgroup/route/bucket signal."

### Why no single defensible sign can be frozen without violating an explicit prohibition

`endpoint_return_bps` is a raw, unflipped, absolute price-return quantity: positive means "price rose over the horizon," independent of whether the anchoring signal is LONG (typically anchored to a down-cascade) or SHORT (typically anchored to an up-cascade). Three candidate resolutions were considered, outcome-blind, and all three are blocked by an explicit prohibition in this gate:

| Path | Description | Why blocked |
|---|---|---|
| 1. Restrict population to one direction | Mirror the CVD/Absorption precedent (LONG-only base universe) | **No earlier accepted gate** (readiness, rehearsal, freeze, migration) scoped this family by direction. Introducing the restriction now, for the first time, immediately before modeling, would be a *new* researcher degree of freedom — precisely what Phase 3 exists to prevent. It would also produce a population that is not the frozen 196/97 the migration materialized, which the operator's own population-freeze phase treats as normally invalidating. |
| 2. Direction-flip the outcome (`endpoint_return_bps` for LONG, `-endpoint_return_bps` for SHORT) | Make a pooled additive coefficient sign-coherent across both directions | Explicitly forbidden: "Do not invent a new outcome. Do not derive an outcome specifically for this family." A direction-flip is a derived, family-specific transform of the canonical outcome. |
| 3. Add a `direction × spread_change_bps_w300` interaction (or, equivalently, fit two separate per-direction models — the repository's own established convention for direction-mixed populations, see `ami/research/w8_hold_baseline.py`'s per-direction cell design) | Recover a coherent, direction-conditional sign | Explicitly forbidden by both the model-freeze phase ("Do not create interactions") and the TEST-access policy ("no subgroup rescue"). |

With all three paths blocked, the two candidate physical mechanisms for spread expansion — (a) liquidity-withdrawal amplification of whatever move follows, and (b) amplification of the cascade's own known mean-reversion base rate — **both imply opposite algebraic signs on the raw, unflipped outcome depending on whether the anchor is LONG or SHORT**. This is inherently an interaction effect, not an additive one, and no additive pooled model (the only kind permitted here) can carry a single coherent sign for it. Choosing a sign anyway would be arbitrary; choosing it after inspecting TRAIN or TEST would violate the no-result-inspection requirement that gates this entire phase.

**This is not an oversight or a lack of effort — it is a structural, provable property of this family's frozen, direction-mixed population combined with this gate's single-primary-model, no-interaction, no-new-outcome, no-late-population-restriction constraints.** Per the gate's own instruction: *"If the outcome definition and mechanism do not imply one defensible sign without result inspection, stop INCOMPLETE rather than marking either sign as favorable after TEST."*

## Phases 4–24: not attempted

Per the enforced preregistration order (Phase 15), direction/expected-sign resolution (position 7) precedes predictor freeze (8), population freeze (9), split resolution (10–14), model freeze (15–17), and all downstream phases. Since Phase 3 could not close, none of the downstream phases were started: no predictor scaling was frozen beyond what is already fixed by the canonical schema, no split version was computed, no model was specified, no TEST authorization was created, no diagnostics were listed. This avoids any partial, silently-inconsistent freeze that a later amendment could exploit.

## Amendment path forward (not authorized by this document)

A future preregistration attempt for this exact family/child could resolve Phase 3 by:
- An explicit, **new**, separately operator-approved scoping decision restricting the base universe to one direction (mirroring the CVD/Absorption precedent) — this would need to be authorized *before* re-attempting Phase 3, not decided unilaterally inside a preregistration script.
- A repository-wide, outcome-blind, pre-existing "signal-relative" or "direction-aligned" outcome convention, if one is ever established independently of this family (none currently exists — confirmed by search this session).

Neither path is exercised here. This document authorizes nothing beyond the record of the block itself.

## Input manifest

| Field | Value |
|---|---|
| `canonical.sqlite` sha256 (at this attempt) | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` |
| `canonical.sqlite` schema_version | 14 |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged by this attempt) |
| Readiness/contract commit | `f115b9c1` |
| Disposable rehearsal commit | `6a449a64` |
| Row-accounting freeze commit | `54d00dca` |
| Canonical migration commit (M-0036) | `5267a15a` |

## Governance state (before this attempt = after this attempt)

| Field | Before | After |
|---|---|---|
| `schema_version` | 14 | 14 |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `epistemic_test_nullifiers` | 2 | 2 |
| `experiment_gate_receipts` | 2 | 2 |

**Zero governance-table writes.** An INCOMPLETE closure is, by design, governance-indistinguishable from never having attempted it — nothing is authorized, nothing is consumed, nothing is registered.

## Focused tests

`tests/test_ami_research_book_spread_dynamics_preregistration_v1.py` — 17/17 passed. Covers: family/child identity constants and determinism, graveyard-clean against the real knowledge DB, prior-exposure-clean, read-only enforcement (no write from identity resolution), population accounting matching the frozen 196/97, direction-mixed proof (LONG 70 / SHORT 27 of 97 representatives), row-accounting-root drift rejected (both by the schema's own CHECK constraint and the module's own guard), outcome reused verbatim (not derived) with an AST-scoped guard proving no `endpoint_return_bps`/`mfe_bps` value is ever selected, the Phase-3 INCOMPLETE resolution itself (both the mixed-population blocked path and the trivial single-direction sanity branch), zero database mutation across the full identity→population→outcome→direction chain, and confirmation that the real DB's governance counts are exactly the pre-attempt baseline with no orphaned receipt/nullifier row under this family's resolved `canonical_family_id`.

## Regression

Full accepted-baseline comparison in the state-transition proof. This batch is additive-only (one new read-only module + one new test file); it does not touch `ami/warehouse/schema.py`, any canonical/knowledge table DDL, or any other family's code.

## Storage report

No temporary database copy exceeded a few hundred KB (disposable copies of `canonical.sqlite`/`knowledge.sqlite` under pytest's own `--basetemp`, deleted by the test runner). No copy of `data/microstructure.db` was made. No canonical or knowledge write occurred against the real files (confirmed by hash/count comparison, see state-transition proof).

## Status

**`BOOK_SPREAD_DYNAMICS_PREREGISTRATION_V1_INCOMPLETE`**

## Recommended next step

Not a controlled research gate — an **operator scoping decision**: whether to authorize a direction-restricted re-attempt (mirroring the CVD/Absorption LONG-only precedent) as a new, explicitly-approved population narrowing, before any further preregistration work on this family. This batch does not recommend a default; it surfaces the fork and stops.
