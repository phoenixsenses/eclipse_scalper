# S34_CASCADE_ABSORPTION_IMPACT_EXECUTION_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-GOVERNED-EXECUTION-V1
**Status:** EXECUTED. TEST outcome accessed exactly once, for `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` only.
**Preregistration:** `S34_CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1.md`, commit `fb002a75`.
**TEST access timestamp:** 2026-07-07T11:13:29Z UTC.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Pre-execution identity check (zero drift, confirmed before any TEST access)

| Element | Frozen (preregistration) | Reproduced (this batch, before TEST access) | Match |
|---|---|---|---|
| `family_id` | `FAMv1:3e2dfe63f9e271bf` | `FAMv1:3e2dfe63f9e271bf` | ✅ |
| `experiment_id` | `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` | (used as key throughout) | ✅ |
| Specification hash | `531b16232a88d5a6c692055bd00fa59bd508b7b69cd7fd45cf8e666772fb6608` | not re-derived (frozen constants copied verbatim into code, per module docstring discipline) | ✅ |
| `split_version` | `SPLITv1:16ea98c239034593` | `SPLITv1:16ea98c239034593` | ✅ |
| TRAIN cycle-set hash | `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5` | `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5` | ✅ |
| TEST cycle-set hash | `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04` | `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04` | ✅ |
| TEST nullifier | `4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4` | `4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4` | ✅ |
| Gate receipt ID/hash | `6dbe0f59416977fce75b20a13876ff4d54dddae171d1fa8b07613135550e06e4` | receipt found, identity fields match | ✅ |
| Nullifier prior state | unused | 0 prior consumptions, `is_rerun_of_self=False` | ✅ |
| TRAIN count | 91 | 91 | ✅ |
| TEST count | 40 | 40 | ✅ |
| `schema_version` | 13 | 13 | ✅ |
| Absorption W300 coverage | 324/324 `EXACT_RECONSTRUCTABLE` | 324/324 | ✅ |
| Known-at violations | 0 | 0 | ✅ |
| Proxy tables | none | none present | ✅ |

**Zero identity drift, zero cycle-set drift, nullifier still unused, gate receipt state `PREREGISTERED_NOT_EXECUTED`, migrated absorption rows/hashes unchanged, exact/proxy pooling absent.** Execution proceeded.

---

## Code freeze

`ami/research/cascade_absorption_impact_001.py` — new module, mirroring `ami/research/cvd_windowed_flow_001.py`'s exact lifecycle (population resolution → pre-execution verification → TRAIN diagnostics → nullifier consumption → TEST access → model fit → verdict → registry/results write → gate-receipt reissue), with one deliberate, preregistration-mandated deviation: **no pseudo-inverse**. The CVD execution kept an always-zero `session_EUROPE` column and used `np.linalg.pinv`; this preregistration explicitly froze the opposite policy (`pseudo_inverse_permitted: false`), so `EUROPE` is dropped from the design matrix entirely, a genuine `np.linalg.inv` is used, and an explicit pre-fit matrix-rank check (`check_design_rank`) is mandatory before any fit.

**Focused tests before any TEST access:** `tests/test_ami_research_cascade_absorption_impact_001.py` — 19/19 passed, including a full disposable-copy dress rehearsal of `execute_governed_run` (nullifier CONSUMED→NOOP_IDENTICAL round-trip, registry/results INSERTED→NOOP_IDENTICAL, gate receipt reissue) against copies of the real databases, never the real files. No code was changed after this point.

---

## Zero-variance / rank policy (applied exactly as frozen)

| Check | Result |
|---|---|
| `EUROPE` TRAIN observations | 0/91 |
| `EUROPE` TEST observations | 0/40 |
| `EUROPE` column in design matrix | absent (dropped structurally, not via pseudo-inverse) |
| Design columns | `const, price_response_w300, event_notional_per_100k, session_US, session_OFF, day_trend_bps` (6) |
| TRAIN design rank | 6/6 — full rank |
| TEST design rank | 6/6 — full rank |
| Pseudo-inverse used | **no** |

No design adjustment was made after viewing any coefficient or p-value — the rank checks ran before the coefficient was ever computed.

---

## Nullifier consumption (the point of no return)

| Step | Result |
|---|---|
| Identity/specification/split/TEST-cycle validation | passed (0 errors) |
| Nullifier unused check | confirmed, 0 prior consumptions |
| Consumption | `gates.consume_test_evidence(...)` → **`CONSUMED`** |
| Bound to | `family_id=FAMv1:3e2dfe63f9e271bf`, `split_version=SPLITv1:16ea98c239034593`, `test_cycle_set_hash=98174ed3…` |
| TEST access permitted | only after successful consumption, per the module's own ordering (consume → then read TEST rows) |
| Second TEST execution | **did not occur** — idempotency was proven separately, on a disposable copy, before this real run (see Code freeze §) |
| Nullifier consumed count (real DB) | exactly 1 |

---

## Cross-family holdout exposure (disclosed transparently)

The ordered TEST cycle set (40 cycles, hash `98174ed3…`) is **byte-identical** to the TEST cycle set previously used by the closed CVD experiment (`E-CVD-PRIMARY-LONG-W300-PREREG-001`). This is permitted by the frozen family-specific nullifier design — the nullifier is derived from `family_id + split_version + test_cycle_ids`, so identical cycles under a different family/nullifier are not a reuse violation — but is recorded here and in `researcher_exposure_ledger` (category `CROSS_FAMILY_TEST_CYCLE_REUSE_DISCLOSURE`, exposure_id `EXP-56a8d3e1eb5a4c6abbf2f22b`):

1. Same TEST cycle set as the prior CVD family, byte-identical hash.
2. Different canonical family (`FAMv1:3e2dfe63f9e271bf`) and different nullifier (`4e3d1229…`) than CVD (`FAMv1:bec99d8d36f7d6a1` / `085397f3…`).
3. **Not an independent market-period replication** — both experiments observe the same underlying 40 holdout cycles; this result and the closed CVD result are not statistically independent confirmations of a shared claim.
4. No CVD result was used to alter this absorption specification — the preregistration (`fb002a75`) was frozen before this execution and reused only CVD's outcome/controls/split-algorithm *identity*, never any CVD outcome *value*.
5. No new multiplicity correction or verdict rule is introduced post-preregistration — the verdict rule applied is exactly the one frozen in the preregistration.

---

## Result

| Statistic | Value |
|---|---|
| TEST N | 40 (0 dropped for missingness) |
| TRAIN N | 91 (0 dropped) |
| Primary predictor coefficient | −3.4285074465436134 |
| Units | bps of `endpoint_return_bps` per 1-unit change in `price_response_per_signed_notional` (a bps-per-$1M ratio) |
| Cluster-robust SE | 2.3954324586247613 |
| 95% CI | [−8.27372693, 1.41671204] |
| t-stat / df | −1.4312686772692182 / 39 |
| p-value | 0.16031838015391875 |
| Design rank | 6/6 (both TRAIN and TEST) |
| VIF (predictor / event_notional / day_trend_bps) | 1.034 / 1.029 / 1.019 — no collinearity, 0 drops applied |
| TRAIN-side descriptive coefficient (non-promotable) | 1.841861197678332 (p=0.1107) |
| Secondary `mfe_bps` diagnostic (non-promotable) | −2.350901577615649 (p=0.0595) |

**Missing-row / quality accounting:** 40/40 TEST representative signals used, 0 dropped; 91/91 TRAIN representative signals used, 0 dropped; 324/324 `EXACT_RECONSTRUCTABLE` W300 coverage; 131 representative cycles, 0 duplicates. **Known-at result:** 0 violations, re-verified at the pre-execution identity check. **Exact/proxy separation:** no proxy table exists for this family; 0 proxy rows; pooling absent.

---

## Verdict-rule evaluation

| Condition | Result |
|---|---|
| CI excludes 0 | **false** |
| p < 0.05 | **false** (p=0.160) |
| \|coef × TRAIN stdev(10.7011)\| ≥ 5bps | **true** (36.69 ≥ 5) |
| n ≥ MIN_BUCKET_N(20) | true |
| CI half-width ≤ 2× floor(10) | true (4.85 ≤ 10) |

A favorable effect *magnitude* alone (36.69 bps, well above the 5bps relevance floor) is **not sufficient** — the frozen verdict rule requires CI-excludes-zero AND p<0.05 AND the magnitude floor jointly. Since the confidence interval includes zero and p exceeds 0.05, the disposition is:

**`NO_RELIABLE_INCREMENTAL_ASSOCIATION`**

Reason string (frozen, machine-recorded): `ci_excludes_zero=False p=0.1603 |coef*10.7011|=36.6887`

---

## Exact DB/table deltas

| Table | Before | After |
|---|---|---|
| `canonical.sqlite` sha256 | `815f35d0…` | `3aefce83…` |
| `knowledge.sqlite` sha256 | `d435c3a2…` | `710b3f68…` |
| `experiment_registry` | 23 | **24** |
| `experiment_results` (total) | 350 | **381** (+31, all bound to this `experiment_id`) |
| `epistemic_test_nullifiers` | 1 | **2** (+1, this experiment's nullifier) |
| `experiment_gate_receipts` (row count) | 2 | 2 (same row, state updated) |
| Gate receipt state (this experiment) | `PREREGISTERED_NOT_EXECUTED` | **`EXECUTED`** |
| `researcher_exposure_ledger` | 1,176 | **1,180** (+4: identity check, TRAIN fetch, TEST fetch, cross-family disclosure) |
| `schema_version` | 13 | 13 (unchanged) |
| `ami_absorption_impact_windowed_flow`/`_quality`/`_exclusions` | 1,619/1,620/1 | 1,619/1,620/1 (unchanged) |
| `integrity_check` | — | ok |
| `foreign_key_check` | — | [] |

## Protected delta

| Table | Before | After |
|---|---|---|
| `ami_events` | 252 | 252 |
| `ami_signal_lifecycle` | 324 | 324 |
| `ami_cycles` | 167 | 167 |
| `ami_birth_truncated_cascade_geometry` | 220 | 220 |
| `ami_agg_trades_repaired` | 40,934 | 40,934 |
| `ami_cvd_windowed_flow` / `_proxy` | 1,840 / 1,840 | 1,840 / 1,840 |

**Protected delta = ZERO.** No runtime/risk/execution file was read or modified. No route or bucket was promoted.

---

## Remaining limitations

1. `NO_RELIABLE_INCREMENTAL_ASSOCIATION` — the pre-birth W300 price-impact-per-notional ratio does not show a statistically reliable incremental association with the LONG `endpoint_return_bps@swing_24h` outcome in this population, despite a large point-estimate magnitude (the CI is simply too wide, and includes zero).
2. This experiment's TEST cycles are **not independent** of the closed CVD experiment's TEST cycles (identical 40 holdout cycles) — this result and CVD's own closed result are not two independent confirmations of unrelated market periods (disclosed in full above and in the exposure ledger).
3. W60/W600/W1800/W3600 remain entirely untested against any outcome, per the frozen Amendment 2 scope restriction — no information exists about whether a different window would show an association.
4. SHORT-direction absorption/impact remains entirely untested.
5. The secondary `mfe_bps` diagnostic (p=0.059) sits close to conventional significance but is explicitly `NON_PROMOTABLE` and does not alter the primary verdict — it may inform, but must not justify, any future independently preregistered follow-up.

---

## Success verdicts

**Operational:** `CASCADE_ABSORPTION_IMPACT_GOVERNED_EXECUTION_V1_COMPLETE`

**Scientific disposition:** `NO_RELIABLE_INCREMENTAL_ASSOCIATION`

Stopping after recording the result. No follow-up hypothesis, window, subgroup, or bucket is opened by this batch.
