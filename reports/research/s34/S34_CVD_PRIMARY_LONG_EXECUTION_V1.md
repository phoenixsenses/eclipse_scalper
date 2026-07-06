# S34_CVD_PRIMARY_LONG_EXECUTION_V1

**Gate:** G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1
**Preregistration:** `S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md` (commit `749520b3`), experiment `E-CVD-PRIMARY-LONG-W300-PREREG-001`
**Status:** EXECUTED. Single, authorized, one-shot TEST-outcome access. Frozen specification, no amendment.
**Date:** 2026-07-06 · **Author:** Sonnet 5

This document is immutable once committed. A revised computation on this same TEST evidence requires a new preregistration and a supersession token — never an edit here.

---

## 0. Pre-execution verification (before any TEST access)

Every frozen identity element from the preregistration was independently reproduced against the real `data/ami/canonical.sqlite`/`data/ami/knowledge.sqlite`, **before the nullifier was consumed**:

| Element | Frozen value | Reproduced | Match |
|---|---|---|---|
| Canonical family ID | `FAMv1:bec99d8d36f7d6a1` | `FAMv1:bec99d8d36f7d6a1` | ✅ |
| Split version | `SPLITv1:0a1b96fd74dd281e` | (used as frozen constant, not re-derived from prose — see §6) | ✅ |
| Representative cycles | 131 | 131 | ✅ |
| TRAIN / TEST cycle count | 91 / 40 | 91 / 40 | ✅ |
| TRAIN cycle-set hash | `61486bc6…` | `61486bc6…` | ✅ |
| TEST cycle-set hash | `98174ed3…` | `98174ed3…` | ✅ |
| TEST-evidence nullifier | `085397f3…` | `085397f3…` | ✅ |
| Nullifier prior consumption | 0 | 0 | ✅ (unconsumed) |
| Gate receipt | `PREREGISTERED_NOT_EXECUTED`, family/split/nullifier match | present, matching | ✅ |
| W300 quality population | 324/324 EXACT_RECONSTRUCTABLE, 0 SOURCE_GAPPED | 324/324 | ✅ |
| Known-at violations (W300) | 0 | 0 | ✅ |
| Bucket exclusions | 100% SHORT (104) | `{'SHORT': 104}` | ✅ |
| `experiment_registry` / `experiment_results` (before) | 22 / 323 | 22 / 323 | ✅ |
| `schema_version` | 12 | 12 | ✅ |

**Zero errors.** `CVD_PRIMARY_LONG_GOVERNED_EXECUTION_V1_BLOCKED` was not triggered — execution proceeded.

## 1. Split-version note (disclosed, not a deviation)

The preregistration's `split_version` token (`SPLITv1:0a1b96fd74dd281e`) is an output of `epistemic_gates.resolve_split_version(frozen_splits)`, a one-way hash of a `frozen_splits` free-text description. That exact description string was never recorded verbatim in any committed artifact (only the resulting token was). Rather than author a new description and risk silently deriving a **different** split_version/nullifier pair, this execution treats the already-issued token as a frozen constant and reproduces the population/split/nullifier from it directly — exactly the same approach the preregistration's own committed test suite (`tests/test_ami_cvd_primary_long_preregistration_v1.py::test_nullifier_reproducible_from_frozen_cycle_sets`) already uses. The nullifier match above (`085397f3…`, computed from `family_id` + this frozen `split_version` + the reproduced TEST cycle set) is the operative proof that the correct split identity was used — it could not match if the split methodology had silently changed.

## 2. Authorization and TEST access

1. `consume_test_evidence()` was called with `family_id=FAMv1:bec99d8d36f7d6a1`, `split_version=SPLITv1:0a1b96fd74dd281e`, the reproduced 40-cycle TEST set, `experiment_id=E-CVD-PRIMARY-LONG-W300-PREREG-001` — **result: `CONSUMED`** (first use, real database).
2. Only after that call did any code path read `endpoint_return_bps`/`mfe_bps` for the 40 TEST-representative signals (scoped SQL by exact signal_id list — see `ami/research/cvd_windowed_flow_001.py::_fetch_effective_outcome_for_signals`).
3. TRAIN outcome (91 rows) was read **before** authorization — permitted by the preregistration (§7: "TRAIN may be used only for... model implementation validation... never to select anything") and used only for the predictor-scaling check (§3 below), VIF, and the secondary TRAIN-side descriptive coefficient (never compared to TEST to choose anything).

## 3. TRAIN-side diagnostics (pre-authorization, non-promotable)

- **Predictor distribution (TRAIN, n=91, USD, before /1,000,000 scaling):** min=-114,507,036.88, max=20,161,411.40, mean=-23,002,972.44, median=-16,216,250.98 — **byte-identical to the frozen preregistration's own recorded TRAIN distribution** (§4 of the prereg document), independently reproduced.
- **VIF (TRAIN):** cvd=1.174, event_notional=1.024, session_US=1.329, session_OFF=1.315, day_trend_bps=1.033. All well under the 10.0 threshold.
- **`session_EUROPE` has zero variance in TRAIN (and, discovered subsequently, in TEST too — 0/91 and 0/40 respectively).** No LONG signal in either split falls in the 07:00–13:00 UTC EUROPE session window. This was **not anticipated by the preregistration** (which specifies EUROPE as one of three session dummies) and is a genuine, TRAIN-discovered data condition, not a researcher choice. VIF for `session_EUROPE` is undefined (`None`, reported honestly rather than a spurious `inf`/`nan`). The regression estimator (`np.linalg.pinv` in place of a strict inverse) handles the resulting exact singularity without altering the model's column set: `session_EUROPE`'s own coefficient/SE come out as ≈0/0.0 (there is no information to estimate it), and — because pinv reduces to the ordinary inverse whenever the design is otherwise full rank — every other coefficient, **including the primary predictor**, is unaffected. This is disclosed as a required-validation finding, not concealed.
- **Collinearity policy:** no VIF exceeded 10.0 → **no controls dropped** (`collinearity_drops_applied = []`).
- **TRAIN-side descriptive coefficient (secondary, non-promotable, never used to select anything):** -0.646 bps per $1,000,000, p=0.668 (TRAIN only, for descriptive continuity — not compared to TEST to choose a specification).

## 4. Primary confirmatory result (TEST, n=40)

Model: `endpoint_return_bps ~ cvd_notional_w300_per_1M + event_notional_per_100k + session_EUROPE + session_US + session_OFF + day_trend_bps`, OLS, cluster-robust (CR1) SE clustered by `independent_cycle_id` (G=40, one signal per cluster by construction), inference via Student-t with df=G-1=39.

| Quantity | Value |
|---|---|
| TEST n used | 40 (0 dropped to missingness) |
| Primary predictor coefficient | **-0.9356** bps per $1,000,000 net taker-flow notional |
| Cluster-robust SE | 2.6620 |
| t-statistic (df=39) | -0.3515 |
| 95% CI | (-6.3200, 4.4488) |
| p-value (two-sided) | **0.7271** |
| Effect-size relevance check | `\|coefficient × 10\| = 9.356` bps per $10M — **clears** the 5bps/$10M floor on magnitude alone |
| CI excludes 0? | **No** |

Full coefficient vector (`const, cvd, event_notional, session_EUROPE, session_US, session_OFF, day_trend_bps`):
`[-37.188, -0.936, 2.807, 5.9e-14, -32.177, 202.417, -0.316]`, SE `[96.242, 2.662, 23.356, 0.0, 94.868, 112.926, 0.326]`.

## 5. Secondary checks (non-promotable, reported alongside — never replacing — the primary result)

- **Same model, `mfe_bps@swing_24h` as outcome (TEST):** coefficient +2.353, p=0.145. Not compared to the primary result to select anything; reported per §6 of the preregistration.
- **TRAIN-side descriptive coefficient:** see §3 above.

## 6. Preregistered verdict rule, applied exactly

> `EVIDENCE_SUPPORTS_INCREMENTAL_ASSOCIATION` requires: CI excludes 0 **AND** p<0.05 **AND** `\|coefficient×10\|>=5` **AND** n>=20 **AND** no invalidation.

- n=40 ≥ 20 ✓
- CI half-width = 5.384, well under 2×5=10 (not `INSUFFICIENT_SAMPLE_OR_INCONCLUSIVE` on width)
- CI **includes** 0 (-6.32, 4.45) → fails the first conjunct
- p=0.727 ≥ 0.05 → fails the second conjunct
- Effect-size magnitude (9.356) clears the floor, but a cleared floor alone is explicitly **not sufficient** per the preregistration ("A positive/significant coefficient alone, without clearing the effect-size floor, is `NO_RELIABLE_ASSOCIATION`, not support" — and symmetrically here, clearing the floor without CI-exclusion-of-zero and significance is not support either)

**Verdict: `NO_RELIABLE_ASSOCIATION`.**

No route, entry rule, threshold, or trading promotion is implied by this result — none was ever in scope for this preregistration.

## 7. Required validations (proven at execution time)

- Exact/proxy CVD pooling: **did not occur** — only `ami_cvd_windowed_flow` (`window_id='W300'`) was queried; `ami_cvd_windowed_flow_proxy` was never opened by this module (confirmed by source inspection — the module has no reference to that table name).
- No threshold scan, no predictor binning, no alternative transformation, no subgroup rescue, no session-specific rerun, no alternative outcome swap, no interaction/nonlinear search: **none performed** (the module runs exactly one primary model + the two preregistered secondary checks).
- Known-at violations: 0 (re-verified at execution time, same query as preregistration).
- Cycle representatives unique: 131 = 131 distinct `independent_cycle_id` values (by construction).
- Missing-data accounting: TRAIN 0/91 dropped, TEST 0/40 dropped (listwise deletion had nothing to remove).
- No second model was run after seeing the primary TEST result (the `mfe_bps` secondary check is pre-specified, computed in the same pass, not a reaction to the primary outcome).
- No code changed after TEST access (the module was fully tested — 15/15 tests, including a full dress rehearsal against disposable copies of the real data producing byte-identical TRAIN diagnostics — before this real, one-shot execution).

## 8. Real database state, before → after

| Check | Before | After |
|---|---|---|
| `canonical.sqlite` sha256 | `fdda663d…` | `25a56a98…` (exposure-ledger + new experiment_registry/results rows only) |
| `schema_version` | 12 | 12 (unchanged) |
| `experiment_registry` | 22 | 23 (+1, this experiment only) |
| `experiment_results` | 323 | 350 (+27, this experiment's metrics only) |
| Protected counts (events/signal_lifecycle/cycles/geometry) | 252/324/167/220 | 252/324/167/220 (unchanged) |
| CVD frozen tables (repaired/exact/proxy/exclusions/quality) | 40934/1840/1840/104/1840 | unchanged |
| `researcher_exposure_ledger` | 1173 | 1176 (+3, expected by-design exposure-audit appends from this batch's 3 gateway calls) |
| `integrity_check` | — | ok |
| `foreign_key_check` | — | clean (0 rows) |
| `knowledge.sqlite` sha256 | `ef7f8cde…` | `2a5abc28…` |
| `epistemic_test_nullifiers` (this nullifier) | 0 rows | 1 row, `consumed_by_experiment_id=E-CVD-PRIMARY-LONG-W300-PREREG-001` |
| `experiment_gate_receipts` (`registry_result`) | `PREREGISTERED_NOT_EXECUTED` | `EXECUTED` |
| `graveyard_slash_fingerprints` | 31 | 31 (unchanged) |

Full backups taken before execution: `data/ami/backups/canonical_pre_G2_governed_execution_20260706.sqlite`, `data/ami/backups/knowledge_pre_G2_governed_execution_20260706.sqlite`.

## 9. Scope discipline

`execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` were not opened by this batch. No route was promoted, no live/paper/shadow/forward runner was touched. This is a single, governed, one-shot statistical execution of an already-frozen preregistration — nothing else.

## 10. Remaining limitations (carried over / newly disclosed)

1. **`session_EUROPE` structural absence** (§3) — disclosed here for the first time; does not affect the primary predictor's identification but is a genuine gap in the preregistered model's applicability to this population. A future preregistration touching session controls should account for this.
2. Carried over from the preregistration: the outcome-identity interpretation risk (§2 item 1, no single named prior `experiment_id` for "the primary LONG reversal outcome") and the representative-selection rule being newly authored (§2 item 5) are unaffected by execution — both were operator-disclosed before TEST access and neither was revisited after seeing the result.
3. Family/split identity adapters remain text-hash-based (paraphrase-bypass risk), unchanged by this batch.

---

## Verdict

**Operational: `CVD_PRIMARY_LONG_GOVERNED_EXECUTION_V1_COMPLETE`.**
**Scientific: `NO_RELIABLE_ASSOCIATION`** — the TEST-set confidence interval on the primary CVD coefficient includes zero and p=0.727; no further follow-up hypothesis or research wave is opened from this result.
