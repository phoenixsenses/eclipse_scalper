# CASCADE_ABSORPTION_IMPACT_EXECUTION_V1_EVIDENCE_AND_TEST_STATE_CLOSURE

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-EXECUTION-EVIDENCE-AND-TEST-STATE-CLOSURE-V1
**Nature:** Test-architecture repair and evidence-closure documentation only. No TEST reread, no scientific module/model/predictor/outcome/coefficient/verdict/window/control/population change, no follow-up hypothesis or rescue analysis.
**Depends on (source of truth, unedited):** preregistration `fb002a75`, execution `5e9e2e33`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Frozen scientific result (preserved without rerun)

| Field | Value |
|---|---|
| Operational execution commit | `5e9e2e33` |
| Scientific disposition | **`NO_RELIABLE_INCREMENTAL_ASSOCIATION`** |
| Coefficient | −3.4285074465436134 |
| Cluster-robust SE | 2.3954324586247613 |
| 95% CI | [−8.27372693, 1.41671204] |
| p-value | 0.16031838015391875 |
| TRAIN / TEST | 91 / 40 |
| `experiment_registry` | 23 → 24 |
| `experiment_results` | 350 → 381 |
| Nullifier | consumed exactly once |
| Gate receipt | `EXECUTED` |

TEST was not reread. No scientific code, model, predictor, outcome, coefficient, verdict, window, control, or population was altered by this batch. No follow-up hypothesis or rescue analysis was opened.

---

## PART 1 — Stale test inventory

Root cause (common to all 5): the real `data/ami/canonical.sqlite`/`data/ami/knowledge.sqlite` files now permanently contain the completed execution (commit `5e9e2e33`). Tests written and dress-rehearsed *before* that execution, which read the ambient live-DB state (directly, or via conftest's session-scoped disposable copy — itself a snapshot of whatever the real files currently contain) and asserted the *pre-execution* condition, now correctly observe the *post-execution* condition instead. This is the permanent, expected record of a real, one-time, accepted state transition — not a scientific or enforcement failure.

| # | File | Node ID | Old expected state | Current legitimate state | Reads live DB? | Why stale | Proof it is not a failure |
|---|---|---|---|---|---|---|---|
| 1 | `tests/test_ami_research_cascade_absorption_impact_001.py` | `test_verify_pre_execution_reports_zero_errors_against_real_db` | `is_rerun_of_self is False` | `True` | Yes — conftest disposable copy of the real files | The disposable copy conftest makes for the whole test session is a snapshot of the real files, which now include this experiment's completed nullifier consumption | `verify_pre_execution`'s own `errors == []` and `family_id`/`nullifier` match were never violated — only the *interpretation* of an already-consumed nullifier changed, correctly, from "not yet" to "already" |
| 2 | `tests/test_ami_research_cascade_absorption_impact_001.py` | `test_execute_governed_run_blocks_on_identity_mismatch` | nullifier row count `== 0` before the blocked attempt | `1` (the real, unrelated, legitimate consumption) | Yes — same disposable copy | Same root cause | The test's actual subject — that a blocked (identity-mismatched) attempt raises `ProtocolInvalidation` and consumes nothing *of its own* — still held; only the ambient baseline count changed |
| 3 | `tests/test_ami_research_cascade_absorption_impact_001.py` | `test_governed_execution_dress_rehearsal_on_disposable_copies` | `pre_verify["is_rerun_of_self"] is False`, then fresh `CONSUMED`/`INSERTED` | starts already `is_rerun_of_self=True`; a fresh call now returns `NOOP_IDENTICAL`/`NOOP_IDENTICAL` instead | Yes — same disposable copy | Same root cause | The idempotency mechanism itself is provably correct either way — `NOOP_IDENTICAL` on an already-executed copy is the *correct* response, not a bug |
| 4 | `tests/test_ami_absorption_impact_preregistration_v1.py` | `test_gate_receipt_mechanism_round_trips_on_disposable_copy` | nullifier count `== 0` before and after issuing an unrelated `TEST-COPY-` receipt | `1` before and after (unchanged, just no longer zero) | Yes — raw `shutil.copyfile` of the real `knowledge.sqlite` | Same root cause, via a direct file copy rather than conftest redirection | The invariant actually under test — issuing a gate receipt alone never itself changes the nullifier count — held both before (`0→0`) and after (`1→1`) real execution |
| 5 | `tests/test_ami_absorption_impact_preregistration_v1.py` | `test_real_nullifier_and_receipt_state` | `receipt == ('PREREGISTERED_NOT_EXECUTED',)`, nullifier count `== 0` | `receipt == ('EXECUTED',)`, nullifier bound to this exact experiment_id | Yes — direct `mode=ro` connection to the real `knowledge.sqlite` | This test explicitly checks the *real, current* state of the real file — which has legitimately, permanently changed | The real file's `EXECUTED` state is precisely the accepted, intended outcome of commit `5e9e2e33` — asserting anything else would itself be the false statement |

None of the 5 indicate a scientific or enforcement failure: in every case, the underlying mechanism (identity resolution, nullifier single-consumption, gate-receipt validation, idempotent rerun, blocked-attempt rejection) is proven correct by the *new* assertions written for this closure (Part 2) — only the *ambient starting condition* the old assertions assumed no longer holds, permanently and correctly.

An additional, sixth, genuinely new failure was found during validation (Part 6) and is documented and fixed separately, since it falls outside this 5-test inventory (it was not part of the original transition proof's disclosure) — see Part 6.

---

## PART 2 — Test-state repair

**Approach taken (matches the requested preferred approach exactly):**

- **Pre-execution behavior** is now tested against a deterministic disposable fixture: `_reset_experiment_state_to_preregistered()` + the `fresh_experiment_conns` pytest fixture (`tests/test_ami_research_cascade_absorption_impact_001.py`), which explicitly deletes this experiment's own `experiment_results`/`experiment_registry` rows and resets its `epistemic_test_nullifiers`/`experiment_gate_receipts` rows to `PREREGISTERED_NOT_EXECUTED` **on the disposable copy only**, before any pre-execution assertion runs — deterministic regardless of what the real source files currently contain.
- **Post-execution behavior** is tested against an explicit EXECUTED-state fixture: a new test, `test_verify_pre_execution_detects_already_executed_state`, constructs the EXECUTED state deterministically (by calling the real `execute_governed_run` once, on the disposable copy, via the same fresh-state fixture) and then proves `verify_pre_execution` correctly reports `is_rerun_of_self is True` and the gate receipt as `EXECUTED`.
- **The lifecycle transition itself** (`PREREGISTERED_NOT_EXECUTED → EXECUTED`) is now tested explicitly: `test_governed_execution_dress_rehearsal_on_disposable_copies` was extended with an explicit before/after receipt-state assertion around the real `execute_governed_run` call, and the idempotent rerun (`r2`) additionally asserts the receipt stays `EXECUTED` (not re-transitioned).
- For the two preregistration-file tests, the real-database-dependent assertions were updated to be either **relative** (`test_gate_receipt_mechanism_round_trips_on_disposable_copy` now asserts the nullifier count is *unchanged* by issuing a receipt, rather than asserting it equals a specific absolute number) or **accurate to current, permanent, real-world truth** (`test_real_nullifier_and_receipt_state`, renamed `test_real_nullifier_and_receipt_state_post_execution`, now asserts `EXECUTED` + the nullifier bound to this exact experiment_id — a strictly *more* specific check than the original, not a weaker one).

**Requirements verified:**

| Requirement | How satisfied |
|---|---|
| Preserve all pre-execution invariants | `test_verify_pre_execution_reports_zero_errors_against_fresh_disposable_state` (renamed) still asserts `errors==[]`, `family_id`/`nullifier` match, `is_rerun_of_self is False`, `already_has_results_before==0`, `schema_version==13` — all against the deterministic fresh-state fixture |
| Preserve nullifier single-consumption checks | Unchanged in substance across all 5 repaired tests; `test_execute_governed_run_blocks_on_identity_mismatch` still proves a blocked attempt consumes nothing |
| Preserve gate receipt validation | Both the fresh-state and EXECUTED-state paths now explicitly check `experiment_gate_receipts.registry_result` |
| Preserve idempotency checks | `test_governed_execution_dress_rehearsal_on_disposable_copies`'s `r1`→`r2` `CONSUMED`→`NOOP_IDENTICAL` / `INSERTED`→`NOOP_IDENTICAL` round-trip is unchanged, now against a guaranteed-fresh starting state |
| Preserve rejection of a second TEST execution | `test_execute_governed_run_blocks_on_identity_mismatch` unchanged in mechanism, now deterministic |
| Do not weaken, skip, xfail, or delete meaningful assertions | No assertion was removed; two were made *more* specific (post-execution receipt test now also checks the exact consuming experiment_id) |
| Do not modify scientific production code | `ami/research/cascade_absorption_impact_001.py` was not touched by this batch |
| Do not mutate the live canonical or knowledge DB during tests | All resets/mutations target only the conftest-redirected disposable copy or a `shutil.copyfile`d disposable file — confirmed by a before/after hash comparison of the real files (Part 5) showing zero change across the entire test-repair and validation process |

---

## PART 3 — 31-row result accounting

All 31 rows share a single `created_ms=1783422809955` (**2026-07-07T11:13:29.955Z UTC**, one atomic write inside `record_experiment_results`) and a single `provenance` string: `BATCH-CASCADE-ABSORPTION-IMPACT-GOVERNED-EXECUTION-V1; preregistration_commit=fb002a75; spec_hash=531b16232a…; test_nullifier=4e3d1229…; gate_receipt_hash_at_preregistration=6dbe0f59…; collinearity_drops=[]; design_rank_check=full_rank_no_pinv`.

| # | Metric key | Role | Content hash (16 hex) |
|---|---|---|---|
| 1 | `primary_predictor_coefficient_bps_per_unit` | PRIMARY | `a3da3a3913c38e0a` |
| 2 | `primary_predictor_se_cluster_robust` | PRIMARY | `bc550b51a536b9bf` |
| 3 | `primary_predictor_ci95_lo` | PRIMARY | `fbe518ca962a1b25` |
| 4 | `primary_predictor_ci95_hi` | PRIMARY | `a016d101c8f9c3c1` |
| 5 | `primary_predictor_p_value` | PRIMARY | `a23ec31914575ce5` |
| 6 | `primary_predictor_t_stat` | PRIMARY | `7fec0380acdf2457` |
| 7 | `primary_predictor_df` | PRIMARY | `f204f5c3b487cb9b` |
| 8 | `test_n_used` | PRIMARY (sample accounting) | `7d8b99797f8b8e28` |
| 9 | `test_n_total_representative` | PRIMARY (sample accounting) | `ba2d5916795538b2` |
| 10 | `test_n_dropped_missing` | PRIMARY (sample accounting) | `f412eb197f1608a9` |
| 11 | `test_n_clusters` | PRIMARY (sample accounting) | `831f4ee05f364bcd` |
| 12 | `test_design_rank` | PRIMARY (rank proof) | `41a6f1e9ce5cab38` |
| 13 | `verdict_reason` | PRIMARY (verdict binding) | `332d03e678409208` |
| 14 | `test_cycle_set_hash` | PRIMARY (identity binding) | `3136faa0304109e4` |
| 15 | `test_nullifier_sha256` | PRIMARY (identity binding) | `4f81a88af6d253bb` |
| 16 | `collinearity_drops_applied` | DIAGNOSTIC/ACCOUNTING | `d1b3866ca642cbc5` |
| 17 | `cross_family_test_cycle_reuse_disclosure` | DIAGNOSTIC/ACCOUNTING | `b1d9dd00f27172eb` |
| 18 | `design_columns` | DIAGNOSTIC/ACCOUNTING | `6957af01ac4b43a3` |
| 19 | `full_beta_vector` | DIAGNOSTIC/ACCOUNTING | `a44003e1f484e3b8` |
| 20 | `full_se_vector` | DIAGNOSTIC/ACCOUNTING | `b341c379090264d0` |
| 21 | `predictor_train_scale_stats` | DIAGNOSTIC/ACCOUNTING | `87cab1468fd4659e` |
| 22 | `secondary_mfe_bps_coefficient` | DIAGNOSTIC (non-promotable) | `392e73610f7ce666` |
| 23 | `secondary_mfe_bps_p_value` | DIAGNOSTIC (non-promotable) | `adf6314dc275a41b` |
| 24 | `train_cycle_set_hash` | DIAGNOSTIC/ACCOUNTING | `68bdf7192423f17a` |
| 25 | `train_design_rank` | DIAGNOSTIC/ACCOUNTING | `284a1ff99be627c2` |
| 26 | `train_n_dropped_missing` | DIAGNOSTIC/ACCOUNTING | `99f35442f41b8612` |
| 27 | `train_n_used` | DIAGNOSTIC/ACCOUNTING | `9bbe72a81f5a2e2a` |
| 28 | `train_predictor_stdev_reverified` | DIAGNOSTIC/ACCOUNTING | `7b31eec579967620` |
| 29 | `train_side_descriptive_coefficient` | DIAGNOSTIC (non-promotable) | `f5722a7b2a820549` |
| 30 | `train_side_descriptive_p_value` | DIAGNOSTIC (non-promotable) | `1712fbcde654abc2` |
| 31 | `vif` | DIAGNOSTIC/ACCOUNTING | `041a171c7b474524` |

**15 PRIMARY + 16 DIAGNOSTIC/ACCOUNTING = 31.** Source function for all 31: `ami/research/cascade_absorption_impact_001.py::execute_governed_run`, written to `canonical.sqlite` via `record_experiment_results(... schema_version=13, provenance=..., created_ms=1783422809955)`.

**Proof of scope discipline:**

- **One experiment ID**: all 31 rows share `experiment_id='E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001'` — no other experiment_id was written by this batch.
- **One primary TEST fit**: exactly one `run_cluster_robust_ols` call on `window_id='W300'` TEST design produced rows 1–15; the `mfe_bps` fit (rows 22–23) is explicitly `secondary_*`, and the TRAIN-side fit (rows 29–30) is explicitly `train_side_descriptive_*` — both are `NON_PROMOTABLE_DIAGNOSTIC` per the frozen preregistration, never fed into `apply_verdict_rule`.
- **No threshold result**: no row name contains "threshold"; `apply_verdict_rule` takes no threshold argument.
- **No subgroup/session result**: no row is scoped to a session/subgroup; `session_US`/`session_OFF` appear only inside `design_columns`/`full_beta_vector`/`full_se_vector` as design-matrix bookkeeping, never as a separate fit.
- **No alternative window**: `design_columns`/`frozen_features` (registry) and every `test_*`/`train_*` row reference `window_id='W300'` exclusively; no row name or value contains `W60`/`W600`/`W1800`/`W3600`.
- **No proxy result**: no proxy table exists for this family; 0 proxy rows anywhere.
- **No alternative outcome**: the primary fit used `endpoint_return_bps` exclusively; `mfe_bps` is the one explicitly-frozen, non-promotable secondary check.
- **No second TEST model**: `test_fit`/`test_design` were computed exactly once in `execute_governed_run`; `test_design_mfe`/`secondary_fit` is a second *outcome column* on the *same* TEST design matrix, not a second model specification or a second nullifier consumption.
- **Diagnostics did not modify the verdict**: `apply_verdict_rule(test_design["n"], coef, se_coef, ci_lo, ci_hi, p_value)` takes only the primary fit's own five values as input — no diagnostic row is referenced anywhere in its signature or body (`ami/research/cascade_absorption_impact_001.py`, verified by direct source inspection).

---

## PART 4 — Effect-magnitude reconciliation (36.7 bps)

**Formula (frozen at preregistration, `fb002a75`, before any TEST access):**

```
relevant iff |coefficient × TRAIN_stdev(predictor)| >= 5 bps
```

**Inputs:**

| Input | Value | Source |
|---|---|---|
| Coefficient | −3.4285074465436134 | TEST-side primary OLS fit, column 1 (`price_response_w300`) |
| Coefficient units | bps of `endpoint_return_bps` per 1-unit change in `price_response_per_signed_notional` (itself a bps-per-$1M ratio) | frozen predictor definition |
| Predictor scaling | none (raw units, no rescaling) | frozen at preregistration — "already well-scaled... no transform needed" |
| Reference predictor difference used | **one TRAIN standard deviation** of the predictor, `10.701083978672228` | frozen at preregistration as `10.70108397867223` (see reconciliation below), independently **re-verified** (not re-selected) from TRAIN data at execution time |

**Exact calculation:**

```
magnitude = |coefficient × TRAIN_stdev| = |-3.4285074465436134 × 10.701083978672228| = 36.68874610696629
```

Rounded to 4 decimal places for the machine-recorded `verdict_reason` string: **`36.6887`**.

**Preregistration clause defining the relevance floor** (`S34_CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1.md`, §8 Model specification): *"relevant iff |coefficient × TRAIN_stdev(predictor)| ≥ 5 bps, i.e. a one-TRAIN-standard-deviation move in the predictor (10.70108397867223 bps-per-$1M, frozen from TRAIN this session, outcome-blind) implies at least 5bps of expected `endpoint_return_bps` change."*

**Proof the magnitude was not selected from TEST data post hoc:**

1. The **anchor value** (`TRAIN_stdev = 10.70108397867223`) was computed and written into the committed preregistration document (`fb002a75`) — **before this execution batch existed, before the TEST-evidence nullifier was consumed, before any TEST row was ever read.**
2. At execution time, the module re-derives the SAME statistic from TRAIN data only (`np.std(train_design["X"][:, 1], ddof=1) = 10.701083978672228`) and asserts it matches the frozen value within `1e-6` **before** consuming the nullifier — a drift check, not a re-selection. The two values differ by `1.7763568394002505e-15` (pure floating-point summation-order noise between the original `statistics.stdev` computation and `numpy`'s `ddof=1` estimator — both are the unbiased sample standard deviation of the same 91 TRAIN values), far inside the tolerance.
3. The **relevance floor itself** (5 bps) is a fixed constant, also frozen at preregistration, never touched by TEST data.
4. The **coefficient** is the only TEST-derived quantity in the formula, and it enters only *after* nullifier consumption, in the one, single, frozen primary model fit (Part 3).
5. No alternative reference-move size, no alternative predictor scaling, and no alternative relevance floor was ever computed or considered at execution time — `RELEVANCE_FLOOR_BPS` and `PREDICTOR_TRAIN_STDEV` are both hardcoded module-level constants in `ami/research/cascade_absorption_impact_001.py`, copied verbatim from the preregistration, never recomputed from TEST.

**No wording or scaling error was found.** The 36.7 bps figure (36.6887, as reported) is correct and requires no additive evidence note beyond this reconciliation.

---

## PART 5 — State and hash verification

| Check | Value | Required |
|---|---|---|
| `schema_version` | 13 | 13 ✓ |
| `ami_absorption_impact_windowed_flow` / `_quality` / `_exclusions` | 1,619 / 1,620 / 1 | unchanged ✓ |
| `experiment_registry` | 24 | 24 ✓ |
| `experiment_results` | 381 | 381 ✓ |
| Nullifier consumed | exactly once, bound to `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` | ✓ |
| Gate receipt | `EXECUTED` | ✓ |
| Prior experiments/results (23/350 as of pre-execution) | unchanged | ✓ (immutability-guard-enforced) |
| `known_at_violations` (W300) | 0 | 0 ✓ |
| Alternative-window outcome reads | 0 | 0 ✓ |
| Route promotion | none | none ✓ |
| Runtime/risk/execution delta | 0 | 0 ✓ |
| `integrity_check` | ok | ok ✓ |
| `foreign_key_check` | `[]` | `[]` (0) ✓ |
| `ami_events` / `ami_signal_lifecycle` / `ami_cycles` / `ami_birth_truncated_cascade_geometry` | 252 / 324 / 167 / 220 | unchanged ✓ |
| `ami_agg_trades_repaired` / `ami_cvd_windowed_flow` / `_proxy` | 40,934 / 1,840 / 1,840 | unchanged ✓ |

### Hash reconciliation (execution proof → this closure batch)

| File | Execution proof "after" | This closure batch (current) | Match |
|---|---|---|---|
| `canonical.sqlite` sha256 | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` | ✅ byte-identical |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | ✅ byte-identical |

**Both real files are byte-for-byte unchanged since the execution batch closed** — confirming this entire closure batch (test repairs, validation, two full regression passes) never wrote to the real database files, only to disposable copies.

---

## PART 6 — Validation

**Collect-only:** 1,027 tests, 78 files (`pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py --collect-only`) — up from the M-0035 baseline's 987/76 by exactly the 2 new test files added between M-0035 and this closure (20 preregistration tests + 20 execution tests, +40 net after this closure's own additions).

**Repaired preregistration test file:** `tests/test_ami_absorption_impact_preregistration_v1.py` — 20/20 passed.
**Repaired execution test file:** `tests/test_ami_research_cascade_absorption_impact_001.py` — 21/21 passed (20 original + 1 new: `test_verify_pre_execution_detects_already_executed_state`).
**Focused lifecycle/enforcement tests:** both files run together, twice independently, deterministic — 40/40 (then 41/41 after the new test was added) passed both times, and confirmed the real files were never mutated (Part 5).

### Deterministic regression comparison against the accepted M-0035 baseline

**First clean pass** (before the 6th-failure fix): 1,027 collected, **1,013 passed, 14 failed** — but one of those 14 was a **genuinely new** failure not in the M-0035 waived set: `tests/test_ami_absorption_cascade_impact_rehearsal.py::test_real_data_no_experiment_result_nullifier_delta`, hardcoding `experiment_registry==23`/`experiment_results==350`/`epistemic_test_nullifiers==1` — stale because of this family's *own* governed execution (registry 23→24, results 350→381, nullifiers 1→2), not because of any unrelated batch. Fixed (Part 2 of this closure covers only the 5 originally-disclosed tests; this 6th, additionally-discovered one, caused directly by this family's own execution, was fixed under the same "necessary test-only repairs" commit-policy allowance — a one-line snapshot-value bump with a disclosure docstring, identical in kind to every other "protected snapshot" update already established throughout this codebase, e.g. `test_ami_lifecycle_provenance_rehearsal.py`'s `schema_version` tuple).

That same first pass also showed one instance of the already-known, non-deterministic live-`microstructure.db`-collector timing flake (`test_ami_lifecycle_short_noisy_v1_rehearsal.py::test_disposable_db_and_microstructure_db_untouched`) — not counted as a deterministic regression failure, per the operator's own standing instruction to keep live-collector health separate from deterministic acceptance.

**Second clean pass** (after the fix, isolated re-verification of the fixed file, then one more full clean pass): **1,027 collected, 1,013 passed, 14 failed — exactly the 14 tests already named and accepted in the M-0035 regression-baseline waiver (`5ab89f63`)**, batch-for-batch identical in content (`test_ami_cvd_primary_long_preregistration_v1.py` ×3, `test_ami_epistemic_nullifier_enforcement_wiring.py`+`test_ami_epistemic_nullifier_legacy_bypass_closure.py` ×8, `test_ami_research_cvd_windowed_flow_001.py` ×3). **Zero new deterministic failures. The live-collector flake did not even reappear on this pass.**

**Result: `NEW_DETERMINISTIC_FAILURES_INTRODUCED = 0`**, matching or improving on (this run: 0 flake instances vs. 1 in the prior pass, purely a timing artifact either way) the accepted baseline.

---

## Remaining limitations

Unchanged from the execution report: the closed CVD experiment's TEST cycles are not independent of this experiment's own (disclosed both in the execution report and the `researcher_exposure_ledger`); W60/W600/W1800/W3600 remain entirely untested against any outcome; SHORT-direction absorption/impact remains entirely untested; the non-promotable `mfe_bps` diagnostic (p=0.059) does not alter and must not be used to justify any future work. Additionally: the 14 pre-existing G2-execution-caused test failures (already documented in the M-0035 waiver) remain unresolved and outside every batch's scope to date — a dedicated remediation batch updating their hardcoded G2-era checkpoints remains a separate, not-yet-scheduled task.

---

## Storage guardrail

| Item | Value |
|---|---|
| Temporary files created | `closure_test_file_list.txt` (~2KB), `run_closure_regression.sh` (~1KB), `closure_regression.log` (~30KB, first pass), `closure_regression_final.log` (~28KB, second/final pass) — all under the OS scratchpad, none under the repo |
| Peak temporary disk usage | ~65KB (log files) + transient pytest `--basetemp` fixture directories (each deleted immediately after its own paired batch, per the regression script's own `rm -rf "$BT"` step) |
| Full database copies created | 0 beyond pytest's own conftest session-scoped disposable copies (cleaned up automatically at session end) |
| `data/microstructure.db` copied | never |
| Files retained | none of the scratch files above — all information is captured in this closure artifact |
| Files deleted | `closure_test_file_list.txt`, `run_closure_regression.sh`, `closure_regression.log`, `closure_regression_final.log`, and all intermediate `pytest_*` basetemp directories |
| Remaining under `.runtime_temp` | unchanged from the execution-batch checkpoint (`absorption_impact_rehearsal_v1/` + the 4 M-0035 evidence JSONs) |
| Remaining under `.pytest_temp` | none |

---

## Success verdicts

**`CASCADE_ABSORPTION_IMPACT_GOVERNED_EXECUTION_V1_COMPLETE`**

**`ABSORPTION_IMPACT_EXECUTION_TEST_STATE_CLOSURE_COMPLETE`**

**`NO_RELIABLE_INCREMENTAL_ASSOCIATION`**

All 5 originally-named stale tests were corrected without weakening enforcement (two assertions became strictly more specific; none were skipped, xfailed, or deleted). One additional, genuinely new failure — caused directly by this family's own governed execution, not by any unrelated batch — was found during validation and fixed under the same test-only-repair scope. All accounting closes: 31/31 result rows reconciled and classified, the 36.7bps effect magnitude fully reconciled to its frozen, outcome-blind formula, and two independent full regression passes confirm exactly the accepted 14-failure baseline with zero new deterministic failures.
