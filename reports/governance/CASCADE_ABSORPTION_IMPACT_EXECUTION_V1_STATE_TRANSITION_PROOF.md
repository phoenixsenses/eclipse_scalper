# CASCADE_ABSORPTION_IMPACT_EXECUTION_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-CASCADE-ABSORPTION-IMPACT-GOVERNED-EXECUTION-V1
**Purpose:** Execute the frozen preregistration `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` exactly once, consuming its TEST-evidence nullifier and recording one scientific disposition.
**Prior checkpoint (unchanged in identity, not reopened):** commit `fb002a75` (`ABSORPTION_IMPACT_PREREGISTERED_NOT_EXECUTED`), `schema_version=13`, `experiment_registry=23`, `experiment_results=350`.
**Nature:** Governed TEST execution. Exactly one model fit, one nullifier consumption, one experiment_registry/results write, one gate-receipt state transition. No amendment, optimization, reinterpretation, or rescue of the experiment.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Code authored and dress-rehearsed, before any TEST access:** `ami/research/cascade_absorption_impact_001.py`, mirroring `ami/research/cvd_windowed_flow_001.py`'s exact lifecycle behavior (per the operator's explicit consistency requirement), with one deliberate, preregistration-mandated deviation: `EUROPE` dropped structurally from the design matrix (no pseudo-inverse), plus a mandatory pre-fit `check_design_rank()` validation absent from the CVD precedent.
2. **Focused tests, before TEST access:** `tests/test_ami_research_cascade_absorption_impact_001.py` (19 tests: pure statistics on synthetic data, TEST-outcome-blind population/split reproduction, and a full disposable-copy dress rehearsal of `execute_governed_run` proving `CONSUMED`→`NOOP_IDENTICAL` idempotency, `INSERTED`→`NOOP_IDENTICAL` registry/results idempotency, and gate-receipt reissue) — **19/19 passed**. One test-fixture bug (missing `signal_id` key in a synthetic row) was found and fixed at this stage — before any TEST access, permitted under the code-freeze discipline.
3. **Pre-execution identity check against the real live databases:** `verify_pre_execution(canonical_conn, knowledge_conn)` — **0 errors**. Reproduced `family_id`, `nullifier`, TRAIN/TEST cycle-set hashes, `schema_version=13`, W300 coverage (324/324 `EXACT_RECONSTRUCTABLE`), 0 known-at violations, gate receipt state `PREREGISTERED_NOT_EXECUTED` with matching identity fields, `is_rerun_of_self=False`. Zero identity drift, zero cycle-set drift.
4. **Pre-execution checkpoint recorded:** `canonical.sqlite` sha256 `815f35d0…`, `knowledge.sqlite` sha256 `d435c3a2…`, `experiment_registry=23`, `experiment_results=350`, `epistemic_test_nullifiers=1`, `experiment_gate_receipts=2`, `researcher_exposure_ledger=1176→1177` (the identity check's own `fetch_lifecycle_signals` call, expected by-design exposure).
5. **Real governed execution:** `execute_governed_run(canonical_conn, knowledge_conn)` against the live databases, at **2026-07-07T11:13:29Z UTC**. Internally: re-ran the same pre-execution verification (0 errors) → computed TRAIN diagnostics (scaling stats, VIF, TRAIN-only descriptive fit, TRAIN design-rank check: 6/6 full rank, TRAIN predictor stdev re-verified against the frozen `10.701083978672228` value from the preregistration — matched exactly) → **consumed the TEST-evidence nullifier** (`CONSUMED`, the point of no return) → read TEST rows (first and only TEST-outcome access for this experiment_id) → fit the frozen primary model (TEST design-rank check: 6/6 full rank) → applied the frozen verdict rule → wrote `experiment_registry` (`INSERTED`) and `experiment_results` (`INSERTED`, 31 rows) to `canonical.sqlite`, committed → reissued the gate receipt with `registry_result='EXECUTED'` to `knowledge.sqlite`, committed.
6. **Post-execution checkpoint recorded:** `canonical.sqlite` sha256 `b1aff8a5…` (pre-disclosure-row) then `3aefce83…` (post-disclosure-row, see step 7), `experiment_registry=24`, `experiment_results=381`, `epistemic_test_nullifiers=2`, gate receipt state `EXECUTED`, `integrity_check=ok`, `foreign_key_check=[]`.
7. **Cross-family holdout exposure disclosure:** one explicit `researcher_exposure_ledger` row inserted (`exposure_id=EXP-56a8d3e1eb5a4c6abbf2f22b`, category `CROSS_FAMILY_TEST_CYCLE_REUSE_DISCLOSURE`), recording verbatim the five required disclosures (same TEST cycle set as CVD, different family/nullifier, not an independent replication, no CVD outcome used to alter this spec, no new multiplicity correction introduced). `researcher_exposure_ledger` final count: 1,180.
8. **No second TEST execution:** `execute_governed_run` was **not** called a second time against the real database. Idempotency (`CONSUMED`→`NOOP_IDENTICAL`, `INSERTED`→`NOOP_IDENTICAL`, stable verdict) was already proven in step 2 against a disposable copy — re-invoking against the real, now-executed database was judged unnecessary and avoided, per the operator's "no second TEST execution occurs" invariant.

---

## Identity/hash record

| Field | Value |
|---|---|
| `canonical_family_id` | `FAMv1:3e2dfe63f9e271bf` |
| `experiment_id` | `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` |
| `split_version` | `SPLITv1:16ea98c239034593` |
| TEST nullifier | `4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4` |
| Gate receipt hash | `6dbe0f59416977fce75b20a13876ff4d54dddae171d1fa8b07613135550e06e4` (unchanged — the receipt's identity hash is a function of experiment_id/family/split/nullifier, all unchanged; only `registry_result` transitioned) |
| `canonical.sqlite` sha256, pre-execution | `815f35d0619e293d64b7a2d34057a0679d52818116e97fd86fd85408e84e9252` |
| `canonical.sqlite` sha256, post-execution (final, after disclosure row) | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` |
| `knowledge.sqlite` sha256, pre-execution | `d435c3a294a286a18a7900d42824f3d4ad020ddedbadf878fcca2a18865c03a9` |
| `knowledge.sqlite` sha256, post-execution | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` |

---

## Known-at and access proof

- `known_at_violations = 0` (re-verified at the pre-execution identity check: 0/324 W300 rows with `window_end_ts_ms > signal_birth_ts` or non-`KNOWN_AT_SAFE` classification).
- `resolve_population()`'s own source is statically guarded (`test_verify_pre_execution_never_selects_outcome_columns`) to never name `endpoint_return_bps`/`mfe_bps` — confirmed by direct source inspection, not merely by convention.
- The **only** function in this module (or the codebase, for this experiment) that reads `endpoint_return_bps`/`mfe_bps` is `_fetch_effective_outcome_for_signals`, and it was called exactly twice: once for the 91 TRAIN signal_ids (permitted, pre-authorization) and once for the 40 TEST signal_ids (only after `consume_test_evidence` returned `CONSUMED`).
- TRAIN outcome access is **not** a TEST-evidence violation: TRAIN rows are outside the frozen TEST cycle set by construction (chronologically earlier, disjoint cycle-key sets, verified 0 overlap at both preregistration and pre-execution time) — reading TRAIN's own outcome value is a routine, expected step of any TRAIN-side diagnostic/descriptive fit, never gated by the nullifier (which protects only the specific frozen TEST cycle set).

---

## Exact/proxy separation

No proxy table exists for `FAM_CASCADE_ABSORPTION_IMPACT` (confirmed by `verify_pre_execution`'s own structural check: `SELECT name FROM sqlite_master WHERE name LIKE '%absorption_impact%proxy%'` → empty). 0 proxy rows read or written. Pooling absent by construction, not merely by discipline.

---

## Design/rank proof (frozen policy, no pseudo-inverse)

| Check | TRAIN | TEST |
|---|---|---|
| n | 91 | 40 |
| k (columns) | 6 | 6 |
| rank | 6 | 6 |
| full rank | true | true |
| `EUROPE` observations | 0 | 0 |
| `EUROPE` column present | no | no |

Both design matrices were confirmed full-rank via `np.linalg.matrix_rank` **before** any coefficient was computed — `run_cluster_robust_ols` uses a genuine `np.linalg.inv`, never `np.linalg.pinv`, consistent with the preregistration's explicit `pseudo_inverse_permitted: false` freeze.

---

## Result summary (full detail in the execution report/JSON)

| Field | Value |
|---|---|
| Coefficient | −3.4285074465436134 |
| Cluster-robust SE | 2.3954324586247613 |
| 95% CI | [−8.27372693, 1.41671204] |
| p-value | 0.16031838015391875 |
| TEST n / TRAIN n | 40 / 91 (0 dropped, both) |
| Verdict | **`NO_RELIABLE_INCREMENTAL_ASSOCIATION`** |

---

## Result recording (lifecycle consistency with the accepted CVD governed execution)

| Step | CVD precedent | This execution |
|---|---|---|
| Registry write | `record_experiment_registry` → `INSERTED` | `record_experiment_registry` → `INSERTED` (identical call pattern) |
| Results write | `record_experiment_results` → `INSERTED` | `record_experiment_results` → `INSERTED`, 31 rows |
| Nullifier consumption | `gates.consume_test_evidence(...)` before TEST read | identical call pattern, before TEST read |
| Gate receipt reissue | `gates.issue_gate_receipt(..., registry_result="EXECUTED")` | identical call pattern |
| Atomicity | two separate commits (knowledge.sqlite nullifier commit, then canonical.sqlite registry/results commit, then knowledge.sqlite receipt commit) — not a single cross-DB transaction | **identical** ordering and commit boundaries, deliberately matching the accepted precedent rather than introducing `register_experiment_with_gates`'s stricter atomicity, per the explicit consistency requirement |

Exactly one experiment registration/result set was created. No second experiment or result attempt occurred. All prior experiment/result rows (23 registry rows, 350 result rows) are preserved byte/content-identically — verified by the `_REGISTRY_CONTENT_COLUMNS`/result-set immutability guard inside `record_experiment_registry`/`record_experiment_results` themselves (any diff would have raised `ImmutableExperimentConflict`; none did, since this was a first `INSERT`, not a rewrite).

---

## Full DB/table deltas

| Table | Before | After |
|---|---|---|
| `experiment_registry` | 23 | 24 |
| `experiment_results` (total) | 350 | 381 |
| `experiment_results` (this experiment_id) | 0 | 31 |
| `epistemic_test_nullifiers` | 1 | 2 |
| `experiment_gate_receipts` (rows) | 2 | 2 (state updated, no new row) |
| `researcher_exposure_ledger` | 1,176 | 1,180 |
| `schema_version` | 13 | 13 |
| `ami_absorption_impact_windowed_flow` | 1,619 | 1,619 |
| `ami_absorption_impact_window_quality_v1` | 1,620 | 1,620 |
| `ami_absorption_impact_exclusions` | 1 | 1 |
| `integrity_check` | — | ok |
| `foreign_key_check` | — | [] |

## Protected delta (zero)

| Table | Before | After |
|---|---|---|
| `ami_events` | 252 | 252 |
| `ami_signal_lifecycle` | 324 | 324 |
| `ami_cycles` | 167 | 167 |
| `ami_birth_truncated_cascade_geometry` | 220 | 220 |
| `ami_agg_trades_repaired` | 40,934 | 40,934 |
| `ami_cvd_windowed_flow` / `_proxy` | 1,840 / 1,840 | 1,840 / 1,840 |

No file under `execution/`, `risk/`, `brain/`, `.env` was read or modified. No route or bucket was promoted.

---

## Required validations (proven)

| Check | Result |
|---|---|
| `schema_version` remains 13 | ✅ |
| Absorption feature/quality/exclusion rows unchanged | ✅ (1,619/1,620/1, both before and after) |
| TEST nullifier consumed exactly once | ✅ |
| Gate receipt moved to `EXECUTED` | ✅ |
| Exactly one experiment/result set added | ✅ (registry +1, all 31 new result rows bound to the one new experiment_id) |
| Prior experiment/result history unchanged | ✅ (immutability-guard-enforced, no `ImmutableExperimentConflict` raised, since this was a first insert) |
| `known_at_violations = 0` | ✅ |
| Exact/proxy pooling = 0 | ✅ |
| No alternative-window outcome access | ✅ (only `window_id='W300'` ever queried against `ami_absorption_impact_windowed_flow` in this module) |
| No second TEST fit | ✅ |
| No route promotion | ✅ |
| No runtime/risk/execution delta | ✅ |
| `integrity_check = ok` | ✅ |
| `foreign_key_check = 0` (empty) | ✅ |

---

## Expected post-execution test staleness (disclosed, not a regression, not fixed)

Both new test files written for this and the preregistration batch (`tests/test_ami_absorption_impact_preregistration_v1.py`, `tests/test_ami_research_cascade_absorption_impact_001.py`) contain assertions of the **pre-execution** state (`registry_result == 'PREREGISTERED_NOT_EXECUTED'`, nullifier row count `== 0`). A real, successful, one-time governed execution permanently and correctly transitions that state to `EXECUTED`/nullifier-consumed — these specific assertions are now structurally stale **by design**, exactly the same category of staleness already documented for `tests/test_ami_research_cvd_windowed_flow_001.py` in the M-0035 regression-baseline waiver (`5ab89f63`) after the CVD family's own governed execution.

Affected tests (5, verified this session, not silently discovered): `test_ami_research_cascade_absorption_impact_001.py::test_verify_pre_execution_reports_zero_errors_against_real_db` (asserts `is_rerun_of_self is False`, now `True`), `::test_execute_governed_run_blocks_on_identity_mismatch` and `::test_governed_execution_dress_rehearsal_on_disposable_copies` (assert the real nullifier row count is 0 before their own disposable-copy work, now 1 for the real, unrelated consumption), `test_ami_absorption_impact_preregistration_v1.py::test_gate_receipt_mechanism_round_trips_on_disposable_copy` and `::test_real_nullifier_and_receipt_state` (assert the real receipt state is `PREREGISTERED_NOT_EXECUTED`, now `EXECUTED`).

**Not fixed, per the code-freeze requirement** ("No scientific code may change after first TEST access") — editing these assertions after TEST access would itself be a post-execution code change to the accepted execution package. These tests remain committed exactly as they were written and dress-rehearsed *before* TEST access; their staleness is the expected, permanent record of a one-time state transition, not a defect. Future work on this family (if any) must use a new experiment_id and new test file, never edit these in place.

---

## Storage guardrail

| Item | Value |
|---|---|
| Temporary files created | `.runtime_temp/exec_before_state.json`, `.runtime_temp/exec_result.json`, `.runtime_temp/exec_after_state.json` (~1KB each, checkpoint scratch) |
| Peak temporary disk usage | <5KB |
| Full database copies created | 0 against the real files this batch (the focused-test dress rehearsal used pytest's own conftest session-scoped disposable copies, cleaned up automatically) |
| `data/microstructure.db` copied | never |
| Files retained | none of the scratch JSONs — all values folded into the committed execution report/JSON/this proof |
| Files deleted | the three `.runtime_temp/exec_*.json` scratch files |
| Remaining under `.runtime_temp` | unchanged from the prior checkpoint (`absorption_impact_rehearsal_v1/` + the M-0035 evidence JSONs) |
| Remaining under `.pytest_temp` | none |

---

## Remaining limitations

See the execution report's own "Remaining limitations" section (cross-family TEST-cycle non-independence with the closed CVD result, untested windows/direction, the near-significant but non-promotable `mfe_bps` diagnostic).

---

## Verdict

**Operational:** `CASCADE_ABSORPTION_IMPACT_GOVERNED_EXECUTION_V1_COMPLETE`

**Scientific disposition:** `NO_RELIABLE_INCREMENTAL_ASSOCIATION`

Stopping after recording the result. No follow-up hypothesis, window, subgroup, or bucket is opened by this batch. Any further work on `FAM_CASCADE_ABSORPTION_IMPACT` requires new, separate operator instruction and its own preregistration.
