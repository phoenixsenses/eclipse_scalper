# CVD_PRIMARY_LONG_EXECUTION_V1_STATE_TRANSITION_PROOF

**Batch:** G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1
**Purpose:** Execute the frozen preregistration `E-CVD-PRIMARY-LONG-W300-PREREG-001` exactly once — the first real (non-disposable) TEST-evidence consumption under the M-0033/M-0034 epistemic gate mechanism.
**Prior checkpoint:** `CVD_PRIMARY_LONG_PREREGISTRATION_V1_COMPLETE`, commit `749520b3`.
**Nature:** Single governed execution. One TEST outcome access, one experiment_registry row, one experiment_results set, one nullifier consumption. No production code touched outside this batch's own new module.
**Author:** Sonnet 5 · **Date:** 2026-07-06

---

## Sequence executed (matches the operator's required 6-step authorization order + 11-step gate order)

1. **Pre-execution verification** (`ami/research/cvd_windowed_flow_001.py::verify_pre_execution`) — reproduced family_id, split_version, TRAIN/TEST cycle-set hashes, nullifier, gate-receipt identity, W300 quality population, known-at contract, bucket-exclusion population, and canonical invariants, all against the **real** `data/ami/canonical.sqlite` and `data/ami/knowledge.sqlite`, all **before** any TEST outcome was read. Zero errors.
2. **Family identity, split identity, TEST cycle set, nullifier**: all frozen constants copied verbatim from the committed preregistration artifacts (not re-derived from a `frozen_splits` free-text description never recorded verbatim — see the execution report §1 for the disclosed reasoning) — reproduced, matched.
3. **Nullifier confirmed unused**: `SELECT ... FROM epistemic_test_nullifiers WHERE nullifier=?` → 0 rows (real DB, verified immediately before authorization).
4. **Nullifier atomically consumed**: `epistemic_gates.consume_test_evidence()` → `CONSUMED`, bound to `experiment_id=E-CVD-PRIMARY-LONG-W300-PREREG-001`, `family_id=FAMv1:bec99d8d36f7d6a1`, `split_version=SPLITv1:0a1b96fd74dd281e`. This call happened **before** any TEST-row `endpoint_return_bps`/`mfe_bps` value was read — the point of no return.
5. **TEST outcome access**: `_fetch_effective_outcome_for_signals()`, scoped by the exact 40 TEST signal_ids, executed for the first and only time immediately after step 4.
6. **Primary model run once**: OLS + cluster-robust (CR1) SE on the 40 TEST rows. One secondary check (`mfe_bps` outcome) computed in the same pass, per the preregistration's own permitted-secondary-checks list — not a reaction to the primary result.
7. **Verdict rule applied exactly as frozen**: `NO_RELIABLE_ASSOCIATION` (CI includes 0, p=0.727 ≥ 0.05).
8. **Result recording**: `record_experiment_registry`/`record_experiment_results` (the mandatory immutable writers) → `INSERTED` / `INSERTED`, bound to the **same** `experiment_id` used at preregistration (no new experiment_id, no `supersedes_experiment_id`).
9. **Gate receipt reissued**: `issue_gate_receipt(..., registry_result="EXECUTED")` — same experiment/family/split/nullifier identity, `registry_result` updated from `PREREGISTERED_NOT_EXECUTED` to `EXECUTED` (the receipt's own hash is an identity hash over experiment/family/split/nullifier, unaffected by `registry_result`, and is therefore unchanged: `d46f7e2c…`, same as at preregistration).

## Why the lower-level gate functions were used directly (not `register_experiment_with_gates`)

`ami.warehouse.experiment_ledger.register_experiment_with_gates()` derives `split_version` internally via `resolve_split_version(frozen_splits)` from a caller-supplied free-text description. The exact `frozen_splits` string hashed into the already-issued `SPLITv1:0a1b96fd74dd281e` token was never recorded verbatim in any committed artifact (only the resulting token was, per the preregistration's own committed test suite, which itself reuses the token directly rather than re-deriving it — see `test_nullifier_reproducible_from_frozen_cycle_sets`). Authoring a new description and feeding it through the orchestrator risks silently producing a **different** split_version/nullifier pair than the one already frozen and receipted. This execution therefore calls `epistemic_gates.consume_test_evidence()`, `record_experiment_registry()`, `record_experiment_results()`, and `epistemic_gates.issue_gate_receipt()` directly — the same lower-level functions the preregistration batch itself used, and for an analogous, disclosed reason (see the preregistration's own transition proof §"Why not the full `register_experiment_with_gates()`...").

## Dress rehearsal before the real execution

Before touching the real database, the full flow was exercised **twice** against disposable, conftest-session-isolated copies of both `canonical.sqlite` and `knowledge.sqlite` (`tests/test_ami_research_cvd_windowed_flow_001.py::test_governed_execution_dress_rehearsal_on_disposable_copies`): first call → `CONSUMED`/`INSERTED`/`INSERTED`; second call (idempotent rerun) → `NOOP_IDENTICAL`/`NOOP_IDENTICAL`/`NOOP_IDENTICAL`, zero duplicate rows. A separate test (`test_execute_governed_run_blocks_on_identity_mismatch`) proved a deliberately corrupted family_id raises `ProtocolInvalidation` and consumes nothing. **15/15 tests passed** before the real execution ran, including a standalone script dry-run against fresh disposable copies whose TRAIN predictor-distribution statistics came out **byte-identical** to the frozen preregistration's own recorded TRAIN distribution (min/max/mean/median), independently confirming the population-reconstruction logic before it ever touched the real files.

## A real, TRAIN-discovered condition handled before TEST access

TRAIN diagnostics (computed before nullifier consumption, therefore not TEST-driven) revealed that `session_EUROPE` has **zero variance in both TRAIN (0/91) and TEST (0/40)** — no LONG signal in this population falls in the 07:00–13:00 UTC EUROPE window. This was not anticipated by the preregistration. It was handled as a numerical-estimator concern (Moore-Penrose pseudo-inverse in place of a strict matrix inverse), not a model-specification change: the frozen formula's six terms are all still present in the design matrix; the degenerate column's own coefficient/SE come out as ≈0/0.0 (there is nothing to estimate), and the primary predictor's coefficient/SE/CI/p-value are unaffected (pinv reduces to the ordinary inverse wherever the design is otherwise full rank). This was discovered and fixed in the module's own test suite, entirely before any TEST outcome was read for this experiment — consistent with the "already tested frozen code" requirement (code may be built and debugged pre-TEST-access; it may not be patched after).

## Real database state, before → after

| Check | Before | After |
|---|---|---|
| `data/ami/canonical.sqlite` sha256 | `fdda663dcc331053f6351d6acb7117eeb266fda5cf5d5691a799e48416be724c` | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| `canonical_warehouse` schema_version | 12 | 12 (unchanged) |
| `experiment_registry` | 22 | 23 (+1, this experiment only) |
| `experiment_results` | 323 | 350 (+27, this experiment's metrics only) |
| Protected counts (events/signal_lifecycle/cycles/geometry) | 252/324/167/220 | 252/324/167/220 (unchanged) |
| CVD frozen counts (repaired/exact/proxy/exclusions/quality) | 40934/1840/1840/104/1840 | unchanged |
| `researcher_exposure_ledger` | 1173 | 1176 (+3 — this batch's own 3 gateway calls: one `fetch_lifecycle_signals` in the standalone pre-check, one in `execute_governed_run`'s internal re-verification, one `fetch_events`; the accepted by-design exception, same as every prior batch) |
| `integrity_check` | — | ok |
| `foreign_key_check` | — | clean (0 rows) |
| `data/ami/knowledge.sqlite` sha256 | `ef7f8cde5e790ef765498861e2bdbc561f8ac64e3044eeb84afe3786125a8b6c` | `2a5abc280889eac91a5ec5e9c82f63d024670b6735f8c4a77b10597c9029b93e` |
| `epistemic_test_nullifiers` (this nullifier) | 0 rows | 1 row, `consumed_by_experiment_id=E-CVD-PRIMARY-LONG-W300-PREREG-001` |
| `experiment_gate_receipts.registry_result` (this experiment) | `PREREGISTERED_NOT_EXECUTED` | `EXECUTED` (receipt hash unchanged: `d46f7e2c…`) |
| `graveyard_slash_fingerprints` | 31 | 31 (unchanged) |

Backups taken **before** this batch touched the real files (both verified against the real file's own hash immediately prior): `data/ami/backups/canonical_pre_G2_governed_execution_20260706.sqlite`, `data/ami/backups/knowledge_pre_G2_governed_execution_20260706.sqlite`.

## Required invariants — verified

- **Failed authorization consumes nothing**: proved by `test_execute_governed_run_blocks_on_identity_mismatch` (0 nullifier rows after a blocked attempt).
- **Crash/stop before TEST access consumes nothing**: `verify_pre_execution` performs zero writes; the nullifier is only ever touched by `consume_test_evidence`, called once, after all pre-checks pass.
- **Successful TEST access cannot leave the nullifier unconsumed**: the real run's `consume_result="CONSUMED"` was confirmed by an immediate post-hoc read (`epistemic_test_nullifiers` — 1 row, this nullifier, this experiment_id) before any report was written.
- **A second experiment cannot use the same family/split/TEST set**: enforced structurally by `consume_test_evidence`'s own partial-unique-index/`TestEvidenceReuseBlocked` mechanism (unchanged, not touched by this batch).
- **No new experiment identity was substituted**: `experiment_id` is the same `E-CVD-PRIMARY-LONG-W300-PREREG-001` used at preregistration throughout.
- **No supersession or retry token was created**: neither `issue_retry_authorization` nor `issue_supersession_authorization` was called anywhere in this batch (graveyard was clean; nullifier had zero prior consumers).
- **Identical rerun after result finalization creates no second result**: proved twice — in the dress rehearsal (disposable copies) and structurally (both `record_experiment_registry`/`record_experiment_results` are content-addressed, `NOOP_IDENTICAL` on a byte-identical rerun, `ImmutableExperimentConflict` on any divergent one).

## Exact changed/added-file manifest (this commit)

| File | Status | Content |
|---|---|---|
| `ami/research/cvd_windowed_flow_001.py` | New | population/split/identity resolution, gated TEST access, OLS + cluster-robust regression, verdict rule, immutable-writer integration |
| `tests/test_ami_research_cvd_windowed_flow_001.py` | New | 15 tests: pure statistics (synthetic), real-data population/identity reproduction, disposable-copy dress rehearsal + idempotency + blocked-path proof |
| `reports/research/s34/S34_CVD_PRIMARY_LONG_EXECUTION_V1.md` | New | the execution report (this batch's scientific artifact) |
| `reports/research/s34/S34_CVD_PRIMARY_LONG_EXECUTION_V1.json` | New | machine-readable result manifest |
| `reports/governance/CVD_PRIMARY_LONG_EXECUTION_V1_STATE_TRANSITION_PROOF.md` | New | this document |

No shared governance Markdown file (`SYSTEM_STATE.md`/`IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md`/`MIGRATION_LOG.md`) is included in this commit, matching the preregistration commit's own precedent. No route/runtime/risk/execution/paper/shadow/forward/live file was touched.

## New state root

| Field | Value |
|---|---|
| `canonical.sqlite` hash | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| `canonical.sqlite` schema_version | 12 (unchanged) |
| `experiment_registry` / `experiment_results` | 23 / 350 |
| `epistemic_test_nullifiers` (real) | 1 row (this experiment, consumed) |
| `experiment_gate_receipts` (this experiment) | `EXECUTED` |
| Scientific disposition | `NO_RELIABLE_ASSOCIATION` |

## Verdict

**`CVD_PRIMARY_LONG_GOVERNED_EXECUTION_V1_COMPLETE`**

The single authorized TEST-evidence access was performed exactly once, bound to the pre-existing frozen experiment_id, with the nullifier moved from issued/unused to consumed exactly once and exactly one new experiment_results set recorded. Scientific disposition: `NO_RELIABLE_ASSOCIATION`. No follow-up hypothesis or research wave is opened from this result — this batch stops here.
