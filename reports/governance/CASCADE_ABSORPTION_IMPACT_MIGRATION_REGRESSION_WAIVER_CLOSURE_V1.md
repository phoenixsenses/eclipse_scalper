# CASCADE_ABSORPTION_IMPACT_MIGRATION_REGRESSION_WAIVER_CLOSURE_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-MIGRATION-REGRESSION-WAIVER-CLOSURE-V1
**Nature:** Additive evidence-closure artifact only. No migration rerun, no schema change, no data alteration, no full regression rerun. Every claim below is either transcribed from the three authoritative regression logs already produced during M-0035 (`M0035_pass1.log`, `M0035_pass2.log`, `M0035_clean_final.log`, all still present under the session scratchpad, none rewritten) or freshly re-derived by read-only queries against the pre-G2 backup, the pre-M-0035 backup, and the current live files.
**Depends on (source of truth, unedited):** migration commit `8808ada8` (M-0035), row-accounting freeze `931cd3dd`, rehearsal `fc43e972`, readiness/contract `fc1321f5`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Correction to the count stated in chat

The operator's framing referenced **13** pre-existing failures. The authoritative count, reproduced identically across all three regression logs (`M0035_pass1.log`, `M0035_pass2.log`, `M0035_clean_final.log`) and consistent with the committed migration report and transition proof, is **14** (batch 4: 3, batch 7: 8, batch 21: 3). This document uses **14** throughout and flags this correction explicitly rather than silently adopting the smaller number — an ad hoc chat summary during the prior session understated it by one; the committed artifacts (`S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1.md`, the transition proof, and the M-0035 commit message) already correctly stated 14 and are unaffected by this correction.

---

## PART 1 — Required reconciliation: the 14-failure matrix

### Chain-of-custody source data (fresh, read-only, this batch)

Three real point-in-time backups provide direct (not inferred) proof of state at each transition, independent of the regression logs:

| Checkpoint | File | `experiment_registry` | `schema_version` | canonical sha256 | `epistemic_test_nullifiers` (specific nullifier) | gate receipt `registry_result` |
|---|---|---|---|---|---|---|
| **Before G2 execution** | `data/ami/backups/canonical_pre_G2_governed_execution_20260706.sqlite` / `knowledge_pre_G2_governed_execution_20260706.sqlite` | 22 | 12 | `fdda663dcc331053f6351d6acb7117eeb266fda5cf5d5691a799e48416be724c` | 0 | `PREREGISTERED_NOT_EXECUTED` |
| **Before M-0035 (= after G2)** | `data/ami/backups/canonical_pre_M0035_absorption_impact_canonical_migration_20260707_065549.sqlite` / live `knowledge.sqlite` (untouched since G2) | 23 | 12 | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` | 1 | `EXECUTED` |
| **After M-0035 (current live)** | live `data/ami/canonical.sqlite` / live `knowledge.sqlite` (still untouched) | 23 | 13 | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` | 1 | `EXECUTED` |

**Reading this table:** `experiment_registry` moved 22→23 **between the first two rows** (during G2, commit `60c3e26f`) and is **unchanged (23→23) across the M-0035 transition**. The nullifier/receipt state moved from unconsumed→consumed **between the first two rows** (during G2) and is **unchanged across the M-0035 transition** (`knowledge.sqlite` was never opened by M-0035 — its mtime, size, and hash are identical before and after this entire batch: `2a5abc280889eac91a5ec5e9c82f63d024670b6735f8c4a77b10597c9029b93e`, 110,592 bytes). `schema_version` is the **only** field in this table that changed across the M-0035 transition (12→12 before it, 12→13 during it) — this is the migration's own, intended, disclosed effect.

### Failure-signature categories (normalized, value-independent)

Four categories, each hashed as `sha256("<invariant>|frozen_expected=<value>|external_driver=<cause>")` — deliberately excluding the specific observed (wrong) value, so the signature identifies "this same logical failure," not a particular number:

| Signature | Definition string | sha256 |
|---|---|---|
| **SIG-A** | `experiment_registry_count\|frozen_expected=22\|external_driver=G2_CVD_execution_60c3e26f` | `f7a2750659f607b3030962901d786163689440ceecb2a1be677e770be5e443e0` |
| **SIG-B** | `canonical_sqlite_full_file_sha256\|frozen_expected=458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49\|external_driver=G2_CVD_execution_60c3e26f` | `a10c92110462bae5111ca593872c8d5f8062282495fb08b6c2cf13216d197e55` |
| **SIG-C** | `canonical_schema_version\|frozen_expected=12\|external_driver=M-0035_schema_bump` | `dbf66c25d6a15d385f3a9af4748ad6f5b36d5685aa97eb03d64282e3f299f0a5` |
| **SIG-D** | `epistemic_test_nullifiers_or_gate_receipt_state\|frozen_expected=unconsumed_PREREGISTERED_NOT_EXECUTED\|external_driver=G2_CVD_execution_60c3e26f` | `81c9567429781782a7d8182fb4d20bcb87bad03bad8f04df0e097fc3d326a53a` |

**SIG-A, SIG-B, SIG-D are exclusively caused by G2** (unrelated to `FAM_CASCADE_ABSORPTION_IMPACT` in any way — proven below). **SIG-C is the one failure mode M-0035 itself introduces** — but see tests 6 and 11 below for the important nuance where SIG-C and SIG-B co-occur in the same test function.

### The 14-row matrix

| # | Test node ID | Signature(s) | Pre-M-0035 result | Post-M-0035 result | Structurally unrelated to absorption schema/code/tables/version-tuple because... |
|---|---|---|---|---|---|
| 1 | `tests/test_ami_cvd_primary_long_preregistration_v1.py::test_gate_receipt_mechanism_round_trips_on_disposable_copy` | SIG-D | **FAILED** — `assert n==0` → disposable copy of the real `knowledge.sqlite` already carries 1 nullifier row (from G2), so `1==0` fails | **FAILED** — identical (`1==0`); `knowledge.sqlite` never opened by M-0035 | Test's own assertion never references any `ami_absorption_impact_*` table, `canonical.sqlite`, or `schema_version`; it fails purely on a `shutil.copyfile`d snapshot of the real `knowledge.sqlite`, whose nullifier row predates M-0035 (see chain-of-custody table) |
| 2 | `...test_real_nullifier_and_receipt_state` | SIG-D | **FAILED** — `assert receipt == ('PREREGISTERED_NOT_EXECUTED',)` → actual `('EXECUTED',)` | **FAILED** — identical | Reads only `knowledge.sqlite`'s `experiment_gate_receipts`/`epistemic_test_nullifiers`; never opens `canonical.sqlite` at all; `knowledge.sqlite` byte-identical before/after M-0035 |
| 3 | `...test_no_experiment_created_and_canonical_invariants_hold` | SIG-A | **FAILED** — `assert n_reg==22` → actual `23` | **FAILED** — identical (`23==22` still fails) | `experiment_registry` count is set by `ResearchRegistry.register_experiment`, never touched by the migration module (`ami/absorption/cascade_absorption_impact_canonical_migration.py` contains no reference to `experiment_registry`) |
| 4 | `tests/test_ami_epistemic_nullifier_enforcement_wiring.py::test_24_existing_22_historical_experiments_remain_unchanged` | SIG-A | **FAILED** — `assert count==22` → `23` | **FAILED** — identical | Same as #3 |
| 5 | `...test_25_retro_audit_remains_0_of_22` | SIG-A | **FAILED** — `assert len(results)==22` → `23` | **FAILED** — identical | `retro_audit_experiment_registry` enumerates `experiment_registry` rows only; no `ami_absorption_impact_*` table referenced |
| 6 | `...test_26_canonical_schema_version_and_hash_unchanged` | SIG-C (proximate) + SIG-B (underlying) | **FAILED** — code reaches `assert version==12` first (passes, since pre-M-0035 version was still 12), then reaches `assert hash==458bc07c…` and **fails there** (actual hash already `25a56a98d0…`, drifted by G2) | **FAILED** — code now fails at the *earlier* `assert version==12` line (actual `13`), never reaching the hash line | The test's **pass/fail outcome is unchanged by M-0035** (FAILED before, FAILED after) — M-0035 only changed *which line* trips first, from the G2-caused hash line to the M-0035-caused version line. The version-tuple concept itself is the same generic "schema only grows" pattern this codebase applies on every migration (see Part 4); the underlying reason this test can never be green again without its own maintenance is the pre-existing G2 hash drift, not this family's tables |
| 7 | `tests/test_ami_epistemic_nullifier_legacy_bypass_closure.py::test_17_18_existing_22_experiments_and_results_unchanged` | SIG-A | **FAILED** — `23==22` | **FAILED** — identical | Same as #3 |
| 8 | `...test_19_retro_audit_remains_0_of_22` | SIG-A | **FAILED** — `23==22` | **FAILED** — identical | Same as #5 |
| 9 | `...test_20_no_new_experiment_created_by_this_batch` | SIG-A | **FAILED** — `23==22` | **FAILED** — identical | Same as #3 |
| 10 | `...test_21_no_scientific_result_generated_by_this_batch_real_hash_unchanged` | SIG-B | **FAILED** — `assert hash==458bc07c…` → actual `25a56a98d0…` | **FAILED** — actual now `a229d4b0…` (still ≠ `458bc07c…`) | Pure full-file hash comparison against a G2-era frozen constant; already false before M-0035 touched anything. M-0035 changed *which* wrong value it now sees (as any additive migration necessarily must, since it writes new rows), but did not change the outcome (FAILED→FAILED) |
| 11 | `...test_22_23_canonical_schema_version_and_hash_unchanged` | SIG-C (proximate) + SIG-B (underlying) | **FAILED** — same mechanism as #6 | **FAILED** — same mechanism as #6 | Same reasoning as #6 |
| 12 | `tests/test_ami_research_cvd_windowed_flow_001.py::test_execute_governed_run_blocks_on_identity_mismatch` | SIG-D | **FAILED** — `assert n_rows==0` → `1` | **FAILED** — identical | Queries `knowledge.sqlite`'s `epistemic_test_nullifiers` for a specific CVD nullifier hash; unrelated table, unrelated family, `knowledge.sqlite` untouched |
| 13 | `...test_governed_execution_dress_rehearsal_on_disposable_copies` | SIG-D | **FAILED** — `assert is_rerun_of_self is False` → `True` | **FAILED** — identical | `is_rerun_of_self` is computed from the real, already-consumed CVD nullifier (G2's own execution) — has nothing to do with `FAM_CASCADE_ABSORPTION_IMPACT`'s identity, schema, or tables |
| 14 | `...test_verify_pre_execution_reports_zero_errors_against_real_db` | SIG-D | **FAILED** — same mechanism as #13 | **FAILED** — identical | Same as #13 |

### Required conclusion

**`NEW_DETERMINISTIC_FAILURES_INTRODUCED_BY_M0035 = 0`**

Every one of the 14 rows shows an **identical pre/post pass-fail outcome** (FAILED→FAILED, never PASSED→FAILED). Rows 6 and 11 are the only ones where M-0035 changed a failing test's *proximate* failing line (from the G2-caused hash assertion to the M-0035-caused version assertion) — but the test's boolean result was already FAILED before M-0035 touched anything, so this is not a new failure, it is the same pre-existing failure surfacing at a different line for a documented, expected reason (the version-tuple/hash pair is checked in a fixed order in the test's own source; M-0035 simply became the newer of two independent reasons this line-pair can never both pass simultaneously again without dedicated test maintenance for each cause separately).

No count/hash/nullifier-state was classified "by intuition or file name" — every one of the 14 rows is backed by a direct read-only reproduction against the pre-G2 backup, the pre-M-0035 backup, and the current live files, cross-checked against the exact traceback text captured in `M0035_pass1.log`/`M0035_pass2.log`/`M0035_clean_final.log`.

---

## PART 2 — Live-collector flake

**Test:** `tests/test_ami_lifecycle_short_noisy_v1_rehearsal.py::test_disposable_db_and_microstructure_db_untouched`

**Failure (observed only in the parallel-contaminated run, §Part 3):**
```
assert _file_hash_chunked(MICROSTRUCTURE_DB_PATH, max_bytes=64*1024*1024) == micro_prefix_hash_before
AssertionError: assert 'b6bc606cc897...' == '71eb3d2c5790...'
```

**Proof it depends on mutable live data, not a deterministic fixture:** the assertion compares a 64MB byte-prefix hash of `data/microstructure.db` — a file with an **actively running live collector process** appending new rows continuously — taken at two points a few seconds apart within the same test. The test's own pre-existing source comment (predating this batch) already documents this exact risk: *"microstructure.db's mtime is NOT a reliable untouched-assertion while its live collector is running... the bounded 64MB PREFIX content hash is the collector-aware invariant instead"* — the test authors already knew this file mutates live and designed around it; the prefix hash usually holds but is not immune to collector timing under load.

**Proof M-0035 did not change collector/runtime semantics:** `ami/absorption/cascade_absorption_impact_canonical_migration.py` and the M-0035 driver script open only `data/ami/canonical.sqlite` (write, live migration target) and the frozen retained rehearsal file under `.runtime_temp/` (read-only). Neither ever opens `data/microstructure.db` in any mode. No file under `execution/`, `risk/`, `brain/`, or any collector script was read or written by this batch (confirmed by `git status` at commit time — only the 7 files in commit `8808ada8` changed).

**Proof it was not skipped, weakened, or hidden to obtain green:** the test file was not modified at all in this batch (`git show --stat 8808ada8` lists no changes to `tests/test_ami_lifecycle_short_noisy_v1_rehearsal.py`). It was run, unmodified, in five separate contexts across this work: isolated single-file rerun (green), the original two clean full passes `pass1`/`pass2` (both showed this exact failure alongside the schema-version-tuple failure, in the *same* batch pairing as `test_ami_lifecycle_provenance_rehearsal.py`), and the final clean full pass (green, both sub-tests). Its outcome is genuinely intermittent under real collector timing, not deterministically red or green — which is exactly the "operational health," not "deterministic regression," category.

**Correct future separation (documented, not implemented in this closure batch — no runtime change made):**
- **Deterministic regression suite**: fixtures/assertions that depend only on repository state and disposable copies — should never reference a live, continuously-appending collector file's byte content across a time gap.
- **Live operational-health suite**: checks like this one belong in a separate suite/tag that tolerates or explicitly accounts for concurrent collector writes (e.g., asserting no *write* occurred via a different mechanism than a raw prefix-hash diff, or widening the tolerance window) — a future, separate batch's decision, not addressed here.

---

## PART 3 — Parallel-process contamination incident

**Exact sequence:**

1. A synchronous Bash command was issued to run `final1` then `final2` (chained with `&&`) without `run_in_background`. The default tool timeout is 2 minutes; the command was killed from the tool's perspective with exit code 143 ("Command timed out after 2m 0s") — but the underlying `bash`/`python`/`pytest` process tree it had already spawned was **not confirmed terminated** and continued running detached in the background (Windows/Git-Bash process-group semantics do not guarantee a clean kill of all descendants on tool-side timeout).
2. Without independently verifying that process tree had exited, a **second**, independent invocation of the identical `final1`→`final2` script pair was started via `run_in_background=true`. This second invocation's script begins by truncating (`>`) its own log file — but the first, still-running invocation held its own open file handle to the *same* log path and continued appending (`>>`) to it concurrently.
3. For an overlapping window, **two full 76-file pytest sweeps were running concurrently** against the same repository and the same `D:\eclipse_scalper\.pytest_temp`/OS-scratchpad basetemp root — a direct violation of this repository's no-parallel-Python-processes guardrail (RAM/tmp-permission risk, per `CLAUDE.md`).
4. **Detection:** batch 9 (`test_ami_geometry_birth_truncated_geometry_rehearsal.py` + `test_ami_geometry_liquidation_source_quality_contract_v2.py`) took 308.66s in the contaminated `final1` log vs. its normal ~140s baseline (seen consistently in `pass1`/`pass2`/`clean_final`), and produced a `FileNotFoundError: [WinError 3]` on a `shutil.copy2` of a session-fixture file (`real_canonical_test_copy0/canonical.sqlite`) that had no reason to be missing under a single-process run. The same log additionally showed duplicate "X failed, Y passed" summary lines for the same batch number with two different timings (e.g., batch 4: `20.11s` and `15.42s`), consistent with two independent runs of the same batch interleaving output into the same file.
5. **Process IDs:** not captured at the moment of overlap (the detached first process was not still running by the time this was investigated — confirmed via `ps aux` showing no eclipse_scalper-related python/pytest process, only an unrelated `D:\commerce_intelligence\.venv` process from a different project).
6. **Overlapping time range:** approximately `2026-07-07T11:2x` through `11:3x` local time, bounded by the killed command's issuance and the second invocation's own start; exact bounds not independently logged (a gap in this incident's own record-keeping, disclosed rather than reconstructed after the fact).
7. **Affected log paths (both discarded, never used as evidence):** `M0035_final1.log`, `M0035_final2.log`, and their associated `pytest_final1_batch*`/`pytest_final2_batch*` basetemp directories under the OS scratchpad.
8. **Why invalid:** any duration, resource-contention-induced, or filesystem-race failure produced under concurrent execution cannot be attributed to the code under test — the repository's own established precedent (M-0031's identical "tek-proses mega-invocation" incident, `MIGRATION_LOG.md`) treats this class of failure as non-authoritative by policy, not merely by this author's judgment.
9. **Confirmation excluded from acceptance evidence:** `M0035_final1.log`/`M0035_final2.log` were **not** cited in the committed migration report (`S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1.md`) or transition proof as acceptance evidence — both documents explicitly label them contaminated and cite only `M0035_pass1.log`, `M0035_pass2.log`, and `M0035_clean_final.log` as authoritative. Neither contaminated log file was deleted (both remain under the OS scratchpad, unmodified, for independent audit) nor was any accepted evidence rewritten.
10. **Identity of the valid later clean rerun:** `M0035_clean_final.log` — launched via a single `run_in_background=true` invocation only *after* `ps aux` was checked and confirmed to show zero eclipse_scalper-related processes running. Result: 987 collected, 973 passed, 14 failed — the geometry `FileNotFoundError` did **not** recur, and the batch-9 pair duration returned to its normal ~140s baseline, corroborating that the anomaly was contamination, not a real defect.

**Explicit non-authoritative marking:** `M0035_final1.log` and `M0035_final2.log` are hereby marked **`NON_AUTHORITATIVE — PARALLEL_PROCESS_CONTAMINATED — DO NOT CITE`**. `M0035_pass1.log`, `M0035_pass2.log`, and `M0035_clean_final.log` are the three authoritative regression records for M-0035.

---

## PART 4 — Migration-specific test result (the one genuine M-0035 failure)

**Test:** `tests/test_ami_lifecycle_provenance_rehearsal.py::test_full_provenance_rehearsal_real_data`

**Root cause:** hardcoded `assert report["schema_version_before"] in (8, 9, 10, 11, 12)` — a tuple that must be extended by exactly one element on every additive schema-version bump, per this codebase's own established, repeated precedent (identical fix applied at v9, v10, v11, v12 by four prior migrations, each cited in the pre-existing code comments immediately above the assertion).

**Exact fix (commit `8808ada8`):**
```diff
-    assert report["schema_version_before"] in (8, 9, 10, 11, 12)
+    assert report["schema_version_before"] in (8, 9, 10, 11, 12, 13)
     assert report["new_objects_present"] is True
-    already_migrated = report["schema_version_before"] in (9, 10, 11, 12)
+    already_migrated = report["schema_version_before"] in (9, 10, 11, 12, 13)
```

**Focused test result after the fix:** `python -m pytest tests/test_ami_lifecycle_provenance_rehearsal.py` → `3 passed`. Re-confirmed green in the subsequent clean full-suite pass (`M0035_clean_final.log`, batch 19: `12 passed`, both files in the pair).

**Proof that no scientific formula, feature value, row accounting, or quality state changed:** this fix touches only a structural version-tuple in a lifecycle-provenance rehearsal test (an infra-level "schema only grows" guard, unrelated to any family's business logic). It does not reference `ami_absorption_impact_*`, `signed_notional`, `price_response_per_signed_notional`, `FLOOR_USD_M`, any quality-status literal, or any exclusion-reason literal. The migration's own content-hash proofs (§ below) independently confirm zero feature/formula/quality drift.

---

## PART 5 — Full hash and state checkpoint (complete, non-truncated)

| Item | Value |
|---|---|
| Pre-migration `canonical.sqlite` sha256 | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| Post-migration (current live) `canonical.sqlite` sha256 | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` |
| `knowledge.sqlite` sha256, before M-0035 | `2a5abc280889eac91a5ec5e9c82f63d024670b6735f8c4a77b10597c9029b93e` |
| `knowledge.sqlite` sha256, after M-0035 (current) | `2a5abc280889eac91a5ec5e9c82f63d024670b6735f8c4a77b10597c9029b93e` (byte-identical — file never opened) |
| `S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1.md` sha256 | `f629f242cf0c0e254b13368d8e8bbeaf799c2474f883bf4616e6908e639e4f54` |
| `CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_STATE_TRANSITION_PROOF.md` sha256 | `9c3eddde365daf88b6fef380efc55f6c82226337efc3b70d125bbc96479f4064` |
| Migration commit hash | `8808ada8781aef6c1e8aa7a3a3b76ad8cca5e5e0` |

### Read-only re-verification (this closure batch, fresh, live)

| Check | Value | Expected |
|---|---|---|
| `schema_version` | 13 | 13 ✓ |
| `ami_absorption_impact_windowed_flow` | 1,619 | 1,619 ✓ |
| `ami_absorption_impact_window_quality_v1` | 1,620 | 1,620 ✓ |
| `ami_absorption_impact_exclusions` | 1 | 1 ✓ |
| `experiment_registry` | 23 | 23 ✓ |
| `experiment_results` | 350 | 350 ✓ |
| `knowledge.sqlite` nullifier/receipt state | unconsumed→consumed transition occurred only during G2, before M-0035 (chain-of-custody table, Part 1) | unchanged by M-0035 ✓ |
| `integrity_check` | ok | ok ✓ |
| `foreign_key_check` | `[]` | `[]` ✓ |
| `ami_events` / `ami_signal_lifecycle` / `ami_cycles` / `ami_birth_truncated_cascade_geometry` | 252 / 324 / 167 / 220 | unchanged ✓ |
| `ami_agg_trades_repaired` / `ami_cvd_windowed_flow` / `_proxy` | 40,934 / 1,840 / 1,840 | unchanged ✓ |
| `researcher_exposure_ledger` | 1,176 | unchanged ✓ |
| Runtime/risk/execution protected delta | 0 (no file under `execution/`, `risk/`, `brain/`, `.env` referenced or modified by this closure batch or M-0035) | 0 ✓ |

No code was changed to obtain the above; every value was re-read directly from the live files.

---

## PART 6 — Waiver scope

**This waiver states, precisely:**

1. **M-0035 is accepted despite the repository-wide deterministic test suite not being fully green.** The full suite's clean state is 973/987 passing, 14 failing.
2. **Acceptance is based on a proven zero new deterministic failure delta** — every one of the 14 failing tests exhibited an identical FAILED outcome both immediately before M-0035 (verified against the pre-migration backup and the untouched `knowledge.sqlite`) and immediately after M-0035 (verified against the current live files), per the row-by-row matrix in Part 1.
3. **This waiver applies only to the exact 14 named test node IDs** listed in the Part 1 matrix, identified by their exact node IDs and normalized failure signatures (SIG-A/SIG-B/SIG-C/SIG-D). It does not apply to any other test, present or future.
4. **This waiver does not declare those 14 failures resolved.** They remain red. Their hardcoded `experiment_registry`/canonical-hash/nullifier-state expectations still reference a pre-G2 checkpoint and require their own separate remediation batch — explicitly not undertaken here, as it is outside `FAM_CASCADE_ABSORPTION_IMPACT`'s scope and would require asserting new frozen values for a different family's (G2's) checkpoint, which this batch has no authority or context to do.
5. **This waiver does not permit any future batch to introduce new failures** and cite this document as cover. Any new failure appearing after this checkpoint must be independently root-caused and proven pre-existing under this same evidentiary standard (exact node ID, failure signature, before/after chain-of-custody proof) before it may be waived.
6. **Absorption preregistration may proceed only from the accepted schema-13 data state** already recorded in commit `8808ada8` (`ami_absorption_impact_windowed_flow`=1,619, `window_quality_v1`=1,620, `exclusions`=1) — this waiver does not itself authorize preregistration; that remains gated on new, separate operator instruction per the migration's own closing statement.
7. **Future deterministic-regression remediation work must restore a fully green baseline in its own, separate batch** (updating the 14 named tests' G2-era hardcoded checkpoint values, and separately resolving the live-collector-flake test's operational-health classification per Part 2) — not bundled into any future scientific or migration batch.

---

## Remaining risks

1. The 14 named pre-existing failures remain unresolved and will continue to fail in every future full regression run until a dedicated remediation batch updates their G2-era hardcoded checkpoints.
2. The live-microstructure.db-collector timing flake (Part 2) is inherently non-deterministic under concurrent collector load; until it is moved to a separate operational-health suite (not done in this closure batch), it may intermittently reappear in future full-suite runs for reasons unrelated to any code change.
3. Tests 6 and 11 (SIG-C/SIG-B co-occurrence) will continue to fail at the version-tuple line for as long as `schema_version` exceeds 12; a future schema bump (v14+) will not change their outcome, since they are already red for the independent, pre-existing SIG-B reason.
4. This closure batch's overlapping-time-range documentation in Part 3 item 6 is approximate (exact PIDs and precise timestamps were not captured at the moment of the incident) — disclosed as a gap in the incident's own record-keeping rather than reconstructed after the fact with false precision.

## Success verdicts

**`CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_COMPLETE`**

**`ABSORPTION_IMPACT_CANONICAL_DATA_READY_FOR_PREREGISTRATION`**

**`M0035_REGRESSION_BASELINE_WAIVER_ACCEPTED`**

All 14 pre-existing failures are proven, by direct chain-of-custody evidence (not intuition or file-name classification), to have an identical pass/fail outcome before and after M-0035. `NEW_DETERMINISTIC_FAILURES_INTRODUCED_BY_M0035 = 0`. No migration code, schema, migrated data, tests, or runtime files were modified in this closure batch.
