# CVD_PRIMARY_LONG_EXECUTION_V1_EVIDENCE_CLOSURE

**Gate:** G2-CVD-PRIMARY-LONG-EXECUTION-EVIDENCE-CLOSURE-V1
**Nature:** Evidence reconciliation only. No TEST rerun, no new model, no new experiment, no nullifier action, no code change with scientific effect. This document is **additive** — it does not edit `S34_CVD_PRIMARY_LONG_EXECUTION_V1.md`/`.json` or the transition proof from commit `60c3e26f`.
**Prior state (preserved, unchanged):** commit `60c3e26f`, `CVD_PRIMARY_LONG_GOVERNED_EXECUTION_V1_COMPLETE`, TEST n=40, coefficient -0.9356320563432652 bps/$1M, SE 2.662003862077086, 95% CI [-6.320043097269376, 4.448778984582845], p=0.7271227120001349, nullifier consumed exactly once, `experiment_registry` 22→23, `experiment_results` 323→350.
**Author:** Sonnet 5 · **Date:** 2026-07-06

All checks below were performed read-only against the real database files and the existing pre-execution backup (`data/ami/backups/canonical_pre_G2_governed_execution_20260706.sqlite`, `..._knowledge_...`). No test suite was rerun (no code changed since commit `60c3e26f`); only integrity/reconciliation queries and one non-promotable numerical parity check (GAP 4) were executed, none of which write to the real database.

---

## GAP 1 — Pre-TEST code freeze proof

Exact sequence, with millisecond-precision timestamps pulled from git (`git log`), filesystem mtimes, and the real database's own bookkeeping columns (`epistemic_test_nullifiers.consumed_ms`, `experiment_registry.created_ms`, `experiment_gate_receipts.issued_ms`, `researcher_exposure_ledger.created_ms`):

| # | Step | Timestamp | Evidence |
|---|---|---|---|
| 1 | Preregistration committed (frozen scientific specification) | 2026-07-06 21:25:58 +03:00 | `git log` commit `749520b3` |
| 2 | Module `ami/research/cvd_windowed_flow_001.py` written (model/spec implementation) | mtime 2026-07-06 22:16:36.357868 | filesystem `stat` |
| 3 | Test file `tests/test_ami_research_cvd_windowed_flow_001.py` written | mtime 2026-07-06 22:11:07.767817 | filesystem `stat` |
| 4 | **Module content frozen** — no edit after this mtime, proved by (a) `git diff HEAD -- ami/research/cvd_windowed_flow_001.py` = empty (working tree byte-identical to the committed blob) and (b) file `mtime == ctime` (no metadata-only touch since last content write) | 2026-07-06 22:16:36.357868 | `git diff`, `stat` |
| 5 | TRAIN-only validation completed (15/15 tests, incl. a full dress rehearsal of `execute_governed_run` against disposable copies of both real databases, and a standalone dry run whose TRAIN predictor-distribution statistics matched the frozen preregistration's own recorded TRAIN distribution byte-for-byte) | before 22:20:56 (all pre-real-execution) | pytest run log (this session); see execution report §"Dress rehearsal" |
| 6 | Input/specification/code commitments recorded | `code_commit=09104298` (the HEAD at preregistration time, embedded in `registry_values["code_commit"]`), `spec_hash=a2fd9e5b…` (from the frozen preregistration JSON) | `experiment_registry.code_commit`, `provenance` column (real DB, read-only-verified below) |
| 7 | **TEST authorization granted** — `epistemic_gates.consume_test_evidence()` returns `CONSUMED` | 2026-07-06 22:20:56.277 | `epistemic_test_nullifiers.consumed_ms` (real `knowledge.sqlite`) |
| 8 | **First TEST outcome access** — `_fetch_effective_outcome_for_signals()` scoped to the 40 TEST signal_ids, called immediately after step 7 in the same synchronous script run | between 22:20:56.277 and 22:20:56.284 | code ordering (`execute_governed_run`, line 459: `consume_test_evidence()` call; line 465: the first TEST-scoped `_fetch_predictors_controls()` call, textually and causally after) + the two `researcher_exposure_ledger` rows at 22:20:56.254/.263 (TRAIN-phase gateway calls, both *before* .277) and .269 (the `fetch_events` call, also before .277) — no exposure-ledger row exists between .277 and .284, confirming no gateway call intervened |
| 9 | **Result recording** — `record_experiment_registry`/`record_experiment_results` | 2026-07-06 22:20:56.284 | `experiment_registry.created_ms`/`started_at`/`completed_at`, `experiment_results.created_ms` (all four columns identical, real DB) |
| 10 | Gate receipt reissued (`registry_result="EXECUTED"`) | 2026-07-06 22:20:56.286 | `experiment_gate_receipts.issued_ms` (real `knowledge.sqlite`) |
| 11 | Reports written | 22:22:55 (MD), 22:25:06 (JSON), 22:26:40 (transition proof) | filesystem mtimes |
| 12 | **Final commit** | 2026-07-06 22:30:36 +03:00 | `git log` commit `60c3e26f` |

**Proof that no scientific code changed after first TEST access (step 8, ≈22:20:56.28):**
- The module's last content modification (step 4, mtime 22:16:36.358) precedes TEST access by **4 minutes 20 seconds**, with zero intervening writes (mtime==ctime, no subsequent Edit calls in this session's tool-call log).
- `git diff HEAD -- ami/research/cvd_windowed_flow_001.py` → **empty** (verified this session). The working-tree file is byte-identical to the blob committed at 22:30:36, which is the exact file that ran at 22:20:56 — nothing was inserted between execution and commit.
- No model, control, scaling, or exclusion logic exists anywhere outside this one module (`resolve_population`, `build_design`, `run_cluster_robust_ols`, `compute_vif`, `apply_collinearity_policy`, `apply_verdict_rule` — all in the single frozen file), and none of it changed post-TEST.

**No protocol deviation of this kind occurred.** (GAP 4 below covers a real, but pre-TEST, TRAIN-discovered numerical condition — not a post-TEST code change.)

---

## GAP 2 — Database and hash reconciliation

Two physical SQLite files are involved. Full table-by-table reconciliation (every table in both files, not a spot check) was run against the pre-execution backups taken immediately before the real run.

### `data/ami/canonical.sqlite`

| Field | Value |
|---|---|
| Absolute path | `D:\eclipse_scalper\data\ami\canonical.sqlite` |
| Role | Scientific/data warehouse — holds `experiment_registry`, `experiment_results`, `researcher_exposure_ledger`, all AMI signal/event/CVD tables |
| sha256 before | `fdda663dcc331053f6351d6acb7117eeb266fda5cf5d5691a799e48416be724c` |
| sha256 after | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| Journal mode | WAL; `-wal` file 0 bytes (fully checkpointed, no pending frames); `-shm` present (32KB, normal) |
| Backup path | `D:\eclipse_scalper\data\ami\backups\canonical_pre_G2_governed_execution_20260706.sqlite` |
| Backup sha256 | `fdda663dcc331053f6351d6acb7117eeb266fda5cf5d5691a799e48416be724c` (re-verified this batch, matches "before") |

**Tables checked: 39 (all tables in the schema). Tables changed: 3, plus one internal SQLite bookkeeping table.**

| Table | Rows before → after | Delta | Content hash before | Content hash after |
|---|---|---|---|---|
| `experiment_registry` | 22 → 23 | +1 | `d8f15beb…` | `ab54c33a…` |
| `experiment_results` | 323 → 350 | +27 | `5f0efb32…` | `91fcfd58…` |
| `researcher_exposure_ledger` | 1173 → 1176 | +3 | `567bda91…` | `5f2a084a…` |
| `sqlite_sequence` | 3 → 3 (same count) | 0 | `2263b649…` | `0f016adc…` |

`sqlite_sequence` is SQLite's own internal `AUTOINCREMENT` counter table — its content change is the **mechanical, automatic consequence** of the `experiment_registry`/`experiment_results` inserts above (SQLite updates the stored max-rowid counter for those two tables), not an independent write path. **All other 35 tables in `canonical.sqlite` are byte-for-byte identical** (row count and content hash both unchanged), including every protected table (`ami_events`, `ami_signal_lifecycle`, `ami_cycles`, `ami_birth_truncated_cascade_geometry`, `ami_agg_trades_repaired`, `ami_cvd_windowed_flow`, `ami_cvd_windowed_flow_proxy`, `ami_cvd_window_quality_v1`, `ami_cvd_bucket_exclusions`, `ami_lifecycle_path_observations`, and all 25 others).

### `data/ami/knowledge.sqlite`

| Field | Value |
|---|---|
| Absolute path | `D:\eclipse_scalper\data\ami\knowledge.sqlite` |
| Role | Epistemic governance — `epistemic_test_nullifiers`, `experiment_gate_receipts`, `graveyard_slash_fingerprints`, `audit_log` |
| sha256 before | `ef7f8cde5e790ef765498861e2bdbc561f8ac64e3044eeb84afe3786125a8b6c` |
| sha256 after | `2a5abc280889eac91a5ec5e9c82f63d024670b6735f8c4a77b10597c9029b93e` |
| Journal mode | WAL; `-wal` 0 bytes; `-shm` present (32KB) |
| Backup path | `D:\eclipse_scalper\data\ami\backups\knowledge_pre_G2_governed_execution_20260706.sqlite` |
| Backup sha256 | `ef7f8cde5e790ef765498861e2bdbc561f8ac64e3044eeb84afe3786125a8b6c` (re-verified, matches "before") |

Tables changed: **2**.

| Table | Rows before → after | Delta |
|---|---|---|
| `epistemic_test_nullifiers` | 0 → 1 | +1 (this nullifier, `consumed_by_experiment_id=E-CVD-PRIMARY-LONG-W300-PREREG-001`) |
| `experiment_gate_receipts` | 1 → 1 (same count) | content changed: `registry_result` `PREREGISTERED_NOT_EXECUTED`→`EXECUTED` (receipt hash unchanged, `d46f7e2c…`, since the hash is over `experiment_id\|family_id\|split_version\|nullifier`, not `registry_result`) |

`graveyard_slash_fingerprints` (31 rows) and `audit_log` are byte-for-byte unchanged (no supersession/retry token was created, so nothing was appended to `audit_log` by this batch).

### Where each item lives

| Item | Physical file | Table |
|---|---|---|
| `experiment_registry` | `canonical.sqlite` | `experiment_registry` |
| `experiment_results` (= "result metrics") | `canonical.sqlite` | `experiment_results` (same table; `metric_name`/`metric_value` columns) |
| `researcher_exposure_ledger` | `canonical.sqlite` | `researcher_exposure_ledger` |
| Nullifier state | `knowledge.sqlite` | `epistemic_test_nullifiers` |
| Gate receipt | `knowledge.sqlite` | `experiment_gate_receipts` |

### Wording correction (additive, not a rewrite of history)

The final chat-message summary of the execution stated: *"canonical hash delta = expected exposure-ledger appends only."* Taken as a standalone sentence, this is **incomplete** — the canonical.sqlite hash delta is fully explained by **three** table changes (`experiment_registry` +1, `experiment_results` +27, `researcher_exposure_ledger` +3) plus the mechanical `sqlite_sequence` side effect, not by the exposure ledger alone. The registry/results write is not an incidental side effect to be minimized — it **is** the batch's entire purpose. The committed execution report and JSON manifest (`S34_CVD_PRIMARY_LONG_EXECUTION_V1.md`/`.json`, commit `60c3e26f`) already itemized `experiment_registry`/`experiment_results` as their own separate, correctly-labeled line items in the before/after table — **those files are accurate and are not amended here**. This section exists solely to correct the imprecise chat-summary phrasing, additively, per instruction.

---

## GAP 3 — 27 result-row accounting

All 27 rows share **one** `experiment_id` (`E-CVD-PRIMARY-LONG-W300-PREREG-001`) and **one** `created_ms` (`1783365656284`, confirmed by `SELECT DISTINCT experiment_id FROM experiment_results WHERE created_ms=1783365656284` → exactly one value) — i.e., all 27 rows were written in the single result-recording transaction of the single TEST model fit. Aggregate content hash of the full 27-row (name, value) set: `b2097631be1b655e9e5f4bbac24bd5b38ca2e1e43cb3052e5cb8e218c870f5c9`.

| # | Metric key | Role | Preregistered? | From which fit | Source line |
|---|---|---|---|---|---|
| 1 | `primary_predictor_coefficient_bps_per_1m` | **PRIMARY** | Yes (§4/§8) | TEST model | L546 |
| 2 | `primary_predictor_se_cluster_robust` | **PRIMARY** | Yes | TEST model | L547 |
| 3 | `primary_predictor_ci95_lo` | **PRIMARY** | Yes | TEST model | L548 |
| 4 | `primary_predictor_ci95_hi` | **PRIMARY** | Yes | TEST model | L549 |
| 5 | `primary_predictor_p_value` | **PRIMARY** | Yes | TEST model | L550 |
| 6 | `primary_predictor_t_stat` | primary diagnostic | Implied by model | TEST model | L551 |
| 7 | `primary_predictor_df` | primary diagnostic | Implied by model | TEST model | L552 |
| 8 | `test_n_used` | accounting | Yes (missing-data policy) | TEST model | L553 |
| 9 | `test_n_total_representative` | accounting | Yes | population resolution | L554 |
| 10 | `test_n_dropped_missing` | accounting | Yes | TEST model | L555 |
| 11 | `test_n_clusters` | accounting | Yes | TEST model | L556 |
| 12 | `train_n_used` | TRAIN diagnostic | Yes | TRAIN model | L557 |
| 13 | `train_n_dropped_missing` | TRAIN diagnostic | Yes | TRAIN model | L558 |
| 14 | `vif` | TRAIN diagnostic | Yes (§5 collinearity policy) | TRAIN model | L559 |
| 15 | `collinearity_drops_applied` | TRAIN diagnostic | Yes (§5) | TRAIN model | L560 |
| 16 | `predictor_train_scale_stats_usd` | TRAIN diagnostic | Yes (§4) | TRAIN model | L561 |
| 17 | `design_columns` | metadata | N/A (documentation) | both | L562 |
| 18 | `full_beta_vector` | **PRIMARY** (full vector, includes #1) | Yes | TEST model | L563 |
| 19 | `full_se_vector` | **PRIMARY** (full vector, includes #2) | Yes | TEST model | L564 |
| 20 | `verdict_reason` | verdict documentation | N/A (documentation) | derived from TEST model | L565 |
| 21 | `secondary_mfe_bps_coefficient` | **SECONDARY, non-promotable** | Yes (§6, permitted secondary check #1) | TEST model, `mfe_bps` outcome | L566 |
| 22 | `secondary_mfe_bps_p_value` | **SECONDARY, non-promotable** | Yes | TEST model, `mfe_bps` outcome | L567 |
| 23 | `train_side_descriptive_coefficient` | **SECONDARY, non-promotable** | Yes (§6, permitted secondary check #2) | TRAIN model | L568 |
| 24 | `train_side_descriptive_p_value` | **SECONDARY, non-promotable** | Yes | TRAIN model | L569 |
| 25 | `test_cycle_set_hash` | identity/provenance | N/A | population resolution | L570 |
| 26 | `train_cycle_set_hash` | identity/provenance | N/A | population resolution | L571 |
| 27 | `test_nullifier_sha256` | identity/provenance | N/A | gate resolution | L572 |

(Source file: `ami/research/cvd_windowed_flow_001.py`, function `execute_governed_run`, lines 546–572, list literal `results_rows`.)

**Required proof:**
- **One experiment ID only** — confirmed (`SELECT DISTINCT experiment_id ...` above → 1 value).
- **One TEST model fit only** — `run_cluster_robust_ols` is called exactly twice in `execute_governed_run`: once on TRAIN (`train_fit`, diagnostic/descriptive, rows 12–16, 23–24) and once on TEST with `endpoint_return_bps` as outcome (`test_fit`, rows 1–11, 18–20). A third call, `secondary_fit` (rows 21–22), reuses the **same TEST design matrix construction** with `mfe_bps` substituted as the outcome column — this is the single preregistered secondary check (§6, item 1 of "Permitted secondary checks"), not an independent or alternative specification.
- **No hidden alternative specification, no subgroup model, no threshold result, no proxy result, no alternative outcome beyond the one preregistered `mfe_bps` secondary, no repeated TEST evaluation**: verified by direct inspection of `execute_governed_run` — it contains exactly 3 `run_cluster_robust_ols` calls total (TRAIN, TEST-primary, TEST-secondary-mfe), 0 references to `ami_cvd_windowed_flow_proxy` (grep count = 0), 0 threshold/subgroup filtering code paths.
- **Diagnostic rows did not influence the scientific verdict**: `apply_verdict_rule()` (module line 395) takes exactly five scalar arguments — `n_test, coef, se, ci_lo, ci_hi, p_value` — all five sourced only from `test_fit` (rows 1–5 above). No VIF, TRAIN, or secondary-check value is passed into or referenced by `apply_verdict_rule`.

**No row represents an unregistered alternative analysis.**

---

## GAP 4 — Zero-variance session control

**Discovery timing:** the `session_EUROPE` zero-variance condition was found during this session's TRAIN-only diagnostic development (in the standalone dress-rehearsal script run against disposable database copies), **before** the module's frozen mtime (22:16:36) and therefore before any TEST access (22:20:56.277+). It is not a TEST-driven discovery.

**Numerical facts (recomputed this batch, read-only, real database, mode=ro — no gateway writes):**

| Quantity | Value |
|---|---|
| TRAIN design matrix shape / rank | (91, 7) / **rank 6** (rank-deficient by exactly 1) |
| TEST design matrix shape / rank | (40, 7) / **rank 6** |
| `session_EUROPE` column | identically zero in both TRAIN (91/91 zero) and TEST (40/40 zero) |
| Encoded columns | `const, cvd_notional_w300_per_1m, event_notional_per_100k, session_EUROPE, session_US, session_OFF, day_trend_bps` |
| Reference category | ASIA (per the frozen preregistration §5) |
| Pseudo-inverse tolerance | `numpy.linalg.pinv(X.T @ X)` called with no `rcond`/`rtol` override → numpy 2.3.5 default: `rtol = max(M, N) * eps(float64)` (≈ 7×7 matrix, eps≈2.22e-16) |
| Primary coefficient (frozen implementation, pinv, k=7) | -0.9356320563432652 (SE 2.662003862077086, p=0.7271227120001349, CI [-6.320, 4.449]) |

**Was pseudo-inverse covered by the preregistered numerical-stability/collinearity policy?**

**No — this must be classified transparently, as instructed.** The preregistration's §5 collinearity policy addresses **high VIF** among the continuous controls (`day_trend_bps`, `event_notional`) via a fixed drop order, and explicitly states "session is never dropped" — but it does not anticipate or authorize a numerical-estimator substitution (pinv vs. strict inverse) for a **structurally empty** session category. **This is a disclosed protocol/implementation gap, not a violation**, for two reasons: (1) it was discovered and resolved entirely pre-TEST, using only TRAIN diagnostics, with no TEST-driven amendment; (2) it changes nothing about the model's column set, control set, or the primary predictor's identification — it only replaces one instance of `numpy.linalg.inv` with `numpy.linalg.pinv` to avoid a hard `LinAlgError: Singular matrix` crash on an exactly-singular `X'X`. It was already disclosed in the original execution report (§3) and transition proof, not concealed.

**Was any session category dropped or changed after seeing TEST outcomes?** No. `session_EUROPE` remains present as a formula term in both the TRAIN and TEST design matrices, in the single frozen implementation, throughout — its coefficient is estimated (as ≈0), never removed from the design.

**Non-promotable algebraic parity check** (computed this batch, read-only, does not create a second scientific model, does not revisit the verdict, not written to `experiment_results`):

Dropping the exactly-zero `session_EUROPE` column entirely and using an ordinary matrix inverse (`np.linalg.inv`) on the resulting full-rank 6-column TEST design:

| Quantity | Frozen implementation (pinv, k=7) | Reduced encoding (inv, k=6, EUROPE dropped) | Difference |
|---|---|---|---|
| Primary predictor coefficient | -0.9356320563432652 | -0.9356320563432894 | 7.9×10⁻¹² (floating-point noise — **invariant**) |
| Primary predictor SE | 2.662003862077086 | 2.6225645875327155 | 1.5% relative — fully explained by the CR1 finite-sample correction factor `(N-1)/(N-k)·G/(G-1)` using k=7 vs k=6 (a structurally-empty parameter still "spends" one degree of freedom in the frozen k=7 implementation) |
| p-value | 0.7271227120001349 | 0.7231929041168732 | both far from 0.05 |
| 95% CI | [-6.320, 4.449] | [-6.240, 4.369] | both include 0 by a wide margin |

**The primary coefficient — the object of the significance test — is invariant** under the permitted equivalent encoding (agrees to 1e-12, i.e., identical up to floating-point precision). The SE/p/CI shift is a well-understood, mechanically-explained property of the degrees-of-freedom term in the CR1 correction, not a computational error, and **does not change the qualitative conclusion under either encoding**: both give p≫0.05 and a CI that includes zero by several multiples of the coefficient's own magnitude. **`NO_RELIABLE_ASSOCIATION` is robust to this numerical-method choice.**

---

## Scientific label normalization

The frozen preregistration document (`reports/research/s34/S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md`, commit `749520b3`, §8, and the companion `.json` §`verdict_rule`) **literally specifies the negative-result label as `NO_RELIABLE_ASSOCIATION`** (verified by direct grep of the committed, immutable file — three occurrences in the MD, two in the JSON) — not `NO_RELIABLE_INCREMENTAL_ASSOCIATION`. Only the positive-result label includes "INCREMENTAL" (`EVIDENCE_SUPPORTS_INCREMENTAL_ASSOCIATION`); the frozen spec's negative label is asymmetric with respect to that word.

This is disclosed rather than silently reconciled: **the execution report's use of `NO_RELIABLE_ASSOCIATION` (commit `60c3e26f`) is the literal, correct, frozen term** — it was not an error or a deviation from the preregistration.

Per the instruction to normalize forward and preserve the original wording as an alias (without creating a new experiment result or editing the frozen artifacts), the following mapping is recorded as the canonical reporting convention **from this evidence-closure document forward**:

| Canonical label (this closure forward) | Frozen-artifact literal wording (verbatim, commit `749520b3`/`60c3e26f`, unchanged) |
|---|---|
| `NO_RELIABLE_INCREMENTAL_ASSOCIATION` | `NO_RELIABLE_ASSOCIATION` (alias — the exact string in the frozen preregistration's verdict rule and the committed execution report/manifest) |

No row in `experiment_results` was added or changed to effect this — it is a documentation-layer alias only, recorded here additively.

---

## Final validation (read-only, this batch)

| Check | Result |
|---|---|
| One TEST access only | ✅ (single `created_ms`/`consumed_ms` pair; no second nullifier-consumption row exists) |
| Nullifier consumed exactly once | ✅ (`epistemic_test_nullifiers`: 1 row for this nullifier) |
| `experiment_registry` delta | ✅ exactly +1 |
| `experiment_results` delta | ✅ exactly +27, all preregistered/derived from the single TEST fit (GAP 3) |
| Prior 22 experiments byte/content identical | ✅ (hash of all `experiment_registry` rows excl. the new one: identical before/after) |
| Prior 323 result rows byte/content identical | ✅ (hash of all `experiment_results` rows excl. the new experiment: identical before/after) |
| Known-at violations | ✅ 0 (rechecked this batch) |
| Exact/proxy pooling | ✅ 0 (module has zero references to `ami_cvd_windowed_flow_proxy`) |
| No route promotion | ✅ (no route/paper/shadow/forward/live file referenced or touched) |
| No runtime/risk/execution delta | ✅ (`git status` on `execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` shows no change attributable to this work; the one pre-existing untracked file listed there predates this entire session and was never opened by it) |
| `schema_version` | ✅ remains 12 |
| `integrity_check` / `foreign_key_check` | ✅ `ok` / 0 rows |
| Commit `60c3e26f` unchanged | ✅ (`git cat-file -t 60c3e26f` → `commit`; the 5 files it introduced show no working-tree modifications) |

---

## Storage guardrail compliance

This session's test runs (before this closure batch) created disposable database copies for the conftest session-isolation fixture under the harness-provided scratchpad (`C:\Users\...\AppData\Local\Temp\claude\...`), reaching a **peak of ~1.5 GB** across six `--basetemp` directories plus one stray full-copy diagnostic file. **All of it has now been deleted** (none of it was accepted evidence — it was disposable pytest fixture data and throwaway diagnostic copies). Exact accounting:

| Item | Size | Disposition |
|---|---|---|
| `pytest_g2`, `pytest_g2b`..`pytest_g2f` (6 basetemp dirs, each holding a disposable copy of `canonical.sqlite`+`knowledge.sqlite`) | ~212–230 MB each, ~1.3 GB total | Deleted this batch |
| `canonical_diag_copy.sqlite` (stray full copy from an early diagnostic attempt) | 211 MB | Deleted this batch |
| Two small standalone scripts (`verify_g2.py`, `diag_g2.py`) | 4 KB total | Retained (negligible, no DB content) |
| `D:\eclipse_scalper_scratch_canonical_diag.sqlite`, `..._knowledge_diag.sqlite` (used for the dry-run preview and the first diagnostic) | ~221 MB + ~110 KB | Already deleted immediately after use, earlier in this work (before this closure batch even started) |

**Remaining at batch completion:** `C:\Users\...\AppData\Local\Temp\claude\...\scratchpad\` now contains only the two 4KB scripts — 0 bytes of database copies. `D:\eclipse_scalper\.runtime_temp\` and `D:\eclipse_scalper\.pytest_temp\` were created this batch (both empty) for any future large-copy needs, per the new guardrail; **no full database copy was created under them in this batch**, since every check here was either read-only against the real files or against the already-existing pre-execution backups. `data\microstructure.db`, `canonical.sqlite`, `knowledge.sqlite`, both accepted `data\ami\backups\*` files, and all immutable evidence artifacts remain untouched and undeleted.

---

## Closure artifact

- Path: `reports/governance/CVD_PRIMARY_LONG_EXECUTION_V1_EVIDENCE_CLOSURE.md` (this file)
- No code, schema, or data changed as part of producing it — a full regression suite was correctly **not** rerun, per instruction.

## Remaining limitations

1. `session_EUROPE` structural absence (GAP 4) remains a disclosed, unresolved gap in the preregistered model's applicability — carried over, not newly introduced by this closure.
2. The label-normalization table above is a documentation-layer convention only; the frozen artifacts' literal text (`NO_RELIABLE_ASSOCIATION`) remains authoritative and unedited.
3. Family/split identity adapters remain text-hash-based (paraphrase-bypass risk) — carried over from preregistration, unaffected by this closure.
4. This closure batch performed no new statistical inference; the GAP 4 parity check is explicitly non-promotable and does not constitute independent confirmatory evidence.

---

## Verdict

**`CVD_PRIMARY_LONG_GOVERNED_EXECUTION_V1_COMPLETE`**

**`NO_RELIABLE_INCREMENTAL_ASSOCIATION`** (canonical label; frozen-artifact literal alias: `NO_RELIABLE_ASSOCIATION`)

All four gaps close. No follow-up hypothesis, threshold scan, subgroup rescue, or second TEST pass was performed or opened.
