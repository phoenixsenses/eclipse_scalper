# CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-CASCADE-ABSORPTION-IMPACT-PREREGISTRATION-V1
**Purpose:** Preregister exactly one primary hypothesis for `FAM_CASCADE_ABSORPTION_IMPACT` through the enforced epistemic-gate system, without accessing any TEST outcome.
**Prior checkpoint (unchanged, not reopened):** commit `5ab89f63` (`M0035_REGRESSION_BASELINE_WAIVER_ACCEPTED`), migration `8808ada8` (`CASCADE_ABSORPTION_IMPACT_CANONICAL_MIGRATION_V1_COMPLETE`), `schema_version=13`, `experiment_registry=23`, `experiment_results=350`.
**Nature:** Preregistration only. No TEST execution, no TEST outcome read, no nullifier consumption, no window/threshold/direction scan, no proxy pooling, no migrated-row alteration, no schema change, no runtime/risk/execution modification, no route/bucket promotion.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Window ruling (operator-approved before this batch)

The primary window (W300) was proposed strictly outcome-blind (comparing all five frozen windows only on mechanism timescale, coverage, known-at safety, measurement stability, and cross-mechanism comparability — no outcome, correlation, or TEST value was read) and approved by the operator with two mandatory amendments: (1) the unverified W60 "1-2 prints" claim removed and replaced with a defensible, narrower statement; (2) only W300 may be joined to any outcome in this first experiment — W60/W600/W1800/W3600 remain canonical data products only. The exact frozen ruling text is reproduced in full in the preregistration MD/JSON (§1 / `window_ruling`).

## Sequence executed (enforced order, per Phase 11)

1. **Resolve canonical family and child identity:** `question_ids="FAM_CASCADE_ABSORPTION_IMPACT_LONG_REVERSAL"`, `hypothesis_id="H-CASCADE-ABSORPTION-IMPACT-LONG-W300-EXACT-PRICE-RESPONSE-PER-SIGNED-NOTIONAL-V1"` → `family_id = gates.resolve_canonical_family_id(...) = "FAMv1:3e2dfe63f9e271bf"`.
2. **Graveyard check:** `match_graveyard()` against the full spec text (question_ids + hypothesis_id + frozen_population + frozen_features + frozen_target + frozen_thresholds) run against the real `data/ami/knowledge.sqlite` `graveyard_slash_fingerprints` (31 curated) — **0 hits**.
3. **Retry requirements:** none — clean graveyard result requires no retry token (`authorization_state = NOT_REQUIRED_CLEAN_GRAVEYARD`).
4. **Freeze specification:** `specification_hash_sha256 = 531b16232a88d5a6c692055bd00fa59bd508b7b69cd7fd45cf8e666772fb6608` (sha256 of the 7 frozen identity/spec fields joined).
5. **Freeze outcome:** `endpoint_return_bps@swing_24h` — reused verbatim from the closed CVD preregistration/test, not redefined.
6. **Freeze predictor/window:** `ami_absorption_impact_windowed_flow.price_response_per_signed_notional` WHERE `window_id='W300'` — the sole frozen, formula-unaltered M-0035 canonical feature, per the operator's window ruling.
7. **Freeze population:** `ami_signal_lifecycle direction='LONG'` (220) → 194 outcome-eligible → 131 representative independent cycles — reproduced independently this session by direct SQL, not copied from any prior document.
8. **Freeze split:** cycle-grouped chronological 70/30, `split_version = SPLITv1:16ea98c239034593` (freshly computed for this family; the underlying TRAIN/TEST cycle **sets** are proven byte-identical to the closed CVD preregistration's own, via independently reproduced hash matches — not assumed).
9. **Freeze ordered TRAIN set:** 91 cycles, hash `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5` (byte-identical to CVD's own).
10. **Freeze ordered TEST set:** 40 cycles, hash `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04` (byte-identical to CVD's own).
11. **Compute TEST nullifier:** `gates.derive_test_nullifier(family_id, split_version, test_cycle_ids) = "4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4"` — genuinely different from CVD's own nullifier (`085397f3…`) despite the identical underlying TEST cycle set, because `family_id` differs (a different mechanism/hypothesis) — proving the single-use law is correctly scoped per family, not merely per cycle-set.
12. **Prove nullifier unused:** `SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?` against the real, read-only `knowledge.sqlite` → **0**, both before and immediately after this batch's own write (the write only ever touches `experiment_gate_receipts`, never `epistemic_test_nullifiers`).
13. **Create experiment registration (gate-system, not `experiment_registry`):** the CVD precedent's own preregistration never wrote an `experiment_registry`/`experiment_results` row either — that table is populated only at TEST-execution time via `register_experiment_with_gates`. "New experiment count = exactly 1" in this batch refers to the one new `experiment_gate_receipts` row (below), matching the established precedent exactly.
14. **Issue gate receipt:** `gates.issue_gate_receipt(conn, experiment_id="E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001", canonical_family_id="FAMv1:3e2dfe63f9e271bf", split_version="SPLITv1:16ea98c239034593", nullifier="4e3d1229…", registry_result="PREREGISTERED_NOT_EXECUTED")` — written to the **real** `data/ami/knowledge.sqlite` (not a disposable copy). Returned `receipt_hash = "6dbe0f59416977fce75b20a13876ff4d54dddae171d1fa8b07613135550e06e4"`, matching the value independently pre-computed and frozen into the preregistration JSON **before** the write (proving determinism, not post-hoc adjustment).
15. **Leave nullifier unconsumed:** confirmed — `epistemic_test_nullifiers` row count unchanged (1 → 1, the pre-existing CVD nullifier only; 0 rows for this family's nullifier both before and after).
16. **Do not access TEST outcomes:** confirmed structurally — every query issued this batch against `ami_lifecycle_path_observations` selected only `signal_id`, `path_definition_version`, `observation_status` (data-availability columns); `endpoint_return_bps`/`mfe_bps`/`mae_bps` were never named in any SQL statement executed this session (verified by direct review of every script run).

**Expected state after preregistration: `PREREGISTERED_NOT_EXECUTED`.** Confirmed both in the JSON manifest and by a real, read-only re-query of `experiment_gate_receipts` against the live `knowledge.sqlite`.

---

## Identity record

| Field | Value |
|---|---|
| Canonical family ID | `FAMv1:3e2dfe63f9e271bf` |
| Child-hypothesis ID | `H-CASCADE-ABSORPTION-IMPACT-LONG-W300-EXACT-PRICE-RESPONSE-PER-SIGNED-NOTIONAL-V1` |
| Experiment ID | `E-CASCADE-ABSORPTION-IMPACT-LONG-W300-PREREG-001` |
| Specification hash | `531b16232a88d5a6c692055bd00fa59bd508b7b69cd7fd45cf8e666772fb6608` |
| Outcome ID | `endpoint_return_bps@swing_24h` |
| Predictor identity | `ami_absorption_impact_windowed_flow.price_response_per_signed_notional`, `window_id='W300'` |
| Window | W300 (operator-ruled) |
| Split version | `SPLITv1:16ea98c239034593` |
| TRAIN cycle-set hash | `61486bc62392eed7b7fc038715f2cd9775e270a568e5c1f728dc2d60417671a5` |
| TEST cycle-set hash | `98174ed356826b15bd8513584015447b68d18718bb933d75380a4d6b2c4f7b04` |
| TEST nullifier | `4e3d1229edc04a946ef29994f1562444fd7c9e77b6ff3ecf3004677f919df7d4` |
| Gate receipt ID (hash) | `6dbe0f59416977fce75b20a13876ff4d54dddae171d1fa8b07613135550e06e4` |
| Graveyard decision | `CLEAN`, 0 hits |
| Input manifest root | `canonical.sqlite` sha256 `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` @ schema_version=13 |
| Code commitment | `5ab89f63` (repository HEAD at preregistration time) |
| Authorization state | `NOT_REQUIRED_CLEAN_GRAVEYARD` |

---

## Full state checkpoint (before → after this batch's real write)

| Field | Before | After |
|---|---|---|
| `data/ami/knowledge.sqlite` sha256 | `2a5abc280889eac91a5ec5e9c82f63d024670b6735f8c4a77b10597c9029b93e` | `d435c3a294a286a18a7900d42824f3d4ad020ddedbadf878fcca2a18865c03a9` |
| `epistemic_test_nullifiers` (row count) | 1 | 1 (unchanged — the pre-existing CVD nullifier only) |
| `experiment_gate_receipts` (row count) | 1 | **2** (exactly 1 new row, this experiment's) |
| `data/ami/canonical.sqlite` sha256 | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` | `a229d4b0a7ed82c0ec8411f767a3cba031414e61e32b42ace3e7f6ef390aaaf7` (**byte-identical, untouched**) |
| `experiment_registry` | 23 | 23 (unchanged) |
| `experiment_results` | 350 | 350 (unchanged) |
| `schema_version` | 13 | 13 (unchanged) |
| `integrity_check` | ok | ok |
| `foreign_key_check` | [] | [] |

The only hash delta anywhere in this batch is `knowledge.sqlite`'s, and it is fully explained and scoped: exactly one new `experiment_gate_receipts` row. `canonical.sqlite` was never opened for writing at any point this batch.

---

## Validations (proven)

| Check | Result |
|---|---|
| TEST outcome reads | **0** (no query this session ever named `endpoint_return_bps`, `mfe_bps`, or `mae_bps`) |
| TEST nullifier consumed | **0** (`epistemic_test_nullifiers` unchanged, 1→1) |
| New experiment count | **exactly 1** (`experiment_gate_receipts`, 1→2) |
| New experiment-result count | **0** (`experiment_results`, canonical.sqlite untouched) |
| Exact/proxy pooling | **0** (no proxy layer exists for this family at all; never referenced) |
| `known_at_violations` | **0** (0/324 W300 rows with `window_end_ts_ms > signal_birth_ts` or non-`KNOWN_AT_SAFE` classification, reverified this session) |
| Duplicate cycle representatives | **0** (131 representative cycles = 131 distinct `independent_cycle_id` values, one signal each) |
| Split overlap | **0** (TRAIN's latest representative precedes TEST's earliest; no straddling) |
| Migrated absorption rows unchanged | confirmed — `ami_absorption_impact_windowed_flow`=1,619 / `_window_quality_v1`=1,620 / `_exclusions`=1, identical to the M-0035 final checkpoint |
| `schema_version` | remains **13** |
| Route promotion | **0** |
| Runtime/risk/execution delta | **0** (no file under `execution/`, `risk/`, `brain/`, `.env` referenced) |
| Prior experiment/result history | immutable — `experiment_registry`=23, `experiment_results`=350, byte-identical before/after |

Any expected hash delta was limited to the governed `experiment_gate_receipts` insertion in `knowledge.sqlite`, exactly as required.

---

## Testing

Focused test file `tests/test_ami_absorption_impact_preregistration_v1.py` (20 tests, new) — **20/20 passed**, run twice: once before the real gate-receipt write (19/20 passed, the one expected gap being `test_real_nullifier_and_receipt_state` since the receipt did not exist yet), and once after (20/20 passed).

Paired with `tests/test_ami_cvd_primary_long_preregistration_v1.py` (≤2-file guardrail): **3 failed, 33 passed**. The 3 failures are `test_gate_receipt_mechanism_round_trips_on_disposable_copy`, `test_real_nullifier_and_receipt_state`, `test_no_experiment_created_and_canonical_invariants_hold` — exactly the same 3 pre-existing, already-waived failures documented as part of the M-0035 regression-baseline waiver (`5ab89f63`, batch 4, SIG-D/SIG-A: caused by the separate G2-CVD-PRIMARY-LONG-GOVERNED-EXECUTION-V1 batch's nullifier consumption and `experiment_registry` 22→23 drift, unrelated to and unaffected by this preregistration). **Zero new deterministic failures were introduced** — confirmed by exact match against the already-documented waived baseline, not by re-deriving root causes from scratch. This preregistration's own 20 tests are not part of that waived baseline and are fully green.

No repository-wide full regression (987 tests) was rerun for this documentation-plus-one-governed-write batch — not required, since the only production-code changes are additive (one new test file; no migration code, schema, or migrated-data changes), and the one real write (`experiment_gate_receipts`) is proven scoped and correct by the focused tests and the full before/after state checkpoint above.

---

## Storage guardrail

| Item | Value |
|---|---|
| Temporary files created | `.runtime_temp/absorption_prereg_identity.json` (~1KB, intermediate identity-computation scratch), `.runtime_temp/prereg_before_state.json` / `prereg_after_state.json` (~300 bytes each, before/after checkpoint scratch) |
| Peak temporary disk usage | <2KB |
| Full database copies created | 0 (all population/split computation used bounded, read-only SQL against the real files; only the focused tests' own `tmp_path` fixtures made disposable `knowledge.sqlite` copies, cleaned up automatically by pytest) |
| Files retained | none of the above scratch files — all deleted after their values were folded into the committed JSON/MD/proof |
| Files deleted | `.runtime_temp/absorption_prereg_identity.json`, `.runtime_temp/prereg_before_state.json`, `.runtime_temp/prereg_after_state.json` |
| Remaining under `.runtime_temp` | unchanged from the M-0035 checkpoint (`absorption_impact_rehearsal_v1/` + the 4 M-0035 evidence JSONs) |
| Remaining under `.pytest_temp` | none (all `--basetemp` runs targeted the OS scratchpad, not the repo) |
| `data/microstructure.db` copied | never |

---

## Verdict

**`CASCADE_ABSORPTION_IMPACT_PREREGISTRATION_V1_COMPLETE`**

**`ABSORPTION_IMPACT_PREREGISTERED_NOT_EXECUTED`**

The experiment is frozen through the enforced gate (graveyard-clean, nullifier-derived-and-unused, gate-receipt-issued) with no TEST outcome accessed at any point. Stopping after preregistration. Do not execute TEST without new, separate operator instruction.
