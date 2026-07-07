# BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-BOOK-SPREAD-DYNAMICS-DISPOSABLE-REHEARSAL-V1
**Purpose:** Prove the operator-approved W300 additive spread-change feature (`BOOK_SPREAD_CHANGE_BPS_W300_V1`) can be constructed deterministically, safely and reproducibly on the canonical anchor universe — entirely in disposable space, outcome-blind.
**Prior checkpoint (unchanged, not reopened):** readiness commit `f115b9c1` (`SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS`) + operator ruling `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1`. `schema_version=13`, `experiment_registry=24`, `experiment_results=381`, `epistemic_test_nullifiers=2`, `experiment_gate_receipts=2`.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Phase 1 (verify input contract):** reconfirmed the 5 accepted readiness artifacts (`f115b9c1`) by full sha256; confirmed identity `FAM_BOOK_SPREAD_DYNAMICS`, exact L1 `book_ticker`, `id`-DESC tie-break, `FEED_LIMITS["book_ticker"]=5min`; no identifier conflict; graveyard clean; feature/window not reopened (operator ruled).
2. **Phase 2-4 (spec + module):** new `ami/research/book_spread_dynamics_rehearsal.py` reusing the accepted readiness quote-selection/quality primitives verbatim; disposable schema with the full Phase-4 row shape (endpoint provenance + single feature column + immutable quality/exclusion codes); `specification_hash` pinning the frozen definition.
3. **Phase 3/5/6/7/8 (build twice):** ran the rehearsal against the real (mode=ro) canonical + microstructure DBs, writing only to disposable SQLite files under `.runtime_temp/spread_rehearsal_v1/`, with the SQLite access-guard authorizer installed on the canonical connection for the whole run. Two independent full builds.
4. **Phase 9-10:** carried the accepted family-distinctness argument and drafted the future scientific question (no outcome read, no outcome selected).
5. **Focused tests:** `tests/test_ami_research_book_spread_dynamics_rehearsal.py` (24 tests) — **24/24 passed**.

## Results

- **Accounting (reconciles):** 324 = 196 `EXACT_RECONSTRUCTABLE` + 22 `STALE_SOURCE` + 106 `UNAVAILABLE_BEFORE_COLLECTION` (0 crossed/locked/zero/repaired/gapped/proxy). 196 exact → 97 independent cycles → 97 representatives, 0 duplicates. Est. TEST ≈ 30 ≥ MIN_BUCKET_N=20. Matches the readiness estimate exactly.
- **Known-at:** 0 violations, 0 field violations, both endpoints, both runs.
- **Access guard:** `authorizer_violations = []` on both runs — 0 outcome/experiment/nullifier/gate-receipt access.
- **Determinism:** content hash `5e9ee58c…` and row-manifest hash `8e8e23ff…` **identical** across both builds; only the `.sqlite` file hash differs (created_ms bookkeeping) → `REBUILD_IDENTICAL`.
- **Numerical:** 0 non-finite, 0 non-positive mid, additive identity exact; changes −0.00084→+0.12807 bps (170 expansion / 26 compression).

## Full hashes (non-truncated)

### Databases (unchanged — mode=ro throughout)

| File | Before | After |
|---|---|---|
| `data/ami/canonical.sqlite` | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` |
| `data/ami/knowledge.sqlite` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` |

### Accepted input artifacts used (readiness commit `f115b9c1`)

| Artifact | sha256 |
|---|---|
| `S34_SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1.md` | `127e6a4f9dae1a0043e1b3e5396b3f5ec96a44fba110d6c5076eb9f58b2dabc1` |
| `S34_SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1.json` | `fe08c9f4a7f44884f6ed3c118549dc12c83dc14069445a3d7e2b1ad695f51bbe` |
| `SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF.md` | `405e82a919e20a2a11d4b740d3908559ab823396e834359ce52b75af67dc02b4` |
| `ami/research/spread_dynamics_readiness_audit.py` | `b338f435afa3e150122e851ba7c9ed95f6d5b7b1fb8f0ccec2e33b84ce40a494` |
| `tests/test_ami_research_spread_dynamics_readiness_audit.py` | `b7bc724fc7f6d96414a21cda5a8ffd05042b825d1c8ab151616d2b544f59bedf` |

### This batch's artifacts

| Item | sha256 |
|---|---|
| `ami/research/book_spread_dynamics_rehearsal.py` (code) | `b4a45a5342ba161dc7d749de79e1fa6783781117c1a613cb4e9f1f047755160a` |
| Disposable specification (`specification_hash`) | `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212` |
| Final ordered content hash | `5e9ee58cd9c260c2877b05ed803dbf51767ecedc579bdc90c37b5391a867bcbb` |
| Final ordered row-manifest hash | `8e8e23ff8af6dfd1c11199f963698d4a148583fd2b9c979dffa7f4e4fdec72f2` |
| Disposable dataset `rehearsal_run1.sqlite` | `341677679426af38336393e89369df586be00714a43ebe0e892bb555557b11e5` |
| Disposable dataset `rehearsal_run2.sqlite` | `227b26040fdae02157caa983f20ee98ccbcaab79a73517c72d4b9076dd69c941` |
| `rehearsal_result.json` | `bbefff65bd8f5642598c82e5944f0f9e59c8ca4e840592f63a7166f788e9f2c2` |

## Required validations (proven)

| Check | Result |
|---|---|
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| All outcome-table reads | 0 (authorizer-proven) |
| Outcome-table writes | 0 |
| Experiment creation | 0 |
| Experiment-result creation | 0 |
| Nullifier creation | 0 |
| Nullifier consumption | 0 |
| Gate-receipt creation / update | 0 |
| Canonical migration | 0 |
| Schema-version change | 0 (remains 13) |
| Route promotion | 0 |
| Runtime/risk/execution delta | 0 |
| Shadow/paper/forward/live delta | 0 |
| Canonical DB mutation | 0 (hash unchanged) |
| Knowledge DB mutation | 0 (hash unchanged) |
| Prior experiment/result history mutation | 0 |
| Prior nullifier/gate-receipt mutation | 0 |
| `experiment_registry` | remains **24** |
| `schema_version` | remains **13** |
| `epistemic_test_nullifiers` | remains **2** |
| `experiment_gate_receipts` | remains **2** |

## Exact changed/added-file manifest (this commit)

| File | Status |
|---|---|
| `reports/governance/FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1.md` | New (operator ruling record) |
| `reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1.md` | New |
| `reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1.json` | New |
| `reports/governance/BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_STATE_TRANSITION_PROOF.md` | New (this document) |
| `ami/research/book_spread_dynamics_rehearsal.py` | New (rehearsal module) |
| `tests/test_ami_research_book_spread_dynamics_rehearsal.py` | New (24 focused tests) |

Not included: canonical migration, schema changes, preregistration, experiment records, TEST execution, outcome results, bucket construction, route/runtime/risk/execution/shadow/paper/forward/live changes, unrelated cleanup, another mechanism family, unrelated-waived-failure repairs, or the disposable databases (retained under `.runtime_temp`, not committed to git).

## Regression policy

The accepted deterministic baseline (1,027 collected / 1,013 passed / 14 narrowly waived) is not perturbed: this batch adds only two new files (a read-only rehearsal module with no import side effects, plus its own 24-test file) and touches no existing code, test, or DB. A full paired regression sweep was **not** rerun for this additive, read-only, disposable-only batch (no production-code or shared-state change exists that could shift the baseline); the new file's own 24 tests are green. The 14 waived pre-existing failures remain exactly as in the M-0035 waiver (`5ab89f63`); no new deterministic failure is introduced and none is hidden under that waiver. Mutable live-collector health checks remain a separate concern.

## Storage guardrail

| Item | Value |
|---|---|
| Full database copies created | 0 |
| Full-table scan/copy of `book_ticker` (~2×10⁹ rows) | never (index-backed per-anchor at-or-before seeks only) |
| Peak temporary disk usage | ~462 KB (two 229 KB disposable SQLite files + small JSON) under `.runtime_temp` |
| Temp files created | `.runtime_temp/spread_rehearsal_v1/{rehearsal_run1.sqlite, rehearsal_run2.sqlite, rehearsal_result.json, manifest.json}`; one OS-scratchpad driver script |
| Temp files deleted | the OS-scratchpad driver script (after evidence recorded) |
| Temp files retained (hashed immutable evidence) | the 4 files under `.runtime_temp/spread_rehearsal_v1/` (no outcome data; for the future row-accounting-freeze gate) |
| Remaining under `.runtime_temp` | `spread_rehearsal_v1/` (this batch) + `absorption_impact_rehearsal_v1/` + the 4 M-0035 evidence JSONs (prior accepted evidence, untouched) |
| Remaining under `.pytest_temp` | none |
| Full database copy created | **no** |

---

## Verdict

**`BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_COMPLETE`**

**Readiness disposition:** `BOOK_SPREAD_DYNAMICS_DISPOSABLE_DATA_READY_FOR_ROW_ACCOUNTING_FREEZE` (authorizes no automatic next step).

Recommended next gate: `BATCH-BOOK-SPREAD-DYNAMICS-ROW-ACCOUNTING-FREEZE-V1` — not begun automatically; still prohibits outcome access unless separately authorized. Stopping after the disposable-rehearsal verdict and dedicated commit.
