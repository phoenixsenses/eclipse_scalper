# BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-BOOK-SPREAD-DYNAMICS-ROW-ACCOUNTING-FREEZE-V1
**Purpose:** Bind the operator-approved `BOOK_SPREAD_CHANGE_BPS_W300_V1` definition to one exact, ordered, reproducible anchor population and one exact set of selected source quotes — outcome-blind, before any canonical migration.
**Prior checkpoint (unchanged, not reopened):** rehearsal commit `6a449a64` (`BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_COMPLETE`) + operator ruling `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1`. `schema_version=13`, `experiment_registry=24`, `experiment_results=381`, `epistemic_test_nullifiers=2`, `experiment_gate_receipts=2`.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Phase 1 (reconcile):** recomputed the content/row-manifest/spec hashes from the retained accepted evidence (`.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite`) → reproduced `5e9ee58c…` / `8e8e23ff…` / `ea611121…` exactly; identity/definition/counts intact; no internal disagreement.
2. **Phases 2-7 (freeze module):** new `ami/research/book_spread_dynamics_row_accounting_freeze.py` — frozen ordering (`signal_birth_ts ASC, anchor_id ASC`) and serialization (`repr()` + U+001F/U+001E + sha256); 5 ordered manifests (anchor 324 / exact-feature 196 / exclusion 128 / cycle-membership 196 / representative 97); accounting-identity and known-at revalidation helpers; root-hash builder.
3. **Phase 8 (independent replay):** two fresh rebuilds via the accepted rehearsal builder (not copies), each with the SQLite authorizer installed → A ≡ B ≡ accepted rehearsal (content, row-manifest, all 5 manifests), exact serialized equality.
4. **Phases 9-10 (known-at + access guard):** all no-lookahead/identity checks zero at both endpoints, both replays; authorizer violations `[]`.
5. **Phases 11-13 (immutability + identities):** all Phase-13 accounting identities true; immutable scope + amendment/repair policy recorded.
6. **Phase 12 (migration draft):** drafted the future canonical-migration contract — no ID assigned, no DDL, no write.
7. **Phase 16 (tests):** `tests/test_ami_research_book_spread_dynamics_row_accounting_freeze.py` (15 tests) — **15/15 passed** (one iteration: a naive `"ratio"` substring guard matched "mig**ratio**n" in prose; narrowed to specific transform identifiers).

## Required validations (proven)

| Check | Result |
|---|---|
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| All outcome reads | 0 (authorizer-proven) |
| Outcome writes | 0 |
| Experiment creation / experiment-result creation | 0 / 0 |
| Nullifier creation / consumption | 0 / 0 |
| Gate-receipt creation / update | 0 / 0 |
| Canonical feature migration | 0 |
| Migration ID creation | 0 |
| Schema-version change | 0 (remains 13) |
| Experiment-registry change | 0 (remains 24) |
| Route / bucket promotion | 0 |
| Runtime / risk / execution delta | 0 |
| Shadow / paper / forward / live delta | 0 |
| Prior experiment/result history mutation | 0 |
| Prior nullifier / gate-receipt mutation | 0 |
| Accepted rehearsal evidence mutation | 0 (read-only; hashes unchanged) |
| `canonical.sqlite` mutation | 0 (hash unchanged) |
| `knowledge.sqlite` mutation | 0 (hash unchanged) |

Post-batch: `schema_version=13`, `experiment_registry=24`, `epistemic_test_nullifiers=2`, `experiment_gate_receipts=2`.

## Full hashes (non-truncated)

### Databases (unchanged — mode=ro throughout)

| File | Before | After |
|---|---|---|
| `data/ami/canonical.sqlite` | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` |
| `data/ami/knowledge.sqlite` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` |

### Accepted input artifacts (rehearsal commit `6a449a64`)

| Artifact | sha256 |
|---|---|
| `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1.md` | `d98f6793e4bc92f835ce7deed7954b0e96496bdbf3ea69d96e088210d2394c70` |
| `S34_BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1.md` | `75e4e30afdf8dd661cf44f878d55f5a8a74a950d53f3b9bee36fa6ebbb2a2b5c` |
| `S34_BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1.json` | `444d6bd3ab4eca184ffb9b738625c1fde6d565f4404db23de88357a1d4798d39` |
| `BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_STATE_TRANSITION_PROOF.md` | `3f9dd2a280a54055662ee4b932e1dffdf2f96dc5ed91e76e3ab0120464e672a4` |
| `ami/research/book_spread_dynamics_rehearsal.py` | `b4a45a5342ba161dc7d749de79e1fa6783781117c1a613cb4e9f1f047755160a` |
| `tests/test_ami_research_book_spread_dynamics_rehearsal.py` | `ff7738982ab36dc90828aad049bd944363175598657651e34c1d1e609bc561a4` |

### Accepted retained rehearsal evidence (`.runtime_temp/spread_rehearsal_v1/`, unchanged)

| File | sha256 |
|---|---|
| `rehearsal_run1.sqlite` | `341677679426af38336393e89369df586be00714a43ebe0e892bb555557b11e5` |
| `rehearsal_run2.sqlite` | `227b26040fdae02157caa983f20ee98ccbcaab79a73517c72d4b9076dd69c941` |
| `rehearsal_result.json` | `bbefff65bd8f5642598c82e5944f0f9e59c8ca4e840592f63a7166f788e9f2c2` |
| `manifest.json` | `0419ce7c528b255ec8e10b4d367aec2a575a01ab183322822385adf9c1daeab5` |

### This batch's manifests and root

| Item | sha256 |
|---|---|
| Ordered anchor manifest (324) | `a77a8daf2a8d198d775436674a20a9bd5328dc071e2883938b7c331c17c534bb` |
| Ordered exact-feature manifest (196) | `b1eb902f5b3d1ea0f19b4b60d0ad999907a042b228adf506bbe09800a81e155b` |
| Ordered exclusion manifest (128) | `0694e43300710e1204c1b23643d9eacb9f10188c21aa0ceda572c28229cc8449` |
| Ordered cycle-membership manifest (196) | `e692ff1c8ce37b54a3349a501a38bd44f24865e75a51accc81c7e97399d29e18` |
| Ordered representative manifest (97) | `edadf5972cbbdddb0efa1db8234473ee089972f504d3bfbfafbae508238db246` |
| Rehearsal content hash (reproduced) | `5e9ee58cd9c260c2877b05ed803dbf51767ecedc579bdc90c37b5391a867bcbb` |
| Rehearsal row-manifest hash (reproduced) | `8e8e23ff8af6dfd1c11199f963698d4a148583fd2b9c979dffa7f4e4fdec72f2` |
| Specification hash | `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212` |
| Committed detail manifest JSON | `0a65c45ffba906414c7a484e3f966e2405017eaea8990aded429dc35ed142c89` |
| Freeze module code | `0c892d88b3744da4f6b41f88fa68c03dc4e34cc5949be60e8c84f164bdd37892` |
| **Root** `BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_ROOT` | **`33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31`** |

## Exact changed/added-file manifest (this commit)

| File | Status |
|---|---|
| `reports/governance/FAM_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1.md` | New |
| `reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1.md` | New |
| `reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1.json` | New |
| `reports/research/s34/S34_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_MANIFEST.json` | New (335 KB immutable manifest) |
| `reports/governance/BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_STATE_TRANSITION_PROOF.md` | New (this document) |
| `ami/research/book_spread_dynamics_row_accounting_freeze.py` | New (freeze/replay module) |
| `tests/test_ami_research_book_spread_dynamics_row_accounting_freeze.py` | New (15 focused tests) |

Not included: canonical migration, migration scripts/ID, schema changes, canonical DB changes, preregistration, experiment records, TEST execution, outcome results, bucket construction, route promotion, runtime/risk/execution/shadow/paper/forward/live changes, unrelated cleanup, another mechanism family, unrelated-waived-failure repairs, large database copies, or the disposable replay databases (retained under `.runtime_temp`, not committed).

## Regression policy

The accepted deterministic baseline (1,027 collected / 1,013 passed / 14 narrowly waived) is not perturbed: this batch adds two new files (a read-only freeze module with no import side effects + its own 15-test file) and touches no existing code, test, or DB. A full paired sweep was **not** rerun for this additive, read-only, disposable-only batch (no production-code or shared-state change could shift the baseline); the new file's 15 tests are green. The 14 waived pre-existing failures remain exactly as in the M-0035 waiver (`5ab89f63`); no new deterministic failure is introduced and none is hidden under it. Mutable live-collector health checks remain a separate concern.

## Storage guardrail

| Item | Value |
|---|---|
| Full database copies | 0 |
| Full-table scan/copy of `book_ticker` (~2×10⁹ rows) | never (index-backed per-anchor seeks only) |
| Peak temporary disk usage | ~0.9 MB (`.runtime_temp/spread_freeze_v1/`: two ~230 KB replay SQLites + result/manifest JSON) |
| Temp files created | `.runtime_temp/spread_freeze_v1/{replay_A.sqlite, replay_B.sqlite, freeze_result.json, manifest.json}`; one OS-scratchpad driver |
| Temp files deleted | OS-scratchpad driver (after evidence recorded) |
| Temp files retained | `.runtime_temp/spread_freeze_v1/` (small, hashed; the 335 KB detail manifest also copied into the committed repo artifact) |
| Remaining under `.runtime_temp` | `spread_freeze_v1/` (this batch) + `spread_rehearsal_v1/` + `absorption_impact_rehearsal_v1/` + 4 M-0035 JSONs (prior evidence, untouched) |
| Remaining under `.pytest_temp` | none |
| Accepted rehearsal evidence modified | **no** (read-only; hashes unchanged) |
| Full database copy created | **no** |

---

## Verdict

**`BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_COMPLETE`**

**Disposition:** `BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FROZEN_FOR_CANONICAL_MIGRATION` (authorizes no automatic next step).

Recommended next gate: `BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1` — not begun automatically. Stopping after the freeze verdict and dedicated commit.
