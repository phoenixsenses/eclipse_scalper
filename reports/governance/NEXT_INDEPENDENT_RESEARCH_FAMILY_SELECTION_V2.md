# NEXT_INDEPENDENT_RESEARCH_FAMILY_SELECTION_V2

**Gate:** BATCH-NEXT-INDEPENDENT-RESEARCH-FAMILY-SELECTION-V2
**Nature:** Governance-only, outcome-blind portfolio reconciliation. No source audit, no readiness build, no TRAIN/TEST access, no feature construction, no experiment/nullifier/gate-receipt, no route/bucket promotion.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## 0. Accepted checkpoint

| Chain | Latest commit / verdict |
|---|---|
| CVD | closed `NO_RELIABLE_ASSOCIATION`, execution `60c3e26f` |
| Absorption | closed `NO_RELIABLE_INCREMENTAL_ASSOCIATION`, execution `5e9e2e33`, closure `ba3ab906` |
| Spot-perp basis | `SPOT_PERP_BASIS_BLOCKED_BY_COVERAGE`, `1630f0a1` — 54/324 aligned anchors, 38 independent cycles |
| Book-spread dynamics | canonical migration `5267a15a` (M-0036); mixed-direction preregistration `a4722117` (INCOMPLETE, direction-mixed sign unresolvable); LONG preregistration `93b7296d` (INCOMPLETE, `PARKED_FOR_SAMPLE_GROWTH` — TEST=18<20, need eligible n≥67) |
| Governance state | `schema_version=14`, `experiment_registry=24`, `experiment_results=381`, `epistemic_test_nullifiers=2`, `experiment_gate_receipts=2` |

## Operator ruling: book-spread LONG parking (recorded, not reopened here)

`FAM_BOOK_SPREAD_DYNAMICS` LONG (`H-BOOK-SPREAD-CHANGE-BPS-W300-LONG-V1`) is parked for sample growth. This batch does not touch it, does not recompute its population, and does not access any outcome for it.

---

## Phase 1 — Authoritative roadmap resolution

**`reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md` (commit `0c976e21`) remains the sole authoritative roadmap.** No superseding artifact exists (`find . -iname "*FAMILY_SELECTION*" -o -iname "*HYPOTHESIS_SELECTION*"` returns only this V1 document and this V2 batch's own new artifacts). `FAILURE_ARCHIVE.md` was checked for any `SATISFIED`/`RETRY_MET`/`CONDITION_MET` marker on a previously-graveyarded retry condition — **none found**. `OPERATOR_DECISION_QUEUE.md` OD-006/OD-012/OD-014/OD-016/OD-017 were re-read — all remain `OPEN` or `IMPLEMENTED`-as-deferred with no status change since V1 was written. `S34_CVD_NEXT_BATCHES_PLAN_2026-07-06.md`'s own BATCH-CVD-A/B/C sequence remains **plan-only, never built** (no `IMPLEMENTED`/`BUILT`/`EXECUTED` token found in that file).

`SYSTEM_STATE.md` (ends §97, 2026-07-06) and `IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md` do not yet contain entries for the G2-CVD execution, absorption chain, spot-perp basis readiness, or the full book-spread chain (a pre-existing, disclosed bookkeeping gap carried forward from V1's own §"Unresolved uncertainties" item 2 — not this batch's to close). Direct commit/report evidence for each chain (readiness/rehearsal/freeze/migration/preregistration/execution/closure artifacts, and the real `canonical.sqlite`/`knowledge.sqlite` state) was used as authoritative in every case, per the gate's own precedence rule ("repository state wins").

---

## Phase 2-3 — Portfolio status matrix and eligibility

18 candidates reconciled: the 15 non-shortlisted candidates from V1's Phase 1 inventory, plus the 3 V1-shortlisted candidates (Absorption, Basis, Spread) that have since been resolved.

| Family / candidate | Rank | Status | Evidence |
|---|---|---|---|
| `FAM_CASCADE_ABSORPTION_IMPACT` | 1 | `SCIENTIFICALLY_CLOSED` | execution `5e9e2e33`, closure `ba3ab906` |
| `FAM_SPOT_PERP_BASIS_REVERSAL` | 2 | `BLOCKED_BY_COVERAGE` | `1630f0a1`, 54/324, 38 cycles |
| `FAM_BOOK_SPREAD_DYNAMICS` | 3 | `PARKED_FOR_SAMPLE_GROWTH` | M-0036 `5267a15a`; mixed `a4722117`; LONG `93b7296d` |
| CVD/taker-flow W300 LONG | — | `SCIENTIFICALLY_CLOSED` | `60c3e26f` |
| CVD other windows (BATCH-CVD-A/B/C) | — | `DUPLICATE_OR_NONINDEPENDENT` | plan-only, forbidden follow-up |
| OFI momentum | — | `GRAVEYARDED` | graveyard fingerprint |
| OFI event-anchored non-momentum | — | `GRAVEYARDED` | same fingerprint family, deprioritized |
| Pull/refill liquidity | — | `GRAVEYARDED` | failure_archive id=6, retry unmet |
| Funding level/velocity | — | `BLOCKED_BY_SOURCE_QUALITY` | OD-006 OPEN, both collectors dead |
| Open interest | — | `BLOCKED_BY_COVERAGE` | OD-012, 38/252 (15%), unresolved |
| Cross-asset transfer | — | `GRAVEYARDED` | graveyard id=9, NO_EDGE |
| Cross-exchange (new collector) | — | `BLOCKED_BY_SOURCE_QUALITY` | no collector exists |
| LONG-anchor event asymmetry | — | `BLOCKED_BY_COVERAGE` | OD-017, `ami_events` 100% SELL-cascade |
| Entry timing/hold-exit/MFE-MAE/reversal/regime | — | `SCIENTIFICALLY_CLOSED` | W3/W4/W7A/W8/W10A already answered |
| Failed breakouts/sweep-retest/exhaustion | — | `BLOCKED_BY_SOURCE_QUALITY` | OD-014 OPEN, infra not implemented |
| Forward/shadow validation | — | `ACTIVE_GATE_IN_PROGRESS` | already accumulating, leave alone |
| Birth-truncated cascade geometry | — | `BLOCKED_BY_SOURCE_QUALITY` | source dead, operator-excluded |
| Pre-cascade dip-recovery | — | `GRAVEYARDED` | failure_archive id=22, retry unmet |

**Eligible candidates: 0 of 18.**

---

## Phase 4 — Selection rule applied

No candidate has status `UNTOUCHED_ELIGIBLE`. The three V1-shortlisted, originally-ranked candidates — the only ones ever cleared for a readiness/prereg gate — are each now resolved to a non-selectable state:

- **Absorption (rank 1):** family-level scientific closure. No rescue permitted (alternate windows/thresholds/bins/subgroups/sessions/regimes/interactions/alternate outcomes/repeat TEST access all explicitly prohibited by its own closure).
- **Basis (rank 2):** coverage-blocked, not closed — retryable once forward spot coverage grows, but not now.
- **Spread (rank 3):** parked for sample growth on its only retryable child (LONG); its mixed-direction child is closed INCOMPLETE (a structural sign-resolution impossibility, not a data shortfall — not retryable without a new authorized population-scoping decision); SHORT is explicitly not authorized.

No lower-ranked (originally-excluded) candidate has had its blocking condition satisfied since V1: funding and cross-exchange remain source-dead; open interest's coverage recheck is itself forbidden inside a selection-only batch (would require an anchor-coverage join); the LONG-anchor asymmetry population remains structurally 0/252; the graveyarded candidates (OFI, pull/refill, cross-asset-transfer, pre-cascade dip-recovery) have no satisfied retry condition on record; the already-answered families (entry timing, hold/exit, MFE/MAE, reversal, regime) have no fresh un-fished sub-question identified; failed-breakouts/exhaustion remain infra-blocked; forward/shadow validation is intentionally left running, not a new-selection target.

**No profitability, remembered result, or outcome behavior was used to rank or exclude any candidate** — every status above is copied verbatim from an already-accepted, already-committed closure/block/graveyard record.

---

## Phase 5 — Family distinctness (not applicable — no family selected)

Not performed: no family was selected, so there is no new candidate requiring a distinctness argument against CVD/Absorption/Basis/Spread/liquidation-geometry/day-trend/funding/OFI/depth-refill-pull.

## Phase 6 — Source-path feasibility screen (not applicable)

Not performed for the same reason. No bounded metadata screen was run against any table for a "proposed selected family," since none exists this batch.

## Phase 7 — Graveyard/retry/exposure (performed at the portfolio level, Phase 2-3 above)

Every graveyarded/blocked candidate's exact retry condition is listed in the JSON companion's `retry_conditions` array (also reproduced in Phase 9 below). No outcome value was read for any of these checks — all are status-string lookups against already-recorded artifacts.

---

## Phase 8 — Freeze (no family frozen)

**No canonical family ID, name, predictor, window, formula, direction, outcome, split, model, controls, nullifier, or gate receipt is frozen by this batch.** There is nothing to freeze — the disposition is that no family is currently eligible.

## Phase 9 — Next controlled gate (deferred)

**No next readiness gate is defined by this batch**, since none of the 18 tracked candidates is currently eligible to receive one. The next controlled action is **not** a readiness gate but one of the retry conditions below becoming satisfied, most plausibly (in order of proximity):

1. **`FAM_BOOK_SPREAD_DYNAMICS` LONG** — the closest to unblocking: needs 9 more eligible LONG independent cycles (current 58, need ≥67) under the frozen 70/30 split. This accrues automatically as new signals are collected and their `swing_24h` path observations mature — no new engineering required, only time/data.
2. **`FAM_SPOT_PERP_BASIS_REVERSAL`** — needs sufficient new forward spot-price coverage growth to raise the aligned-anchor count past the frozen minimum.
3. **`OPEN_INTEREST`** — needs its own dedicated, separately-authorized anchor-coverage recheck (the last measurement, 15%, is 3+ days stale and this batch is not permitted to refresh it).

Full retry-condition table for all 18 candidates is in the JSON companion.

---

## Phase 10-11 — No outcome access / immutability proof

| Check | Result |
|---|---|
| TRAIN outcome reads | 0 |
| TEST outcome reads | 0 |
| All outcome-value reads | 0 |
| Experiment/result/nullifier/gate-receipt creation | 0 |
| Schema change | 0 |
| Canonical data migration | 0 |
| Feature-row creation | 0 |
| Route/bucket promotion | 0 |
| Runtime/risk/execution/paper/shadow/forward/live delta | 0 |
| `canonical.sqlite` sha256 | `0604b0da93238388451eb23203e1b12806f6e627d4d599168877e1abcb8d57a0` (unchanged) |
| `knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged) |
| `schema_version` / `experiment_registry` / `experiment_results` / nullifiers / receipts | 14 / 24 / 381 / 2 / 2 (all unchanged) |
| M-0036 rows, book-spread INCOMPLETE artifacts, prior family artifacts | unchanged (confirmed by dedicated tests) |

## Phase 12 — Focused tests

`tests/test_ami_governance_next_independent_research_family_selection_v2.py` — **22/22 passed**. Covers: candidate-count/no-duplicates, status/reason/evidence completeness, enum-membership, per-candidate classification (absorption closed, basis coverage-blocked not graveyarded, spread parked distinct from closed, CVD-alt-windows flagged duplicate, forward/shadow active, graveyard tagging), zero-eligibility proof, deterministic disposition, no-profitability-field structural guard, rank-restricted-to-shortlist proof, retry-condition completeness, no-`execute()`/no-`sqlite3`-import AST guards (the module never opens a database), and real-DB/prior-artifact immutability.

## Phase 13 — Regression

Additive-only batch (one new pure-Python module with zero database access, one new test file). Touches no schema, no shared governance-write path, no other family's code. Established 18-pre-existing-failure baseline unaffected (not re-run in full — nothing in this batch's write-set overlaps with any test that pins governance counts or schema version).

## Storage report

No temporary database created — the module opens no connection at all (proven by AST guard). Real-DB reads in the test file are `mode=ro` hash/count checks only. No `microstructure.db` copy. Peak temporary disk usage: 0 (pytest's own bytecode cache aside).

---

## Verdict

**`NEXT_INDEPENDENT_RESEARCH_FAMILY_SELECTION_V2_COMPLETE`**

**Disposition: `NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY`**

Portfolio accounting is exact: 18 tracked candidates, 0 eligible, every status and evidence citation verified against an already-accepted repository record. No new readiness gate is opened. The nearest path forward is passive time/data accrual on the parked `FAM_BOOK_SPREAD_DYNAMICS` LONG child (9 more eligible cycles needed), not new engineering.

Stopping after this selection accounting. No readiness gate begins without new, separate operator instruction.
