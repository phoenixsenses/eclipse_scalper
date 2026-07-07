# SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF

**Batch:** SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1
**Purpose:** Determine whether the repository contains sufficient exact, known-at-safe bid/ask evidence to construct a canonical spread expansion/compression feature family (`FAM_BOOK_SPREAD_DYNAMICS`) on the existing signal-anchor universe.
**Prior checkpoint (unchanged, not reopened):** commit `1630f0a1` (`SPOT_PERP_BASIS_BLOCKED_BY_COVERAGE`), which itself followed `ba3ab906` (`ABSORPTION_IMPACT_EXECUTION_TEST_STATE_CLOSURE_COMPLETE`). `schema_version=13`, `experiment_registry=24`, `experiment_results=381`, `epistemic_test_nullifiers=2`, `experiment_gate_receipts=2`.
**Nature:** Readiness, source audit and scientific-contract only. No TRAIN/TEST outcome access, no experiment, no gate receipt, no nullifier, no preregistration, no migration, no route promotion, no runtime/risk/execution/paper/shadow/forward/live change, no unrelated legacy-regression repair.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Family identity:** read `NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md` (commit `0c976e21`) — `FAM_BOOK_SPREAD_DYNAMICS`, candidate 3 (ranked 47/60), hypothesis verbatim. No conflicting identity exists; not invented.
2. **Graveyard/prior-exposure:** scanned `graveyard_slash_fingerprints` (0 hits), `knowledge` (0 spread KOs; book-pull/refill are depth, not spread), `failure_archive` (0 direct; 1 incidental MFE50-giveback mention ruled non-blocking). Clean.
3. **Source audit (read-only, index-backed):** `book_ticker` schema, indexes, and MIN/MAX coverage per symbol (`SELECT MIN/MAX(ts_ms) WHERE symbol=?` — index seeks, never a full scan). Coverage begins 2026-04-11T17:08:42.005Z. Confirmed venue (Binance perp websocket) and single-`ts_ms` receipt-time semantics.
4. **Data-quality audit (bounded):** examined the actual quote at all 324 anchors + a 598,261-quote sample across 5 anchors' pre-birth 5-minute windows. Found the material duplicate-`ts_ms` property (~75% of rows, ~6.5% of collisions with differing bid/ask) requiring an `id`-tie-break; 0 crossed/locked/zero/out-of-order.
5. **New audit module:** `ami/research/spread_dynamics_readiness_audit.py` — pure, read-only, outcome-blind: deterministic at-or-before quote selection with `id`-tie-break, spread/mid formula, quality classification (immutable codes), level-at-birth accounting, windowed-pair accounting, known-at and duplicate-cycle verification. Reuses `FEED_LIMITS["book_ticker"]=5min`.
6. **Focused tests first:** `tests/test_ami_research_spread_dynamics_readiness_audit.py` (25 tests). One iteration: a redundant second outcome-guard test matched the module docstring's prose ("selects any outcome column") — fixed to only inspect string literals that *begin* with a SQL verb (real statements), matching the established false-positive fix pattern. **25/25 passed.**
7. **Anchor accounting (outcome-blind):** 324 → 196 `EXACT_RECONSTRUCTABLE` / 22 `STALE_SOURCE` / 106 `UNAVAILABLE_BEFORE_COLLECTION` (reconciles), collapsing to 97 independent cycles. Windowed-pair coverage window-invariant (97 cycles at every candidate window).
8. **Verdict:** `SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS` — data is sufficient (not coverage- or quality-blocked), but the family's expansion/compression concept has an unresolved feature-form + baseline-window that is not selectable outcome-blind and requires an operator ruling before rehearsal.

---

## Why this batch could rule on readiness without touching outcome data

Every fact was obtained via `mode=ro` SQL over identity columns (`signal_id`/`direction`/`independent_cycle_id`/`signal_birth_ts`/`source_event_id` from `ami_signal_lifecycle`) and quote columns (`ts_ms`/`id`/`bid_price`/`ask_price`/`bid_qty`/`ask_qty` from `book_ticker`). No query in this batch, the module, or the tests ever named `ami_lifecycle_path_observations`, selected `endpoint_return_bps`/`mfe_bps`/`mae_bps`, or touched any experiment/nullifier/gate-receipt table — proven by the committed AST-based static guards over the module source. `book_ticker` was opened `mode=ro` throughout; every query was bounded (index-backed MIN/MAX, per-anchor at-or-before seeks, or a small bounded pre-birth-window range) — **no full-table copy or full-table scan** of the ~2×10⁹-row table.

## Real database state — unchanged (proof)

| Check | Before this batch | After this batch |
|---|---|---|
| `data/ami/canonical.sqlite` sha256 | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` | `3aefce833a67b8d43b841619f97667a56e182822e167aa606320ca8c52043d59` (unchanged) |
| `data/ami/knowledge.sqlite` sha256 | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` | `710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (unchanged) |
| `experiment_registry` | 24 | 24 |
| `experiment_results` | 381 | 381 |
| `schema_version` | 13 | 13 |
| `epistemic_test_nullifiers` | 2 | 2 |
| `experiment_gate_receipts` | 2 | 2 |
| `execution/`, `risk/`, `brain/`, `.env`, live executor | untouched | untouched |

Both real DB files are **byte-for-byte identical** before and after this batch (every connection was `mode=ro`, structurally incapable of writing). `microstructure.db` was likewise `mode=ro` and never copied.

## Required validations (proven)

- No TRAIN outcome access · No TEST outcome access · No spread-feature/outcome join
- No experiment creation · No experiment-result creation
- No nullifier creation · No nullifier consumption
- No gate-receipt creation or update
- No canonical migration · No schema-version change (remains 13)
- No route/bucket promotion
- No runtime/risk/execution delta · No shadow/paper/forward/live change
- No unrelated legacy-regression repair
- `canonical.sqlite`/`knowledge.sqlite` hashes unchanged (full, non-truncated, above)
- `experiment_registry` remains **24** · `schema_version` remains **13**

## Exact changed/added-file manifest (this commit)

| File | Status |
|---|---|
| `reports/research/s34/S34_SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1.md` | New |
| `reports/research/s34/S34_SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1.json` | New |
| `reports/governance/SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF.md` | New |
| `ami/research/spread_dynamics_readiness_audit.py` | New |
| `tests/test_ami_research_spread_dynamics_readiness_audit.py` | New |

Not included: rehearsal output, migration code, schema changes, preregistration, TEST execution, unrelated cleanup, unrelated governance projections, runtime changes, or another research family.

## Focused test results

`tests/test_ami_research_spread_dynamics_readiness_audit.py` — **25/25 passed**. Coverage: spread/mid formula correctness; crossed/zero/negative formula rejection; at-or-before quote selection; no-future-quote; boundary-inclusive selection; **duplicate-`ts_ms` deterministic `id`-tie-break**; unavailable/crossed/zero/locked/stale/exact classification with boundary cases; real-data reconciliation (196/22/106, 97 cycles); idempotent rebuild; known-at cleanliness; duplicate-cycle dedup; window-invariant windowed-pair coverage; symbol/venue/segment identity; no-proxy-tier taxonomy; and two AST-based outcome/governance access-denial guards.

## Regression policy

The accepted deterministic baseline (1,027 collected / 1,013 passed / 14 narrowly waived pre-existing failures) is **not perturbed** by this batch: it adds only two new files (a read-only audit module with no import side effects, and its own 25-test file), touches no existing code or test, and makes no DB write. A full paired regression sweep was **not** rerun for this readiness-only, additive batch (no production-code or shared-state change exists that could shift the baseline); the new file's own 25 tests are green. The 14 waived pre-existing failures remain exactly as documented in the M-0035 waiver (`5ab89f63`); no new deterministic failure is introduced, and none is hidden under that waiver. Any mutable live-collector health check remains a separate concern, not part of this deterministic accounting.

## Storage guardrail

| Item | Value |
|---|---|
| Full database copies created | 0 |
| Full-table scan/copy of `book_ticker` (~2×10⁹ rows) | never (index-backed MIN/MAX + bounded per-anchor seeks only) |
| Large temp files under OS `C:` temp | 0 |
| `D:\eclipse_scalper\.runtime_temp` usage this batch | one transient snapshot JSON (deleted) during audit; no retained temp file |
| Peak temporary disk usage | <10 KB |
| Files retained | none beyond the 5 committed files |
| Files deleted | the transient audit snapshot JSON |
| Remaining under `.runtime_temp` | unchanged from the prior checkpoint (`absorption_impact_rehearsal_v1/` + the 4 M-0035 evidence JSONs) |
| Remaining under `.pytest_temp` | none |

---

## Verdict

**`SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS`**

Data is sufficient (97 independent cycles, TEST ≈ 30 ≥ MIN_BUCKET_N=20; exact L1, 0 anomalies at anchors, deterministic quote selection) — this is a **definition** stop, not a data stop. The family's expansion/compression concept has an unresolved feature-form (level/ratio/log-ratio/difference/z-score) and a baseline window that is coverage-indistinguishable across candidates and therefore not selectable outcome-blind. One operator ruling on feature-form + window is required before a disposable-rehearsal gate opens.

Stopping after the readiness verdict. No disposable rehearsal, row-accounting freeze, migration, preregistration, TEST execution, or bucket construction begins without new, separate operator instruction.
