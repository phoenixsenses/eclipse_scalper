# SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF

**Batch:** SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1
**Purpose:** Determine whether the repository contains enough source-quality, known-at-safe spot and perpetual market data to construct a canonical `FAM_SPOT_PERP_BASIS_REVERSAL` feature family on the existing signal-anchor population.
**Prior checkpoint (unchanged, not reopened):** commit `ba3ab906` (`ABSORPTION_IMPACT_EXECUTION_TEST_STATE_CLOSURE_COMPLETE`), `schema_version=13`, `experiment_registry=24`, `experiment_results=381`.
**Nature:** Readiness and contract only. No TRAIN/TEST outcome access, no experiment, no nullifier, no preregistration, no migration, no route promotion, no runtime/risk/execution/shadow/paper/forward/live modification.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Family identity resolution:** read `reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md` (commit `0c976e21`) — candidate 2 already defines `FAM_SPOT_PERP_BASIS_REVERSAL` verbatim, with its hypothesis text, predictor/outcome/population/controls pattern, and explicit prerequisite blockers (level-vs-slope design decision). Not invented this batch.
2. **Graveyard/Knowledge-Object check:** `match_graveyard()`-equivalent full-table scan of `graveyard_slash_fingerprints` (31 curated) and `failure_archive` (22 rows) for `basis`/`spot`/`arb`/`funding`/`perp` stems — 0 hits. Read all 11 `knowledge` table rows — no existing Knowledge Object for spot-perp basis; `K-S34-FUNDING-LEVEL-001` identified and ruled a distinct, adjacent (not identical) mechanism.
3. **Source audit (read-only, live):** enumerated `microstructure.db` tables; inspected schema and coverage of `spot_prices`, `mark_prices`, `funding_rates`; confirmed venue identity from `data/oi_spot_poller.py` source code (`api.binance.com` for spot, `fapi.binance.com` for perp — same exchange, different market segment, not a cross-venue mismatch). Discovered the poller's own header comment discloses the `spot_prices` producer died 2026-06-05 — independent, first-party confirmation of the gap measured empirically in step 5.
4. **Ungoverned-precedent audit:** located `tools/s34_mechanism_feature_store.py`'s `basis_spot_bps`/`basis_spot_slope` formula (reused as a mechanical shape precedent only) and the unrelated `basis_reversion_candidates` table (1,430 rows, real `long_return`/`long_win` outcome columns, belonging to a different exploratory script — **never queried for its outcome columns by this audit**).
5. **New audit module:** `ami/research/spot_perp_basis_readiness_audit.py` — pure, read-only, outcome-blind functions: `fetch_anchor_universe` (identity columns only), `fetch_sorted_timestamps`, `nearest_at_or_before` (structurally incapable of returning a future timestamp), `inter_sample_gap_stats`, `classify_signal_spot_coverage`, `anchor_accounting`, `verify_no_lookahead`, `verify_duplicate_cycle_free`. Reuses the existing `ami.states.engine.FEED_LIMITS["spot_prices"]=10min` staleness convention rather than inventing a new one.
6. **Focused tests first:** `tests/test_ami_research_spot_perp_basis_readiness_audit.py` (21 tests: pure-function timestamp-alignment/boundary tests on synthetic data, real-data anchor-accounting reconciliation and idempotency, symbol/venue-identity scoping, exact/proxy-taxonomy checks, an AST-based static guard proving no `.execute()`-style call anywhere in the module ever names the outcome table/columns, known-at re-verification, duplicate-cycle collapsing, and gap-statistics sanity including a direct assertion that the known ~27-day outage is not silently smoothed away). One iteration required: a naive substring-based outcome-reference guard produced a false positive against the module's own explanatory docstring (identical false-positive class already found and fixed in the absorption-impact rehearsal batch, `fc43e972`) — replaced with the same AST-based, execute-call-scoped approach used there. **21/21 passed** after the fix.
7. **Anchor accounting (real, outcome-blind):** 324 total anchors; 49 `SOURCE_ABSENT_BEFORE_COLLECTION`, 221 `SOURCE_STALE_BEYOND_HEALTHY_AGE`, 54 `EXACT_RECONSTRUCTABLE` (reconciles exactly). The 54 fresh rows collapse to 38 independent cycles. Traced the dominant cause to one ~26.97-day collector outage (2026-06-05T15:59:11.295Z → 2026-07-02T15:12:58.399Z), affecting 159 anchors directly, cross-confirmed against the collector script's own disclosed incident.
8. **Known-at proof:** re-verified 0 known-at violations across all 324 anchors, both legs (spot and perp), reproduced independently twice in this session (byte-identical accounting both times).
9. **MIN_BUCKET_N check:** 38 independent cycles → a cycle-grouped 70/30 split (the same algorithm every prior W-series/CVD/absorption family uses) would leave a TEST fold (~11 cycles) already below the `MIN_BUCKET_N=20` convention this codebase applies uniformly, before any outcome-eligibility gate is even applied.
10. **Readiness verdict:** `SPOT_PERP_BASIS_BLOCKED_BY_COVERAGE` — the existing anchor population's spot-leg coverage is inadequate; perp-leg data is not the problem.

---

## Why this batch could rule on readiness without touching outcome data

Every fact above was obtained via `mode=ro` SQL against `signal_id`/`direction`/`independent_cycle_id`/`signal_birth_ts`/`source_event_id` (from `ami_signal_lifecycle`) and `ts_ms`/`spot_price`/`mark_price`/`funding_rate` (from `microstructure.db`). No query in this batch, in the new module, or in the focused tests ever named `ami_lifecycle_path_observations` or selected `endpoint_return_bps`/`mfe_bps`/`mae_bps` — proven both by direct review of every script executed this session and by the committed test file's AST-based static guard (`test_module_never_executes_sql_naming_the_outcome_table`), which parses the module's source and inspects only string-literal arguments passed to `.execute()`-family calls.

## Real database state — unchanged (proof)

| Check | Before this batch | After this batch |
|---|---|---|
| `data/ami/canonical.sqlite`: `experiment_registry` | 24 | 24 (unchanged) |
| `data/ami/canonical.sqlite`: `experiment_results` | 381 | 381 (unchanged) |
| `data/ami/canonical.sqlite`: `schema_version` | 13 | 13 (unchanged) |
| `data/ami/knowledge.sqlite`: `epistemic_test_nullifiers` | 2 | 2 (unchanged) |
| `data/ami/knowledge.sqlite`: `experiment_gate_receipts` | 2 | 2 (unchanged) |
| `execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` | untouched | untouched |

No file hash comparison beyond the counts above is needed for `canonical.sqlite`/`knowledge.sqlite`: every connection this batch opened to those two files was `mode=ro`, structurally incapable of writing. `microstructure.db` was also opened `mode=ro` throughout — every query was a bounded, column-scoped read (identity/timestamp/price columns only), never a full-table copy of `spot_prices`/`mark_prices` (only the `ts_ms` column was materialized in memory for gap/alignment statistics, via `fetch_sorted_timestamps`).

## Exact changed/added-file manifest (this commit)

| File | Status | Content |
|---|---|---|
| `reports/research/s34/S34_SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1.md` | New | full readiness/contract report (source audit, formula, feature contract, anchor accounting, known-at proof, family-distinctness, verdict) |
| `reports/research/s34/S34_SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1.json` | New | machine-readable companion |
| `reports/governance/SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF.md` | New | this document |
| `ami/research/spot_perp_basis_readiness_audit.py` | New | narrowly-scoped, read-only, outcome-blind audit module (no schema, no data write) |
| `tests/test_ami_research_spot_perp_basis_readiness_audit.py` | New | 21 focused tests |

Not included: migration code, preregistration artifacts, TEST results, shared unrelated governance-projection changes, runtime modifications, or repository-wide cleanup.

## Storage guardrail compliance

| Item | Value |
|---|---|
| Full database copies created | 0 |
| Large temp files created under `C:\Users\...\AppData\Local\Temp` | 0 |
| `D:\eclipse_scalper\.runtime_temp` usage | one small (~5KB) diagnostic snapshot JSON (`spot_perp_basis_audit_snapshot.json`), created to cross-check report figures against a single consistent capture, then deleted |
| `D:\eclipse_scalper\.pytest_temp` usage | pytest `--basetemp` fixture scratch under the OS session scratchpad, not the repo; cleaned up automatically |
| Full `microstructure.db` copy | never made |
| Peak temporary disk usage | <5KB |
| Files retained | none beyond the 5 committed files above |
| Files deleted | `spot_perp_basis_audit_snapshot.json` |
| Remaining under `.runtime_temp` | unchanged from the prior checkpoint (`absorption_impact_rehearsal_v1/` + the 4 M-0035 evidence JSONs) |
| Remaining under `.pytest_temp` | none |

## Required validations (proven)

- TEST/outcome reads: **0**
- New experiment count: **0**
- New nullifier count: **0**
- Existing nullifier consumed: **none**
- Preregistration created: **none**
- Migration performed: **none**
- Route or bucket promoted: **0**
- Runtime/risk/execution delta: **0**
- `experiment_registry`/`experiment_results`/`schema_version`: unchanged (24/381/13)
- Known-at violations: **0** (re-verified, reproducible, idempotent)
- Duplicate cycle representatives: **0** (verified structurally, both synthetic and real-data cases)

---

## Verdict

**`SPOT_PERP_BASIS_BLOCKED_BY_COVERAGE`**

Basis: 270/324 anchors (83.3%) lack an adequate spot price at `signal_birth_ts` under the pre-existing, non-invented 10-minute staleness tolerance (`FEED_LIMITS["spot_prices"]`), dominated by one ~27-day collector outage independently disclosed in the collector's own source code. The 54 fresh anchors collapse to only 38 independent cycles — below any usable TRAIN/TEST split threshold this codebase applies uniformly. Perp-side data is exact and not the limiting factor.

Stopping after readiness. No disposable rehearsal, preregistration, or migration begins without new, separate operator instruction.
