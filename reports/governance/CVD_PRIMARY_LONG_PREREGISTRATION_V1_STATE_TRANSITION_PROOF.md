# CVD_PRIMARY_LONG_PREREGISTRATION_V1_STATE_TRANSITION_PROOF

**Batch:** G2-CVD-PRIMARY-LONG-PREREGISTRATION-V1
**Purpose:** Preregister (not execute) one continuous-predictor test of whether exact 300-second net taker-flow notional carries incremental predictive information for the primary LONG endpoint-return outcome, and be the first real research consumer of the M-0033/M-0034 enforcement mechanism.
**Prior checkpoint:** `EPISTEMIC_NULLIFIER_LEGACY_BYPASS_CLOSURE_V1_COMPLETE`, commits `e8576900` + `09104298`.
**Nature:** Preregistration only. No TEST outcome read, no experiment executed, no experiment_registry row written, no CVD data changed, no runtime/risk/execution file touched.
**Author:** Sonnet 5 · **Date:** 2026-07-06

---

## Identity resolution summary

Full reasoning and sourcing in `reports/research/s34/S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md` §2. Two elements required constructing a minimal, disclosed rule where no single pre-existing artifact covered the exact question (both flagged, not silently invented):

1. **The "primary LONG reversal outcome"** has no single named `experiment_id` yet — this preregistration reuses the already-accepted `classify_path`/`endpoint_return_bps` classification (verbatim across 6 downstream experiments: W5A/W6/W6RS-Confirmation/W6RS-Confound-Resolution/W7A/W10A, all reading "reused, not redefined") as its continuous source value, at the `swing_24h` horizon.
2. **The independent-cycle representative rule** (one signal per cycle for a population that has up to 5 LONG signals per cycle) has no dedicated pre-existing function; this preregistration defines "earliest eligible `signal_birth_ts` per `independent_cycle_id`", consistent with (not copied from) the one existing convention that touches this question — `ami.research.w8_short_expanded_baseline.compute_global_cycle_split`'s own cycle-ordering-by-earliest-signal_birth_ts convention.

Every other identity element (outcome definition, fee/slippage=none, signal universe, split algorithm, CVD feature column/window, known-at contract, session/day_trend_bps definitions) was located verbatim in existing, already-accepted code and re-verified against the real canonical database this session (not assumed from documentation alone).

## Population and split (computed, not read for outcomes)

| Step | Count |
|---|---|
| `ami_signal_lifecycle` total | 324 (LONG=220, SHORT=104) |
| CVD `W300` EXACT_RECONSTRUCTABLE (all directions) | 324/324 (0 SOURCE_GAPPED for this window) |
| BUCKET exclusions (104 total) touching LONG | 0 (100% SHORT) |
| LONG signals with `swing_24h` `observation_status='OK'` (effective/corrected selection) | 194 (23 MISSING_INTERNAL_GAP, 3 EXCLUDED_NO_HORIZON_DATA) |
| Distinct `independent_cycle_id` among the 220 LONG signals | 142 |
| Representative cycles after eligibility + earliest-per-cycle dedup | 131 |
| TRAIN cycles (cycle-grouped chronological 70/30, cut by count) | 91 |
| TEST cycles | 40 |

Computed using the real, canonical `ami.research.feature_gateway.fetch_lifecycle_signals` and `ami.lifecycle.path_candle_repair_correction.fetch_effective_path_observations` — the SAME functions the eventual execution code must use, not ad-hoc SQL, so this preregistration's numbers are provably reproducible by whatever runs the actual TEST later. These calls write to `researcher_exposure_ledger` by design (existing, accepted mechanism, see Protected delta below) — no outcome value (`endpoint_return_bps`/`mfe_bps`) was ever printed, read, or inspected; only `observation_status`, `signal_birth_ts`, `independent_cycle_id`, `direction`, and CVD quality/window metadata were read.

## Nullifier and gate enforcement (real, first-use)

This preregistration is the first real (non-test, non-disposable-copy) exercise of the M-0033/M-0034 mechanism against the real `data/ami/knowledge.sqlite`, following the required 12-step sequence:

1. **Family identity resolved:** `resolve_canonical_family_id("FAM_CVD_PRIMARY_LONG_REVERSAL", "H-CVD-PRIMARY-LONG-W300-EXACT-NET-TAKER-FLOW-NOTIONAL-V1")` → `FAMv1:bec99d8d36f7d6a1`.
2. **Graveyard checked:** `match_graveyard()` against the real 31-fingerprint list → 0 hits.
3. **Retry requirements:** none (clean graveyard, no token needed).
4. **Specification frozen:** spec_text sha256 `a2fd9e5b08ed2a716ac0c1cae0658740f24b48024d5b7524eb843e4441940b57` (recorded in the preregistration document, immutable once committed).
5. **Split version frozen:** `resolve_split_version(...)` → `SPLITv1:0a1b96fd74dd281e`.
6. **TRAIN cycle set frozen:** 91 cycles, hash `61486bc6…`.
7. **TEST cycle set frozen:** 40 cycles, hash `98174ed3…`.
8. **TEST-evidence nullifier computed:** `derive_test_nullifier(family_id, split_version, test_cycle_ids)` → `085397f3…`.
9. **Nullifier confirmed unused:** `SELECT COUNT(*) FROM epistemic_test_nullifiers WHERE nullifier=?` → 0 (real DB, verified).
10. **Enforced preregistration receipt created:** `issue_gate_receipt(experiment_id="E-CVD-PRIMARY-LONG-W300-PREREG-001", canonical_family_id=..., split_version=..., nullifier=..., registry_result="PREREGISTERED_NOT_EXECUTED")` → receipt hash `d46f7e2c…`, written for real to `experiment_gate_receipts` in `data/ami/knowledge.sqlite`, confirmed queryable via `has_gate_receipt()` immediately after.
11. **Nullifier NOT consumed at preregistration time** (deliberate — `consume_test_evidence()`/`register_experiment_with_gates()` were never called; the accepted protocol for THIS repository, per the W8 precedent documented in `S34_CVD_NEXT_BATCHES_PLAN_2026-07-06.md`'s own BATCH-CVD-B DoD ["prereg MD exists, registered nowhere in SQL yet"], defers registration+nullifier-consumption to actual execution time, not preregistration time). Re-verified after receipt issuance: `epistemic_test_nullifiers` row count for this nullifier = 0.
12. **No TEST outcome accessed** — confirmed throughout (see Required validations below).

**Why not the full `register_experiment_with_gates()`/`register_legacy_snapshot_with_gates()` orchestrators:** both require real `results` (metric_name/metric_value pairs) to write to `experiment_results` in the same transaction as the registry row — this preregistration has no results (no computation has been run). Calling either would either require inventing placeholder results (forbidden) or leave the orchestrator's "register now, nullifier-consume now" coupling triggered prematurely (violating item 11). Using the lower-level `epistemic_gates` functions directly (`resolve_canonical_family_id`, `resolve_split_version`, `derive_test_nullifier`, `match_graveyard`, `issue_gate_receipt`, `has_gate_receipt`) gives the exact 12-step sequence the operator specified, with the nullifier genuinely uncommitted until real execution.

## Real database state, before/during/after

| Check | Before | After |
|---|---|---|
| `data/ami/canonical.sqlite` sha256 | `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` | `fdda663dcc331053f6351d6acb7117eeb266fda5cf5d5691a799e48416be724c` (changed — see below) |
| `canonical_warehouse` schema_version | 12 | 12 (unchanged) |
| `experiment_registry` count | 22 | 22 (unchanged) |
| `experiment_results` count | 323 | 323 (unchanged) |
| Protected counts (events/signal_lifecycle/cycles/geometry) | 252/324/167/220 | 252/324/167/220 (unchanged) |
| CVD frozen counts (repaired_trades/batch_ledger/exact/proxy/exclusions/quality/EXACT_RECONSTRUCTABLE/SOURCE_GAPPED) | 40934/8/1840/1840/104/1840/1828/12 | unchanged (all re-verified) |
| `ami_lifecycle_path_observations` count | 1466 | 1466 (unchanged) |
| `integrity_check` | ok | ok |
| `data/ami/knowledge.sqlite` | schema deployed (M-0033/M-0034), 0 receipt rows | schema unchanged, **1 receipt row** (this preregistration's, non-scientific) |

**canonical.sqlite hash changed for exactly one, already-accepted reason:** `researcher_exposure_ledger` append rows from `fetch_lifecycle_signals`/`fetch_effective_path_observations` calls made during identity/population resolution — the SAME by-design exception already documented in `S34_CVD_NEXT_BATCHES_PLAN_2026-07-06.md`'s BATCH-CVD-A DoD ("canonical.sqlite hash unchanged except `researcher_exposure_ledger` append rows (known by-design exception)"). The hash shown above ("After") passed through several intermediate values during this session, each corresponding to one more accepted exposure-ledger append from a further identity-verification query (e.g. re-confirming counts) — the table records only the final value at the point this proof was written; every intermediate value differed from its predecessor by exposure-ledger appends only, verified at each step. Every OTHER table's row count is byte-for-byte/row-for-row identical before and after, confirmed by direct query, not assumed.

## Required validations (proven, read-only, this session)

- Exact and proxy populations were not pooled — only `ami_cvd_windowed_flow` (`window_id='W300'`) was queried; `ami_cvd_windowed_flow_proxy` was never opened.
- No TEST outcome was read — confirmed: no `endpoint_return_bps`/`mfe_bps` value was ever selected/printed in this session; only `observation_status`, timestamps, `independent_cycle_id`, `direction`, and CVD quality/window metadata were read.
- No experiment result was written — `experiment_results` count unchanged (323/323).
- No threshold scan was performed — no threshold exists in this design (continuous predictor, continuous outcome).
- No subgroup was selected — full eligible LONG population used.
- Known-at violations = 0 — verified `0/324` `W300` rows with `window_end_ts_ms > signal_birth_ts`; all 324 rows `known_at_classification='KNOWN_AT_SAFE'`.
- Cycle representatives are unique — 131 representative cycles = 131 distinct `independent_cycle_id` values by construction (dict keyed on cycle id).
- Split identity is frozen — `SPLITv1:0a1b96fd74dd281e`, recorded in the preregistration document and this proof.
- Outcome identity is pre-existing and unchanged — `classify_path`/`endpoint_return_bps` in `ami/lifecycle/path_metrics.py`, not modified by this batch.
- Canonical `schema_version` remains 12; canonical hash's only change is the accepted exposure-ledger exception (proven above).
- Protected runtime/risk/execution delta = 0 — `execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` not opened this session.

## Exact changed-file manifest

| File | Status | Content |
|---|---|---|
| `reports/research/s34/S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.md` | New | the preregistration artifact |
| `reports/research/s34/S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1.json` | New | machine-readable companion manifest |
| `tests/test_ami_cvd_primary_long_preregistration_v1.py` | New | focused validation tests (population/split/hash/nullifier/no-outcome-read/no-experiment-created proofs) |
| `reports/governance/CVD_PRIMARY_LONG_PREREGISTRATION_V1_STATE_TRANSITION_PROOF.md` | New | this document |

No production code was changed by this batch. No shared governance Markdown file (`SYSTEM_STATE.md`/`IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md`/`MIGRATION_LOG.md`) is included in this commit.

## New state root

| Field | Value |
|---|---|
| canonical.sqlite hash | `fdda663dcc331053f6351d6acb7117eeb266fda5cf5d5691a799e48416be724c` (exposure-ledger-only delta from `458bc07c…`) |
| canonical.sqlite schema_version | 12 (unchanged) |
| `experiment_registry` / `experiment_results` | 22 / 323 (unchanged — no experiment created) |
| Real `experiment_gate_receipts` | 1 row (`E-CVD-PRIMARY-LONG-W300-PREREG-001`, `PREREGISTERED_NOT_EXECUTED`) |
| Real `epistemic_test_nullifiers` | 0 rows (unconsumed) |
| Preregistration status | `S34_CVD_PRIMARY_LONG_PREREGISTRATION_V1`, immutable, awaiting a separate, future, explicitly-authorized execution batch |

## Remaining risks / open items for operator review

1. **Outcome-identity interpretation risk** (disclosed in the preregistration §2, item 1): no single named `experiment_id` exists yet for "the primary LONG reversal outcome" — this preregistration's choice (`endpoint_return_bps@swing_24h`, the exact source of the existing `REVERSAL` classification) is well-evidenced but is an interpretation, not a literal reuse of a pre-existing outcome_id. If the operator intends a different outcome (e.g. `mfe_bps` as primary rather than secondary), a new preregistration version is required before execution.
2. **New (not reused) representative-selection rule** (§2 item 5) — minimal and consistent with existing convention, but newly authored for this population; flagged for explicit operator sign-off before execution, same as BATCH-CVD-B's own graveyard-gate sign-off pattern.
3. Carried over, unchanged: family/split identity adapters remain text-hash-based (paraphrase-bypass risk, previously disclosed); the 10-legacy-module/`research.sqlite` closure's own residual risks are unaffected by this batch.

---

## Verdict

**`CVD_PRIMARY_LONG_PREREGISTRATION_V1_COMPLETE`**

The complete frozen specification is registered through the enforced gate (family/graveyard/split/TEST-cycle-set/nullifier/receipt all resolved and recorded for real against the live knowledge.sqlite), and zero TEST outcomes were accessed. Execution requires a new, separate operator instruction — this batch stops here.
