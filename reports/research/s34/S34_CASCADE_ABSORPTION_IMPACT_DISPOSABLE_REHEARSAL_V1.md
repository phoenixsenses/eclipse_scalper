# S34_CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-DISPOSABLE-REHEARSAL-V1
**Nature:** Feature reconstruction, quality validation, and row-accounting rehearsal only. No preregistration, no experiment ID, no nullifier action, no TEST/outcome access, no performance calculation, no canonical migration, no schema_version change, no runtime/risk/execution modification.
**Depends on (source of truth, unedited):** `S34_CASCADE_ABSORPTION_IMPACT_READINESS_AUDIT_V1.md`/`.json`, `S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_BRIDGE_CONTRACT_V1.md`, `CASCADE_ABSORPTION_IMPACT_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF.md` (all commit `fc1321f5`).
**Date:** 2026-07-07 · **Author:** Sonnet 5

No contract ambiguity was found that blocked deterministic implementation (see §"Contract ambiguity check" below) — the rehearsal proceeded in full.

---

## Implementation identity (Phase 1)

The single frozen `PROPOSED_NOT_YET_ACCEPTED` formula, implemented **exactly** as contracted, no alternative formulas added, no multi-definition comparison performed:

```
price_response_per_signed_notional_w =
    mark_price_return_bps([T-W, T]) / max(|signed_notional_w| / 1_000_000, FLOOR_USD_M)

signed_notional_w = sum(taker_sign(trade) * notional(trade)) over trades in [T-W, T]
taker_sign(trade) = +1 if is_buyer_maker=0 (taker BUY), -1 if is_buyer_maker=1 (taker SELL)
```

| Element | Value |
|---|---|
| Numerator | `mark_price_return_bps = (mark_price(T) - mark_price(T-W)) / mark_price(T-W) * 1e4`, both prices fetched via "nearest at-or-before" (never a future mark price) |
| Denominator | `max(\|signed_notional_w\| / 1,000,000, FLOOR_USD_M)` — **absolute value**, exactly as the frozen contract text specifies (not a signed denominator) |
| Units | bps of mark-price return per $1,000,000 net signed aggressive notional |
| Sign convention | `is_buyer_maker=0 -> +1` (taker BUY), `is_buyer_maker=1 -> -1` (taker SELL) — identical to `ami/cvd/windowed_taker_flow.py` |
| Window boundaries | `[T-W, T]`, both ends **inclusive** (verified by `test_boundary_timestamps_both_inclusive`) |
| Anchor timestamp | `T = signal_birth_ts` (from `ami_signal_lifecycle`, reused verbatim, no new identity) |
| Price source | `microstructure.db:mark_prices`, "at-or-before" convention identical to `ami.states.engine.StateEngine._px` |
| Signed-notional source | `microstructure.db:agg_trades` (native), merged with `data/ami/canonical.sqlite:ami_agg_trades_repaired` where a window intersects one of its 8 repair spans (repaired row wins on any content conflict for the same `agg_trade_id`) |
| Feature-availability timestamp | `feature_available_ts_ms = signal_birth_ts` (schema `CHECK`-constrained) |
| Latency allowance | none — pre-birth only, `window_end_ts_ms = signal_birth_ts` (schema `CHECK`-constrained) |
| Zero-denominator policy | floored at `FLOOR_USD_M`, never divides by exactly zero, never crashes (`test_zero_signed_notional_floor_applied_no_crash`) |
| Near-zero-denominator policy | same floor applies continuously (`test_near_zero_denominator_floor_applied`) |
| Missing-source policy | a signal/window with zero trades in range still computes (signed_notional=0, floor applied) — never excluded for missingness alone; a signal/window that fails a **gap/coverage** check is excluded with an immutable reason code (§ Exclusions) |
| Duplicate policy | dedup by `agg_trade_id`; native + repaired both present for the same id → repaired value wins (matches CVD repair precedent) |
| Numerical precision | IEEE-754 double throughout (Python `float`); no intermediate rounding; reported figures below rounded only for display |
| Deterministic rounding | none applied to stored values; two independent runs reproduce byte-identical content (see Idempotency, below) |

### `FLOOR_USD_M` — frozen this batch

Per the contract's own deferred-derivation clause: computed from the real `|signed_notional|` distribution at the smallest window (W=60s, the most binding case), across all 324 signals (0 excluded at that window): min=$37,006.70, 1st-percentile=$93,985.10, max=$78,936,758.21. `FLOOR_USD_M = 0.01` ($10,000) — the 1st percentile rounded down to one significant figure, expressed in $M. **This floor never actually bound on any of the 1,619 usable real rows in this rehearsal** (`floor_applied_rows = 0`, see Row Accounting) — it is a pure safety guard against a hypothetical future near-zero-flow signal, not an active adjustment to any real value observed.

### Multiple fixed windows — no "best" window chosen

All five contracted windows (`{60, 300, 600, 1800, 3600}` seconds) were constructed as five **separately identified** representations (`window_id` = `W60`/`W300`/`W600`/`W1800`/`W3600`), each with its own row in `absorption_impact_windowed_flow`/`absorption_impact_window_quality_v1`/`absorption_impact_exclusions`. No comparison, ranking, or selection among them was performed.

### Contract ambiguity check

No blocking ambiguity was found. The one item requiring a decision (`FLOOR_USD_M`'s numeric value) was explicitly pre-authorized by the contract itself to be derived mechanically in this stage ("value to be chosen at rehearsal time from the real signed-notional distribution... documented with its derivation") — this is not an unresolved ambiguity, it is delegated work, completed above.

---

## Disposable environment (Phase 2)

| Field | Value |
|---|---|
| Location | `D:\eclipse_scalper\.runtime_temp\absorption_impact_rehearsal_v1\` |
| Contents | `rehearsal_run1.sqlite` (1,257,472 bytes), `rehearsal_run2.sqlite` (1,257,472 bytes), `manifest.json`, `rehearsal_result.json` |
| Source connections | `mode=ro` throughout — `data/ami/canonical.sqlite` (`ami_signal_lifecycle`, `ami_agg_trades_repaired`), `data/microstructure.db` (`agg_trades`, `mark_prices`, `gaps`) |
| Live canonical writes | **zero** (proven below) |
| `microstructure.db` full copy | **never made** — every query is a bounded `WHERE symbol=? AND ts_ms BETWEEN ? AND ?` range scan against the existing `idx_trade_symbol_ts`/`idx_mark_symbol_ts` indexes; 1,620 signal×window pairs processed in 270.29s (run 1) / 169.82s (run 2) |
| Manifest fields recorded | contract commit (`fc1321f5`), contract MD sha256, readiness-audit MD sha256, code sha256, source DB paths + `canonical.sqlite` sha256, disposable-output paths + sha256, creation timestamp (UTC), reproducibility flag, retention decision |
| Reproducible | Yes — proven by two independent runs (below) |
| Retention | **Retained as accepted immutable rehearsal evidence** pending operator/gate acceptance (not deleted) — 2.5 MB total |

---

## Anchor universe (Phase 3)

Reused verbatim from `ami_signal_lifecycle`: 324 signals (LONG 220 / SHORT 104), `signal_birth_ts`, `independent_cycle_id`, `source_event_id`, `direction`, `symbol` (100% ETHUSDT) — no outcome-dependent eligibility rule of any kind. Every rehearsal row's primary key is content-derived (`sha256(symbol|signal_id|window_id|version)[:24]`, prefixed `AIF-`), verified unique by the schema's own `UNIQUE` constraints (violated inserts would raise `sqlite3.IntegrityError`, none did). No signal/window pair was ever excluded because its feature value looked extreme or inconvenient — the only exclusion in the entire rehearsal (§ Exclusions) is a pre-existing, independently-documented `agg_trades` collector gap, identified **before** any feature value was computed for that row.

---

## Source reconstruction (Phase 4)

For every one of the 1,620 signal×window pairs: (1) confirmed-gap/unresolved-gap/before-collection status checked first (read-only, `microstructure.db:gaps` + `MIN(ts_ms)`); (2) if clear, native `agg_trades` + any intersecting `ami_agg_trades_repaired` rows fetched and merged (bounded range query); (3) signed/total notional constructed; (4) mark-price return constructed from two "at-or-before" lookups; (5) the frozen ratio computed; (6) `feature_available_ts_ms = signal_birth_ts` recorded; (7) quality-state row recorded regardless of outcome (usable or excluded); (8) if construction was impossible or gap-affected, an exclusion row with an immutable reason code was written instead of a feature row — never both, never neither (`test_signal_window_pair_never_in_both_windowed_flow_and_exclusions`).

No post-availability information entered any feature: every trade/mark-price row consumed is independently re-verified against `[window_start_ts_ms, window_end_ts_ms]` inside `fetch_window_trades` (raises `KnownAtViolation` on any violation, defensive re-check beyond the SQL `WHERE` clause itself) and the "at-or-before" mark-price query is structurally incapable of returning a future row. **`known_at_violations = 0`** across all 1,620 pairs, both runs.

---

## Numerical-stability validation (Phase 5)

| Condition | Result |
|---|---|
| Zero signed flow | Handled — floor applied, finite result, no crash (unit test + real-data: 0/1,619 real rows had exactly zero signed notional) |
| Near-zero signed flow | Handled — floor applied continuously (unit test) |
| One-sided flow (all BUY or all SELL) | Handled naturally by the signed-sum construction (no special case needed) |
| Price response exactly zero | Handled — `mark_return_bps=0` when `mark_price(T)==mark_price(T-W)` (unit test) |
| Crossed/invalid price data | Not encountered in real data (`mark_price_start`/`mark_price_end` both always resolved for all 1,619 usable rows — 0 `NULL` prices) |
| Duplicated agg trades | Handled — dedup by `agg_trade_id`, repaired-wins-on-conflict (unit tests, both a genuine content conflict and an identical-content duplicate) |
| Out-of-order timestamps | Handled — construction is order-independent (dict/sum-based, not sequential-dependent); proven by inserting the same 3 trades in reverse order and confirming identical output |
| Gaps | Handled — confirmed/unresolved gap overlap checked before any feature is computed |
| Boundary timestamps | Handled — both ends of `[T-W,T]` inclusive, verified exactly at the boundary (trade at `T-W` and at `T` both included; trades one ms outside excluded) |
| Extreme but valid values | **Not clipped or winsorized** — the frozen contract does not require it, and none was applied (max observed `\|signed_notional\|` at W60 = $78,936,758.21, retained as-is) |
| Floating-point overflow/underflow | Not observed; all values well within IEEE-754 double range for this notional/bps scale |
| Sign consistency | Verified via the manual hand-calculated fixture (`test_manual_fixture_parity_hand_calculated`) — signed notional, mark return, and the resulting ratio all match hand arithmetic exactly |

No value was winsorized, clipped, or discarded for being extreme — consistent with the frozen contract's silence on any such requirement.

---

## Evidence-layer separation (Phase 6)

Only the native `agg_trades`-derived (`EXACT`) representation was rehearsed, per the contract (book-depth proxying is ruled `LOW_FIDELITY_PROXY_ONLY` and explicitly **not** proposed as primary evidence — no book-depth rehearsal was performed, and none was required by the contract). The schema enforces this structurally: `absorption_impact_windowed_flow.evidence_layer` and `absorption_impact_window_quality_v1.evidence_layer` both carry `CHECK (evidence_layer = 'EXACT')` — an attempted `'PROXY'` insert raises `sqlite3.IntegrityError` (`test_schema_rejects_non_exact_evidence_layer`). No proxy table exists in this rehearsal to silently fall back to, and the code contains no reference to `book_ticker` at all (`test_no_silent_proxy_fallback_structural`, static source scan).

---

## Quality taxonomy (Phase 7)

Every one of the 1,620 pairs carries exactly one `absorption_impact_window_quality_v1` row (`UNIQUE(signal_id, window_id, quality_contract_version)`, violated duplicate insert proven to raise `sqlite3.IntegrityError`) recording: quality state, evidence layer (`EXACT` only), confirmed/unresolved gap flags, before-collection flag, repaired/native row counts used, and an assessment timestamp. **Disclosed limitation** (already flagged in the readiness audit, not newly discovered): this rehearsal's `classify_quality()` implements a **conservative, gap-ledger-only** approximation — it does not implement the full CVD-style per-minute cadence-proof/duplicate-unresolved machinery (`cvd_source_quality_contract_v1.py`'s five-status decision function in full). A row is called `EXACT_RECONSTRUCTABLE` here only on the basis of "no known collector gap overlaps this window," which is real, verified evidence but a narrower proof than CVD's full contract. This is reported honestly, not silently upgraded to the stronger claim.

Quality breakdown (both runs, identical):

| `quality_status` | Count |
|---|---|
| `EXACT_RECONSTRUCTABLE` | 1,619 |
| `SOURCE_GAPPED` | 1 |
| `PROXY_ONLY` | 0 |
| `SOURCE_COVERAGE_UNRESOLVED` | 0 |
| `UNREPAIRABLE` | 0 |

No row belongs to two terminal quality partitions (enforced by the same `UNIQUE` constraint, and independently proven by `test_signal_window_pair_never_in_both_windowed_flow_and_exclusions`).

---

## Row accounting (Phase 8) — exact, outcome-blind, reconciles every row

Reconciliation equation, per window: `candidate anchor universe (324) = usable constructed rows + immutable exclusions + quarantined/unresolved rows`.

| Window | Usable | Excl. (before collection) | Excl. (confirmed gap) | Excl. (unresolved gap) | Reconciled | Quality rows |
|---|---|---|---|---|---|---|
| W60 | 324 | 0 | 0 | 0 | 324 | 324 |
| W300 | 324 | 0 | 0 | 0 | 324 | 324 |
| W600 | 324 | 0 | 0 | 0 | 324 | 324 |
| W1800 | 324 | 0 | 0 | 0 | 324 | 324 |
| W3600 | 323 | 0 | 1 | 0 | 324 | 324 |
| **Total (5 windows)** | **1,619** | **0** | **1** | **0** | **1,620** | **1,620** |

Additional accounting:

| Metric | Value |
|---|---|
| Total signal×window pairs | 1,620 (324 signals × 5 windows) |
| Constructed feature rows (`EXACT`) | 1,619 |
| Proxy-only rows | 0 (no proxy representation rehearsed) |
| Source-gapped rows | 1 (W3600 only, 1 LONG signal) |
| Timestamp-invalid rows | 0 |
| Duplicate/conflict rows | 0 in the real population (unit-tested separately with synthetic duplicates, since none occur in the real 8 repair spans intersecting 0 of the 324 signals' windows) |
| Zero/near-zero-denominator rows (floor applied) | 0 |
| Zero signed-notional rows | 0 |

Exact and proxy populations were never added together (no proxy population exists in this rehearsal to sum).

---

## Known-at proof (Phase 9)

- `known_at_violations = 0` (both runs, all 1,620 pairs).
- `feature_available_ts_ms` deterministic and equal to `signal_birth_ts` for every row (schema `CHECK`-enforced, cannot be violated by construction).
- No future trade or mark-price row was read — the "at-or-before" mark-price query and the explicit range-bound trade query, plus the defensive in-code re-check, jointly guarantee this.
- **No outcome table was opened; no outcome column was selected.** Proven with a real SQLite authorizer (`install_outcome_access_guard`, `SQLITE_DENY` on any reference to `ami_lifecycle_path_observations` or `endpoint_return_bps`/`mfe_bps`/`mae_bps`), installed on the live read-only `canonical.sqlite` connection for the entire rehearsal run: **`outcome_access_violations = []` (0 attempts, let alone successes)**. A static-source-scan test additionally confirms no `.execute()`-style call anywhere in the module's source ever names the outcome table or an outcome column.
- No post-signal data is included beyond what the frozen contract explicitly permits (pre-birth `[T-W,T]` only).

---

## Idempotency and parity (Phase 10)

Two independent full reconstructions were run from the identical real-data source state (`canonical.sqlite` sha256 `25a56a98d0…`, unchanged between and after both runs):

| Check | Run 1 | Run 2 | Match |
|---|---|---|---|
| Total pairs / usable / excluded | 1,620 / 1,619 / 1 | 1,620 / 1,619 / 1 | ✅ identical |
| Content hash, `absorption_impact_windowed_flow` (bookkeeping timestamp columns excluded) | `f7c834cc…` | `f7c834cc…` | ✅ identical |
| Content hash, `absorption_impact_window_quality_v1` | `5d1a205c…` | `5d1a205c…` | ✅ identical |
| Content hash, `absorption_impact_exclusions` | `5e3ae2e5…` | `5e3ae2e5…` | ✅ identical |
| Overall | **`REBUILD_IDENTICAL`** | | |

**Note on the disposable `.sqlite` file-level hashes** (`run1_sha256`/`run2_sha256` in `manifest.json`): these differ (`b42972c7…` vs `e842f694…`) because each row's `created_ms`/`assessed_at_ms` bookkeeping column captures the wall-clock time of that specific run — this is expected and is exactly why `content_hash_of_disposable()` excludes those columns from the *content* comparison above (identical discipline to `ami/warehouse/experiment_ledger.py`'s `_VOLATILE_BOOKKEEPING_COLUMNS`). The scientific/structural content is byte-identical; only wall-clock bookkeeping differs.

**Manual fixture parity:** a hand-calculated 3-trade example (2 SELL + 1 BUY, net -$1,500,000 signed notional, -50bps mark move over 10 minutes) reproduces the module's output to floating-point precision: `signed_notional=-1,500,000`, `mark_return_bps=-50.0`, `price_response = -50.0 / 1.5 = -33.333...` — confirmed exactly (`test_manual_fixture_parity_hand_calculated`).

---

## Source-gap reconciliation (Phase 11)

The readiness audit (commit `fc1321f5`) reported: complete native ETHUSDT coverage through 1800 seconds across 324 signals, one exclusion at 3600 seconds. This rehearsal **independently reproduced that result from the frozen source manifest**, not by assuming the prior document was correct: `test_real_data_coverage_reconciliation_matches_readiness_audit` re-derives the exact same exclusion counts `{60: 0, 300: 0, 600: 0, 1800: 0, 3600: 1}` from a fresh read of `ami_signal_lifecycle` + `microstructure.db:gaps`, and the full rehearsal run above reproduces the identical 323/324 split at W3600. **No discrepancy found — the readiness audit's coverage claim is confirmed, not merely repeated.**

---

## Required tests — all 25 areas covered, 26/26 passed

| # | Area | Test(s) |
|---|---|---|
| 1 | Deterministic signed-notional construction | `test_signed_notional_construction_deterministic` |
| 2 | Buy/sell sign convention | `test_taker_sign_convention` |
| 3 | Deterministic price-response construction | `test_price_response_exact_frozen_formula` |
| 4 | Exact frozen formula | `test_price_response_exact_frozen_formula`, `test_manual_fixture_parity_hand_calculated` |
| 5 | Zero denominator | `test_zero_signed_notional_floor_applied_no_crash` |
| 6 | Near-zero denominator | `test_near_zero_denominator_floor_applied` |
| 7 | Zero price response | `test_zero_price_response_when_price_unchanged` |
| 8 | Missing agg-trade interval | `test_missing_agg_trade_interval_zero_trades` |
| 9 | Partial source gap | `test_partial_source_gap_overlap_detected` |
| 10 | Duplicate trade | `test_duplicate_trade_repaired_wins_no_double_count`, `test_duplicate_trade_identical_content_not_flagged_as_conflict` |
| 11 | Out-of-order source rows | `test_out_of_order_rows_do_not_affect_result` |
| 12 | Boundary timestamp inclusion/exclusion | `test_boundary_timestamps_both_inclusive` |
| 13 | Feature availability timestamp | `test_feature_available_ts_equals_signal_birth_ts_in_schema` |
| 14 | Known-at enforcement | `test_known_at_violation_raises_on_out_of_bound_trade` |
| 15 | Outcome table access denied | `test_outcome_table_access_raises`, `test_outcome_column_access_raises_even_via_different_table_alias`, `test_rehearsal_functions_never_execute_sql_naming_the_outcome_table` |
| 16 | Exact/proxy non-pooling | `test_schema_rejects_non_exact_evidence_layer` |
| 17 | No silent proxy fallback | `test_no_silent_proxy_fallback_structural` |
| 18 | Quality partition uniqueness | `test_quality_partition_uniqueness_constraint`, `test_signal_window_pair_never_in_both_windowed_flow_and_exclusions` |
| 19 | Row-accounting reconciliation | `test_row_accounting_reconciles_synthetic_population` |
| 20 | Independent rerun parity | `test_full_real_data_rehearsal_idempotent_and_known_at_clean` |
| 21 | Manual fixture parity | `test_manual_fixture_parity_hand_calculated` |
| 22 | Existing 324-anchor source-coverage reconciliation | `test_real_data_coverage_reconciliation_matches_readiness_audit` |
| 23 | Live canonical DB unchanged | `test_full_real_data_rehearsal_idempotent_and_known_at_clean` (in-test) + standalone rehearsal script proof |
| 24 | No experiment/result/nullifier delta | `test_real_data_no_experiment_result_nullifier_delta` |
| 25 | Runtime/risk/execution protected delta = 0 | verified via `git status` (see transition proof) — no test file imports or references `execution/`, `risk/`, `brain/`, `.env`, or the live executor |

**Final: 26/26 passed** (25 required areas + 1 additional structural guard).

---

## Remaining risks

1. The quality classification in this rehearsal is a **conservative gap-ledger-only approximation**, not the full CVD-style cadence-proof/duplicate-unresolved machinery. A future canonical migration (stage A5) should decide whether this narrower proof is sufficient for `EXACT_RECONSTRUCTABLE` or whether the fuller CVD-style contract should be ported over first.
2. `FLOOR_USD_M=0.01` is frozen from data but has never actually bound on real data (0/1,619 rows) — its correctness at truly near-zero flow remains untested against a real example (only synthetically).
3. The denominator's absolute-value convention (not signed) is implemented exactly as contracted, but remains a disclosed methodological choice for a future preregistration to revisit if desired (not a defect).
4. This rehearsal used only `agg_trades`+`mark_prices`; no cross-symbol (BTC/SOL) context was rehearsed, consistent with the frozen contract's ETHUSDT-only scope.

## Recommendation for the next gate

The population is fully constructed, quality-classified, known-at-clean, outcome-blind, and reproducible. The next controlled step (per the contract's own Phase 9 stage plan) is **A4 — row-accounting freeze** (a formally hash-stamped freeze of this exact accounting), followed by operator review of whether the conservative quality-classification approximation (Remaining risk #1) needs strengthening before **A5 — controlled canonical migration**. No research execution (A6-A8) should begin before A4/A5 are complete.

---

## Success verdict

**`ABSORPTION_IMPACT_REHEARSAL_READY_FOR_ROW_ACCOUNTING_FREEZE`**

All required conditions met: deterministic implementation ✅, exact row reconciliation ✅ (1,620 = 1,619 + 1 + 0, every window), `known_at_violations = 0` ✅, outcome reads = 0 ✅ (SQLite-authorizer-proven), no exact/proxy pooling ✅ (schema-enforced), reproducible identical rerun ✅ (content-hash-proven), live canonical state unchanged ✅, all 26 focused tests green ✅.

**`CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1_COMPLETE`**

Stopping after rehearsal. No row-accounting freeze, canonical migration, preregistration, or execution begins without new operator instruction.
